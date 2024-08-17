import torch
from torch.distributed import ReduceOp
import contextlib
import numpy as np


class GNOM_noised(torch.optim.Optimizer):
    # init is required for any custom optimizer class
    def __init__(self, params, base_optimizer, model, rho_scheduler=None,
                 adaptive=False, perturb_eps=1e-12, args=None,
                 grad_reduce='mean', **kwargs):
        # Set up the defaults dictionary with the adaptive flag = false and any additional hyper parameter arguments (e.g. weight decay)
        defaults = dict(adaptive=adaptive, **kwargs)

        # Initialize the base class (torch.optim.Optimizer) with the model parameters and default dictionary
        super(GNOM_noised, self).__init__(params, defaults)

        # initilizing instance variables/direct attributes
        self.rho_scheduler = rho_scheduler
        self.perturb_eps = perturb_eps # small number to stop division by zero
        self.model = model
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups # param groupings will be defined as they are in base optimizer
        self.adaptive = adaptive
        self.noise_threshold = args.noise_threshold
        self.noise_radius = args.noise_radius
        self.args = args

        # Add counters for tracking noise addition
        self.noise_applied_count = 0
        self.total_batches = 0

        # uses get grad reduce to check the passed gradient reduction type (mean or sum)
        self.get_grad_reduce(grad_reduce)


    # checks argument to see if sum or mean have been declared as reduction type and sets them, throws error if neither
    def get_grad_reduce(self, grad_reduce: str):
        if grad_reduce.lower() == 'mean':
            if hasattr(ReduceOp, 'AVG'):
                self.grad_reduce = ReduceOp.AVG
                self.manual_average = False
            else:  # PyTorch <= 1.11.0 does not have AVG, need to manually average across processes
                self.grad_reduce = ReduceOp.SUM
                self.manual_average = True
        elif grad_reduce.lower() == 'sum':
            self.grad_reduce = ReduceOp.SUM
            self.manual_average = False
        else:
            raise ValueError('"grad_reduce" should be one of ["mean", "sum"].')

    # calculates the norm of the gradient
    def grad_norm(self):
        # Compute gradient vector
        grad_vec = torch.cat([p.grad.contiguous().view(-1) for p in self.model.parameters()])

        # Compute gradient vector norm
        grad_vec_norm = torch.norm(grad_vec)

        return grad_vec_norm

    @torch.no_grad()
    def perturb_weights(self):
        # Compute random number U from the uniform distribution [0, 1]
        U = np.random.uniform(0, 1)

        # Compute d random numbers from the multivariate normal distribution Y ~ N(0, I_d)
        d = sum(p.numel() for p in self.model.parameters()) 
        Y = np.random.normal(0, 1, d)
        
        # Normalize the norm distribution vector
        norm_Y = np.linalg.norm(Y)
        Y_normalized = Y / norm_Y
        
        # Combine to get the sample point in the ball: radius * U^(1/d) * (Y / ||Y||)
        perturbation_vec = self.noise_radius * U**(1/d) * Y_normalized
        
        # Apply perturbation to the model parameters
        start_idx = 0
        for p in self.model.parameters():
            numel = p.numel()
            perturbation = torch.tensor(perturbation_vec[start_idx:start_idx + numel], dtype=p.dtype, device=p.device).view_as(p)
            p.add_(perturbation)
            start_idx += numel

    # calculates an approximation of the gradient of the norm of the gradient, f_t
    def grad_norm_grad(self):
        # Compute gradient vector
        grad_vec = torch.cat([p.grad.contiguous().view(-1) for p in self.model.parameters()])

        # Compute Hessian-vector product
        hessian_vec_prod_dict = torch.autograd.grad(
            grad_vec, self.model.parameters(), grad_outputs=grad_vec, only_inputs=True
        )

        # Concatenate the results over the different components of the network
        hessian_vec_prod = torch.cat([g.contiguous().view(-1) for g in hessian_vec_prod_dict])

        return hessian_vec_prod

    # performs scaled combination of calculated gradients and sets p.grad values to new gradient
    def set_gradients(self, g):
        # Set the gradients manually for optimizer step
        start_idx = 0
        for p in self.model.parameters():
            numel = p.numel()
            p.grad = g[start_idx:start_idx + numel].view_as(p).clone()
            start_idx += numel


    # syncs gradients if distributed training has been chosen
    @torch.no_grad()
    def _sync_grad(self):
        if torch.distributed.is_initialized():  # synchronize final gradients
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    if self.manual_average:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
                        world_size = torch.distributed.get_world_size()
                        p.grad.div_(float(world_size))
                    else:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
        return

    def maybe_no_sync(self):
        if torch.distributed.is_initialized():
            return self.model.no_sync()
        else:
            return contextlib.ExitStack()

    # closure re-evaluates the function and returns loss, used in algos with multiple evaluations of objective function
    @torch.no_grad()
    def set_closure(self, loss_fn, inputs, targets, **kwargs):
        # create self.forward_backward_func, which is a function such that
        # self.forward_backward_func() automatically performs forward and backward passes.

        def get_grad():
            self.zero_grad()
            with torch.enable_grad():
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets, **kwargs)
            loss_value = loss.data.clone().detach()
            loss.backward(create_graph = True)
            return outputs, loss_value

        self.forward_backward_func = get_grad

    # uses closure as argument, calling it internally multiple times as needed, this is why closure is used
    def step(self, closure=None):

        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_backward_func

        with self.maybe_no_sync():
            self.total_batches += 1

            # calculate oracle loss gradient/gradient at original weights
            outputs, loss_value = get_grad()

            # calculate gradient norm to determine if noise needs to be added
            grad_norm = self.grad_norm()

            # check if norm exceeds threshold, if not add noise
            if grad_norm < self.noise_threshold:
                self.noise_applied_count += 1
                self.perturb_weights()

            # calculate g (gradient of gradient norm squared, g = hessian@w * grad@w)
            g = self.grad_norm_grad()

            # clear gradient manually to ensure graph is not being tracked
            for p in self.model.parameters():
                p.grad = None

            # set gradient to hessian vector product, g
            self.set_gradients(g)

        # synchronize gradients across workers
        self._sync_grad()

        # update with new directions
        self.base_optimizer.step()

        return outputs, loss_value

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none)

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)

    # Method to reset counters (e.g., at the beginning of an epoch)
    def reset_counters(self):
        self.noise_applied_count = 0
        self.total_batches = 0

    # Method to get noise statistics after an epoch
    def get_noise_statistics(self):
        if self.total_batches == 0:
            return 0.0, 0
        percentage_noise = (self.noise_applied_count / self.total_batches) * 100
        return percentage_noise, self.noise_applied_count

    # def add_param_group(self, param_group):
    #     self.base_optimizer.add_param_group(param_group)

    def __repr__(self):
        return f'GNOM_noised({self.base_optimizer.__class__.__name__})'