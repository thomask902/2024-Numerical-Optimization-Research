import torch
from torch.distributed import ReduceOp
import contextlib


class GAM_nonaccel(torch.optim.Optimizer):
    # init is required for any custom optimizer class
    def __init__(self, params, base_optimizer, model, rho_scheduler=None,
                 adaptive=False, perturb_eps=1e-12, args=None,
                 grad_reduce='mean', **kwargs):
        # Set up the defaults dictionary with the adaptive flag = false and any additional hyper parameter arguments (e.g. weight decay)
        defaults = dict(adaptive=adaptive, **kwargs)

        # Initialize the base class (torch.optim.Optimizer) with the model parameters and default dictionary
        super(GAM_nonaccel, self).__init__(params, defaults)

        # initilizing instance variables/direct attributes
        self.rho_scheduler = rho_scheduler
        self.perturb_eps = perturb_eps # small number to stop division by zero
        self.model = model
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups # param groupings will be defined as they are in base optimizer
        self.adaptive = adaptive
        self.args = args

        # gam non-accelerated parameters
        self.rho = args.rho
        self.alpha = args.alpha

        # uses get grad reduce to check the passed gradient reduction type (mean or sum)
        self.get_grad_reduce(grad_reduce)

        # sets rho scheduler for gradient and gradient norm if there is one
        self.update_rho_t()

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

    # updates rho "radius/ball" for rho if scheduler exists
    @torch.no_grad() # Stops gradient from being automatically calculated and tracked by pytorch. 
                     # Important for performing updates to model parameters or other tensors but do not need to compute gradients
    def update_rho_t(self):
        if self.rho_scheduler is not None:
            self.rho = self.rho_scheduler.step()

    # calculates an approximation of the gradient of the norm of the gradient, f_t
    def grad_norm_grad(self):
        # Compute gradient vector
        grad_vec = torch.cat([p.grad.contiguous().view(-1) for p in self.model.parameters()])

        # Compute gradient vector norm
        grad_vec_norm = torch.norm(grad_vec)

        # Compute Hessian-vector product
        hessian_vec_prod_dict = torch.autograd.grad(
            grad_vec, self.model.parameters(), grad_outputs=grad_vec, only_inputs=True
        )

        # Concatenate the results over the different components of the network
        hessian_vec_prod = torch.cat([g.contiguous().view(-1) for g in hessian_vec_prod_dict])

        # Compute f_t and its norm
        f_t = hessian_vec_prod / grad_vec_norm

        return grad_vec, f_t

    # perturbs weights based on f_t and returns the perturbation vector rho * f_t / ||f_t||
    @torch.no_grad()
    def perturb_weights(self, f_t):
        f_t_norm = torch.norm(f_t)

        # Compute perturbation scale
        scale = self.rho / (f_t_norm + self.perturb_eps)

        perturbation_vec = f_t * scale

        # Apply perturbation to the model parameters
        start_idx = 0
        for p in self.model.parameters():
            numel = p.numel()
            perturbation = perturbation_vec[start_idx:start_idx + numel].view_as(p)
            p.add_(perturbation)
            start_idx += numel

        return perturbation_vec

    # takes perturbation vectors and returns weights to original values
    @torch.no_grad()
    def unperturb(self, perturbation_vec):
        start_idx = 0
        for p in self.model.parameters():
            numel = p.numel()
            perturbation = perturbation_vec[start_idx:start_idx + numel].view_as(p)
            p.sub_(perturbation)
            start_idx += numel

    # performs scaled combination of calculated gradients and sets p.grad values to new gradient
    def set_gradients(self, g_0, f_t_adv):
        combined_grad = g_0 + self.alpha * self.rho * f_t_adv

        # Set the gradients manually for optimizer step
        start_idx = 0
        for p in self.model.parameters():
            numel = p.numel()
            p.grad = combined_grad[start_idx:start_idx + numel].view_as(p).clone()
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
            # calculate oracle loss gradient/gradient at original weights (g_0)
            outputs, loss_value = get_grad()

            # calculate g_0 and ft (norm of gradient norm, ft = hessian@w * grad@w / ||grad@w||)
            g_0, f_t = self.grad_norm_grad()

            # adjusts weights to the GAM adversarial point using ft, w_adv (max gradient norm within rho of weights, weights_adv = weights + rho * ft/||ft||)
            perturbation_vec = self.perturb_weights(f_t)

            # clear gradient manually to ensure graph is not being tracked
            for p in self.model.parameters():
                p.grad = None

            # get gradient from GAM adv weights (g_adv)
            get_grad()

            # calculate ft again from adversarial weights
            g_adv, f_t_adv = self.grad_norm_grad()

            # print("difference in f_t:", torch.norm(f_t_adv - f_t))

            # reverse perturbation from GAM adv point, returning parameter tensors to their original value
            self.unperturb(perturbation_vec)

            # combine to find g_2 = g_0 + alpha * g_1 (g_1 = rho * ft@w_adv) and set to p.grad
            self.set_gradients(g_0, f_t_adv)

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

    # def add_param_group(self, param_group):
    #     self.base_optimizer.add_param_group(param_group)

    def __repr__(self):
        return f'GAM({self.base_optimizer.__class__.__name__})'
