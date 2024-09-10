import torch
from torch.distributed import ReduceOp
import contextlib
import time


class newtonMR(torch.optim.Optimizer):
    # init is required for any custom optimizer class
    def __init__(self, params, base_optimizer, model, rho_scheduler=None,
                 adaptive=False, perturb_eps=1e-12, args=None,
                 grad_reduce='mean', **kwargs):
        # Set up the defaults dictionary with the adaptive flag = false and any additional hyper parameter arguments (e.g. weight decay)
        defaults = dict(adaptive=adaptive, **kwargs)

        # Initialize the base class (torch.optim.Optimizer) with the model parameters and default dictionary
        super(newtonMR, self).__init__(params, defaults)

        # initilizing instance variables/direct attributes
        self.rho_scheduler = rho_scheduler
        self.perturb_eps = perturb_eps # small number to stop division by zero
        self.model = model
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups # param groupings will be defined as they are in base optimizer
        self.adaptive = adaptive
        self.args = args

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

    # calculates hessian of the loss function
    def hessian(self, grads):

        hessian = torch.zeros(grads.shape[0], grads.shape[0])
        
        for idx, grad in enumerate(grads):
            second_der = torch.autograd.grad(grad, self.model.parameters(), retain_graph=True, allow_unused=True)
            second_der = torch.cat(([grad.flatten() for grad in second_der]))
            hessian[idx, :] = second_der
        
        return hessian

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
            # calculate oracle loss gradient/gradient at original weights (g_0)
            outputs, loss_value = get_grad()

            # calculate g (gradient of gradient norm squared, g = hessian@w * grad@w)
            g = torch.cat([p.grad.flatten() for p in self.model.parameters()])

            # get Hessian
            start_time = time.time()
            H = self.hessian(g)
            end_time = time.time()
            print("Hessian Calculation Took", end_time - start_time, "seconds")

            # take pseudo-inverse of the hessian
            H_inv = torch.linalg.pinv(H)
            start_time2 = time.time()
            H = self.hessian(g)
            end_time2 = time.time()
            print("Hessian Pseduo-Inverse Took", end_time2 - start_time2, "seconds")

            # multiply negative pseudo-inverse of hessian with gradient
            update_dir = - torch.matmul(H_inv, g)

            # clear gradient manually to ensure graph is not being tracked
            for p in self.model.parameters():
                p.grad = None

            # set gradient to update direction
            self.set_gradients(update_dir)

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
        return f'newtonMR({self.base_optimizer.__class__.__name__})'