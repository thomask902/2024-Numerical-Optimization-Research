import torch
# from .utils import enable_running_stats, disable_running_stats
from torch.distributed import ReduceOp
import contextlib


class GAM(torch.optim.Optimizer):
    # init is required for any custom optimizer class
    def __init__(self, params, base_optimizer, model, grad_rho_scheduler=None,
                 grad_norm_rho_scheduler=None, adaptive=False, perturb_eps=1e-12, args=None,
                 grad_reduce='mean', **kwargs):
        # Set up the defaults dictionary with the adaptive flag = false and any additional hyper parameter arguments (e.g. weight decay)
        defaults = dict(adaptive=adaptive, **kwargs)

        # Initialize the base class (torch.optim.Optimizer) with the model parameters and default dictionary
        super(GAM, self).__init__(params, defaults)

        # initilizing instance variables/direct attributes
        self.grad_rho_scheduler = grad_rho_scheduler
        self.grad_norm_rho_scheduler = grad_norm_rho_scheduler
        self.perturb_eps = perturb_eps # small number to stop division by zero
        self.model = model
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups # param groupings will be defined as they are in base optimizer
        self.adaptive = adaptive
        self.args = args

        # adding for local run
        self.grad_rho = args.grad_rho
        self.grad_norm_rho = args.grad_norm_rho

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

    # updates rho "radius/ball" for both the gradient and grad norm if scheduler exists
    @torch.no_grad() # Stops gradient from being automatically calculated and tracked by pytorch. 
                     # Important for performing updates to model parameters or other tensors but do not need to compute gradients
    def update_rho_t(self):
        if self.grad_rho_scheduler is not None:
            self.grad_rho = self.grad_rho_scheduler.step()
        if self.grad_norm_rho_scheduler is not None:
            self.grad_norm_rho = self.grad_norm_rho_scheduler.step()

    # used to perturb weights at original point or adversarial, saves original gradient value and perturbation value
    @torch.no_grad()
    def perturb_weights(self, perturb_idx: int):
        # compute gradient norm
        grad_norm = self._grad_norm(weight_adaptive=self.adaptive)

        # computing constant to scale gradient by
        scale = self.grad_rho / (grad_norm + self.perturb_eps)

        # save first set of gradients g_0, and do the first perturbation to get e_w_0 (adversarial weights)
        if perturb_idx == 0:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    self.state[p]["g_0"] = p.grad.data.clone() # save current gradients as g_0
                    e_w = p.grad * scale.to(p) # compute perturbation of gradient norm (scale.to ensures they are on same device)
                    if self.adaptive: # scales differently if self adaptive mode is on
                        e_w *= torch.pow(p, 2)
                    p.add_(e_w) # add pertubation, p is now at SAM adversarial point (max loss within rho of weights)
                    self.state[p]['e_w_0'] = e_w

        # save second set of gradients g_2, and apply perturbation at adversarial weights e_w_1_2
        elif perturb_idx == 1:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    self.state[p]["g_2"] = p.grad.data.clone() # save grad at grad ascent adv point as g_2
                    
                    # scales by rho in direction of gradient from the grad ascent adverse point 
                    e_w = p.grad * scale.to(p)
                    if self.adaptive: # scales differently if self adaptive mode is on
                        e_w *= torch.pow(p, 2)
                    p.add_(e_w)
                    self.state[p]['e_w_1_2'] += e_w # adds this perturbation to the previous grad ascent one

        # raise error if neither set of gradients declared
        else:
            raise ValueError('"perturb_idx" should be one of [0, 1].')

    
    @torch.no_grad()
    def grad_norm_ascent(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["g_1"] = p.grad.data.clone() # save current gradients as g_1, as these are gradients at SAM adv point
                p.grad.data -= self.state[p]["g_0"] # subtract original gradient to get change from the SAM adv point adjustment (this is the manual calculation of the gradient)

        # compute gradient norm of (g_1 - g_0)
        grad_norm = self._grad_norm(weight_adaptive=self.adaptive)

        # scaling by the gradient norm rho, this is our very small perturbation to compute the manual derivative/gradient
        scale = self.grad_norm_rho / (grad_norm + self.perturb_eps)

        # loop through and adjust perturbation, apply to parameter and save as e_w_1_2
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                if self.adaptive:
                    e_w *= torch.pow(p, 2)
                p.add_(e_w)
                self.state[p]['e_w_1_2'] = e_w

    # subtracts pertubation given in perturb_key parameter from each param p, reversing adversarial weight adjustment
    @torch.no_grad()
    def unperturb(self, perturb_key: str):
        for group in self.param_groups:
            for p in group['params']:
                if perturb_key in self.state[p].keys():
                    p.data.sub_(self.state[p][perturb_key])

    # performs a weighted sum of the calculated gradients to determine update direction
    # grad_beta_1/2/3 defined for each model and dataset in main_cifar
    @torch.no_grad()
    def gradient_decompose(self, args=None):
        inner_prod = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                # defines pro_m as weighted sum of original gradient g_0 and beta * gradient (grad ascent adv point)
                self.state[p]['pro_m'] = self.state[p]['g_0'] + abs(args.grad_beta_2) * self.state[p]['g_2']

                # creates weighted sum of g_1 (grad at SAM adv point) and the current gradient at the perturbed grad ascent point
                p.grad.data = args.grad_beta_1 * self.state[p][
                    "g_1"] + args.grad_beta_3 * p.grad.data.detach().clone()

                # calculates the inner product between the two scaled gradients
                inner_prod += torch.sum(
                    self.state[p]['pro_m'] * p.grad.data
                )

        # gets norm of scaled gradient between SAM adv point and perturbed grad ascent point (as done in the paper)
        new_grad_norm = self._grad_norm()

        # gets norm of the gradient at the point scaled between original gradient and grad ascent adv point
        old_grad_norm = self._grad_norm(by='pro_m')

        # gets cosine similarity of the gradient and pro_m
        cosine = inner_prod / (new_grad_norm * old_grad_norm + self.perturb_eps)

        # gradient decomposition (helps convergence)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                # vertical component represents the part of the gradient that is orthogonal to the direction of pro_m
                vertical = self.state[p]['pro_m'] - cosine * old_grad_norm * p.grad.data / (
                        new_grad_norm + self.perturb_eps)
                # moves in a direction not just aligned with pro_m, potentially avoiding local minima or saddle points
                p.grad.data.add_(vertical, alpha=-args.grad_gamma)

    # function declared to calculate the gradient norm
    @torch.no_grad()
    def _grad_norm(self, weight_adaptive: bool = False, by: str = 'grad'):
        norm = 0.0
        # iterating over each group of params and each p=param in group
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                if by == 'grad':
                    g = p.grad.data
                elif by == 'pro_m':
                    g = self.state[p]['pro_m']
                # elif by == 'e_w':
                #     g = self.state[p]['e_w_0'] + self.state[p]['e_w_1_2'] + self.state[p]['e_w_2']
                elif by == 'p':
                    g = p.data
                else:
                    raise ValueError("Invalid 'by' argument in _grad_norm")

                if weight_adaptive:
                    norm += torch.sum((g * torch.abs(p.data)) ** 2)
                else:
                    norm += torch.sum(g ** 2)

        return torch.sqrt(norm)

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
            loss.backward()
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

            # adjusts weights to the SAM adversarial point (max loss within rho of weights, weights_adv = weights + rho * grad/||grad||)
            self.perturb_weights(perturb_idx=0)

            # disable running stats for second pass
            # disable_running_stats(self.model)

            # gets gradient from SAM adv weights (g_1)
            get_grad()

            # reverses perturbation from SAM adv point, returning parameter tensors to their original value
            self.unperturb(perturb_key="e_w_0")

            # second interation perturbation in direction of g_1 - g_0
            self.grad_norm_ascent()

            # gets gradient a third time from "grad ascent adv point"
            get_grad()
            
            # perturbs in direction of gradient from the "grad ascent adv point"
            self.perturb_weights(perturb_idx=1)
            
            # gets gradient a fourth time from the perturbed grad ascent adv point
            get_grad()

            # decompose and get new update direction
            self.gradient_decompose(args=self.args)

            # unperturb from perturbed grad ascent point back to original point
            self.unperturb(perturb_key="e_w_1_2")

        # synchronize gradients across workers
        self._sync_grad()

        # update with new directions
        self.base_optimizer.step()

        # enable running stats
        # enable_running_stats(self.model)

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
