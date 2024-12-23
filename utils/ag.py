import torch

class AG(torch.optim.Optimizer):

    def __init__(self, params, model, loss_type, lipschitz, reg=0.0, stochastic=False, args=None):
        defaults = dict()
        super(AG, self).__init__(params, defaults)

        self.model = model

        # loss
        self.loss_type = loss_type
        self.convex = True # default
        if(self.loss_type == "hinge"):
            self.convex = True
        elif(self.loss_type == "sigmoid"):
            self.convex = False

        # stochasticity
        self.stochastic = stochastic

        # initial parameter values
        self.lipschitz = lipschitz
        self.alpha_k = 1.0

        if self.stochastic and not self.convex:
            self.beta_k = 8.0 / (21.0 * self.lipschitz)
        else:
            self.beta_k = 1.0 / (2.0 * self.lipschitz)
        
        if self.convex and not self.stochastic:
            self.lambda_k = (self.beta_k / 2.0)
        elif self.convex and self.stochastic:
            self.lambda_k = (self.lipschitz * self.beta_k ** 2) / 2.0
        else:
            self.lambda_k = self.beta_k
        self.reg = reg

        # other parameters if needed
        self.args = args

    def update_k(self):
        # update alpha
        current_alpha = self.alpha_k
        self.alpha_k = (2.0 * current_alpha) / (2 + current_alpha)

        # update lambda if needed
        if self.convex and not self.stochastic:
            current_lambda = self.lambda_k
            self.lambda_k = (2.0 * current_lambda + self.beta_k) / 2.0
        elif self.convex and self.stochastic:
            current_lambda = self.lambda_k
            self.lambda_k = current_lambda + 0.5 * self.lipschitz * self.beta_k ** 2

    # closure re-evaluates the function and returns loss, used in algos with multiple evaluations of objective function
    def set_closure(self, loss_fn, inputs, targets, create_graph=True, enable_reg=True, **kwargs):

        def get_grad():
            self.zero_grad()
            outputs = self.model(inputs)
            loss = loss_fn(outputs, targets, **kwargs)

            if self.reg != 0.0 and enable_reg:
                # Add L2 regularization term
                #print(f"m = {self.args.m}, lambda (reg) = {self.reg}")
                l2_reg = torch.tensor(0.0, device=loss.device)
                for param in self.model.parameters():
                    l2_reg += torch.norm(param, 2) ** 2
                #print(f"Regularization Weight Added = {(self.reg / 2) * l2_reg}")
                loss += (self.reg / 2) * l2_reg

            loss_value = loss.data.clone().detach()
            loss.backward(create_graph=create_graph)
            return outputs, loss_value

        self.forward_backward_func = get_grad

    # uses closure as argument, calling it internally multiple times as needed, this is why closure is used
    def step(self, closure=None):

        # TESTING
        #print(f'Alpha k: {self.alpha_k}, Beta k: {self.beta_k}, Lambda k: {self.lambda_k}')

        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_backward_func


        # initialize state if not already initialized, this will store our actual and accelerated information
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    # State initialization
                    state['x_k'] = p.data.clone()  # x_k-1
                    state['x_ag_k'] = p.data.clone()  # x^ag_k-1

        # retreive x_k-1 and x^ag_k-1 for each parameter
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                x_k_minus1 = state['x_k']
                x_ag_k_minus1 = state['x_ag_k']

                # compute x_md_k = (1 - alpha_k) * x^ag_k-1 + alpha_k * x_k-1
                x_md_k = (1 - self.alpha_k) * x_ag_k_minus1 + self.alpha_k * x_k_minus1
                state['x_md_k'] = x_md_k.clone()

                # set parameter p to x_md_k for gradient computation
                p.data.copy_(x_md_k)


        # compute gradient at x^md_k (get_grad)
        outputs, loss_value = get_grad()
        x_md_norm = self.grad_norm()

        # now need to update x_k and x^md_k
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                x_k_minus1 = state['x_k']
                x_ag_k_minus1 = state['x_ag_k']
                x_md_k = state['x_md_k']
                grad = p.grad.data

                # update x_k and x_ag_k
                with torch.no_grad():
                    x_k = x_k_minus1 - self.lambda_k * grad
                    x_ag_k = x_md_k - self.beta_k * grad

                # update state
                state['x_k'] = x_k.clone()
                state['x_ag_k'] = x_ag_k.clone()

                # set model parameters to x_ag_k
                with torch.no_grad():
                    p.data.copy_(x_ag_k)

                # set model parameters to x_k to find gradient norm
                #with torch.no_grad():
                #    p.data.copy_(x_k)

        # reporting gradient norm at x_k
        #x_k_norm = self.calc_grad_norm()

        # set parameters to x_ag_k (As they should be)
        #for group in self.param_groups:
        #    for p in group['params']:
        #        state = self.state[p]
        #        x_ag_k = state['x_ag_k'].clone()

        #        # set model parameters to x_ag_k
        #        with torch.no_grad():
        #            p.data.copy_(x_ag_k)

        # set "k = k+1" by updating parameters (last step)
        self.update_k()

        return outputs, loss_value, x_md_norm, 0.0 #x_k_norm[1]
    
    def grad_norm(self):
        grad_norm = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grad_norm += p.grad.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5  
        return grad_norm

    def calc_grad_norm(self):
        outputs, loss_value = self.forward_backward_func()
        grad_norm = self.grad_norm()
        return loss_value, grad_norm

    def calc_x_md_grad_norm(self):
        # set parameters to x_md_k
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                x_md_k = state['x_md_k'].clone()

                # set model parameters to x_ag_k
                with torch.no_grad():
                    p.data.copy_(x_md_k)

        # take gradient at x_md_k
        outputs, loss_value = self.forward_backward_func()
        grad_norm = self.grad_norm()

        # set back to x_ag_k
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                x_ag_k = state['x_ag_k'].clone()

                # set model parameters to x_ag_k
                with torch.no_grad():
                    p.data.copy_(x_ag_k)
        return loss_value, grad_norm

    def zero_grad(self):
        super().zero_grad(set_to_none=False)


    def __repr__(self):
        return (
            f"Model: {repr(self.model)} \n"
            f"Loss Function: {self.args.loss} \n"
            f"Loss function convex: {self.convexity} \n"
            f"Lipschitz Constant of Loss Function: {self.args.lipschitz}"
        )