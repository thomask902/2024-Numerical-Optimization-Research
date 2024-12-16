import torch

class AG_pf(torch.optim.Optimizer):

    def __init__(self, params, model):
        defaults = dict()
        super(AG_pf, self).__init__(params, defaults)

        self.model = model

        # INITIALIZE GLOBAL PARAMETERS
        self.k = 1
        self.gamma_1 = 0.4
        self.gamma_2 = 0.4
        self.gamma_3 = 0.5
        self.beta_hat = 1.0
        self.lambda_hat = 1.0
        self.sigma = 0.0001
        self.delta = 0.001
        self.max_iter = 50
        self.upper_lambda_prev = 0.0
        self.lambda_k_list = []


    # closure re-evaluates the function and returns loss, used in algos with multiple evaluations of objective function
    def set_closure(self, loss_fn, inputs, targets, create_graph=True, **kwargs):

        def get_grad():
            self.zero_grad()
            outputs = self.model(inputs)
            loss = loss_fn(outputs, targets, **kwargs)
            loss_value = loss.data.clone().detach()
            loss.backward(create_graph=create_graph)
            return outputs, loss_value

        self.forward_backward_func = get_grad

    # uses closure as argument, calling it internally multiple times as needed, this is why closure is used
    def step(self, closure=None):

        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_backward_func


        # initialize state if not already initialized, this will store our actual and accelerated information
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                # State initialization for first iteration
                if len(state) == 0:
                    state['x_k_prev'] = p.data.clone()
                    state['x_k'] = p.data.clone()  
                    state['x_ag_prev'] = p.data.clone()  
                    state['x_ag_k'] = p.data.clone()
                    state['x_ag_tilda'] = p.data.clone()
                    state['x_ag_bar'] = p.data.clone()
                    state['x_md_k'] = p.data.clone()  
                
                # x_k and x_ag_k need to be set to prev to be maintained throughout line search procedures
                x_k = state['x_k']
                x_ag_k = state['x_ag_k']
                state['x_k_prev'] = x_k.clone()
                state['x_ag_prev'] = x_ag_k.clone()
                
        
        # ---------------------------------------------------------------------------------------
        # Lambda Line Search

        # lambda k line search loop, tau_1 starts at 0 increasing by 1 each loop iteration
        for tau_1 in range(self.max_iter):
            # find n_k, lambda_k, upper_lambda_k, and alpha_k
            n_k = self.lambda_hat * self.gamma_1 ** tau_1
            lambda_k = (n_k + (n_k ** 2 + 4.0 * n_k * self.upper_lambda_prev) ** 0.5) / 2.0
            upper_lambda_k = sum(self.lambda_k_list) + lambda_k
            alpha_k = lambda_k / upper_lambda_k

            # find and store x_md using alpha_k
            for group in self.param_groups:
                for p in group['params']:
                    # retreive current x_k and set to state x_k-1 to retain it
                    state = self.state[p]
                    x_k_prev = state['x_k_prev']
                    x_ag_prev = state['x_ag_prev']

                    # compute x_md_k = (1 - alpha_k) * x^ag_k-1 + alpha_k * x_k-1
                    x_md_k = (1 - alpha_k) * x_ag_prev + alpha_k * x_k_prev
                    state['x_md_k'] = x_md_k.clone()

                    # set parameter p to x_md_k for gradient computation
                    with torch.no_grad():
                        p.data.copy_(x_md_k)

            # calculate loss and gradient at x_md_k
            outputs, loss_md_k_raw = get_grad()
            loss_md_k = loss_md_k_raw.item()
            grad_md_k = torch.cat([p.grad.contiguous().view(-1) for p in self.model.parameters()])

            # update x_k with this gradient
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]
                    x_k_prev = state['x_k_prev']
                    x_md_k = state['x_md_k']
                    grad = p.grad.data

                    # update x_k
                    with torch.no_grad():
                        x_k = x_k_prev - lambda_k * grad
                    state['x_k'] = x_k.clone()
                    
            # find and store x_ag_tilda using x_ag_prev and x_k
            for group in self.param_groups:
                for p in group['params']:
                    # retreive current x_k and set to state x_k-1 to retain it
                    state = self.state[p]
                    x_k = state['x_k']
                    x_ag_prev = state['x_ag_prev']

                    # compute x_ag_tilda = (1 - alpha_k) * x^ag_k-1 + alpha_k * x_k
                    x_ag_tilda = (1 - alpha_k) * x_ag_prev + alpha_k * x_k
                    state['x_ag_tilda'] = x_ag_tilda.clone()

                    # set parameter p to x_ag_tilda for gradient computation
                    with torch.no_grad():
                        p.data.copy_(x_ag_tilda)

            # calculate loss and gradient at x_ag_tilda
            outputs, loss_ag_tilda_raw = get_grad()
            loss_ag_tilda = loss_ag_tilda_raw.item()
            grad_ag_tilda = torch.cat([p.grad.contiguous().view(-1) for p in self.model.parameters()])

            # get vector form of x_k and x_k_prev
            vec_x_k = torch.cat([self.state[p]['x_k'].contiguous().view(-1) for p in self.model.parameters()])
            vec_x_k_prev = torch.cat([self.state[p]['x_k_prev'].contiguous().view(-1) for p in self.model.parameters()])

            # calculate lhs and rhs of termination condition
            lhs = loss_ag_tilda
            rhs = loss_md_k + alpha_k * torch.dot(grad_md_k, (vec_x_k - vec_x_k_prev)) + alpha_k / (2 * lambda_k) * torch.norm(vec_x_k - vec_x_k_prev) ** 2 + self.delta * alpha_k

            # print(f"lhs={lhs}, rhs={rhs}")

            if lhs <= rhs:
                # print(f"Lambda line search converged in {tau_1} iterations")

                # update previous upper case lambda, and append final lambda_k to list
                self.upper_lambda_prev = upper_lambda_k
                self.lambda_k_list.append(lambda_k)
                break
        
        # If lambda_k line search algorithm ran until max iterations
        if tau_1 == (self.max_iter - 1): 
            raise RuntimeError(f"Lambda line search did not converge within {self.max_iter} iterations")
        
        # ---------------------------------------------------------------------------------------



        # ---------------------------------------------------------------------------------------
        # Beta Line Search

        # beta k line search loop, tau_2 starts at 0 increasing by 1 each loop iteration
        for tau_2 in range(self.max_iter):
            # finding beta_k
            beta_k = self.beta_hat * self.gamma_2 ** tau_2
            
            # set parameters to x_ag_prev and find loss and gradient
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]
                    x_ag_prev = state['x_ag_prev']

                    # set parameter p to x_ag_prev for gradient computation
                    with torch.no_grad():
                        p.data.copy_(x_md_k)

            # calculate loss and gradient at x_ag_prev
            outputs, loss_ag_prev_raw = get_grad()
            loss_ag_prev = loss_ag_prev_raw.item()
            grad_ag_prev = torch.cat([p.grad.contiguous().view(-1) for p in self.model.parameters()])

            # update x_ag_bar with this gradient
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]
                    x_ag_prev = state['x_ag_prev']
                    grad = p.grad.data

                    # update x_ag_bar
                    with torch.no_grad():
                        x_ag_bar = x_ag_prev - beta_k * grad
                    state['x_ag_bar'] = x_ag_bar.clone()

                    # set parameter p to x_ag_bar for gradient computation
                    with torch.no_grad():
                        p.data.copy_(x_ag_bar)
            
            # calculate loss and gradient at x_ag_bar
            outputs, loss_ag_bar_raw = get_grad()
            loss_ag_bar = loss_ag_bar_raw.item()
            grad_ag_bar = torch.cat([p.grad.contiguous().view(-1) for p in self.model.parameters()])

            # get vector form of x_ag_prev and x_ag_bar
            vec_x_ag_prev = torch.cat([self.state[p]['x_ag_prev'].contiguous().view(-1) for p in self.model.parameters()])
            vec_x_ag_bar = torch.cat([self.state[p]['x_ag_bar'].contiguous().view(-1) for p in self.model.parameters()])

            # calculate lhs and rhs of termination condition
            lhs = loss_ag_bar
            rhs = loss_ag_prev - self.gamma_3 / (2.0 ** beta_k) * torch.norm(vec_x_ag_bar - vec_x_ag_prev) ** 2 + 1.0 / self.k

            if tau_1 % 10 == 0:
                print(f"lhs={lhs}, rhs={rhs}")

            if lhs <= rhs:
                # print(f"Beta line search converged in {tau_2} iterations")
                break
        
        # If beta_k line search algorithm ran until max iterations
        if tau_2 == (self.max_iter - 1): 
            raise RuntimeError(f"Beta line search did not converge within {self.max_iter} iterations")

        # ---------------------------------------------------------------------------------------

        

        # ---------------------------------------------------------------------------------------
        # x_ag Update

        # find the minimum loss value, that parameter becomes new x_ag
        losses = {
            "x_ag_prev": loss_ag_prev,
            "x_ag_bar": loss_ag_bar,
            "x_ag_tilda": loss_ag_tilda
        }
        min_key = min(losses, key=losses.get)
        loss_value = losses[min_key]


        # set x_ag_k to lowest loss, and set model parameters to x_k
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                x_k = state['x_k']
                x_ag_k = state[min_key]

                state["x_ag_k"] = x_ag_k

                # set model paramters to x_k
                with torch.no_grad():
                    p.data.copy_(x_k)
        
        # ---------------------------------------------------------------------------------------

        

        # ---------------------------------------------------------------------------------------
        # beta_hat and lambda_hat updates
        
        # find lambda_hat for k = k+1
        vec_x_md = torch.cat([self.state[p]['x_md_k'].contiguous().view(-1) for p in self.model.parameters()])
        vec_x_ag_tilda = torch.cat([self.state[p]['x_ag_tilda'].contiguous().view(-1) for p in self.model.parameters()])

        # these values are actually k-1 because it will be used in next iteration
        s_md_prev = vec_x_md - vec_x_ag_tilda
        y_md_prev = grad_md_k - grad_ag_tilda
        approx_lambda_hat = torch.dot(s_md_prev, y_md_prev) / (torch.dot(y_md_prev, y_md_prev) + 1e-8)
        self.lambda_hat = max(approx_lambda_hat, self.sigma)

        # find beta_hat for k = k+1
        # get gradient of x_ag_k
        grads = {
            "x_ag_prev": grad_ag_prev,
            "x_ag_bar": grad_ag_bar,
            "x_ag_tilda": grad_ag_tilda
        }
        grad_ag_k = grads[min_key]
        vec_ag_k = torch.cat([self.state[p]['x_ag_k'].contiguous().view(-1) for p in self.model.parameters()])

        # this is actually k-1 and k-2 because this will be used in next iteration
        s_ag_prev = vec_ag_k - vec_x_ag_prev
        y_ag_prev = grad_ag_k - grad_ag_prev
        approx_beta_hat = torch.dot(s_ag_prev, y_ag_prev) / (torch.dot(y_ag_prev, y_ag_prev) + 1e-8)
        self.beta_hat = max(approx_beta_hat, self.sigma)

        print(f"For k={self.k+1}: lambda_hat = {self.lambda_hat}, beta_hat = {self.beta_hat}")

        # ---------------------------------------------------------------------------------------

        # set k = k+1
        self.k += 1

        return outputs, loss_value
    
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

    def zero_grad(self):
        super().zero_grad(set_to_none=False)


    def __repr__(self):
        return (
            f"Model: {repr(self.model)} \n"
            f"Loss Function: {self.args.loss} \n"
            f"Loss function convex: {self.convexity} \n"
            f"Lipschitz Constant of Loss Function: {self.args.lipschitz}"
        )