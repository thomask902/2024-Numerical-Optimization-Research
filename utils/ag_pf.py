import torch

class AG_pf(torch.optim.Optimizer):

    def __init__(self, params, model):
        defaults = dict()
        super(AG_pf, self).__init__(params, defaults)

        self.model = model

        # INITIALIZE GLOBAL PARAMETERS
        self.k = 1
        self.gamma_1 = 0.5
        self.gamma_2 = 0.5
        self.gamma_3 = 0.5
        self.beta_hat = 1.0
        self.lambda_hat = 1.0
        self.delta = 0.001
        self.max_iter = 20
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
            outputs, loss_md_k = get_grad()
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
            outputs, loss_ag_tilda = get_grad()
            grad_ag_tilda = torch.cat([p.grad.contiguous().view(-1) for p in self.model.parameters()])

            # get vector form of x_k and x_k_prev
            vec_x_k = torch.cat([self.state[p]['x_k'].contiguous().view(-1) for p in self.model.parameters()])
            vec_x_k_prev = torch.cat([self.state[p]['x_k_prev'].contiguous().view(-1) for p in self.model.parameters()])

            # calculate lhs and rhs of termination condition
            lhs = loss_ag_tilda
            rhs = loss_md_k + alpha_k * torch.dot(grad_md_k, (vec_x_k - vec_x_k_prev)) + alpha_k / (2 * lambda_k) * torch.norm(vec_x_k - vec_x_k_prev) ** 2 + self.delta * alpha_k

            print(f"lhs={lhs}, rhs={rhs}")

            if lhs <= rhs:
                print(f"Lambda line search converged in {tau_1} iterations")

                # update previous upper case lambda, and append final lambda_k to list
                self.upper_lambda_prev = upper_lambda_k
                self.lambda_k_list.append(lambda_k)
                break
        
        # If lambda_k line search algorithm ran until max iterations
        if tau_1 == (self.max_iter - 1): 
            raise RuntimeError(f"Lambda line search did not converge within {self.max_iter} iterations")



        # ---------------------------------------------------------------------------------------
        # Beta Line Search

        # beta k line search loop, tau_2 starts at 0 increasing by 1 each loop iteration
        for tau_2 in range(self.max_iter):
            

            # calculate lhs and rhs of termination condition
            lhs = 0
            rhs = 0

            # print(f"lhs={lhs}, rhs={rhs}")

            if lhs <= rhs:
                print(f"Beta line search converged in {tau_2} iterations")
                break
        
        # If beta_k line search algorithm ran until max iterations
        if tau_2 == (self.max_iter - 1): 
            raise RuntimeError(f"Beta line search did not converge within {self.max_iter} iterations")

        

        # ---------------------------------------------------------------------------------------
        # x_ag Update

        losses = [1,2,3]
        loss_value = min(losses)


        # dummy update for x_k to test algorithm
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                x_k = state['x_k']

                # set model paramters to x_k
                with torch.no_grad():
                    p.data.copy_(x_k)

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