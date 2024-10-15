import torch

class AG(torch.optim.Optimizer):

    def __init__(self, params, base_optimizer, model, args=None):
        defaults = dict()
        super(AG, self).__init__(params, defaults)

        self.model = model
        self.base_optimizer = base_optimizer
        self.args = args
        self.convexity = True # default
        if(self.args.loss == "hinge"):
            self.convexity = True
        elif(self.args.loss == "sigmoid"):
            self.convexity = False

    def step(self):
        # add code here to step
        return ""


    def __repr__(self):
        return (
            f"Model: {repr(self.model)} \n"
            f"Base Optimizer: {repr(self.base_optimizer)} \n"
            f"Loss Function: {self.args.loss} \n"
            f"Loss function convex? {self.convexity} \n"
            f"Lipschitz Constant of Loss Function: {self.args.lipschitz}"
        )