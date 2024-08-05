import torch
import torch.nn as nn
import torch.optim as optim

class ModelWrapper:
    def __init__(self, model, optimizer, rho=0.1, perturb_eps=1e-12, alpha = 0.1):
        self.model = model
        self.optimizer = optimizer
        self.param_groups = optimizer.param_groups
        self.rho = rho
        self.perturb_eps = perturb_eps
        self.alpha = alpha

    def forward_and_backward(self, input, target, criterion):
        # zero gradients to make sure they are clear
        self.optimizer.zero_grad()

        # Forward pass
        output = self.model(input)
        loss = criterion(output, target)

        # Backward pass, tracking computation graph so that second order derivative can be taken
        loss.backward(create_graph=True)


    def grad_norm_grad(self):
        # Compute gradient vector
        grad_vec = torch.cat([p.grad.contiguous().view(-1) for p in self.model.parameters()])

        # Compute gradient vector norm
        grad_vec_norm = torch.norm(grad_vec)

        # Zero gradients again to calculate Hessian
        # self.optimizer.zero_grad()

        # Compute Hessian-vector product
        hessian_vec_prod_dict = torch.autograd.grad(
            grad_vec, self.model.parameters(), grad_outputs=grad_vec, only_inputs=True
        )

        # Concatenate the results over the different components of the network
        hessian_vec_prod = torch.cat([g.contiguous().view(-1) for g in hessian_vec_prod_dict])

        # Compute f_t and its norm
        f_t = hessian_vec_prod / grad_vec_norm

        print("Gradient Vector:", grad_vec)
        print("Gradient Vector Norm:", grad_vec_norm)
        print("Hessian-Vector Product:", hessian_vec_prod)
        print("f_t:", f_t)

        return grad_vec, f_t

    
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
        
        print("||f_t||:", f_t_norm)
        print("Perturbation Scale:", scale)

        return perturbation_vec

    @torch.no_grad()
    def unperturb(self, perturbation_vec):
        # Apply perturbation to the model parameters
        start_idx = 0
        for p in self.model.parameters():
            numel = p.numel()
            perturbation = perturbation_vec[start_idx:start_idx + numel].view_as(p)
            p.sub_(perturbation)
            start_idx += numel

    def set_gradients(self, g_0, f_t_adv):
        combined_grad = g_0 + self.alpha * self.rho * f_t_adv
        print("g_0:", g_0)
        print("g_1:", self.alpha * self.rho * f_t_adv)
        print("Combined gradients:", combined_grad)

        # Set the gradients manually for optimizer step
        start_idx = 0
        for p in self.model.parameters():
            numel = p.numel()
            p.grad = combined_grad[start_idx:start_idx + numel].view_as(p).clone()
            start_idx += numel

# Example usage
model = nn.Linear(10, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
wrapper = ModelWrapper(model, optimizer)

# Set fixed input and target tensors
input = torch.tensor([[0.5, -0.3, 0.8, 0.1, -0.1, 0.2, -0.5, 0.7, -0.6, 0.4]])
target = torch.tensor([[0.1]])

# Set fixed weights for the model
with torch.no_grad():
    model.weight.copy_(torch.tensor([[0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 1.0]]))
    model.bias.copy_(torch.tensor([0.1]))

# Print each parameter value after the optimizer step
print("Parameters before optimizer step:")
for name, param in wrapper.model.named_parameters():
    print(f"Parameter name: {name}")
    print(param)

# Perform the forward and backward pass
wrapper.forward_and_backward(input, target, criterion)

# calcuate f_t
g_0, f_t = wrapper.grad_norm_grad()

# Perform the perturbation to adversariaral point
perturbation_vec = wrapper.perturb_weights(f_t)

# clear gradient manually to ensure graph is not being tracked
for p in wrapper.model.parameters():
    p.grad = None

# calculate new gradient from adversarial point
wrapper.forward_and_backward(input, target, criterion)

# find ft@w_adv 
g_adv, f_t_adv = wrapper.grad_norm_grad()

print("difference in f_t:", torch.norm(f_t_adv - f_t))

# reverse perturbation from GAM adv point, returning parameter tensors to their original value
wrapper.unperturb(perturbation_vec)

# combine to find g_2 = g_0 + alpha * g_1 (g_1 = rho * ft@w_adv) and set to p.grad
wrapper.set_gradients(g_0, f_t_adv)

# step in direction of g_2 using optimizer
optimizer.step()

# Print each parameter value after the optimizer step
print("Parameters after optimizer step:")
for name, param in wrapper.model.named_parameters():
    print(f"Parameter name: {name}")
    print(param)

# Set fixed input and target tensors
input = torch.tensor([[0.6, 0.3, -0.8, 0.2, 0.1, -0.2, -0.5, 0.8, -0.6, 0.4]])
target = torch.tensor([[0.8]])

# Print each parameter value after the optimizer step
print("Parameters before 2nd optimizer step:")
for name, param in wrapper.model.named_parameters():
    print(f"Parameter name: {name}")
    print(param)

# Perform the forward and backward pass
wrapper.forward_and_backward(input, target, criterion)

# calcuate f_t
g_0, f_t = wrapper.grad_norm_grad()

# Perform the perturbation to adversariaral point
perturbation_vec = wrapper.perturb_weights(f_t)

# clear gradient manually to ensure graph is not being tracked
for p in wrapper.model.parameters():
    p.grad = None

# calculate new gradient from adversarial point
wrapper.forward_and_backward(input, target, criterion)

# find ft@w_adv 
g_adv, f_t_adv = wrapper.grad_norm_grad()

print("difference in f_t:", torch.norm(f_t_adv - f_t))

# reverse perturbation from GAM adv point, returning parameter tensors to their original value
wrapper.unperturb(perturbation_vec)

# combine to find g_2 = g_0 + alpha * g_1 (g_1 = rho * ft@w_adv) and set to p.grad
wrapper.set_gradients(g_0, f_t_adv)

# step in direction of g_2 using optimizer
optimizer.step()

# Print each parameter value after the optimizer step
print("Parameters after 2nd optimizer step:")
for name, param in wrapper.model.named_parameters():
    print(f"Parameter name: {name}")
    print(param)