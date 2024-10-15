import torch
import torch.nn as nn
from utils.ag import AG

# Define a simple neural network with one linear layer
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(3, 2)  # Simple linear layer: input dimension = 3, output dimension = 2

    def forward(self, x):
        return self.fc(x)

# Initialize the model
model = SimpleNet()

# Create an instance of the optimizer with some basic configurations
base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Define dummy arguments needed for the GNOM_noised optimizer
class Args:
    loss = "hinge" 
    lipschitz = 70  

args = Args()

# Create the custom GNOM_noised optimizer
optimizer = AG(model.parameters(), base_optimizer, model, args=args)
print(repr(optimizer))

''' TO TEST OPTIMIZER

# Dummy input and target data
inputs = torch.tensor([[0.1, 0.2, 0.3]])
targets = torch.tensor([[0.4, 0.5]])

# Define a simple mean squared error loss function
loss_fn = nn.MSELoss()

# Perform a forward pass to compute the loss
outputs = model(inputs)
loss = loss_fn(outputs, targets)

# calls set_closure from optimizer to set up GNOM
optimizer.set_closure(loss_fn, inputs, targets)

# calls step() from GNOM to use info from closure to run steps
predictions, loss = optimizer.step()

percentage_noise, noise_count = optimizer.get_noise_statistics()

print(f"Noise applied in {noise_count} out of {optimizer.total_batches} batches "
          f"({percentage_noise:.2f}% of batches).")

# Print the updated model parameters
for param in model.parameters():
    print(f"Updated parameter: {param.data}")

'''

'''
if args.GNOM_noised:
            # Get all indices of the train dataset
            all_indices = np.arange(len(train_dataset))

            # Randomly select `grad_approx_size` samples
            np.random.shuffle(all_indices)
            grad_approx_indices = all_indices[:args.grad_approx_size]
            remaining_indices = all_indices[args.grad_approx_size:]

            # Create the grad_approx_dataset with only basic transformations
            grad_approx_dataset = torch.utils.data.Subset(train_dataset, grad_approx_indices)
            train_dataset = torch.utils.data.Subset(train_dataset, remaining_indices)
'''