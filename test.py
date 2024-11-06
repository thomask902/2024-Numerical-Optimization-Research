import torch
import torch.nn as nn
from utils.ag import AG
from torch.utils.data import DataLoader, TensorDataset, random_split

# defining the loss functions
class Squared_Hinge_Loss(nn.Module):    
    def __init__(self):
        super(Squared_Hinge_Loss,self).__init__()
    def forward(self, outputs, labels): 
        print("Hinge loss calcs:")
        print(1 - outputs * labels)
        print(torch.clamp(1 - outputs * labels, min=0))
        print((torch.clamp(1 - outputs * labels, min=0)) ** 2)
        return torch.mean((torch.clamp(1 - outputs * labels, min=0)) ** 2)  

class Sigmoid_Loss(nn.Module):    
    def __init__(self):
        super(Sigmoid_Loss,self).__init__()
    def forward(self, outputs, labels):
        return torch.mean(1 - torch.tanh(outputs * labels))

def main():

    '''
    train_features = torch.tensor([[1.0, -4.0]], dtype=torch.double)
    input_dim = train_features.shape[1]
    train_labels = torch.tensor([-1.0], dtype=torch.double)
    train_dataset = TensorDataset(train_features, train_labels)

    test_features = torch.tensor([[2.0, 1.0]], dtype=torch.double)
    test_labels = torch.tensor([1.0], dtype=torch.double)
    test_dataset = TensorDataset(test_features, test_labels)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    '''

    # linear model for SVM
    model = nn.Linear(in_features=2, out_features=1, bias=False).double()
    
    # for manual testing
    weights = torch.tensor([1.0, -1.0], dtype=torch.double)
    with torch.no_grad():
        model.weight.copy_(weights)

    # Define inputs and labels with requires_grad=True
    inputs = torch.tensor([[1.0, -4.0]], requires_grad=True, dtype=torch.double)
    labels = torch.tensor([-1.0], dtype=torch.double)

    # Forward pass through the model to get outputs
    outputs = model(inputs)
    print("Outputs:", outputs)

    # Step 1: Calculate the margin
    margin = 1 - outputs * labels
    print("Margin (1 - outputs * labels):", margin)

    # Step 2: Apply clamping to ensure positive values (hinge loss behavior)
    clamped = torch.clamp(margin, min=0)
    print("Clamped (torch.clamp(margin, min=0)):", clamped)

    # Step 3: Square the clamped margin
    squared_clamped = clamped ** 2
    print("Squared Clamped (clamped ** 2):", squared_clamped)

    # Step 4: Take the mean to get the final loss
    loss = torch.mean(squared_clamped)
    print("Loss (torch.mean(squared_clamped)):", loss)

    # Compute gradients for intermediate tensors
    grad_squared_clamped = torch.autograd.grad(loss, squared_clamped, retain_graph=True)
    print("Gradient w.r.t. squared_clamped:", grad_squared_clamped)

    grad_clamped = torch.autograd.grad(loss, clamped, retain_graph=True)
    print("Gradient w.r.t. clamped tensor:", grad_clamped)

    grad_margin = torch.autograd.grad(loss, margin, retain_graph=True)
    print("Gradient w.r.t. margin tensor:", grad_margin)

    grad_outputs = torch.autograd.grad(loss, outputs, retain_graph=True)
    print("Gradient w.r.t. outputs tensor:", grad_outputs)

if __name__ == '__main__':
    main()


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