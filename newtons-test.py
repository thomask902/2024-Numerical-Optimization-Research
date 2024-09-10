import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision import models

# Step 1: Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

# Step 2: Load pre-trained ResNet18 and modify for CIFAR-10
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 10)  # Adjust the output layer for 10 classes (CIFAR-10)

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Loss function and learning rate
criterion = nn.CrossEntropyLoss()
lr = 0.1

# Step 3: Define the Newton's Method optimization step
def newtons_method_step(model, x, y):
    # Forward pass
    output = model(x)
    loss = criterion(output, y)
    
    # Step 2: Calculate gradient of the loss
    loss.backward(create_graph=True)  # Create graph for higher-order derivatives
    gradients = torch.cat([param.grad.view(-1) for param in model.parameters() if param.grad is not None])

    # Step 3: Calculate the Hessian matrix
    hessian = []
    for grad_i in gradients:
        hessian_row = []
        for param in model.parameters():
            grad2_ij = torch.autograd.grad(grad_i, param, retain_graph=True, create_graph=True)
            hessian_row.append(grad2_ij[0].view(-1).detach())  # Only take the first gradient (w.r.t param)
        hessian.append(torch.cat(hessian_row))
    hessian = torch.stack(hessian)

    # Step 4: Compute the pseudo-inverse of the Hessian
    hessian_pinv = torch.linalg.pinv(hessian)

    # Step 5: Calculate the update direction
    update_direction = -torch.matmul(hessian_pinv, gradients)

    # Step 6: Update the weights
    start_idx = 0
    for param in model.parameters():
        param_size = param.numel()  # Number of elements in the parameter tensor
        param_update = update_direction[start_idx:start_idx + param_size].view_as(param)
        param.data.add_(lr * param_update)  # Update the weights
        start_idx += param_size

    # Zero the gradients for the next step
    model.zero_grad()

# Step 4: Training loop using Newton's method
num_epochs = 1

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Move data to GPU if available
        images = images.to(device)
        labels = labels.to(device)

        # Perform Newton's method step
        newtons_method_step(model, images, labels)

        # Forward pass to compute loss for monitoring
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        if (i+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

# Step 5: Testing the model on test set (optional, just for validation purposes)
model.eval()  # Set model to evaluation mode
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the 10000 test images: {100 * correct / total}%')