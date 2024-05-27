# Defining training functions

import torch
import torch.optim as optim
from utils.smooth_cross_entropy import get_smooth_crossentropy

# vanilla SGD optimizer with momentum
def get_optimizer(model, lr=0.001, momentum=0.9):
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum)

def train_epoch(model, trainloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize the model parameters
        
        running_loss += loss.item()
        if (i + 1) % 100 == 0:  # Print every 10 mini-batches
            print(f'Batch {i + 1}, Loss: {running_loss / 100:.3f}')
            running_loss = 0.0

def train_epoch_smooth(model, trainloader, optimizer, device):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(inputs)  # Forward pass
        loss = get_smooth_crossentropy(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize the model parameters
        
        running_loss += loss.item()
        if (i + 1) % 100 == 0:  # Print every 100 mini-batches
            print(f'Batch {i + 1}, Loss: {running_loss / 100:.3f}')
            running_loss = 0.0

def evaluate_model(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Accuracy: {100 * correct / total}%')