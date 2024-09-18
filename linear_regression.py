import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.gnom import GNOM
import argparse

# taking in arguments to determine how the model will be run
parser = argparse.ArgumentParser(description='PyTorch Linear Regression Training with GNOM')

parser.add_argument('--optimizer', default='GD', help='Choose from GD, GNOM, or GAM')
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate', dest='lr')
parser.add_argument('--gpu', default=False, type=bool, help='Set to true to train with GPU.')
parser.add_argument('--log_base', default='./linear_regression', type=str, help='path to save logs (default: none)')

# defining the model
class linearRegression(nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

def main():

    args = parser.parse_args()

    # importing the dataset
    wine_quality = fetch_ucirepo(id=186) 
    
    X = wine_quality.data.features 
    y = wine_quality.data.targets 

    # normalize features and test train split
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    # Split the dataset into train and test sets (80% train, 20% test), random state fixes seed and gives same split each time
    X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)

    inputDim = X_train.shape[1]
    outputDim = y_train.shape[1]
    print("Input Parameters:", inputDim)
    print("Output Parameters:", outputDim)

    # convert data into PyTorch Tensors
    X_train_vals = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_vals = torch.tensor(X_test.values, dtype=torch.float32)
    y_train_vals = torch.tensor(y_train.values, dtype=torch.float32)
    y_test_vals = torch.tensor(y_test.values, dtype=torch.float32)

    # create datasets and loaders
    train_torch = torch.utils.data.TensorDataset(X_train_vals, y_train_vals)
    test_torch = torch.utils.data.TensorDataset(X_test_vals, y_test_vals)
    train_loader = torch.utils.data.DataLoader(dataset=train_torch, batch_size=len(train_torch), shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_torch, batch_size=len(test_torch), shuffle=False)

    # set up model
    learningRate = args.lr
    epochs = args.epochs
    model = linearRegression(inputDim, outputDim)

    device = torch.device('cuda' if args.gpu else 'cpu')
    model.to(device)

    # initialize loss and optimzer
    criterion = torch.nn.MSELoss() 
    base_optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

    if args.optimizer == "GD":
        optimizer = base_optimizer
    elif args.optimizer == "GNOM":
        optimizer = GNOM(params=model.parameters(), base_optimizer=base_optimizer, model=model)
    else:
        print("Please enter a valid optimizer")
        return # exits program

    # training
    print(f"Training with {args.optimizer}") # this will be in name of output file

    test_accuracy = []

    for epoch in range(epochs):
        # train model for epoch
        if args.optimizer == "GD":
            train_loss, train_grad_norm, train_time = train_epoch_base(model, optimizer, train_loader, device, criterion)
        else:
            train_loss, train_grad_norm, train_time = train_epoch_closure(model, optimizer, train_loader, device, criterion)

        # evaluate model after training
        test_loss, test_grad_norm = evaluate(model, test_loader, criterion, device)
        
        # output epoch results
        print(f'Epoch {epoch+1}, Training Loss: {train_loss}, Training Time (s): {train_time}, Test Loss: {test_loss}')

def train_epoch_closure(model, optimizer, train_loader, device, criterion):
    start_time = time.time()
    model.train()  # Set model to training mode

    # TO SET UP
    grad_norm = 0

    total_loss = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()

        # calls set_closure from optimizer to set up GNOM with information
        optimizer.set_closure(criterion, inputs, labels)

        # calls step() from GNOM to use info from closure to run steps
        predictions, loss = optimizer.step()
        
        total_loss += loss.item()

    end_time = time.time()
    train_loss = total_loss/len(train_loader)

    return train_loss, grad_norm, (end_time - start_time)

def train_epoch_base(model, optimizer, train_loader, device, criterion):
    start_time = time.time()
    model.train()  # Set model to training mode

    # TO SET UP
    grad_norm = 0
    
    total_loss = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    end_time = time.time()
    train_loss = total_loss/len(train_loader)

    return train_loss, grad_norm, (end_time - start_time)

def evaluate(model, test_loader, criterion, device):
    model.eval()  # Set model to evaluation mode

    # TO SET UP
    grad_norm = 0

    with torch.no_grad():
        total_loss = 0
        for test_inputs, test_labels in test_loader:
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)

            test_outputs = model(test_inputs)
            total_loss += criterion(test_outputs, test_labels).item()

        test_loss = total_loss / len(test_loader)
    
    return test_loss, grad_norm

if __name__ == '__main__':
    main()