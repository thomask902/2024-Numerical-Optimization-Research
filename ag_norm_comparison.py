from functools import _make_key
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import pandas as pd
from utils.ag import AG
from utils.ag_pf import AG_pf
import argparse
import os
from datetime import datetime
import math

# defining the loss functions
class Squared_Hinge_Loss(nn.Module):
    def __init__(self):
        super(Squared_Hinge_Loss, self).__init__()

    def forward(self, outputs, labels):
        return torch.mean((torch.clamp(1 - outputs * labels, min=0)) ** 2)

class Sigmoid_Loss(nn.Module):
    def __init__(self):
        super(Sigmoid_Loss, self).__init__()

    def forward(self, outputs, labels):
        return torch.mean(1 - torch.tanh(outputs * labels))

# evaluation function
def evaluate(model, test_loader, criterion, device):
    model.eval()  # Set model to evaluation mode

    # Get the entire test dataset
    all_inputs = test_loader.dataset.tensors[0].to(device)
    all_labels = test_loader.dataset.tensors[1].to(device)

    # computing accuracy as the paper demanded
    with torch.no_grad():
        all_outputs = model(all_inputs)
        prod = all_outputs * all_labels
        num_correct = (prod > 0).sum().item()
        acc = float(num_correct) / float(all_labels.shape[0])
        loss = criterion(all_outputs, all_labels)

    # compute gradient norm over the entire test dataset
    model.train()
    model.zero_grad()
    all_outputs = model(all_inputs)
    total_dataset_loss = criterion(all_outputs, all_labels)
    test_loss = total_dataset_loss.item()
    total_dataset_loss.backward()
    test_grad_norm = get_grad_norm(model)
    model.zero_grad()

    return test_loss, test_grad_norm, acc

def get_grad_norm(model):
        grad_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5  # Take the square root of the sum of squares
        return grad_norm

def main():
    # Manually set hyperparameters
    epochs = 200  # Maximum number of subproblems
    input_dim = 2000
    train_size = 1000
    loss_type = "hinge"  # Choose between "hinge" and "sigmoid"
    gpu = False  # Set to True to use GPU if available
    log_base = './svm'

    # Lipschitz approximations
    data_name = f'n_{input_dim}_m_{train_size}'
    lipschitz_dict = {
        "n_2000_m_1000": {
            "no_batching": {"hinge": 2.066, "sigmoid": 0.310},
            "256": {"hinge": 2.5, "sigmoid": 0.32},
            "128": {"hinge": 2.1, "sigmoid": 0.32},
            "32": {"hinge": 7.6, "sigmoid": 0.45},
        }
    }

    # Set up device
    device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')

    # Load in train dataset
    train_data = torch.load(f'generated_data/{data_name}.pt')
    train_features = train_data['features']
    train_labels = train_data['labels'].unsqueeze(1).float()
    train_dataset = TensorDataset(train_features, train_labels)

    # Load in test dataset
    test_data = torch.load(f'generated_data/n_{input_dim}_test.pt')
    test_features = test_data['features']
    test_labels = test_data['labels'].unsqueeze(1).float()
    test_dataset = TensorDataset(test_features, test_labels)

    # Set batch_size for loaders
    batch_test = len(test_dataset)
    batch_train = len(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_train, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_test, shuffle=False)

    # Initialize model
    model = nn.Linear(input_dim, 1, bias=False).to(device)

    # Define loss
    if loss_type == "hinge":
        criterion = Squared_Hinge_Loss()
        lipschitz = lipschitz_dict[data_name]["no_batching"]["hinge"]
    elif loss_type == "sigmoid":
        criterion = Sigmoid_Loss()
        lipschitz = lipschitz_dict[data_name]["no_batching"]["sigmoid"]
    else:
        raise ValueError("Invalid loss function specified.")

    optimizer = AG(params=model.parameters(), model=model, loss_type=loss_type, lipschitz=lipschitz)

    epoch_stats = []

    for epoch in range(1, epochs + 1):

        total_loss = 0.0
        train_loss = 0.0
        train_grad_norm = 0.0
        train_time = 0.0
        test_loss = 0.0
        x_k_diff = 0.0
        x_ag_k_diff = 0.0

        
        total_loss, train_loss, x_ag_norm, train_time, x_md_norm, x_k_norm = train_epoch_closure_ag(model, optimizer, train_loader, device, criterion)

        # evaluate model after training
        test_loss, test_grad_norm, accuracy = evaluate(model, test_loader, criterion, device)

        epoch_stats.append({
            "Epoch": epoch,
            "Training Loss": train_loss,
            "Total Training Loss": total_loss,
            "x_ag_k gradient norm": x_ag_norm,
            "x_md_k gradient norm": x_md_norm,
            "x_k gradient norm": x_k_norm,
            "Training Time (s)": train_time,
            "Test Loss": test_loss,
            "Test Gradient Norm": test_grad_norm,
            "Test Accuracy": accuracy,
            "Test Error": 1 - accuracy
        })

    
    # Set up path to save
    timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    log_path = os.path.join(log_base, "generated", loss_type, data_name, "AG", "no-lr", str(epochs), "no-batching", timestamp, "results.csv")
    log_directory = os.path.dirname(log_path)
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # Save results to CSV
    df_stats = pd.DataFrame(epoch_stats)
    df_stats.to_csv(log_path, index=False)
    print(df_stats)

def train_epoch_closure_ag(model, optimizer, train_loader, device, criterion):
        start_time = time.time()
        model.train()  # Set model to training mode

        total_loss = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()

            # calls set_closure from optimizer to set up AG with information
            optimizer.set_closure(criterion, inputs, labels)

            # calls step() from GNOM to use info from closure to run steps
            predictions, loss, x_md_norm, x_k_norm = optimizer.step()
            
            total_loss += loss.item()
            #if (batch_idx + 1) % 100 == 0:
            #    print(f'Batch {batch_idx + 1}\'s loss is: {loss}')

        end_time = time.time()
        total_loss = total_loss/len(train_loader)

        # Compute gradient norm over the entire training dataset
        optimizer.zero_grad()
        all_inputs = train_loader.dataset.tensors[0].to(device)
        all_labels = train_loader.dataset.tensors[1].to(device)
        optimizer.set_closure(criterion, all_inputs, all_labels, create_graph=False, enable_reg=False)
        train_loss, x_ag_norm = optimizer.calc_grad_norm()
        optimizer.zero_grad()

        return total_loss, train_loss.item(), x_ag_norm, (end_time - start_time), x_md_norm, x_k_norm


if __name__ == '__main__':
    main()
