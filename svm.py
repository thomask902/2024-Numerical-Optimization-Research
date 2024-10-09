import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.gnom import GNOM
import argparse
import os
from datetime import datetime
import sys

# taking in arguments to determine how the model will be run
parser = argparse.ArgumentParser(description='PyTorch Support Vector Machine Training')

parser.add_argument('--optimizer', default='GD', help='Choose from GD, or GNOM')
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate', dest='lr')
parser.add_argument('--gpu', default=False, type=bool, help='Set to true to train with GPU.')
parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 0)')
parser.add_argument('--batch-size', default=0, type=int, help='mini-batch size (default: entire dataset)')
parser.add_argument('--log_base', default='./svm', type=str, help='path to save logs (default: none)')
parser.add_argument('--n', default=2000, type=int, help='number of features in generated dataset')
parser.add_argument('--m', default=1000, type=int, help='number of examples in generated datset')
parser.add_argument('--loss', default="hinge", type=str, help='enter hinge or sigmoid')


# defining the loss functions
class Squared_Hinge_Loss(nn.Module):    
    def __init__(self):
        super(Squared_Hinge_Loss,self).__init__()
    def forward(self, outputs, labels): 
        return torch.mean((torch.clamp(1 - outputs * labels, min=0)) ** 2)  

class Sigmoid_Loss(nn.Module):    
    def __init__(self):
        super(Sigmoid_Loss,self).__init__()
    def forward(self, outputs, labels):
        return torch.mean(1 - torch.sigmoid(outputs * labels))

def main():

    args = parser.parse_args()

    # for generated dataset
    data_name = f'n_{args.n}_m_{args.m}'

    # setting output location
    timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    learning_rate = "lr-" + str(args.lr)
    batch_title = ""
    if args.batch_size == 0:
        batch_title = "no_batching"
    else:
        batch_title = str(args.batch_size)
    log_path = os.path.join(args.log_base, "generated", data_name, args.optimizer, learning_rate, str(args.epochs), batch_title, str(timestamp), "results.csv")
    log_directory = os.path.dirname(log_path)
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)


    # load in train dataset
    train_data = torch.load(f'generated_data/{data_name}.pt')
    train_features = train_data['features']
    input_dim = train_features.shape[1]
    train_labels = train_data['labels'].unsqueeze(1).float()
    train_labels[train_labels == 0] = -1 # added for SVM model
    train_dataset = TensorDataset(train_features, train_labels)

    # load in test dataset
    test_data = torch.load(f'generated_data/n_{args.n}_test.pt')
    test_features = test_data['features']
    test_labels = test_data['labels'].unsqueeze(1).float()
    test_labels[test_labels == 0] = -1 # added for SVM model
    test_dataset = TensorDataset(test_features, test_labels)

    # set batch_size for loaders
    batch_train = 0
    batch_test = 0
    if args.batch_size == 0:
        print("Training Examples:", len(train_dataset))
        batch_train = len(train_dataset)
        batch_test = len(test_dataset)
    else:
        batch_train = args.batch_size
        batch_test = args.batch_size
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_train, num_workers=args.workers, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_test, num_workers=args.workers, shuffle=False)

    # set up model
    learningRate = args.lr
    epochs = args.epochs

    # linear model for SVM
    model = nn.Linear(input_dim, 1)

    device = torch.device('cuda' if args.gpu else 'cpu')
    model.to(device)

    # initialize loss
    if args.loss == "hinge":
        criterion = Squared_Hinge_Loss() 
    elif args.loss == "sigmoid":
        criterion = Sigmoid_Loss()
    else:
        raise ValueError("Please enter a valid loss function!")

    # initialize optimizer
    base_optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

    if args.optimizer == "GD":
        optimizer = base_optimizer
    elif args.optimizer == "GNOM":
        optimizer = GNOM(params=model.parameters(), base_optimizer=base_optimizer, model=model)
    else:
        raise ValueError("Please enter a valid optimizer!")

    # training
    print(f"Training with {args.optimizer} and {args.loss} loss") # this will be in name of output file

    # List to hold stats for each epoch
    epoch_stats = []
    spaces = ""

    for epoch in range(1, epochs + 1):
        if epoch % 10 == 0:
            print(f"Epoch {epoch} underway {spaces}\\(*_*)/")
            spaces += "  "
        
        train_loss = 0.0
        train_grad_norm = 0.0
        train_time = 0.0
        test_loss = 0.0

        # train model for epoch
        if args.optimizer == "GD":
            train_loss, train_grad_norm, train_time = train_epoch_base(model, optimizer, train_loader, device, criterion)
        else:
            train_loss, train_grad_norm, train_time = train_epoch_closure(model, optimizer, train_loader, device, criterion)

        # evaluate model after training
        test_loss, test_grad_norm, accuracy = evaluate(model, test_loader, criterion, device)

        epoch_stats.append({
            "Epoch": epoch,
            "Training Loss": train_loss,
            "Training Gradient Norm": train_grad_norm,
            "Training Time (s)": train_time,
            "Test Loss": test_loss,
            "Test Gradient Norm": test_grad_norm,
            "Test Accuracy": accuracy,
            "Test Error": 1 - accuracy
        })

    # print and save results of run
    df_stats = pd.DataFrame(epoch_stats)
    df_stats.to_csv(log_path, index=False)
    print(df_stats)
        

def train_epoch_closure(model, optimizer, train_loader, device, criterion):
    start_time = time.time()
    model.train()  # Set model to training mode

    total_loss = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()

        # calls set_closure from optimizer to set up GNOM with information
        optimizer.set_closure(criterion, inputs, labels)

        # calls step() from GNOM to use info from closure to run steps
        predictions, loss, grad_norm = optimizer.step()
        
        total_loss += loss.item()
        if (batch_idx + 1) % 100 == 0:
            print(f'Batch {batch_idx + 1}\'s loss is: {loss}')

    end_time = time.time()
    train_loss = total_loss/len(train_loader)

    # Compute gradient norm over the entire training dataset
    optimizer.zero_grad()
    all_inputs = train_loader.dataset.tensors[0].to(device)
    all_labels = train_loader.dataset.tensors[1].to(device)
    optimizer.set_closure(criterion, all_inputs, all_labels, create_graph=False)
    train_grad_norm = optimizer.calc_grad_norm()
    optimizer.zero_grad()

    return train_loss, train_grad_norm, (end_time - start_time)

def train_epoch_base(model, optimizer, train_loader, device, criterion):
    start_time = time.time()
    model.train()  # Set model to training mode

    total_loss = 0.0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()

        # forward pass
        outputs = model(inputs)

        # compute loss
        loss = criterion(outputs, labels)

        # backward pass
        loss.backward()

        optimizer.step()
        
        total_loss += loss.item()
        if (batch_idx + 1) % 100 == 0:
            print(f'Batch {batch_idx + 1}\'s loss is: {loss}')


    end_time = time.time()
    train_loss = total_loss / len(train_loader)

    # compute gradient norm over the entire training dataset
    optimizer.zero_grad()
    all_inputs = train_loader.dataset.tensors[0].to(device)
    all_labels = train_loader.dataset.tensors[1].to(device)
    all_outputs = model(all_inputs)
    total_dataset_loss = criterion(all_outputs, all_labels)
    total_dataset_loss.backward()

    train_grad_norm = get_grad_norm(model)

    optimizer.zero_grad()

    return train_loss, train_grad_norm, (end_time - start_time)

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
        
    test_loss = loss.item()

    # === Compute gradient norm over the entire test dataset ===
    model.zero_grad()
    # Perform forward pass again to compute gradients
    all_outputs = model(all_inputs)
    total_dataset_loss = criterion(all_outputs, all_labels)
    # Backward pass to compute gradients
    total_dataset_loss.backward()
    # Compute gradient norm
    test_grad_norm = get_grad_norm(model)
    model.zero_grad()
    # === End of gradient norm computation ===

    return test_loss, test_grad_norm, acc


def get_grad_norm(model):
        """
        Compute and return the L2 norm of the gradients of the loss function.
        """
        grad_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5  # Take the square root of the sum of squares
        return grad_norm

if __name__ == '__main__':
    main()