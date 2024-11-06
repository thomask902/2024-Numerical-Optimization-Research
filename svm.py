import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import pandas as pd
from utils.gnom import GNOM
from utils.ag import AG
from utils.gnom_manual import GNOM_manual
import argparse
import os
from datetime import datetime

# taking in arguments to determine how the model will be run
parser = argparse.ArgumentParser(description='PyTorch Support Vector Machine Training')

parser.add_argument('--optimizer', default='GD', help='Choose from GD, AG, AG_reg, GNOM_manual or GNOM')
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate', dest='lr')
parser.add_argument('--gpu', default=False, type=bool, help='Set to true to train with GPU.')
parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 0)')
parser.add_argument('--batch-size', default=0, type=int, help='mini-batch size (default: entire dataset)')
parser.add_argument('--log_base', default='./svm', type=str, help='path to save logs (default: none)')
parser.add_argument('--n', default=2000, type=int, help='number of features in generated dataset')
parser.add_argument('--m', default=1000, type=int, help='number of examples in generated datset')
parser.add_argument('--loss', default="hinge", type=str, help='enter hinge or sigmoid')
parser.add_argument('--wd', default=0.0, type=float, help='weight decay, default is none')


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
        return torch.mean(1 - torch.tanh(outputs * labels))

def main():

    args = parser.parse_args()

    # for generated dataset
    data_name = f'n_{args.n}_m_{args.m}'

    # Gradient of loss (NOT GNOM) lipchitz values based on dataset and loss
    lipschitz_dict = { 
        "n_2000_m_1000": {"hinge": 2.066, "sigmoid": 0.310} # point-by-point approximation
        # "n_2000_m_1000": {"hinge": 5.15, "sigmoid": 0.0113} # hessian eigenvalue approximation
    }

    # regularization weight

    regularizer = 0.1

    # setting output location
    timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    learning_rate = "lr-" + str(args.lr)
    if args.batch_size == 0:
        batch_title = "no_batching"
    else:
        batch_title = str(args.batch_size)
    if args.wd == 0.0:
        weight_decay = "no_wd"
    else:
        weight_decay = str(args.wd)
    log_path = os.path.join(args.log_base, "generated", args.loss, data_name, args.optimizer, learning_rate, str(args.epochs), batch_title, weight_decay, str(timestamp), "results.csv")
    log_directory = os.path.dirname(log_path)
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)


    # load in train dataset
    train_data = torch.load(f'generated_data/{data_name}.pt')
    train_features = train_data['features']
    input_dim = train_features.shape[1]
    train_labels = train_data['labels'].unsqueeze(1).float()
    train_dataset = TensorDataset(train_features, train_labels)

    # load in test dataset
    test_data = torch.load(f'generated_data/n_{args.n}_test.pt')
    test_features = test_data['features']
    test_labels = test_data['labels'].unsqueeze(1).float()
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
    model = nn.Linear(in_features=input_dim, out_features=1, bias=False)

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
    base_optimizer = torch.optim.SGD(model.parameters(), lr=learningRate, weight_decay=args.wd)

    if args.optimizer == "GD":
        optimizer = base_optimizer
    elif args.optimizer == "GNOM":
        optimizer = GNOM(params=model.parameters(), base_optimizer=base_optimizer, model=model)
    elif args.optimizer == "GNOM_manual":
        optimizer = GNOM_manual(params=model.parameters(), base_optimizer=base_optimizer, model=model, args=args)
    elif args.optimizer == "AG":
        lipschitz = lipschitz_dict[data_name][args.loss]
        optimizer = AG(params=model.parameters(), model=model, loss_type=args.loss, lipschitz=lipschitz)
    elif args.optimizer == "AG_reg":
        lipschitz = lipschitz_dict[data_name][args.loss] + regularizer
        optimizer = AG(params=model.parameters(), model=model, loss_type=args.loss, lipschitz=lipschitz, reg=regularizer, args=args)
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
        
        total_loss = 0.0
        train_loss = 0.0
        train_grad_norm = 0.0
        train_time = 0.0
        test_loss = 0.0
        x_k_diff = 0.0
        x_ag_k_diff = 0.0

        # train model for epoch
        if args.optimizer == "GD":
            train_loss, train_grad_norm, train_time, x_k_diff = train_epoch_base(model, optimizer, train_loader, device, criterion)
        elif args.optimizer == "AG" or args.optimizer == "AG_reg":
            total_loss, train_loss, train_grad_norm, train_time, x_k_diff, x_ag_k_diff = train_epoch_closure_ag(model, optimizer, train_loader, device, criterion)
        else:
            total_loss, train_loss, train_grad_norm, train_time, x_k_diff = train_epoch_closure(model, optimizer, train_loader, device, criterion)

        # evaluate model after training
        test_loss, test_grad_norm, accuracy = evaluate(model, test_loader, criterion, device)

        epoch_stats.append({
            "Epoch": epoch,
            "Training Loss": train_loss,
            "Total Training Loss": total_loss,
            "Training Gradient Norm": train_grad_norm,
            "Training Time (s)": train_time,
            "Test Loss": test_loss,
            "Test Gradient Norm": test_grad_norm,
            "Test Accuracy": accuracy,
            "Test Error": 1 - accuracy,
            "x_k Comparison": x_k_diff,
            "x_ag_k Comparison": x_ag_k_diff
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
        #if (batch_idx + 1) % 100 == 0:
        #    print(f'Batch {batch_idx + 1}\'s loss is: {loss}')

    end_time = time.time()
    total_loss = total_loss/len(train_loader)

    # track difference between x_bar and x_k
    for p in model.parameters():
        x_bar = torch.load("generated_data/x_bar.pt")
        x_k = p.data.clone().squeeze()
        x_k_diff = torch.norm(x_bar - x_k).item()

    # Compute gradient norm over the entire training dataset
    optimizer.zero_grad()
    all_inputs = train_loader.dataset.tensors[0].to(device)
    all_labels = train_loader.dataset.tensors[1].to(device)
    optimizer.set_closure(criterion, all_inputs, all_labels, create_graph=False, enable_reg=False)
    train_loss, train_grad_norm = optimizer.calc_grad_norm()
    optimizer.zero_grad()

    return total_loss, train_loss.item(), train_grad_norm, (end_time - start_time), x_k_diff

def train_epoch_closure_ag(model, optimizer, train_loader, device, criterion):
    start_time = time.time()
    model.train()  # Set model to training mode

    total_loss = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()

        # calls set_closure from optimizer to set up GNOM with information
        optimizer.set_closure(criterion, inputs, labels)

        # calls step() from GNOM to use info from closure to run steps
        predictions, loss, grad_norm, x_k_diff, x_ag_k_diff = optimizer.step()
        
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
    train_loss, train_grad_norm = optimizer.calc_grad_norm()
    optimizer.zero_grad()

    return total_loss, train_loss.item(), train_grad_norm, (end_time - start_time), x_k_diff, x_ag_k_diff

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
        #if (batch_idx + 1) % 100 == 0:
        #    print(f'Batch {batch_idx + 1}\'s loss is: {loss}')


    end_time = time.time()
    train_loss = total_loss / len(train_loader)

    # track difference between x_bar and x_k
    for p in model.parameters():
        x_bar = torch.load("generated_data/x_bar.pt")
        x_k = p.data.clone().squeeze()
        x_k_diff = torch.norm(x_bar - x_k).item()

    # compute gradient norm over the entire training dataset
    optimizer.zero_grad()
    all_inputs = train_loader.dataset.tensors[0].to(device)
    all_labels = train_loader.dataset.tensors[1].to(device)
    all_outputs = model(all_inputs)
    total_dataset_loss = criterion(all_outputs, all_labels)
    total_dataset_loss.backward()

    train_grad_norm = get_grad_norm(model)

    optimizer.zero_grad()

    return train_loss, train_grad_norm, (end_time - start_time), x_k_diff

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