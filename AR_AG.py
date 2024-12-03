from functools import _make_key
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import pandas as pd
from utils.ag import AG
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

# Evaluation function
def evaluate(model, test_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item()
            correct += (outputs * labels > 0).sum().item()

    accuracy = correct / len(test_loader.dataset)
    return total_loss / len(test_loader), accuracy


def main():
    # fixed hyperparameters
    S = 5
    learning_rate = 0.001
    batch_size = 1000
    input_dim = 2000
    train_size = 1000

    # lipschitz approximations
    lipschitz_dict = {
        "n_2000_m_1000": {
            "no_batching": {"hinge": 2.066, "sigmoid": 0.310},
            "256": {"hinge": 2.5, "sigmoid": 0.32},
            "128": {"hinge": 2.1, "sigmoid": 0.32},
            "32": {"hinge": 7.6, "sigmoid": 0.45},
        }
    }

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load in train dataset (fixed at n 2000 m 1000)
    train_data = torch.load(f'generated_data/n_2000_m_1000.pt')
    train_features = train_data['features']
    train_labels = train_data['labels'].unsqueeze(1).float()
    train_dataset = TensorDataset(train_features, train_labels)

    # load in test dataset
    test_data = torch.load(f'generated_data/n_2000_test.pt')
    test_features = test_data['features']
    test_labels = test_data['labels'].unsqueeze(1).float()
    test_dataset = TensorDataset(test_features, test_labels)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = nn.Linear(input_dim, 1, bias=False).to(device)

    # temporary model identical to the original model for later calculations
    temp_model = nn.Linear(input_dim, 1, bias=False).to(device)

    # Define loss and optimizer
    criterion = Squared_Hinge_Loss()
    # criterion = Sigmoid_Loss()

    # ---------------------------------------------------------------------------------------
    # Subproblem Algorithm

    # Set initial parameters for subproblems
    if len(list(model.parameters())) == 1:
        x_prev = next(iter(model.parameters()))
    else:
        raise RuntimeError("Optimizer can only run on basic models")
    x_bar_prev = x_prev.clone().detach()
    sigma_prev = 0.0
    m_prev = lipschitz_dict["n_2000_m_1000"]["no_batching"]["hinge"]
    # m_prev = lipchitz_dict["n_2000_m_1000"]["no_batching"]["sigmoid"]
    sigma_1 = m_prev / 10.0  # this is a guess, not sure

    # Print initial parameters
    print(f"Initial Parameters:")
    print(f"x_0: {x_prev}")
    print(f"x_bar_0: {x_bar_prev}")
    print(f"sigma_0: {sigma_prev}")
    print(f"m_0: {m_prev}")
    print(f"sigma_1: {sigma_1}\n")

    # Training loop
    for s in range(1, S + 1):
        # Determine parameters of subproblem for iteration s
        # sigma (regularization parameter)
        if s == 1:
            sigma_s = sigma_1
        else:
            sigma_s = 4.0 * sigma_prev

        # gamma (controls new point center weighting)
        gamma_s = 1.0 - sigma_prev / sigma_s

        # new point center
        x_bar_s = ((1.0 - gamma_s) * x_bar_prev + gamma_s * x_prev).detach().to(device)

        # k (number of iterations for subroutine)
        l_s_k = 4.0 * (m_prev + sigma_s)
        k_approx = 8.0 * math.sqrt(2.0 * l_s_k / sigma_s)
        k = math.ceil(k_approx)

        # Print parameters for subproblem iteration
        print(f"Subproblem Iteration s={s}:")
        print(f"sigma_s: {sigma_s}")
        print(f"gamma_s: {gamma_s}")
        print(f"x_bar_s: {x_bar_s}")
        print(f"m_(s-1): {m_prev}")
        print(f"l_s_k: {l_s_k}")
        print(f"k approx.: {k_approx}")
        print(f"k: {k}\n")

        # because lipschitz approximation changes for each s, we redefine AG for each subproblem
        optimizer = AG(params=model.parameters(), model=model, loss_type="hinge", lipschitz=m_prev + sigma_s)
        # optimizer = AG(params=model.parameters(), model=model, loss_type="sigmoid", lipschitz=m_prev)

        # Run subroutine for k iterations, loading in data, computing gradient w/ regularization and stepping
        train_loss = 0.0
        for i in range(0, k):
            model.train()

            inputs, labels = next(iter(train_loader))
            inputs, labels = inputs.to(device), labels.to(device)

            # define a custom closure to run AG at this iteration with these inputs/labels and with regularization from x_bar_s and sigma_s
            def closure():
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # add regularization
                x_s = next(iter(model.parameters()))
                reg_s = (sigma_s / 2.0) * torch.norm(x_s - x_bar_s) ** 2
                loss += reg_s

                # backprop
                loss_value = loss.data.clone().detach()
                loss.backward()
                return outputs, loss_value

            # Perform optimization step with defined closure
            outputs, loss_value, grad_norm, x_k_diff, x_ag_k_diff = optimizer.step(closure=closure)

            train_loss = loss_value.item()

        
        # Eval and output results after running each subproblem
        test_loss, accuracy = evaluate(model, test_loader, criterion, device)
        print(f"Subproblem Iteration s={s} Results:")
        print(f"Train Loss: {train_loss:.8f}")
        print(f"Train Gradient Norm: {grad_norm:.8f}")
        print(f"Test Loss: {test_loss:.8f}")
        print(f"Accuracy: {accuracy:.8f}\n")

        # Set resultant parameters for backtracking function
        x_s = next(iter(model.parameters()))

        # ---------------------------------------------------------------------------------------




        # ---------------------------------------------------------------------------------------
        # Backtracking algorithm (will put into its own function after)
        # inputs: loss function/model, x_s, loss @ x_s, gradient vector @ x_s, sigma_s, m_prev
        # output: new M value, m_s

        # inputs
        max_iter=100
        sigma = sigma_s

        # compute gradient at x_s (x in backtracking algo)
        inputs, labels = next(iter(train_loader))
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        reg_s = (sigma_s / 2.0) * (torch.norm(x_s - x_bar_s) ** 2)
        loss += reg_s
        loss_x = loss.item()
        loss.backward()
        grad_x = torch.cat([p.grad.contiguous().view(-1) for p in model.parameters()])

        # backtracking algorithm
        
        m_j = m_prev / 2.0

        for j in range(max_iter):
            # compute x++
            x_plus = x_s - 1 / (2 * (m_j + sigma)) * grad_x

            # set parameters of temp_model to x_plus
            with torch.no_grad():
                temp_param = next(iter(temp_model.parameters()))
                temp_param.copy_(x_plus)

            # Compute loss at x_plus using temp_model
            temp_model.eval()  # Set to evaluation mode
            inputs, labels = next(iter(train_loader))
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = temp_model(inputs)
            loss = criterion(outputs, labels)

            # add regularization term at x_plus
            reg_j = (sigma_s / 2.0) * torch.norm(x_plus - x_bar_s) ** 2

            loss_x_plus = loss.item() + reg_j

            # calculate lhs and rhs of termination
            lhs = loss_x_plus - loss_x - torch.dot(grad_x, (x_plus - x_s).view(-1))
            rhs = (m_j + sigma) / 2 * torch.norm(x_plus - x_s) ** 2
            print(f"lhs={lhs}, rhs={rhs}")

            if lhs <= rhs:
                print(f"Backtracking converged in {j} iterations")
                m_s = m_j  # Terminate with M = Mj
                break
                # return m_j
            else:
                m_j *= 2  # Update Mj
        
        # backtracking algorithm ran until max_iter
        if j == (max_iter - 1):
            m_s = m_j 
            raise RuntimeError(f"Backtracking did not converge within {max_iter} iterations")
        
        # ---------------------------------------------------------------------------------------



        # Check for stopping condition, if not, set "prev" or s-1 parameters for next subproblem iteration
        if sigma_s >= m_s:
            print("Stopping condition met. Exiting loop.")
            break
        else:
            # Update "prev" parameters for next iteration
            x_prev = x_s.clone()
            x_bar_prev = x_bar_s.clone()
            sigma_prev = sigma_s
            m_prev = m_s

            print(f"Updated Parameters for Next Iteration:")
            print(f"x_(s-1): {x_prev}")
            print(f"x_bar_(s-1): {x_bar_prev}")
            print(f"sigma_(s-1): {sigma_prev}")
            print(f"m_(s-1): {m_prev}\n")


if __name__ == '__main__':
    main()
