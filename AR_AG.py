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
    S = 100  # Maximum number of subproblems
    batch_size = 1000
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

    # Set up logging path
    timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    log_path = os.path.join(log_base, "generated", loss_type, data_name, "AR", "no-lr", str(S), str(batch_size), timestamp, "results.csv")
    log_directory = os.path.dirname(log_path)
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

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

    # Temporary model identical to the original model for later calculations
    temp_model = nn.Linear(input_dim, 1, bias=False).to(device)

    # Define loss
    if loss_type == "hinge":
        criterion = Squared_Hinge_Loss()
        initial_lipschitz = lipschitz_dict[data_name]["no_batching"]["hinge"]
    elif loss_type == "sigmoid":
        criterion = Sigmoid_Loss()
        initial_lipschitz = lipschitz_dict[data_name]["no_batching"]["sigmoid"]
    else:
        raise ValueError("Invalid loss function specified.")

    # ---------------------------------------------------------------------------------------
    # Subproblem Algorithm

    # Set initial parameters for subproblems
    if len(list(model.parameters())) == 1:
        x_prev = next(iter(model.parameters()))
    else:
        raise RuntimeError("Optimizer can only run on basic models")
    x_bar_prev = x_prev.clone().detach()
    sigma_prev = 0.0
    m_prev = initial_lipschitz
    sigma_1 = m_prev / 10.0  # can tune a bit

    # Print initial parameters
    print(f"Initial Parameters: sigma_0={sigma_prev}, m_0={m_prev}, sigma_1={sigma_1}")

    # Initialize variables for logging
    iteration_stats = []
    epoch_counter = 0  # To keep track of epochs across all subproblems

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
        print(f"Subproblem Iteration s={s}: sigma_s={sigma_s}, gamma_s={gamma_s}, m_(s-1)={m_prev}, l_s_k={l_s_k}, k approx.={k_approx}, k={k}")

        # because lipschitz approximation changes for each s, we redefine AG for each subproblem
        optimizer = AG(params=model.parameters(), model=model, loss_type=loss_type, lipschitz=m_prev + sigma_s)
        # optimizer = AG(params=model.parameters(), model=model, loss_type="sigmoid", lipschitz=m_prev)

        # Run subroutine for k iterations, loading in data, computing gradient w/ regularization and stepping
        train_loss = 0.0
        for i in range(0, k):
            if (i + 1) % 50 == 0:
                print (f"Iteration {i+1}/{k} for subproblem {s}")

            epoch_counter += 1
            start_time = time.time()

            inputs, labels = next(iter(train_loader))
            inputs, labels = inputs.to(device), labels.to(device)

            # Define a custom closure
            def closure():
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Add regularization
                x_s = next(iter(model.parameters()))
                reg_s = (sigma_s / 2.0) * torch.norm(x_s - x_bar_s) ** 2
                loss += reg_s

                # Backprop
                loss_value = loss.data.clone().detach()
                loss.backward()
                return outputs, loss_value

            # Perform optimization step with defined closure
            outputs, loss_value, train_norm, x_k_diff, x_ag_k_diff = optimizer.step(closure=closure)
            end_time = time.time()
            train_time = end_time - start_time
            train_loss = loss_value.item()

            # Evaluate on test data
            test_loss, test_norm, accuracy = evaluate(model, test_loader, criterion, device)

            # Log iteration stats
            iteration_stats.append({
                "Subproblem": s,
                "Iteration": i + 1,
                "Epoch": epoch_counter,
                "Training Loss": train_loss,
                "Training Gradient Norm": train_norm,
                "Training Time (s)": train_time,
                "Test Loss": test_loss,
                "Test Gradient Norm": test_norm,
                "Test Accuracy": accuracy,
                "Test Error": 1 - accuracy,
                "x_k Comparison": x_k_diff,
                "x_ag_k Comparison": x_ag_k_diff
            })

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
        model.zero_grad()
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
            #print(f"lhs={lhs}, rhs={rhs}")

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

            print(f"Updated Parameters for Next Iteration: sigma_(s-1)={sigma_prev}, m_(s-1)={m_prev}")
    
    # Save results to CSV
    df_stats = pd.DataFrame(iteration_stats)
    df_stats.to_csv(log_path, index=False)
    print(df_stats)


if __name__ == '__main__':
    main()
