import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd



class linearRegression(nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

class logisticRegression(nn.Module):
    def __init__(self, inputSize):
        super(logisticRegression, self).__init__()
        self.linear = nn.Linear(inputSize, 1)

    def forward(self, x):
        out = torch.sigmoid(self.linear(x))
        return out

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

    # define number of samples to approximate Lipschitz with
    num_samples = 1000

    # importing the dataset
    # y = pd.read_csv("communities+and+crime/targets_cleaned.csv")
    # X = pd.read_csv("communities+and+crime/features_cleaned.csv")
    # y = pd.read_csv("arcene/targets_cleaned.csv")
    # X = pd.read_csv("arcene/features_cleaned.csv")

    # FOR GENERATED .pt DATASETS

    # load in dataset
    data = torch.load(f'generated_data/n_2000_m_1000_balanced.pt')
    features = data['features']
    inputDim = features.shape[1]
    labels = data['labels'].unsqueeze(1).float()
    dataset = torch.utils.data.TensorDataset(features, labels)

  
    inputDim = dataset.tensors[0].shape[1]
    print("dim:", inputDim)
    # inputDim = X.shape[1]
    # outputDim = y.shape[1]

    # convert data to tensors and create loader
    #X_vals = torch.tensor(X.values, dtype=torch.float32)
    #y_vals = torch.tensor(y.values, dtype=torch.float32)
    #data_torch = torch.utils.data.TensorDataset(X_vals, y_vals)

    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, num_workers=0, shuffle=False)
    
    # create model, loss function, and optimizer
    # model = linearRegression(inputDim, outputDim)
    # model = logisticRegression(inputDim)
    model = nn.Linear(inputDim, 1)

    device = torch.device('cpu') # change if GPU
    model.to(device)

    # criterion = torch.nn.MSELoss()
    # criterion = torch.nn.BCELoss()
    # criterion = Squared_Hinge_Loss()
    criterion = Sigmoid_Loss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1) # lr does not matter, will never step

    # create tensors to store results
    samples_data = torch.zeros((num_samples, inputDim))
    hvp_data = torch.zeros((num_samples, inputDim + 1))

    for batch_idx, (inputs, labels) in enumerate(loader):

        inputs, labels = inputs.to(device), labels.to(device)
        
        # calculating gradient
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward(create_graph = True)

        # calculating hessian vector product
        grad_vec = torch.cat([p.grad.contiguous().view(-1) for p in model.parameters()])
        hessian_vec_prod_dict = torch.autograd.grad(
            grad_vec, model.parameters(), grad_outputs=grad_vec, only_inputs=True
        )
        hessian_vec_prod = torch.cat([g.contiguous().view(-1) for g in hessian_vec_prod_dict])

        # add results to tensors
        samples_data[batch_idx] = inputs[0]
        hvp_data[batch_idx] = hessian_vec_prod

        if batch_idx == (num_samples - 1):
            break
    
    # approximate Lipschitz constant for each pair
    L = []

    for i in range(num_samples):
        for j in range(i + 1, num_samples):

            hvp_i = hvp_data[i]
            hvp_j = hvp_data[j]
            hvp_diff = torch.norm(hvp_i - hvp_j)
            
            sample_i = samples_data[i]
            sample_j = samples_data[j]
            diff = torch.norm(sample_i - sample_j)

            lip = hvp_diff.item() / (diff.item() + 0.00001)

            L.append(lip)

    lipschitz = sum(L) / len(L)
    print(f'Approximated Lipschitz: {lipschitz}')
            
            
if __name__ == '__main__':
    main()