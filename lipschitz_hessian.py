import torch
import torch.nn as nn

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
    num_samples = 1

    # load in dataset
    data = torch.load(f'generated_data/n_2000_m_1000.pt')
    features = data['features']
    inputDim = features.shape[1]
    labels = data['labels'].unsqueeze(1).float()
    labels[labels == 0] = -1
    dataset = torch.utils.data.TensorDataset(features, labels)
  
    inputDim = dataset.tensors[0].shape[1]
    print("dim:", inputDim)

    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1000, num_workers=0, shuffle=False)
    
    # create model, loss function, and optimizer
    model = nn.Linear(inputDim, 1)

    device = torch.device('cpu') # change if GPU
    model.to(device)

    criterion = Squared_Hinge_Loss()
    # criterion = Sigmoid_Loss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1) # lr does not matter, will never step

    # create tensors to store results
    L = []

    for batch_idx, (inputs, labels) in enumerate(loader):

        inputs, labels = inputs.to(device), labels.to(device)
        
        # calculating gradient
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward(create_graph = True)

        # calculate hessian
        grad_params = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        grad_vec = torch.cat([g.contiguous().view(-1) for g in grad_params])

        # Initialize Hessian matrix
        hessian_size = grad_vec.size(0)
        hessian = torch.zeros(hessian_size, hessian_size).to(grad_vec.device)

        # Compute the Hessian matrix
        for i in range(hessian_size):
            # Gradient of the i-th element of grad_vec
            grad2_params = torch.autograd.grad(grad_vec[i], model.parameters(), retain_graph=True)
            grad2_vec = torch.cat([g.contiguous().view(-1) for g in grad2_params])
            # Fill the row of the Hessian
            hessian[i] = grad2_vec

        # Compute the eigenvalues of the Hessian
        eigenvalues = torch.linalg.eigvalsh(hessian)
        
        # Find the absolute value of the largest eigenvalue
        max_eigenvalue = torch.max(eigenvalues.abs()).item()

        L.append(max_eigenvalue)

        if batch_idx == (num_samples - 1):
            break
    
    # approximate Lipschitz constant for each pair
    lipschitz = max(L)
    print(f'Approximated Lipschitz: {lipschitz}')
            
            
if __name__ == '__main__':
    main()