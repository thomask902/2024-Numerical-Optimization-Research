import torch
from models.resnet import resnet18_c
from utils.data_loader import get_dataloaders
from utils.train_utils import get_optimizer, train_epoch, train_epoch_smooth, evaluate_model

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Initialize model, criterion, optimizer
    # model = Net().to(device)
    model = resnet18_c().to(device)
    #criterion = torch.nn.CrossEntropyLoss()
    optimizer = get_optimizer(model)
    
    # Get data loaders
    trainloader, testloader = get_dataloaders()
    
    # Train the model
    for epoch in range(2):  # Loop over the dataset multiple times
        print(f'Epoch {epoch + 1}')
        #train_epoch(model, trainloader, criterion, optimizer, device)
        train_epoch_smooth(model, trainloader, optimizer, device)
    
    print('Finished Training')
    
    # Evaluate the model
    evaluate_model(model, testloader, device)

if __name__ == '__main__':
    main()