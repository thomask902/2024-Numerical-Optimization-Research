import torch
import torch.optim as optim
from gam import GAM
from models.resnet import resnet18_c
from utils.gam_schedulers import ProportionScheduler
from utils.data_loader import get_dataloaders
from utils.train_utils_gam import train_epoch_gam, evaluate_model

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Initialize model and base optimizer
    model = resnet18_c().to(device)
    base_optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Initialize GAM optimizer and args
    class Args:
        def __init__(self, grad_norm_rho, grad_rho, grad_beta_0, grad_beta_1, grad_gamma):
            self.grad_norm_rho = grad_norm_rho
            self.grad_rho = grad_rho
            self.grad_beta_0 = grad_beta_0
            self.grad_beta_1 = grad_beta_1
            self.grad_beta_2 = 1 - grad_beta_0
            self.grad_beta_3 = 1 - grad_beta_1
            self.grad_gamma = grad_gamma
    
    args = Args(grad_norm_rho=0.2, grad_rho=0.02, grad_beta_0=0.5, grad_beta_1=0.6, grad_gamma=0.03)


    optimizer = GAM(params=model.parameters(), base_optimizer=base_optimizer, model=model,
                    adaptive=False, args=args)

    # Get data loaders
    trainloader, testloader = get_dataloaders()
    
    # Train the model for 2 epochs
    for epoch in range(2):
        print(f'Epoch {epoch + 1}')
        train_epoch_gam(model, trainloader, optimizer, device)
        # Evaluate the model
        evaluate_model(model, testloader, device)
    
    print('Finished Training')

    PATH = './ResNet18_2_epoch_CIFAR 10.pth'
    torch.save(model.state_dict(), PATH)
    
    

if __name__ == '__main__':
    main()