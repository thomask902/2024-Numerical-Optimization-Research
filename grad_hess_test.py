import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from utils.pyhessian import hessian
from utils.density_plot import get_esd_plot
import os

'''
# function to return gradient and hessian information after training is complete
def grad_hess_info(model, train_loader, optimizer, gpu):
    # first need to use train loader to get one batch and then pass into model and use loss function to compute gradients (get from model validate)
    model.eval()

    def loss_fn(predictions, targets):
        return smooth_crossentropy(predictions, targets).mean()

    for data in train_loader:
        break

    if gpu == -1:
        device = torch.device('cpu')
        images, targets = data[0].to(device), data[1].to(device)
    else:
        images = data[0].cuda(gpu, non_blocking=True)
        targets = data[1].cuda(gpu, non_blocking=True)

    # calls set_closure from optimizer to set up GAM with information
    optimizer.set_closure(loss_fn, images, targets)

    # then take gradients and find norm
    grad_vec = torch.cat([p.grad.contiguous().view(-1) for p in model.parameters()])

    # Compute gradient vector norm
    grad_vec_norm = torch.norm(grad_vec)

    # then pass to PyHessian and find hessian spectra

    return grad_vec_norm
'''


# TESTING

if __name__ == '__main__':

    model = models.resnet18()

    
    # Define the path to the saved model
    model_saved_path = os.path.join("saved_models", "CIFAR10", "GAM", "basicaug", "lr-0.1", "batchsize-128", "2024-08-04-19:31:19", "model.pth")

    # Load the saved model state dictionary
    model.load_state_dict(torch.load(model_saved_path))
    

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # basic data loader
    train_dataset = datasets.CIFAR10(
                root="",
                train=True,
                download=True,
                transform=transform
                )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= 4, shuffle=True, num_workers= 1)

    criterion = nn.CrossEntropyLoss()

    for data in train_loader:
                break


    device = torch.device('cpu')
    images, targets = data[0].to(device), data[1].to(device)
    hessian_comp = hessian(model, criterion, data=(images, targets), cuda=False)
            
            
    top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues()
    print("Largest Hessian Eigenvalue: %.4f"%top_eigenvalues[-1])

    grad_norm = hessian_comp.get_gradient_norm()
    print(f"Norm of the Gradient: {grad_norm:.10e}")


    density_eigen, density_weight = hessian_comp.density()

    # Iterate over each sublist in eigen_list_full to find smallest eigenvalue
    smallest_value = float('inf')

    for eigen_list in density_eigen:
        min_value_in_list = min(eigen_list)
        if min_value_in_list < smallest_value:
            smallest_value = min_value_in_list
    
    print(f"Smallest Hessian Eigenvalue: {smallest_value:.4f}")

    # Get the current file's directory
    current_file_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the path where the plot will be saved
    #plot_path = os.path.join(current_file_dir, "eigenvalue_test")
    # get_esd_plot(density_eigen, density_weight, plot_path)

    # test saving model
    dataset = "CIFAR10"
    log_description = "GAM"
    aug = "basicaug"
    learning_rate = "lr-0.1"
    batch_size = "batchsize-128"
    timestamp = "2024-08-04-19:31:19" 

    '''
    # Construct the path
    model_saved_path = os.path.join("saved_models", dataset, log_description, aug, learning_rate, batch_size, timestamp, "model.pth")
    # saving model to find gradient and hessian information (NOT NECESSARY)
    model_path = os.path.join(current_file_dir, model_saved_path)
    model_directory = os.path.dirname(model_path)
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    torch.save(model.state_dict(), model_path)
    '''

    

