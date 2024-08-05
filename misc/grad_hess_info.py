import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.datasets as datasets
from smooth_cross_entropy import smooth_crossentropy

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



# TESTING

if __name__ == '__main__':
    # Load and save a pretrained ResNet-18 model
    model = models.resnet18(pretrained=True)

    # basic data loader
    train_dataset = datasets.CIFAR10(
                root="",
                train=True,
                download=True
                )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= 32, num_workers= 1)

    optimizer = torch.optim.SGD(params=model.parameters())


    # passing to function to test
    grad_norm_test = grad_hess_info(model, train_loader, optimizer, -1)

    print(grad_norm_test)

