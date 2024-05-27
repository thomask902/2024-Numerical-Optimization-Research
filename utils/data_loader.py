import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data

def get_dataloaders(batch_size=40):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    trainset = datasets.CIFAR10(root='./data', train=True,
                                download=True, transform=transform)
    trainloader = data.DataLoader(trainset, batch_size=batch_size,
                                  shuffle=True, num_workers=2)
    
    testset = datasets.CIFAR10(root='./data', train=False,
                               download=True, transform=transform)
    testloader = data.DataLoader(testset, batch_size=batch_size,
                                 shuffle=False, num_workers=2)
    
    return trainloader, testloader