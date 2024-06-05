# Setting up local training for GAM

import torch
from utils.smooth_cross_entropy import smooth_crossentropy


def train_epoch_gam(model, trainloader, optimizer, gpu):

    def loss_fn(predictions, targets):
        return smooth_crossentropy(predictions, targets).mean()

    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        if gpu == -1:
            device = torch.device('cpu')
            images, target = data[0].to(device), data[1].to(device)
        else:
            images = images.cuda(gpu, non_blocking=True)
            target = target.cuda(gpu, non_blocking=True)
        
        # calls set_closure from optimizer to go through GAM steps within that and return predictions and loss
        optimizer.set_closure(loss_fn, images, target)

        # calls step() from GAM
        predictions, loss = optimizer.step()

        # updates rho "radius/ball" for both the gradient and grad norm within update rho t, and calls accuracy functions and updates meters
        # with torch.no_grad():
        #    optimizer.update_rho_t()

        # zeros gradients to clear them for next batch so they don't add up
        optimizer.zero_grad()
        
        running_loss += loss.item()
        if (i + 1) % 100 == 0:  # Print every 100 mini-batches
            print(f'Batch {i + 1}, Loss: {running_loss / 100:.3f}')
            running_loss = 0.0

    # error catching if no loss value
    if torch.isnan(loss).any():
        raise SystemExit('NaN！')

def evaluate_model(model, testloader, gpu):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            if gpu == -1:
                device = torch.device('cpu')
                images, target = data[0].to(device), data[1].to(device)
            else:
                images = images.cuda(gpu, non_blocking=True)
                target = target.cuda(gpu, non_blocing=True)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    print(f'Accuracy: {100 * correct / total}%')