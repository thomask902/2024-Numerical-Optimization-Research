# Setting up local training for GAM

import torch
from utils.smooth_cross_entropy import smooth_crossentropy


def train_epoch_gam(model, trainloader, optimizer, gpu, args):

    def loss_fn(predictions, targets):
        return smooth_crossentropy(predictions, targets).mean()

    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        if gpu == -1:
            device = torch.device('cpu')
            images, target = data[0].to(device), data[1].to(device)
        else:
            images = data[0].cuda(gpu, non_blocking=True)
            target = data[1].cuda(gpu, non_blocking=True)
        
        # calls set_closure from optimizer to set up GAM with information
        optimizer.set_closure(loss_fn, images, target)

        # calls step() from GAM to use info from closure to run steps
        predictions, loss = optimizer.step()

        # updates rho "radius/ball" for both the gradient and grad norm within update rho t, and calls accuracy functions and updates meters
        # with torch.no_grad():
        #    optimizer.update_rho_t()

        # zeros gradients to clear them for next batch so they don't add up
        optimizer.zero_grad()
        
        running_loss += loss.item()
        if (i + 1) % args.print_freq == 0:
            print(f'Batch {i + 1}, Loss: {running_loss / args.print_freq:.4f}')
            running_loss = 0.0

    if args.GNOM_noised:
        percentage_noise, noise_count = optimizer.get_noise_statistics()
        print(f"Noise applied in {noise_count} out of {optimizer.total_batches} batches, "
            f"{percentage_noise:.2f}")

    # error catching if no loss value
    if torch.isnan(loss).any():
        raise SystemExit('NaN！')

def train_epoch_noised(model, trainloader, gradloader, accum_steps, optimizer, gpu, args):
    def loss_fn(predictions, targets):
        return smooth_crossentropy(predictions, targets).mean()

    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        
        # load data for checking gradient norm
        for data in gradloader:
            if gpu == -1:
                device = torch.device('cpu')
                inputs, targets = data[0].to(device), data[1].to(device)
            else:
                inputs = data[0].cuda(gpu, non_blocking=True)
                targets = data[1].cuda(gpu, non_blocking=True)
            optimizer.accum_grad(loss_fn, inputs, targets, accum_steps)

        # check if noise threshold hit and add noise if so
        optimizer.noise()

        # zero gradients and prepare for normal optimization steps
        optimizer.zero_grad()
        
        # begin normal optimization steps for the batch, after noise has or hasn't been applied
        if gpu == -1:
            device = torch.device('cpu')
            images, target = data[0].to(device), data[1].to(device)
        else:
            images = data[0].cuda(gpu, non_blocking=True)
            target = data[1].cuda(gpu, non_blocking=True)
        
        # calls set_closure from optimizer to set up GAM with information
        optimizer.set_closure(loss_fn, images, target)

        # calls step() from GAM to use info from closure to run steps
        predictions, loss = optimizer.step()

        # zeros gradients to clear them for next batch so they don't add up
        optimizer.zero_grad()
        
        running_loss += loss.item()
        if (i + 1) % args.print_freq == 0:
            print(f'Batch {i + 1}, Loss: {running_loss / args.print_freq:.4f}')
            running_loss = 0.0

    percentage_noise, noise_count = optimizer.get_noise_statistics()
    print(f"Noise applied in {noise_count} out of {optimizer.total_batches} batches, "
        f"{percentage_noise:.2f}")

    # error catching if no loss value
    if torch.isnan(loss).any():
        raise SystemExit('NaN！')

def train_epoch_base(model, trainloader, optimizer, gpu, print_freq):
    
    def loss_fn(predictions, targets):
        return smooth_crossentropy(predictions, targets).mean()
    
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        if gpu == -1:
            device = torch.device('cpu')
            images, target = data[0].to(device), data[1].to(device)
        else:
            images = data[0].cuda(gpu, non_blocking=True)
            target = data[1].cuda(gpu, non_blocking=True)
        
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(images)  # Forward pass
        loss = loss_fn(outputs, target)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize the model parameters
        
        running_loss += loss.item()
        if (i + 1) % print_freq == 0:
            print(f'Batch {i + 1}, Loss: {running_loss / print_freq:.4f}')
            running_loss = 0.0

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
                images = data[0].cuda(gpu, non_blocking=True)
                target = data[1].cuda(gpu, non_blocking=True)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    return 100 * correct / total