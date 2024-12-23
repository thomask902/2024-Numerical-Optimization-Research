# TO RUN:
# python3 main_cifar.py --batch-size 32 --gpu -1 --epochs 1

import argparse
from datetime import datetime
import os
import random
import warnings
from xmlrpc.client import Boolean
import models
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter
from utils.train_utils_gam import train_epoch_gam, evaluate_model, train_epoch_base, train_epoch_noised
from utils.optimizer_helper import get_optim_and_schedulers
from utils.cutout import Cutout
from utils.auto_augment import CIFAR10Policy
from utils.rand_augment import RandAugment

from utils.pyhessian import hessian
from utils.density_plot import get_esd_plot
from utils.smooth_cross_entropy import smooth_crossentropy
# from utils.grad_hess_info import grad_hess_info

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

print('model name space', model_names)
parser = argparse.ArgumentParser(description='PyTorch CIFAR Training with GAM')

# model & training
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18_c',
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr-scheduler', default='cosine', type=str,
                    help='set learning rate scheduler')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--epochs_decay', default=[30, 60], type=int,
                    help='seed for initializing training. ')

parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use. -1 if CPU.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--benchmark', default=True, type=bool,
                    help='GPU id to use.')

# data
parser.add_argument('--cifar10_path', metavar='DIR_C', default='.',
                    help='path to dataset')
parser.add_argument('--cifar100_path', metavar='DIR_C', default='.',
                    help='path to dataset')
parser.add_argument('--dataset', default='CIFAR10', type=str)
parser.add_argument('--cutout', default= False, type=bool)
parser.add_argument('--auto-augment', default=False, type=bool)
parser.add_argument('--rand-augment', default=False, type=bool)
parser.add_argument('--log_base',
                    default='./results', type=str, metavar='PATH',
                    help='path to save logs (default: none)')

# opt
parser.add_argument("--base_opt", default='SGD', type=str, help="")
parser.add_argument("--no_gam", default=False, type=bool, 
                    help="set to true to train only on base optimizer")
parser.add_argument("--gam_nonaccel", default=False, type=bool,
                    help='if set to true will run non-accelerated gam')
parser.add_argument("--GNOM", default=False, type=bool,
                    help='if true will run gradient norm only minimization')
parser.add_argument("--GNOM-noised", default=False, type=bool,
                    help='if true will run gradient norm only minimization with noise')
parser.add_argument("--noise-threshold", default=0.1, type=float,
                    help='sets the gradient norm threshold to add noise')
parser.add_argument("--noise-radius", default=0.01, type=float,
                    help='sets the ball radius to add noise from')
parser.add_argument("--grad-approx-samples", default=1024, type=int,
                    help='sets number of samples to approx gradient')
parser.add_argument("--newtonMR", default=False, type=bool,
                    help='if true will run newtons MR minimization')


parser.add_argument("--grad_beta_0", default=1., type=float, help="scale for g0")
parser.add_argument("--grad_beta_1", default=1., type=float, help="scale for g1")
parser.add_argument("--grad_beta_2", default=-1., type=float, help="scale for g2")
parser.add_argument("--grad_beta_3", default=1., type=float, help="scale for g3")

# parser.add_argument("--grad_rho_max", default=0.04, type=int, help="")
# parser.add_argument("--grad_rho_min", default=0.02, type=int, help="")
parser.add_argument("--grad_rho", default=0.02, type=int, help="")

# parser.add_argument("--grad_norm_rho_max", default=0.04, type=int, help="")
# parser.add_argument("--grad_norm_rho_min", default=0.02, type=int, help="")
parser.add_argument("--grad_norm_rho", default=0.2, type=int, help="")

parser.add_argument("--adaptive", default=False, type=bool, help="")
parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")

parser.add_argument("--grad_gamma", default=0.03, type=int, help="")

# outputs
parser.add_argument("--print-grad-info", default=False, type=bool, 
                    help="if true will output norm of final gradient values and hessian information")

return_acc = 0

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def main():
    torch.backends.cudnn.benchmark = True
    args = parser.parse_args()

    # default hps
    if args.dataset == 'CIFAR100':
        args.rho = 0.1 # GAM non-accel rho
        if args.arch.startswith('resnet'):
            args.grad_norm_rho, args.grad_rho, args.grad_beta_0, args.grad_beta_1, args.grad_gamma = 0.2, 0.02, 0.5, 0.6, 0.03
        elif args.arch.startswith('pyramidnet'):
            args.grad_norm_rho, args.grad_rho, args.grad_beta_0, args.grad_beta_1, args.grad_gamma = 0.2, 0.04, 0.3, 0.5, 0.05
    elif args.dataset == 'CIFAR10':
        args.rho = 0.04 # GAM non-accel rho
        if args.arch.startswith('resnet'):
            args.grad_norm_rho, args.grad_rho, args.grad_beta_0, args.grad_beta_1, args.grad_gamma = 0.2, 0.03, 0.1, 0.1, 0.05
        elif args.arch.startswith('pyramidnet'):
            args.grad_norm_rho, args.grad_rho, args.grad_beta_0, args.grad_beta_1, args.grad_gamma = 0.2, 0.03, 0.1, 0.1, 0.03

    # Non-accelerated GAM alpha/scaling parameter
    if args.arch == 'resnet18_c': 
        args.alpha = 0.3
    else:
        args.alpha = 0.1


    args.grad_beta_2 = 1 - args.grad_beta_0
    args.grad_beta_3 = 1 - args.grad_beta_1
    
    if args.gam_nonaccel:
        log_description = "GAMNonAccelerated"
    elif args.GNOM:
        log_description = "GNOM"
    elif args.GNOM_noised:
        log_description = "GNOM_noised"
    elif args.no_gam:
        log_description = "SGD"
    elif args.newtonMR:
        log_description = "newtonMR"
    else: 
        log_description = 'GAM'

    if args.cutout:
        aug = 'cutout'
    elif args.rand_augment:
        aug = "randaug"
    elif args.auto_augment:
        aug = "autoaug"
    else:
        aug = "basicaug"

    
    timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    learning_rate = "lr-" + str(args.lr)
    batch_size = "batchsize-" + str(args.batch_size)
    args.log_path = os.path.join(args.log_base, args.dataset, log_description, aug, learning_rate, batch_size, str(timestamp), "log.txt")
    args.model_saved_path = os.path.join("saved_models", args.dataset, log_description, aug, learning_rate, batch_size, str(timestamp), "model.pth")
    args.eigenvalue_path = os.path.join("eigenvalues", args.dataset, log_description, aug, learning_rate, batch_size, str(timestamp), "eigenvalue.pdf")



    if args.seed is not None:
        # for reimplement
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None and args.gpu != -1:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count() if args.gpu != -1 else 1
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global return_acc
    args.gpu = gpu

    if args.gpu is not None and args.gpu != -1:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    num_classes = 100 if args.dataset == 'CIFAR100' else 10
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True, num_classes=num_classes)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](num_classes=num_classes)

    if args.distributed:
        if args.gpu is not None and args.gpu != 'cpu':
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and args.gpu != -1:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif args.gpu == -1:
        device = torch.device('cpu')
        model = model.to(device)
    else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu) if args.gpu != -1 else nn.CrossEntropyLoss()
    cudnn.benchmark = args.benchmark

    # image/data augmentations
    if args.dataset == 'CIFAR10':
        data_root = args.cifar10_path
        transform_list_10 = [  # adding basic augmentations
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ]

        if args.auto_augment:
            transform_list_10.append(CIFAR10Policy())

        if args.rand_augment:
            transform_list_10.append(RandAugment(n=3, m=4))
        
        transform_list_10.append(transforms.ToTensor())
        transform_list_10.append(transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124], std=[0.2023, 0.1994, 0.2010]))
        
        if args.cutout or args.auto_augment or args.rand_augment: # adding cutout if specified
            transform_list_10.append(Cutout(n_holes=1, length=16))

        train_dataset = datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose(transform_list_10)
            )

        test_dataset = datasets.CIFAR10(
            root=data_root,
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124], std=[0.2023, 0.1994, 0.2010])
            ]))

    else:
        data_root = args.cifar100_path
        transform_list_100 = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ]

        if args.auto_augment:
            transform_list_100.append(CIFAR10Policy())

        if args.rand_augment:
            transform_list_100.append(RandAugment(n=1, m=2))
        
        transform_list_100.append(transforms.ToTensor())
        transform_list_100.append(transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]))

        if args.cutout or args.auto_augment or args.rand_augment:
            transform_list_100.append(Cutout(n_holes=1, length=8))

        train_dataset = datasets.CIFAR100(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose(transform_list_100)
            )

        test_dataset = datasets.CIFAR100(
            root=data_root,
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
            ]))
    
    if args.gpu == -1:
        args.workers = 1

    # if we are doing gnom noised, taking n samples with no transforms and putting in their own set
    grad_approx_loader = None
    if args.GNOM_noised:
        if args.dataset == 'CIFAR10':
            train_dataset2 = datasets.CIFAR10(
                root=data_root,
                train=True,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124], std=[0.2023, 0.1994, 0.2010])
                ]))
        else:
            train_dataset2 = datasets.CIFAR100(
                root=data_root,
                train=True,
                download=True,
                transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
            ]))

        # Get all indices of the train dataset
        all_indices = np.arange(len(train_dataset2))

        # Randomly select `grad_approx_samples` samples
        np.random.shuffle(all_indices)
        grad_approx_indices = all_indices[:args.grad_approx_samples]
        remaining_indices = all_indices[args.grad_approx_samples:]

        # Create the grad_approx_dataset with only basic transformations
        grad_approx_dataset = torch.utils.data.Subset(train_dataset2, grad_approx_indices)

        print("Gradient Approximation Dataset:", len(grad_approx_dataset))

        grad_approx_loader = torch.utils.data.DataLoader(
            grad_approx_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        # GET RID OF THIS TO INCLUDE GRAD APPROX SAMPLES IN TRAIN DATASET
        train_dataset = torch.utils.data.Subset(train_dataset, remaining_indices)
        print("Train Dataset:", len(train_dataset))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    log_dir = os.path.dirname(args.log_path)
    print('tensorboard dir {}'.format(log_dir))
    tensor_writer = SummaryWriter(log_dir)

    # get base opt and schedulers (will return non-accelerated gam if it has been specified)
    optimizer, base_optimizer, lr_scheduler, grad_rho_scheduler, grad_norm_rho_scheduler = get_optim_and_schedulers(
        model, args)

    # check if training only with base optimizer and overwrite the GAM optimizer
    if args.no_gam:
        optimizer = base_optimizer

    start_time = time.time()

    if args.GNOM_noised:
        print("Gradient Approximation Samples:", args.grad_approx_samples)
        print("Number of Gradient Approximation Accumulation Batches:", int(args.grad_approx_samples / args.batch_size))

    # pass returned optimizers and schedulers into training loop
    for epoch in range(args.epochs):

        start_epoch = time.time()

        if args.distributed:
            train_sampler.set_epoch(epoch)
            
        if args.no_gam:
            train_epoch_base(model, train_loader, optimizer, gpu, args.print_freq)
        elif args.GNOM_noised:
            train_epoch_noised(model, train_loader, grad_approx_loader, int(args.grad_approx_samples / args.batch_size), optimizer, gpu, args)
        else:
            train_epoch_gam(model, train_loader, optimizer, gpu, args)

        if lr_scheduler is not None:
            lr_scheduler.step()
            current_lr = lr_scheduler.get_last_lr()[0]
            print(f"Epoch {epoch + 1} learning rate: {current_lr}")
            tensor_writer.add_scalar('Learning Rate', current_lr, epoch)
        else:
            print(f"Epoch {epoch + 1} learning rate:", args.lr)

        accuracy = evaluate_model(model, val_loader, gpu)

        end_epoch = time.time()
        elapsed_epoch = end_epoch - start_epoch
        print(f"Epoch {epoch + 1} time: {elapsed_epoch} seconds")

        # acc1 = validate(gpu, val_loader, model, criterion, True, args)
        # return_acc = max(return_acc, acc1)
        print(f"Epoch {epoch + 1} accuracy: {accuracy}%")
        tensor_writer.add_scalar('return_ACC@1/test', accuracy, epoch)

    print("rho: ", args.rho, ", alpha: ", args.alpha)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total training time: {elapsed_time} seconds")

    if args.print_grad_info:
        model.eval()

        def loss_fn(predictions, targets):
            return smooth_crossentropy(predictions, targets).mean()
        
        train_loader_info = torch.utils.data.DataLoader(
            train_dataset, batch_size=128, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        for data in train_loader_info:
            break

        if gpu == -1:
            device = torch.device('cpu')
            images, targets = data[0].to(device), data[1].to(device)
            hessian_comp = hessian(model, loss_fn, data=(images, targets), cuda=False)
        else:
            images = data[0].cuda(gpu, non_blocking=True)
            targets = data[1].cuda(gpu, non_blocking=True)
            hessian_comp = hessian(model, loss_fn, data=(images, targets), cuda=True)
        
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

        # Get the current file's directory and move one level up
        current_file_dir = os.path.dirname(os.path.abspath(__file__))

        # Define the path where the plot will be saved
        plot_path = os.path.join(current_file_dir, args.eigenvalue_path)
        get_esd_plot(density_eigen, density_weight, plot_path)

        # Output noisy algorithm details (if applicable)
        if args.GNOM_noised:
            print("Noise Threshold:", args.noise_threshold)
            print("Noise Radius:", args.noise_radius)

    # saving model to find gradient and hessian information (NOT NECESSARY)
    model_path = os.path.join(current_file_dir, args.model_saved_path)
    model_directory = os.path.dirname(model_path)
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
    main()
