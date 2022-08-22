import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
from tensorboardX import SummaryWriter

import argparse
from tqdm import tqdm

import utils.utils as utils
from utils.datasets import ImageDataset
from demo.mlnet import MLNet


def get_args_parser():

    parser = argparse.ArgumentParser(description='MLNet training script', add_help=False)
    # Data
    parser.add_argument('--dataset', default='salicon', type=str, choices=['salicon'])
    parser.add_argument('--root', default='./dataset/SALICON', type=str,
                        help='path to dataset root')
    parser.add_argument('--output-dir', default='./output/mlnet/checkpoint', type=str, help='output dir')
    # Model
    parser.add_argument('--model', default='MLNet', type=str, choices=['MLNet'])
    parser.add_argument('--resume', default='', type=str, help='path to resume from')
    # Training
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--warmup-epochs', default=1, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='number of samples per-device/per-gpu')
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--lr-end', default=1e-5, type=float,
                        help='minimum final lr')
    parser.add_argument('--wd', default=0.1, type=float)
    parser.add_argument('--betas', default=(0.9, 0.98), nargs=2, type=float)
    parser.add_argument('--eval-freq', default=5, type=int)
    # System
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--evaluate', action='store_true', help='eval only')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    return parser


def main(args):
    os.makedirs("./output/mlnet/", exist_ok=True)
    utils.init_distributed_mode(args)

    global best_loss

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # create model
    print("=> creating model: {}".format(args.model))
    model = MLNet()
    model.cuda(args.gpu)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], bucket_cap_mb=200)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=args.betas, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.warmup_epochs, 
                T_mult=args.epochs, eta_min=args.lr_end)
    criterion = nn.MSELoss()

    cudnn.benchmark = True

    print("=> creating dataset")
    transforms_ = transforms.Compose([
        transforms.Resize((256,256), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.471, 0.448, 0.408), std=(0.234, 0.239, 0.242))
    ])

    dataset = ImageDataset(root=args.root, split='train', transform=transforms_)
    val_dataset = ImageDataset(root=args.root, split='val', transform=transforms_)

    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        sampler = None
        val_sampler = None

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=(sampler is None),
                num_workers=args.workers, pin_memory=True, sampler=sampler, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
                num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False)

    if args.evaluate:
        return 

    print(args)

    #  set up logger
    writer = SummaryWriter("./output/mlnet/tb")

    best_loss = 1e2
    print("=> beginning training")
    for epoch in range(args.epochs):
        scheduler.step()
        loss = train(model, dataloader, criterion, optimizer, epoch, args)
        
        writer.add_scalar('train_loss', loss.avg, global_step = epoch)
    

        if epoch % args.eval_freq != 0:
            continue
        
        loss = evaluate(model, val_dataloader, criterion, args)

        writer.add_scalar('test_loss', loss.avg, global_step = epoch)


        is_best = best_loss > loss.avg
        best_loss = loss.avg
        if args.distributed:
            weight = model.module.state_dict()
        else:
            weight = model.state_dict()
        print("=> saving checkpoint")
        utils.save_on_master({
                'epoch': epoch + 1,
                'state_dict': weight,
                'optimizer' : optimizer.state_dict(),
                'best_acc1': best_loss,
                'args': args,
            }, is_best, args.output_dir)
        

def train(model, dataloader, criterion, optimizer, epoch, args):

    train_loss = utils.AverageMeter('train_loss')
    model.train()
    
    for data_iter, data in enumerate(dataloader):

        img, maps = data
        img = img.cuda(args.gpu)
        maps = maps.cuda(args.gpu)
        batchsize = img.size(0)

        output = model(img)
        loss = criterion(output, maps)
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(),batchsize/args.batch_size)

        if data_iter % args.print_freq == 0:
            print('[Epoch: %i/%i] [Batch: %i/%i] train_loss: %f' % (epoch, args.epochs, data_iter, len(dataloader), train_loss.avg))

    return train_loss


def evaluate(model, dataloader, criterion, args):
    
    with torch.no_grad():
        test_loss = utils.AverageMeter('test_loss')
        model.eval()
        
        for data_iter, data in tqdm(enumerate(dataloader), total=len(dataloader)):

            img, maps = data
            img = img.cuda(args.gpu)
            maps = maps.cuda(args.gpu)
            batchsize = img.size(0)
            output = model(img)
            loss = criterion(output, maps)

            test_loss.update(loss.item(),batchsize/args.batch_size)

        print('[Test] test_loss: %f' % (test_loss.avg))

    return test_loss


if __name__=='__main__':
    
    parser = argparse.ArgumentParser('MLNet training script', parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)