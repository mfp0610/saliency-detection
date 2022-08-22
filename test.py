import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from PIL import Image
from tqdm import tqdm

import utils.utils as utils
from utils.datasets import ImageDataset
from demo.mlnet import MLNet


import matplotlib.pyplot as plt
import torchvision.transforms.functional as F


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        img.save("./output/mlnet/" + "mlnet_img.jpg")
        
if __name__ == "__main__":
    # load the data
    bs = 3 # change here for the number of picture testing
    root = './dataset/SALICON'
    transforms_ = transforms.Compose([
        transforms.Resize((256,256), interpolation = Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.471, 0.448, 0.408), std = (0.234, 0.239, 0.242))
    ])
    dataset = ImageDataset(root=root, split = 'val', transform = transforms_)
    dataloader = DataLoader(dataset, batch_size = bs)

    # load the checkpoint
    ckp_path = './output/mlnet/checkpoint/checkpoint_best.pt'
    checkpoint = torch.load(ckp_path, map_location = 'cpu')
    checkpoint_model = checkpoint['state_dict']

    # load the model
    model = MLNet()
    model.load_state_dict(checkpoint_model)
    model.cuda()
    model.eval()

    for i, data in enumerate(dataloader):
        imgs, maps = data
        imgs = imgs.cuda()
        maps = maps.cuda()
        output = model(imgs)
        break

    result = torch.cat((imgs, maps.repeat(1,3,1,1), output.repeat(1,3,1,1)), dim=0)
    grid = make_grid(result, nrow=bs, normalize=True)
    show(grid)