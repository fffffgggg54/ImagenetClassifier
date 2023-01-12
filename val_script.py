# validation script that does a pass over the val set
# currently set up for directml


import torch
import torch.cuda.amp
import torch.nn as nn
import torch.nn.parallel
import torch.profiler
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
from torchvision import models, transforms
from torchvision.io import read_image
import torchvision.utils as vutils
import numpy as np
import time
import glob
import gc
import pandas as pd
import os
import timm
from PIL import Image, ImageOps, ImageDraw
import PIL

torch.set_default_tensor_type('torch.FloatTensor')

FLAGS = {}

FLAGS['rootPath'] = "./data/"
FLAGS['imageRoot'] = FLAGS['rootPath'] + 'val/'

FLAGS['batch_size'] = 64
FLAGS['image_size'] = 224

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

transform = transforms.Compose([
    transforms.Resize((FLAGS['image_size'],FLAGS['image_size'])),
    #transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
])
dataset = torchvision.datasets.ImageFolder(FLAGS['imageRoot'], transform=transform)

loader = torch.utils.data.DataLoader(dataset,
    batch_size = FLAGS['batch_size'],
    shuffle=False,
    num_workers=0,
    prefetch_factor=2, 
    pin_memory = True,  
    generator=torch.Generator().manual_seed(41))
	
model = timm.create_model('tinynet_e', pretrained=True)
model.eval()
print("got model")

import torch_directml
device = torch_directml.device()

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)

print("starting run")

startTime = time.time()
cycleTime = time.time()
samples = 0
correct = 0
stepsPerPrintout = 50

for i, (images, tags) in enumerate(loader):
    imageBatch = images.to(device, non_blocking=True)
    tagBatch = tags.to(device, non_blocking=True)
    #tagsOneHot = torch.nn.functional.one_hot(tags, num_classes = len(classes)).to(device, non_blocking=True)
    with torch.set_grad_enabled(False):
        outputs = model(imageBatch)
        preds = torch.argmax(outputs, dim=1)
        samples += len(images)
        correct += sum(preds == tagBatch)



    if i % stepsPerPrintout == 0:
        accuracy = 100 * (correct/(samples+1e-8))
        imagesPerSecond = (FLAGS['batch_size']*stepsPerPrintout)/(time.time() - cycleTime)
        cycleTime = time.time()

        print('[%d/%d]\tImages/Second: %.4f\ttop-1: %.2f' % (i, len(loader), imagesPerSecond, accuracy))


print(f'top-1: {100 * (correct/samples)}%')
