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
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.metaformers import default_cfgs as default_cfgs






def test_model(model, crop):
    
    
    FLAGS = {}

    FLAGS['interpolation'] = torchvision.transforms.InterpolationMode.BICUBIC
    FLAGS['image_size'] = model.default_cfg['input_size'][2]
    FLAGS['crop'] = crop

    FLAGS['rootPath'] = "/media/fredo/KIOXIA/Datasets/imagenet/data/"
    FLAGS['imageRoot'] = FLAGS['rootPath'] + 'val/'

    FLAGS['batch_size'] = 64

    FLAGS['image_size_initial'] = int(round(FLAGS['image_size'] // FLAGS['crop']))
    
    transform = transforms.Compose([
        transforms.Resize(FLAGS['image_size_initial'], interpolation = FLAGS['interpolation']),
        transforms.CenterCrop((int(FLAGS['image_size']),int(FLAGS['image_size']))),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    dataset = torchvision.datasets.ImageFolder(FLAGS['imageRoot'], transform=transform)

    loader = torch.utils.data.DataLoader(dataset,
        batch_size = FLAGS['batch_size'],
        shuffle=True,
        num_workers=10,
        prefetch_factor=2, 
        pin_memory = True,  
        generator=torch.Generator().manual_seed(42))
        
    #model = timm.create_model('convformer_b36.sail_in22k_ft_in1k_384', pretrained=True)
    #import metaformer_baselines
    #model = metaformer_baselines.identityformer_s12v1(pretrained=True)
    model.eval()
    #print("got model")

    #import torch_directml
    #device = torch_directml.device()
    #device = torch.device('cpu')
    #device = torch.device("mps")

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    #print(device)
    model = model.to(device)
    
    #model = torch.jit.script(model)
    #model = torch.jit.optimize_for_inference(model)

    #print("starting run")

    startTime = time.time()
    cycleTime = time.time()
    samples = 0
    correct = 0
    stepsPerPrintout = 100

    for i, (images, tags) in enumerate(loader):
        imageBatch = images.to(device, non_blocking=True)
        tagBatch = tags.to(device, non_blocking=True)
        #tagsOneHot = torch.nn.functional.one_hot(tags, num_classes = len(classes)).to(device, non_blocking=True)
        with torch.set_grad_enabled(False):
            outputs = model(imageBatch)
            preds = torch.argmax(outputs, dim=1)
            samples += len(images)
            correct += sum(preds == tagBatch)


        '''
        if i % stepsPerPrintout == 0:
            accuracy = 100 * (correct/(samples+1e-8))
            imagesPerSecond = (FLAGS['batch_size']*stepsPerPrintout)/(time.time() - cycleTime)
            cycleTime = time.time()

            print('[%d/%d]\tImages/Second: %.4f\ttop-1: %.2f' % (i, len(loader), imagesPerSecond, accuracy))
        '''

    print(f'model: {model.pretrained_cfg["architechture"]}.{model.pretrained_cfg["tag"]}, crop: {crop}, top-1: {100 * (correct/samples)}%, spent {(time.time() - startTime):.0f} seconds')
    model = model.cpu()
    del model



def main():
    models = default_cfgs.keys()
    crop_bins=[1.00, 0.975, 0.95, 0.925, 0.90, 0.875, 0.85, 0.825, 0.8, 0.75]
    print('starting run')
    for currModel in models:
        for cfg in list(default_cfgs[currModel].cfgs.keys()):
            if 'in1k' in cfg:
                model = timm.create_model(currModel+'.'+cfg, pretrained=True)
                for currCrop in crop_bins:
                    test_model(model, currCrop)


if __name__ == '__main__':
    main()