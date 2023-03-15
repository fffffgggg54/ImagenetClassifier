import torch
import torch.cuda.amp
import torch.nn as nn
import torch.nn.parallel
import torch.profiler
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
from torchvision import models, transforms
from torchvision.io import read_image
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import pandas as pd
import numpy as np
import time
import glob
import gc
import os
import torchvision
import torch_optimizer

import multiprocessing

import timm
import timm.optim
import transformers

import timm.layers.ml_decoder as ml_decoder
import timm.layers
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, AsymmetricLossSingleLabel
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.random_erasing import RandomErasing
from timm.data.auto_augment import rand_augment_transform
from timm.data.transforms import RandomResizedCropAndInterpolation
from timm.data.mixup import FastCollateMixup, Mixup

from PIL import Image, ImageOps, ImageDraw
import PIL
import random


torch.backends.cudnn.benchmark = True

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
#torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
#torch.backends.cudnn.allow_tf32 = True

timm.layers.fast_norm.set_fast_norm(enable=True)

# ================================================
#           CONFIGURATION OPTIONS
# ================================================

# TODO use a configuration file or command line arguments instead of having a bunch of variables

FLAGS = {}

# path config for various directories and files
# TODO replace string appending with os.path.join()

FLAGS['rootPath'] = "/media/fredo/KIOXIA/Datasets/imagenet/"
FLAGS['imageRoot'] = FLAGS['rootPath'] + 'data/'

FLAGS['modelDir'] = FLAGS['rootPath'] + 'models/simpleViT/'



# device config


FLAGS['ngpu'] = torch.cuda.is_available()
FLAGS['device'] = torch.device("cuda:1" if (torch.cuda.is_available() and FLAGS['ngpu'] > 0) else "mps" if (torch.has_mps == True) else "cpu")
FLAGS['device2'] = FLAGS['device']
if(torch.has_mps == True): FLAGS['device2'] = "cpu"
FLAGS['use_AMP'] = False
FLAGS['use_scaler'] = False
#if(FLAGS['device'].type == 'cuda'): FLAGS['use_sclaer'] = True

# dataloader config

FLAGS['num_workers'] = 20


# training config

FLAGS['num_epochs'] = 100
FLAGS['batch_size'] = 64
FLAGS['gradient_accumulation_iterations'] = 32

FLAGS['base_learning_rate'] = 3e-3
FLAGS['base_batch_size'] = 2048
FLAGS['learning_rate'] = ((FLAGS['batch_size'] * FLAGS['gradient_accumulation_iterations']) / FLAGS['base_batch_size']) * FLAGS['base_learning_rate']
FLAGS['lr_warmup_epochs'] = 5

FLAGS['weight_decay'] = 1e-2

FLAGS['resume_epoch'] = 0

FLAGS['finetune'] = False

FLAGS['image_size'] = 224
FLAGS['progressiveImageSize'] = False
FLAGS['progressiveSizeStart'] = 0.5
FLAGS['progressiveAugRatio'] = 3.0

FLAGS['crop'] = 0.875
FLAGS['interpolation'] = torchvision.transforms.InterpolationMode.BICUBIC
FLAGS['image_size_initial'] = int(round(FLAGS['image_size'] // FLAGS['crop']))

# debugging config

FLAGS['verbose_debug'] = False
FLAGS['skip_test_set'] = False
FLAGS['stepsPerPrintout'] = 50
FLAGS['val'] = False

classes = None

"""# custom models"""

from timm.models.layers import LayerNorm2d, to_2tuple
from torch import Tensor
from torch import linalg as LA
import torch.nn.functional as F


class Stem(nn.Module):
    """ Size-agnostic implementation of 2D image to patch embedding,
        allowing input size to be adjusted during model forward operation
    """

    def __init__(
            self,
            in_chs=3,
            out_chs=96,
            stride=4,
            norm_layer=LayerNorm2d,
    ):
        super().__init__()
        stride = to_2tuple(stride)
        self.stride = stride
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.conv = nn.Conv2d(
            in_chs,
            out_chs,
            kernel_size=(stride[0]*2-1, stride[1]*2-1),
            stride=stride,
            padding=(stride[0]-1, stride[1]-1),
        )
        self.norm = norm_layer(out_chs)

    def forward(self, x: Tensor):
        B, C, H, W = x.shape
        x = F.pad(x, (0, (self.stride[1] - W % self.stride[1]) % self.stride[1]))
        x = F.pad(x, (0, 0, 0, (self.stride[0] - H % self.stride[0]) % self.stride[0]))
        x = self.conv(x)
        x = self.norm(x)
        return x

class ConvPosEnc(nn.Module):
    def __init__(self, dim: int, k: int = 3, act: bool = False):
        super(ConvPosEnc, self).__init__()

        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x: Tensor):
        feat = self.proj(x)
        x = x + self.act(feat)
        return x

# stripped timm impl

from functools import partial

from itertools import repeat
import collections.abc


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class TransformerBlock(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        #self.mlp = Tlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class ViT(nn.Module):
    def __init__(
        self,
        in_chs = 3,
        dim=384,
        num_heads=8,
        num_classes=1000,
        depth = 8,
        drop_path = 0.1,
        drop = 0.0
    ):
        super().__init__()

        self.stem = Stem(in_chs = in_chs, out_chs = dim)
        self.cpe = ConvPosEnc(dim=dim, k=3)

        blocks = []

        for i in range(depth):
            blocks.append(
                nn.Sequential(
                    TransformerBlock(dim, num_heads=num_heads, qkv_bias=False, drop_path=drop_path, drop = drop, attn_drop = drop),

                )
            )

        self.blocks = nn.Sequential(*blocks)
        self.depth = depth
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self,x):
        x = self.stem(x)

        # B, C, H, W -> B, N, C
        x=self.cpe(x).flatten(2).transpose(1, 2)
        
        print(x.shape)
        x = self.blocks(x)
        
        x = x.mean(dim=1)
        x = self.norm(x)
        x = self.head(x)
        return x


class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print("Created Directory : ", dir)
    return dir

def getData():
    startTime = time.time()

    trainSet = torchvision.datasets.ImageNet(FLAGS['imageRoot'], split = 'train')
    testSet = torchvision.datasets.ImageNet(FLAGS['imageRoot'], split = 'val')

    global classes
    classes = {classIndex : className for classIndex, className in enumerate(trainSet.classes)}
    
    image_datasets = {'train': trainSet, 'val' : testSet}   # put dataset into a list for easy handling
    return image_datasets

def modelSetup(classes):
    
    '''
    myCvtConfig = transformers.CvtConfig(num_channels=3,
        patch_sizes=[7, 5, 3, 3],
        patch_stride=[4, 3, 2, 2],
        patch_padding=[2, 2, 1, 1],
        embed_dim=[64, 240, 384, 896],
        num_heads=[1, 3, 6, 14],
        depth=[1, 4, 8, 16],
        mlp_ratio=[4.0, 4.0, 4.0, 4.0],
        attention_drop_rate=[0.0, 0.0, 0.0, 0.0],
        drop_rate=[0.0, 0.0, 0.0, 0.0],
        drop_path_rate=[0.0, 0.0, 0.0, 0.1],
        qkv_bias=[True, True, True, True],
        cls_token=[False, False, False, True],
        qkv_projection_method=['dw_bn', 'dw_bn', 'dw_bn', 'dw_bn'],
        kernel_qkv=[3, 3, 3, 3],
        padding_kv=[1, 1, 1, 1],
        stride_kv=[2, 2, 2, 2],
        padding_q=[1, 1, 1, 1],
        stride_q=[1, 1, 1, 1],
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        num_labels=len(classes))
    '''
    
    # custom cvt
    
    #model = transformers.CvtForImageClassification(myCvtConfig)
    
    # pytorch builtin models
    
    #model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    #model = models.resnet152()
    #model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    #model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
    #model = models.resnet34()
    #model = models.resnet34(weights = models.ResNet34_Weights.DEFAULT)
    #model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
    
    #model.fc = nn.Linear(model.fc.in_features, len(classes))
    
    # regular timm models
    
    #model = timm.create_model('maxvit_tiny_tf_224.in1k', pretrained=True, num_classes=len(classes))
    #model = timm.create_model('ghostnet_050', pretrained=True, num_classes=len(classes))
    #model = timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=True, num_classes=len(classes))
    #model = timm.create_model('vit_small_resnet26d_224', pretrained=False, num_classes=len(classes), drop_rate = 0., drop_path_rate = 0.1)
    #model = timm.create_model('lcnet_035', pretrained=False, num_classes=len(classes), drop_rate = 0.0, drop_path_rate = 0.)
    

    model=ViT()
    
    #model = ml_decoder.add_ml_decoder_head(model)
    
    # cvt
    
    #model = transformers.CvtForImageClassification.from_pretrained('microsoft/cvt-13')
    #model.classifier = nn.Linear(model.config.embed_dim[-1], len(classes))

    # regular huggingface models

    #model = transformers.AutoModelForImageClassification.from_pretrained("facebook/levit-384", num_labels=len(classes), ignore_mismatched_sizes=True)
    #model = transformers.AutoModelForImageClassification.from_pretrained("facebook/convnext-tiny-224", num_labels=len(classes), ignore_mismatched_sizes=True)
    
    
    # modified timm models with custom head with hidden layers
    '''
    model = timm.create_model('mixnet_s', pretrained=True, num_classes=-1) # -1 classes for identity head by default
    
    model = nn.Sequential(model,
                          nn.LazyLinear(len(classes)),
                          nn.ReLU(),
                          nn.Linear(len(classes), len(classes)))
    
    '''
    
    if (FLAGS['resume_epoch'] > 0):
        model.load_state_dict(torch.load(FLAGS['modelDir'] + 'saved_model_epoch_' + str(FLAGS['resume_epoch'] - 1) + '.pth'), strict=False)
        
    
    #model.reset_classifier(len(classes))
    
    if FLAGS['finetune'] == True:
        for param in model.parameters():
            param.requires_grad = False
        if hasattr(model, "head"):
            for param in model.head.parameters():
                param.requires_grad = True
        if hasattr(model, "classifier"):
            for param in model.classifier.parameters():
                param.requires_grad = True
        if hasattr(model, "head_dist"):
            for param in model.head_dist.parameters():
                param.requires_grad = True
    
    return model

def trainCycle(image_datasets, model):
    print("starting training")
    startTime = time.time()

    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=FLAGS['batch_size'], shuffle=True, num_workers=FLAGS['num_workers'], persistent_workers = False, prefetch_factor=2, pin_memory = True, drop_last=True, generator=torch.Generator().manual_seed(41)) for x in image_datasets} # set up dataloaders
    
    
    mixup = Mixup(mixup_alpha = 0.2, cutmix_alpha = 0, label_smoothing=0.1)
    #dataloaders['train'].collate_fn = mixup_collate
    
    dataset_sizes = {x: len(image_datasets[x]) for x in image_datasets}
    device = FLAGS['device']
    device2 = FLAGS['device2']
    
    #memory_format = torch.channels_last
    memory_format = torch.contiguous_format
    
    model = model.to(device)

    print("initialized training, time spent: " + str(time.time() - startTime))
    

    criterion = SoftTargetCrossEntropy()
    # CE with ASL (both gammas 0), eps controls label smoothing, pref sum reduction
    #criterion = AsymmetricLossSingleLabel(gamma_pos=0, gamma_neg=0, eps=0.0, reduction = 'mean')
    #criterion = nn.BCEWithLogitsLoss()

    #optimizer = optim.Adam(params=parameters, lr=FLAGS['learning_rate'], weight_decay=FLAGS['weight_decay'])
    #optimizer = optim.SGD(model.parameters(), lr=FLAGS['learning_rate'], weight_decay=FLAGS['weight_decay'], momentum = 0.9)
    optimizer = optim.AdamW(model.parameters(), lr=FLAGS['learning_rate'], weight_decay=FLAGS['weight_decay'])
    #optimizer = timm.optim.Adan(model.parameters(), lr=FLAGS['learning_rate'], weight_decay=FLAGS['weight_decay'])
    if (FLAGS['resume_epoch'] > 0):
        optimizer.load_state_dict(torch.load(FLAGS['modelDir'] + 'optimizer' + '.pth'))
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=FLAGS['learning_rate'], steps_per_epoch=len(dataloaders['train']), epochs=FLAGS['num_epochs'], pct_start=FLAGS['lr_warmup_epochs']/FLAGS['num_epochs'])
    scheduler.last_epoch = len(dataloaders['train'])*FLAGS['resume_epoch']
    
    if (FLAGS['use_scaler'] == True): scaler = torch.cuda.amp.GradScaler()

    
    print("starting training")
    
    startTime = time.time()
    cycleTime = time.time()
    stepsPerPrintout = FLAGS['stepsPerPrintout']
    torch.backends.cudnn.benchmark = True
    
    for epoch in range(FLAGS['resume_epoch'], FLAGS['num_epochs']):
        epochTime = time.time()
        print("starting epoch: " + str(epoch))
        
        
        
        
        if FLAGS['progressiveImageSize'] == True:
                    
                    
            dynamicResizeDim = int(FLAGS['image_size'] * FLAGS['progressiveSizeStart'] + epoch * \
                (FLAGS['image_size']-FLAGS['image_size'] * FLAGS['progressiveSizeStart'])/FLAGS['num_epochs'])
        else:
            dynamicResizeDim = FLAGS['image_size']
        
        
        print(f'Using image size of {dynamicResizeDim}x{dynamicResizeDim}')
        
        trainTransforms = transforms.Compose([
            transforms.Resize((dynamicResizeDim, dynamicResizeDim), interpolation = FLAGS['interpolation']),
            #torchvision.transforms.RandomResizedCrop((dynamicResizeDim, dynamicResizeDim), interpolation = FLAGS['interpolation']),
            transforms.RandomHorizontalFlip(),
            #transforms.TrivialAugmentWide(),
            #transforms.RandAugment(magnitude = epoch, num_magnitude_bins = int(FLAGS['num_epochs'] * FLAGS['progressiveAugRatio'])),
            #transforms.RandAugment(),
            #CutoutPIL(cutout_factor=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])

        
        '''
        trainTransformsRSB = transforms.Compose([transforms.Resize((256)),
            transforms.RandomHorizontalFlip(),
            RandomResizedCropAndInterpolation(size=224),
            rand_augment_transform(
                config_str='rand-m9-mstd0.5', 
                hparams={'translate_const': 117, 'img_mean': (124, 116, 104)}
            ),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            RandomErasing(probability=0.5, mode='pixel', device='cpu'),
            #transforms.GaussianBlur(kernel_size=(7, 7), sigma=(2, 10)),
            #transforms.ToPILImage(),
            #transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        '''
        
        
        image_datasets['train'].transform = trainTransforms
        
        image_datasets['val'].transform = transforms.Compose([
            transforms.Resize(FLAGS['image_size_initial'], interpolation = FLAGS['interpolation']),
            transforms.CenterCrop((int(FLAGS['image_size']),int(FLAGS['image_size']))),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])
        
        for phase in ['train', 'val']:
        
            samples = 0
            correct = 0
            
            if phase == 'train':
                model.train()  # Set model to training mode
                print("training set")
                
                
            if phase == 'val':
                modelDir = create_dir(FLAGS['modelDir'])
                torch.save(model.state_dict(), modelDir + 'saved_model_epoch_' + str(epoch) + '.pth')
                torch.save(optimizer.state_dict(), modelDir + 'optimizer' + '.pth')
                model.eval()   # Set model to evaluate mode
                print("validation set")
                if(FLAGS['skip_test_set'] == True):
                    print("skipping...")
                    break;

            loaderIterable = enumerate(dataloaders[phase])
            for i, (images, tags) in loaderIterable:
                if phase == 'train':
                    images, tags = mixup(images, tags)
                

                imageBatch = images.to(device, memory_format=memory_format, non_blocking=True)
                tagBatch = tags.to(device, non_blocking=True)
                
                
                with torch.set_grad_enabled(phase == 'train'):
                    with torch.cuda.amp.autocast(enabled=FLAGS['use_AMP']):
                        
                        
                        
                        outputs = model(imageBatch)
                        #outputs = model(imageBatch).logits
                        #if phase == 'val':
                        #preds = torch.argmax(outputs, dim=1)
                        preds = torch.softmax(outputs.detach().clone(), dim=1) if phase == 'train' else torch.argmax(outputs.detach().clone(), dim=1)
                        
                        
                        samples += len(images)
                        #correct += sum(preds == tagBatch)
                        correct += (preds * tagBatch.detach().clone()).sum() if phase == 'train' else sum(preds == tagBatch.detach().clone())
                        
                        #print(tagBatch.shape)
                        
                        loss = criterion(outputs.to(device2), tagBatch.to(device2))

                        # backward + optimize only if in training phase
                        if phase == 'train' and (loss.isnan() == False):
                            if (FLAGS['use_scaler'] == True):   # cuda gpu case
                                scaler.scale(loss).backward()   #lotta time spent here
                                if((i+1) % FLAGS['gradient_accumulation_iterations'] == 0):
                                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)
                                    scaler.step(optimizer)
                                    scaler.update()
                                    optimizer.zero_grad()
                            else:                               # apple gpu/cpu case
                                loss.backward()
                                if((i+1) % FLAGS['gradient_accumulation_iterations'] == 0):
                                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)
                                    optimizer.step()
                                    optimizer.zero_grad()
                                    
                if i % stepsPerPrintout == 0:
                    accuracy = 100 * (correct/(samples+1e-8))

                    imagesPerSecond = (FLAGS['batch_size']*stepsPerPrintout)/(time.time() - cycleTime)
                    cycleTime = time.time()

                    print('[%d/%d][%d/%d]\tLoss: %.4f\tImages/Second: %.4f\ttop-1: %.2f' % (epoch, FLAGS['num_epochs'], i, len(dataloaders[phase]), loss, imagesPerSecond, accuracy))
                    torch.cuda.empty_cache()

                if phase == 'train':
                    scheduler.step()
            if phase == 'val':
                print(f'top-1: {100 * (correct/samples)}%')
        
        time_elapsed = time.time() - epochTime
        print(f'epoch {epoch} completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        #print(best)
        

        gc.collect()

        print()



def main():
    #gc.set_debug(gc.DEBUG_LEAK)
    # load json files

    image_datasets = getData()
    model = modelSetup(classes)
    trainCycle(image_datasets, model)


if __name__ == '__main__':
    main()