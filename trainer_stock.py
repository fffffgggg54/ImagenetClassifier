import os
hasTPU = False
if 'XRT_TPU_CONFIG' in os.environ: hasTPU = True


import torch
import torch.cuda.amp
import torch.nn as nn
import torch.nn.parallel
import torch.profiler
import torch.backends.cudnn as cudnn
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

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.xla_backend
import torch_xla.utils.utils as xu
import torch_xla.utils.gcsfs
SERIAL_EXEC = xmp.MpSerialExecutor()


FLAGS = {}

FLAGS['trainSetSize'] = 0.9

# env
FLAGS['num_tpu_cores'] = 1
FLAGS['rootPath'] = "../data/"
FLAGS['imageRoot'] = '../input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/'
FLAGS['modelDir'] = FLAGS['rootPath'] + 'models/resnet50/'

# dataloader
FLAGS['num_workers'] = 11


# traincycle
FLAGS['num_epochs'] = 100
FLAGS['batch_size'] = 256
FLAGS['gradient_accumulation_iterations'] = 4

# hparams
FLAGS['base_learning_rate'] = 3e-3
FLAGS['base_batch_size'] = 2048
FLAGS['learning_rate'] = ((FLAGS['batch_size'] * FLAGS['gradient_accumulation_iterations']) / FLAGS['base_batch_size']) * FLAGS['base_learning_rate']
FLAGS['lr_warmup_epochs'] = 6

FLAGS['weight_decay'] = 2e-2

# util
FLAGS['resume_epoch'] = 1

FLAGS['finetune'] = False

# debugging config

FLAGS['verbose_debug'] = False
FLAGS['skip_test_set'] = False
FLAGS['stepsPerPrintout'] = 50

classes = None



def getData():
    startTime = time.time()

    #trainSet = torchvision.datasets.ImageNet(FLAGS['imageRoot'], split = 'train')
    #testSet = torchvision.datasets.ImageNet(FLAGS['imageRoot'], split = 'val')
    dataset = torchvision.datasets.Caltech101(
        "./",
        download=True
    )
    trainSet, testSet = torch.utils.data.random_split(
        dataset,
        [
            int(FLAGS['trainSetSize'] * len(dataset)),
            len(dataset) - int(FLAGS['trainSetSize'] * len(dataset))
        ],
        generator=torch.Generator().manual_seed(42)
    ) # split dataset


    
    global classes
    #classes = {classIndex : className for classIndex, className in enumerate(dataset.classes)}
    classes = {classIndex : className for classIndex, className in enumerate(dataset.categories)}
    image_datasets = {'train': trainSet, 'val' : testSet}   # put dataset into a list for easy handling
    
    
    image_datasets['train'].transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
        
    image_datasets['val'].transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return image_datasets



def modelSetup(classes):
    #model = timm.create_model('tf_efficientnetv2_b3', pretrained=False, num_classes=len(classes), drop_rate = 0.00, drop_path_rate = 0.0)
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes))

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
                
    if hasTPU == True:
        model = xmp.MpModelWrapper(model)
        
    return model
    
    
def trainCycle(image_datasets, model):
    startTime = time.time()
    device = xm.xla_device()
    print(device)
    samplers = {
        x: torch.utils.data.distributed.DistributedSampler(
            image_datasets[x],
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=True
        ) for x in image_datasets
    }
    
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], 
            batch_size=FLAGS['batch_size'], 
            sampler = samplers[x], 
            num_workers=FLAGS['num_workers'], 
            prefetch_factor=2, 
            drop_last=True,
            generator=torch.Generator().manual_seed(41)
        ) for x in image_datasets
    } # set up dataloaders
    
    lr = FLAGS['learning_rate'] * xm.xrt_world_size() # linear lr scaling wrt world size
    
    
    model = model.to(device)
    
    #mixup = Mixup(mixup_alpha = 0.1, cutmix_alpha = 0, label_smoothing=0)
    #dataloaders['train'].collate_fn = mixup_collate
    
    dataset_sizes = {x: len(image_datasets[x]) for x in image_datasets}

    print("initialized training, time spent: " + str(time.time() - startTime))
    

    criterion = criterion = nn.BCEWithLogitsLoss(reduction='sum')

    optimizer = optim.SGD(model.parameters(), lr=FLAGS['learning_rate'], weight_decay=FLAGS['weight_decay'])
    #optimizer = optim.AdamW(model.parameters(), lr=FLAGS['learning_rate'], weight_decay=FLAGS['weight_decay'])
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=FLAGS['learning_rate'], steps_per_epoch=len(dataloaders['train']), epochs=FLAGS['num_epochs'], pct_start=FLAGS['lr_warmup_epochs']/FLAGS['num_epochs'])

    xm.master_print(("starting training"))
    
    startTime = time.time()
    cycleTime = time.time()
    stepsPerPrintout = FLAGS['stepsPerPrintout']
    
    for epoch in range(FLAGS['num_epochs']):
        epochTime = time.time()
        print("starting epoch: " + str(epoch))


        
        for phase in ['train', 'val']:
        
            samples = 0
            correct = 0
            
            if phase == 'train':
                model.train()  # Set model to training mode
                print("training set")
                
                
            if phase == 'val':
                modelDir = danbooruDataset.create_dir(FLAGS['modelDir'])
                if torch_xla.core.xla_model.is_master_ordinal(local=False) == True:
                    torch.save(model.to('cpu').state_dict(), modelDir + 'saved_model_epoch_' + str(epoch) + '.pth')
                model.eval()   # Set model to evaluate mode
                print("validation set")
                if(FLAGS['skip_test_set'] == True):
                    print("skipping...")
                    break;

            loaderIterable = enumerate(pl.ParallelLoader(dataloaders[phase], [device]).per_device_loader(device))
            for i, (images, tags) in loaderIterable:
                
                

                imageBatch = images.to(device, memory_format=memory_format, non_blocking=True)
                tagBatch = tags.to(device, non_blocking=True)
                tagsOneHot = torch.nn.functional.one_hot(tags, num_classes = len(classes)).to(device, non_blocking=True)
                
                
                
                with torch.set_grad_enabled(phase == 'train'):
                    
                    #if phase == 'train':
                        #imageBatch, tagBatch = mixup(imageBatch, tagBatch)

                    outputs = model(imageBatch)
                    #outputs = model(imageBatch).logits
                    #if phase == 'val':
                    preds = torch.argmax(outputs, dim=1)

                    samples += len(images)
                    correct += sum(preds == tagBatch)
                    tagBatch = torch.eye(len(classes), device=device)[tagBatch]

                    loss = criterion(outputs, tagsOneHot)

                    # backward + optimize only if in training phase
                    if phase == 'train' and (loss.isnan() == False):
                        loss.backward()
                        if(i % FLAGS['gradient_accumulation_iterations'] == 0):
                            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
                            xm.optimizer_step(optimizer)
                            optimizer.zero_grad()
                                    
                if i % stepsPerPrintout == 0:
                    accuracy = 100 * (correct/(samples+1e-8))

                    imagesPerSecond = (FLAGS['batch_size']*stepsPerPrintout)/(time.time() - cycleTime)
                    cycleTime = time.time()

                    print('[%d/%d][%d/%d]\tLoss: %.4f\tImages/Second: %.4f\ttop-1: %.2f' % (epoch, FLAGS['num_epochs'], i, len(dataloaders[phase]), loss, imagesPerSecond, accuracy))

                if phase == 'train':
                    scheduler.step()
            if phase == 'val':
                print(f'top-1: {100 * (correct/samples)}%')
        
        time_elapsed = time.time() - epochTime
        print(f'epoch {epoch} completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        gc.collect()
        if(FLAGS['verbose_debug'] == True):
            xm.master_print(met.metrics_report(), flush=True)
        print()
        
def _mp_fn(rank, flags, image_datasets, model):
    global FLAGS
    FLAGS = flags
    torch.set_default_tensor_type('torch.FloatTensor')
    trainCycle(image_datasets, model)


def main():

    image_datasets = getData()
    print("got datasets")
    
    model = modelSetup(classes)
    print(model)

    xmp.spawn(_mp_fn, args=(FLAGS, image_datasets, model,), nprocs=FLAGS['num_tpu_cores'], start_method='fork')
    




if __name__ == '__main__':
    main()