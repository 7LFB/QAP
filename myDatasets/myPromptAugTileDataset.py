import os

import cv2
import numpy as np
from PIL import Image
import random
import glob

import torch
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from imgaug import augmenters as iaa
from skimage.color import rgb2hed
from utils.utils import *
from utils.spatial_statistics import *
from myDatasets.tools import *

from pdb import set_trace as st

PATTERNS=['none','inflammation','ballooning', 'steatosis']
# PATTERNS=['normal_he_mouse_liver','NAFLD_anomaly_he_mouse_liver']
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)



def _read_array_image(path,resize,float_type=True,rgb=False,pil_format=False):
    if pil_format:
        img = Image.open(path).convert('RGB') # in case there are 4-channels
    else:
        img=cv2.imread(path,-1)
        if float_type: img=img.astype('float32')
        img=cv2.resize(img,(resize[0],resize[1]))
        if rgb: img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class myTileDataset(torch.utils.data.Dataset):
    def __init__(self,XY,dataDir,h=224,w=224,is_training=False,return_path=False,xprompt=False,prompt_index_str_s=None,prompt_index_str_m=None,norm_props=False,augment=False,auto_augment=False,args=None):

        self.dataDir=dataDir
        self.is_training=is_training
        self.return_path=return_path
        self.xprompt=xprompt
        self.prompt_index_str_s=prompt_index_str_s
        self.prompt_index_str_m=prompt_index_str_m
        self.norm_props=norm_props
        self.augment=augment
        self.auto_augment=auto_augment
        self.args=args

        self.XY_list = XY

        self.h = h
        self.w = w

        aug_transforms_list=[]
        # toTensor: 1. HWC->CHW, 2. convert to float32 (3,only for PIL img format in rgb order + uinit8 format, or the type is np.uint8, then scale it to 0-1)
        # self.transforms =   transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x/255.0),transforms.Lambda(lambda x: 2*x-1)])
        base_transforms_list=[transforms.ToTensor(), transforms.Normalize(mean = MEAN, std = STD)]

        self.augPools=[
            transforms.ColorJitter(brightness=(0.5,0.9),contrast=0,saturation=0,hue=0), #0
            transforms.ColorJitter(brightness=0,contrast=(0.1,0.9),saturation=0,hue=0), #1
            transforms.ColorJitter(brightness=0,contrast=0,saturation=(0.1,0.9),hue=0), #2
            transforms.ColorJitter(brightness=0,contrast=0,saturation=0,hue=(-0.2,0.2)), #3                                                                   
            transforms.ColorJitter(brightness=0,contrast=0,saturation=(0.1,0.8),hue=0), #4
            transforms.ColorJitter(brightness=0,contrast=0,saturation=0,hue=(-0.3,0.3)), #5
            transforms.ColorJitter(brightness=0,contrast=0,saturation=0,hue=(-0.1,0.1)), #6
            transforms.GaussianBlur((5,9), sigma=(0.1, 2.0)), #7
            transforms.RandomAdjustSharpness(sharpness_factor=2), #8
            transforms.RandomAutocontrast(), #9
            transforms.RandomEqualize(), #10
            transforms.AugMix(), #11

            ]                                         

        if self.augment and self.args.augment_index!='none':
            for index in self.args.augment_index.split('-'):
                aug_transforms_list.append(self.augPools[int(index)])

        self.autoAugPools=[AutoAugmentPolicy.IMAGENET,AutoAugmentPolicy.CIFAR10,AutoAugmentPolicy.SVHN]
        if self.auto_augment and self.args.auto_augment_index!='none':
            for index in self.args.auto_augment_index.split('-'):
                aug_transforms_list.append(AutoAugment(self.autoAugPools[int(index)]))


        if self.is_training:
            self.transforms=transforms.Compose(aug_transforms_list+base_transforms_list)
        else:
            self.transforms=transforms.Compose(base_transforms_list)


    def __len__(self):
        return len(self.XY_list)

    def get_labels(self):
        labels = []
        for item in self.XY_list:
            ss=item.split(' ')
            label=int(ss[1])
            labels.append(label)

        return labels

    def __getitem__(self, idx):
        # steatosis/##.png 3
        ss=self.XY_list[idx].split(' ')
        _path1,path2=ss[0],ss[1]

        class_subfoler=_path1.split(os.sep)[0]
        path1=os.path.join(self.dataDir,_path1)

        label=int(path2)

        nuclei_path = path1.replace(class_subfoler,self.args.nuclei_folder)
        white_path = path1.replace(class_subfoler,self.args.white_folder)

        img=_read_array_image(path1,[self.h, self.w],pil_format=True)
        img=self.transforms(img)

        white_segment=_read_array_image(white_path,[self.h, self.w],pil_format=True)
        white_segment=self.transforms(white_segment)

        nuclei_segment=_read_array_image(nuclei_path,[self.h, self.w],pil_format=True)
        nuclei_segment=self.transforms(nuclei_segment)


        img=torch.cat((img,nuclei_segment,white_segment),dim=0)

        # read pre-defined prompts
        if self.xprompt:
            prompt=load_or_generate_props(self.dataDir,_path1,self.prompt_index_str_s,self.prompt_index_str_m,self.args)

       
        if self.return_path:
            if self.xprompt:
                return img, prompt, label,path1
            else:
                return img, label, path1
        else:
            if self.xprompt:
                return img, prompt, label
            else:
                return img, label
        

    def test(self,idx):
        ss=self.XY_list[idx].split(' ')
        _path1,path2=ss[0],ss[1]

        class_subfoler=_path1.split(os.sep)[0]

        path1=os.path.join(self.dataDir,_path1)

        label=int(path2)

        nuclei_path = path1.replace(class_subfoler,self.args.nuclei_folder)
        white_path = path1.replace(class_subfoler,self.args.white_folder)

        img=_read_array_image(path1,[self.h, self.w],pil_format=True)
        img=self.transforms(img)


        white_segment=_read_array_image(white_path,[self.h, self.w],pil_format=True)
        white_segment=self.transforms(white_segment)

        nuclei_segment=_read_array_image(nuclei_path,[self.h, self.w],pil_format=True)
        nuclei_segment=self.transforms(nuclei_segment)

        # read pre-defined prompts
        if self.xprompt:
            prompt=load_or_generate_props(self.dataDir,_path1,self.prompt_index_str_s,self.prompt_index_str_m,self.args)

       
        if self.return_path:
            if self.xprompt:
                return img, prompt, label,path1
            else:
                return img, label, path1
        else:
            if self.xprompt:
                return img, prompt, label
            else:
                return img, label


       
