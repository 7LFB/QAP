import os

import cv2
import numpy as np
from PIL import Image
import random

import torch
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
from imgaug import augmenters as iaa
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from skimage.color import rgb2hed
import pandas as pd
from utils.spatial_statistics import *
from datasets.tools import *

from pdb import set_trace as st

PATTERNS=['inflammation','ballooning', 'steatosis']
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
LIVER_DATA_FILE = '/home/comp/chongyin/DataSets/Liver-NASH/SAMPLE20210702.xlsx'
# Real case
LIVER_SCORE_REMAPPING = {'NAS-Steatosis': {'0': 0, '1': 0, '2': 1, '3': 2},
                         'NAS-Inflammation': {'0': 0, '1': 1, '2': 2, '3': 2},
                         'NAS-Ballooning': {'0': 0, '1': 1, '2': 1},
                         'Fibrosis': {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4}}


def _read_array_image(path,resize,float_type=True,rgb=False,pil_format=False):
    if pil_format:
        img = Image.open(path).convert('RGB') # in case there are 4-channels
    else:
        img=cv2.imread(path,-1)
        if float_type: img=img.astype('float32')
        img=cv2.resize(img,(resize[0],resize[1]))
        if rgb: img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img



# XY format
# labno 1.txt
    # labno 1/tile_1_name.png
    # labno 1/tile_2_name.png


class mySlideDataset(torch.utils.data.Dataset):
    def __init__(self,XY,dataDir,priorDir=None,h=224,w=224,is_training=False,return_path=False,xprompt=False,prompt_index_str_s=None,prompt_index_str_m=None,norm_props=False,augment=False,auto_augment=False,args=None):


        self.XY_list = XY
        self.dataDir=dataDir
        self.priorDir = priorDir
        self.is_training=is_training
        self.return_path=return_path
        self.xprompt=xprompt
        self.prompt_index_str_s=prompt_index_str_s
        self.prompt_index_str_m=prompt_index_str_m
        self.norm_props=norm_props
        self.augment = augment
        self.auto_augment = auto_augment
        self.args=args


        self.h = h
        self.w = w

        aug_transforms_list=[]
        # toTensor: 1. HWC->CHW, 2. convert to float32 (3,only for PIL img format in rgb order + uinit8 format, or the type is np.uint8, then scale it to 0-1)
        # self.transforms =   transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x/255.0),transforms.Lambda(lambda x: 2*x-1)])
        base_transforms_list=[transforms.ToTensor(), transforms.Normalize(mean = MEAN, std = STD)]

        if self.augment:
            aug_transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))
            aug_transforms_list.append(transforms.RandomPerspective(distortion_scale=0.3, p=0.5))

        if self.auto_augment:
            aug_transforms_list.append(AutoAugment(AutoAugmentPolicy.IMAGENET))

        if self.is_training:
            self.transforms=transforms.Compose(aug_transforms_list+base_transforms_list)
        else:
            self.transforms=transforms.Compose(base_transforms_list)

       
    def __len__(self):
        return len(self.XY_list)
    
    def __getitem__(self, idx):
        _path1=self.XY_list[idx]

        path1=os.path.join(self.dataDir,_path1)

        img=_read_array_image(path1,[self.h, self.w],pil_format=True)
        img=self.transforms(img)

        # read pre-defined prompts
        if self.xprompt:
            prompt=load_or_generate_props_slide(self.dataDir,self.priorDir,_path1,self.prompt_index_str_s,self.prompt_index_str_m,self.args)
        

        if self.return_path:
            if self.xprompt:
                return img, prompt, _path1
            else:
                return img, _path1
        else:
            if self.xprompt:
                return img, prompt
            else:
                return img
        

    def test(self,idx):
        _path1=self.XY_list[idx]

        path1=os.path.join(self.dataDir,_path1)

        img=_read_array_image(path1,[self.h, self.w],rgb=True)
        img=self.transforms(img)

        # read pre-defined prompts

        prompt=load_and_norm_prompt_slide(self.priorDir,_path1,self.prompt_index_str_s,self.prompt_index_str_m,self.norm_props)


        if self.return_path:
            if self.xprompt:
                return img, prompt, _path1
            else:
                return img, _path1
        else:
            if self.xprompt:
                return img, prompt
            else:
                return img


       
