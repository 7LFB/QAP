import argparse
import os
import pprint
import shutil
import sys
import importlib

import logging
import time
import timeit
from pathlib import Path
import glob

import re

import numpy as np
import json

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter


from torch.optim.lr_scheduler import MultiStepLR, LambdaLR, CosineAnnealingLR, StepLR

import datasets
from core.function import train, validate, test
from utils.params import *
from utils.utils import *
from utils.spatial_statistics import *

from torchsampler import ImbalancedDatasetSampler

from pdb import set_trace as st

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# 设置随机数种子

def main():

    args=parse_args()

    RESUME_FOLDERS=glob.glob(os.path.join(args.resume_from,args.resume_keywords))
    print(toGreen(f'finging {len(RESUME_FOLDERS)} folders'))
    assert len(RESUME_FOLDERS)==1
    RESUME_FOLDER = RESUME_FOLDERS[0]

    with open(os.path.join(RESUME_FOLDER,'commandline_args.json'), 'r') as json_file:
        args_json = json.load(json_file)
        print(toGreen(f'updating args from commandline_args.json'))
        for key, value in args_json.items():
            setattr(args, key, value)

    CKPT_PATH=os.path.join(args.snapshot_dir,'best.pth')

    if args.xprompt:
        from datasets.myPromptTileDataset import myTileDataset
    else:
        from datasets.myTileDataset import myTileDataset

    if 'none' not in args.prompt_index_str_s:
        args.prompt_s_num=len(args.prompt_index_str_s.split('-'))
    else:
        args.prompt_s_num=0
    if 'none' not in args.prompt_index_str_m:
        args.prompt_m_num=len(args.prompt_index_str_m.split('-'))
    else:
        args.prompt_m_num=0

    # 5-fold cross validation
    args.model_name= f'F{args.fold}-{args.model_name}-mv{args.mversion}-seed{args.seed}'


    # cudnn related setting
    gpus = [ii for ii in range(args.gpus)]

    setup_seed(args.seed)
    # prepare data
    if args.build_prompt:
        build_prompt(args.data_dir,args.file_list_folder)

    _x_train = [os.path.join(args.file_list_folder, 'train_fold_'+str(args.fold)+'.txt')]
    _x_val = [os.path.join(args.file_list_folder, 'val_fold_'+str(args.fold)+'.txt')]
    _x_test = [os.path.join(args.file_list_folder, 'test_fold_'+str(args.fold)+'.txt')]
    

    x_train =readlines_from_txt(_x_train)
    if args.ratio<1.0: x_train=balance_sampler(x_train,args.ratio)

    x_val =readlines_from_txt(_x_val)
    x_test =readlines_from_txt(_x_test)


    train_dataset=myTileDataset(x_train,args.data_dir,h=args.img_h,w=args.img_w,is_training=True,xprompt=args.xprompt,prompt_index_str_s=args.prompt_index_str_s,prompt_index_str_m=args.prompt_index_str_m,norm_props=args.norm_props)
    val_dataset=myTileDataset(x_val,args.data_dir,h=args.img_h,w=args.img_w,xprompt=args.xprompt,prompt_index_str_s=args.prompt_index_str_s,prompt_index_str_m=args.prompt_index_str_m,norm_props=args.norm_props)
    test_dataset=myTileDataset(x_test,args.data_dir,h=args.img_h,w=args.img_w,xprompt=args.xprompt,prompt_index_str_s=args.prompt_index_str_s,prompt_index_str_m=args.prompt_index_str_m,norm_props=args.norm_props)

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        prefetch_factor=2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        sampler=ImbalancedDatasetSampler(train_dataset))

    valloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size_for_test,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size_for_test,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)


    # model configuration
    module=importlib.import_module(f'models.model_v{args.mversion}')
    XPrompt = getattr(module, 'XPrompt')
    model=XPrompt(args)
    print(toRed(f'version-{args.mversion}'))

    model = model.cuda()

    ## load the ckpt dict
    model_state_file = CKPT_PATH
    print(toGreen(f'loading ckpt {CKPT_PATH}'))
    pretrained_dict = torch.load(model_state_file)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict.keys()}
    for k, _ in pretrained_dict.items():
        print(toCyan('=> loading {} from pretrained model'.format(k)))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

   
    gts, predicts, probs, accuracy, specificity, recall, f1, auc, con_mat, ovr_mat = test(model, testloader, writer_dict=None, args=args)

    test_result = np.concatenate([np.expand_dims(gts,-1),np.expand_dims(predicts,-1),probs],axis=-1)

    test_msg = f'{args.model_name}:Accuracy:{accuracy*100:.3f}, Specificity:{specificity*100:.3f}, Recall:{recall*100:.3f}, F1:{f1*100:.3f}, AUC:{auc*100:.3f}'

    print(toRed(test_msg))

        



   

if __name__ == '__main__':
    main()
