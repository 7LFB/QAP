import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path
import glob
import pandas as pd
import importlib

import re

import numpy as np
import json

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings("ignore")


from torch.optim.lr_scheduler import MultiStepLR, LambdaLR, CosineAnnealingLR, StepLR

from core.function import train, validate, test
from utils.params import *
from utils.utils import *
from utils.spatial_statistics import *
from myDatasets.myPromptAugTileDataset import myTileDataset

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
    args = parse_args()


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
    args.snapshot_dir = args.snapshot_dir.replace(
        'XPrompt/', 'XPrompt/' + args.model_name + '-')


    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    print(toMagenta(args.snapshot_dir))

    # saving args into txt file for recording
    with open(os.path.join(args.snapshot_dir,'commandline_args.json'), 'wt') as f:
        json.dump(vars(args), f, indent=4)


    writer_dict = {
        'writer': SummaryWriter(args.snapshot_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
        'test_global_steps':0
    }

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

    x_test_tileNames=[x.split(' ')[0] for x in x_test]



    train_dataset=myTileDataset(x_train,args.data_dir,h=args.img_h,w=args.img_w,is_training=True,xprompt=args.xprompt,prompt_index_str_s=args.prompt_index_str_s,prompt_index_str_m=args.prompt_index_str_m,norm_props=args.norm_props,augment=args.augment,auto_augment=args.auto_augment,args=args)
    val_dataset=myTileDataset(x_val,args.data_dir,h=args.img_h,w=args.img_w,xprompt=args.xprompt,prompt_index_str_s=args.prompt_index_str_s,prompt_index_str_m=args.prompt_index_str_m,norm_props=args.norm_props,args=args)
    test_dataset=myTileDataset(x_test,args.data_dir,h=args.img_h,w=args.img_w,xprompt=args.xprompt,prompt_index_str_s=args.prompt_index_str_s,prompt_index_str_m=args.prompt_index_str_m,norm_props=args.norm_props,args=args)

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


    # # model configuration
    module=importlib.import_module(f'models.model_v{args.mversion}')
    XPrompt = getattr(module, 'XPrompt')
    model=XPrompt(args)
    print(toRed(f'version-{args.mversion}'))


    # freeze fundation model
    if args.freeze_pattern:
        patterns=args.freeze_pattern.split('-')
        print(toRed('Freeze fundation model \n'))
        for name, param in model.named_parameters():
            for pattern in patterns:
                if re.search(pattern, name):
                    param.requires_grad = False
                    print(name)
                    break
    
    
    # optimizer
    optimizer = torch.optim.AdamW([{'params':
                                  filter(lambda p: p.requires_grad,
                                         model.parameters()),
                                'lr': args.lr}],
                                lr=args.lr
                                )
    lr_scheduler = CosineAnnealingLR(optimizer, args.num_epochs, eta_min=1e-6)

    model = nn.DataParallel(model, device_ids=gpus).cuda()


    best_val_f1 = 0
    best_val_msg=''
    best_test_msg=''
    best_test_con_mat=0
    best_test_ovr_mat=0
  

    ## load the ckpt dict
    if args.ckpt_path:
        model_state_file = args.ckpt_path
        pretrained_dict = torch.load(model_state_file)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict.keys()}
        for k, _ in pretrained_dict.items():
            print(toCyan('=> loading {} from pretrained model'.format(k)))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

   

    # write final results into specific record.txt
    subf = open(os.path.join(args.snapshot_dir, 'record.txt'),'a+')

    for epoch in range(args.start_epoch, args.num_epochs):
        train(model, trainloader, optimizer, writer_dict, epoch, args)

        lr_scheduler.step()

        print('\t')
        print('---model_name---')
        print(args.model_name)

        val_accuracy, val_specificity, val_recall, val_f1, val_auc = validate(model, valloader, writer_dict, args)

        val_msg = f'{args.model_name}: Epoch:{epoch}, Accuracy:{val_accuracy*100:.3f}, Specificity:{val_specificity*100:.3f}, Recall:{val_recall*100:.3f}, F1:{val_f1*100:.3f}, AUC:{val_auc*100:.3f}'

        gts, predicts, probs, accuracy, specificity, recall, f1, auc, con_mat, ovr_mat = test(model, testloader, writer_dict, args)

        test_result = np.concatenate([np.expand_dims(gts,-1),np.expand_dims(predicts,-1),probs],axis=-1)

        test_result_excel={'tileNames':x_test_tileNames,'gts':gts.tolist(),'predicts':predicts.tolist()}

        test_msg = f'{args.model_name}: Epoch:{epoch}, Accuracy:{accuracy*100:.3f}, Specificity:{specificity*100:.3f}, Recall:{recall*100:.3f}, F1:{f1*100:.3f}, AUC:{auc*100:.3f}'


        if val_f1 > best_val_f1 and epoch > args.record_epoch:
            best_val_f1 = val_f1
            best_test_msg=test_msg
            best_test_con_mat=con_mat
            best_test_ovr_mat=ovr_mat
            print(toMagenta('Best is: '+best_test_msg))
            print('Multi-class one-vs-rest result\t')
            print(str(ovr_mat)+'\n')
            print('Confusion Matrix\t')
            print(str(con_mat)+'\n')
            torch.save(model.module.state_dict(),
                       os.path.join(args.snapshot_dir, 'best.pth'))
            
            np.save(os.path.join(args.snapshot_dir, f'T{args.fold}.npy'),test_result)
            df = pd.DataFrame.from_dict(test_result_excel)
            df.to_excel(os.path.join(args.snapshot_dir, f'T{args.fold}.xlsx'))
            
            
        subf.write('\t')
        subf.write('---'*5 + '\n')
        subf.write('validation results ...\n')
        subf.write(val_msg+'\n')
        subf.write('test results ...\n')
        subf.write(test_msg+'\n')
        subf.write('Multi-class one-vs-rest result\n')
        subf.write(str(ovr_mat)+'\n')
        subf.write('Confusion Matrix\n')
        subf.write(str(con_mat)+'\n')
        subf.write('Best test results ...\n')
        subf.write(best_test_msg+'\n')



    f = open('./log/' + args.logdir+'.txt', 'a+')
    f.write('\t')
    f.write('---'*5 + '\n')
    f.write(args.snapshot_dir+'\n')
    f.write(best_test_msg+'\n')
    f.write('Multi-class one-vs-rest result\n')
    f.write(str(best_test_ovr_mat)+'\n')
    f.write('Confusion Matrix\n')
    f.write(str(best_test_con_mat)+'\n')
    f.close()

  
    torch.save(model.module.state_dict(),
               os.path.join(args.snapshot_dir, 'final_state.pth'))

   

if __name__ == '__main__':
    main()
