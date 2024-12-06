import argparse
import os
import pprint
import shutil
import sys
import glob

import logging
import time
import timeit
from pathlib import Path
import pandas as pd
import importlib

import re

import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim

from datasets.myPromptSlideDataset import mySlideDataset

from utils.params import *
from utils.utils import *
from utils.tools import *
 


from pdb import set_trace as st

ROOT_DIR='/home/comp/chongyin/DataSets/Liver-NASH/'

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

    RESUME_FOLDERS=glob.glob(os.path.join(args.resume_from,args.resume_keywords))
    print(toGreen(f'finging {len(RESUME_FOLDERS)} folders'))
    assert len(RESUME_FOLDERS)==1
    RESUME_FOLDER = RESUME_FOLDERS[0]

    with open(os.path.join(RESUME_FOLDER,'commandline_args.json'), 'r') as json_file:
        args_json = json.load(json_file)
        print(toGreen(f'updating args from commandline_args.json'))
        for key, value in args_json.items():
            if key == 'model_name': continue
            if key == 'generate_props': continue
            if key == 'prior_white': continue
            if key == 'prior_nuclei': continue
            if key == 'dilate': continue
            if key == 'area_thd': continue
            if key == 'convex_hull': continue
            if key == 'clear_border': continue
            setattr(args, key, value)

    CKPT_PATH=os.path.join(RESUME_FOLDER,'best.pth')

    args.pretrained_weights=args.pretrained_weights.replace('/home/comp/chongyin/PyTorch','E:/Datasets/Liver-NAS/LiverBiopsyPatches40X224-classification/methods')


    # PREPARING DATA
    DATADIR='/home/comp/chongyin/DataSets/Liver-NASH/LiverBiopsyPatches40X224'
    PRIORDIR='/home/comp/chongyin/DataSets/Liver-NASH/LiverBiopsyPatches40X224Prior'
    CSVFILE='/home/comp/chongyin/DataSets/Liver-NASH/LiverBiopsyExcels/IJCAI22-Liver-NAS-259.xlsx'


    df=pd.read_excel(CSVFILE)


    setup_seed(args.seed)
    # prepare data


    # # model configuration
    module=importlib.import_module(f'models.model_v{args.mversion}')
    XPrompt = getattr(module, 'XPrompt')
    model=XPrompt(args)
    print(toRed(f'version-{args.mversion}'))

    ## load the ckpt dict
    model_state_file = CKPT_PATH
    print(toGreen(f'loading ckpt {CKPT_PATH}'))
    orig_pretrained_dict = torch.load(model_state_file)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in orig_pretrained_dict.items()
                        if k in model_dict.keys()}
    missing_dict = {k:v for k, v in orig_pretrained_dict.items()
                        if k  not in model_dict.keys()}
    for k, _ in pretrained_dict.items():
        print('=> loading {} from pretrained model'.format(k))

    for k, _ in missing_dict.items():
        print(toRed('=> missing {} from current model'.format(k)))

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model=model.cuda()
    model.eval()

    ## loop over WSIs for NAS estimation
    
    ## loop over WSIs for NAS estimation
    results={'LabNo':[],'NAS-Steatosis':[],'NAS-Inflammation':[],'NAS-Ballooning':[],'totalTiles':[],'Number-Steatosis':[],'Number-Inflammation':[],'Number-Ballooning':[]}
    PATTERNS=['none','NAS-Inflammation','NAS-Ballooning', 'NAS-Steatosis']

    slide_time = AverageMeter()

    for ii in range(len(df)):

        nas_steatosis = df['NAS-Steatosis'][ii]
        nas_inflammation = df['NAS-Inflammation'][ii]
        nas_ballooning = df['NAS-Ballooning'][ii]
        ids = df['LabNo'][ii]

        slidepath_file = os.path.join(DATADIR,ids,f'{ids}.txt')
        if not os.path.exists(slidepath_file):
            continue
        # priorSlidePath = os.path.join(PRIORDIR,'nuclei_segment',ids)
        # if not os.path.exists(priorSlidePath):
        #     continue

        slide_results={'tileName':[],'Others':[],'NAS-Inflammation':[],'NAS-Ballooning':[],'NAS-Steatosis':[],'Prediction':[]}
        tileLists =readlines_from_txt([slidepath_file])

        tilesLoader=mySlideDataset(tileLists,DATADIR,priorDir=PRIORDIR,h=args.img_h,w=args.img_w,is_training=False,return_path=True,xprompt=args.xprompt,prompt_index_str_s=args.prompt_index_str_s,prompt_index_str_m=args.prompt_index_str_m,norm_props=args.norm_props,args=args)

        testloader = torch.utils.data.DataLoader(
        tilesLoader,
        batch_size=128,
        prefetch_factor=32,
        shuffle=False,
        num_workers=12,
        pin_memory=True)
        

        slide_results_path=f'/home/comp/chongyin/DataSets/Liver-NASH/SlidingOnWSI/{args.model_name}/F{args.fold}/{ids}.xlsx'
        mkdirs_if_need(slide_results_path)
        
        tic=time.time()
        # if processed, no need to run the model again, call the file directly
        if os.path.exists(slide_results_path):
            slide_results=pd.read_excel(slide_results_path)
        else:
            for kk, batch in enumerate(testloader):
                if args.xprompt:
                    images, prompts, tileName = batch
                    images = images.float().cuda()
                    prompts = prompts.float().cuda()
                    logits = model(images,prompts)
                else:
                    images, tileName  = batch
                    images = images.float().cuda()
                    logits = model(images)

                pred = torch.argmax(logits,dim=-1)
                logits=logits.detach().cpu().numpy()
                pred=pred.cpu().numpy().tolist()

                slide_results['tileName']+=tileName
                slide_results['Others']+=logits[:,0].tolist()
                slide_results['NAS-Inflammation']+=logits[:,1].tolist()
                slide_results['NAS-Ballooning']+=logits[:,2].tolist()
                slide_results['NAS-Steatosis']+=logits[:,3].tolist()
                slide_results['Prediction']+=pred

                print('\r',toCyan(f'{ids}:{kk}/{len(testloader)}, TileTimeCost: {(time.time()-tic)/(kk+1):.3f}'),end='')

        all_tile_preds=np.array(slide_results['Prediction'])
        N_inflammation = np.sum(all_tile_preds==1)
        N_ballooning = np.sum(all_tile_preds==2)
        N_steatosis = np.sum(all_tile_preds==3)

         # Write DataFrame to an Excel file
       
        slide_df = pd.DataFrame(slide_results)
        slide_df.to_excel(slide_results_path, index=False)

        results['LabNo'].append(ids)
        results['NAS-Steatosis'].append(nas_steatosis)
        results['NAS-Inflammation'].append(nas_inflammation)
        results['NAS-Ballooning'].append(nas_ballooning)
        results['Number-Steatosis'].append(N_steatosis)
        results['Number-Inflammation'].append(N_inflammation)
        results['Number-Ballooning'].append(N_ballooning)
        results['totalTiles'].append(len(tilesLoader))

        print('\t')
        slide_time.update(time.time()-tic)
        print('\r',toGreen(f'{ids}:{ii}/{len(df)}, SlideTimeCost: {slide_time.average():.3f}, ETD: {slide_time.average()*(len(df)-ii)/3600:3f}h'),end='')
        print('\t')


    df = pd.DataFrame(results)
    # Write DataFrame to an Excel file
    model_results_path=f'/home/comp/chongyin/DataSets/Liver-NASH/SlidingOnWSI/{args.model_name}/F{args.fold}.xlsx'
    mkdirs_if_need(model_results_path)
    df.to_excel(model_results_path, index=False)     

   

if __name__ == '__main__':
    main()
