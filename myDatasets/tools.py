import os

import cv2
import numpy as np
import random
import glob

import torch
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
from imgaug import augmenters as iaa
from skimage.color import rgb2hed
from utils.utils import *
from utils.spatial_statistics import *
from utils.histo_lib import *

from pdb import set_trace as st

# PATTERNS=['none','inflammation','ballooning', 'steatosis']
# PATTERNS=['normal_he_mouse_liver','NAFLD_anomaly_he_mouse_liver']
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

PROPS_SPATIAL_TYPES=['nuclei-kfunc','white-kfunc','nuclei2white-kfunc','white2nuclei-kfunc']
PROPS_MORPH_TYPES=['white-props-areas', 'white-props-eccs', 'white-props-circus', 'white-props-intens', 'white-props-entropys', 'white-props-shapes', 'white-props-extents', 'white-props-perimeters', 'white-props-percents']


my_nuclei_extractor=NucleiExtractor(pretrained_data='pannuke')

def load_props(data_dir,img_name,props_index_str_s,props_index_str_m):

    class_name=img_name.split(os.sep)[0]
    orig_img_path = os.path.join(data_dir,img_name)

    props=[]

    if 'none' not in props_index_str_s:
        for s_index in props_index_str_s.split('-'):
            s_props = np.load(orig_img_path.replace(class_name,f'prompt/{PROPS_SPATIAL_TYPES[int(s_index)]}').replace('png','npy').replace('PNG','npy'))
            s_props=pad_and_sampler(s_props,140,int(len(pad(s_props))/10))
            props.append(s_props.reshape(1,10))
    if 'none' not in props_index_str_m:
        for m_index in props_index_str_m.split('-'):
            m_props = np.load(orig_img_path.replace(class_name,f'prompt/{PROPS_MORPH_TYPES[int(m_index)]}').replace('png','npy').replace('PNG','npy'))
            props.append(m_props.reshape(1,10))

    prompt = torch.from_numpy(np.concatenate(props,axis=0))

    return prompt


def load_props_slide(prior_data_dir,img_name,props_index_str_s,props_index_str_m):

    props=[]

    if 'none' not in props_index_str_s:
        for s_index in props_index_str_s.split('-'):
            s_props = np.load(os.path.join(prior_data_dir,f'prompt/{PROPS_SPATIAL_TYPES[int(s_index)]}',img_name.replace('png','npy').replace('PNG','npy')))
            s_props=pad_and_sampler(s_props,140,int(len(pad(s_props))/10))
            props.append(s_props.reshape(1,10))
    if 'none' not in props_index_str_m:
        for m_index in props_index_str_m.split('-'):
            m_props = np.load(os.path.join(prior_data_dir,f'prompt/{PROPS_MORPH_TYPES[int(m_index)]}',img_name.replace('png','npy').replace('PNG','npy')))
            props.append(m_props.reshape(1,10))

    prompt = torch.from_numpy(np.concatenate(props,axis=0))

    return prompt



def load_and_norm_prompt(data_dir,img_name,props_index_str_s,props_index_str_m,norm=0):

    class_name=img_name.split(os.sep)[0]
    orig_img_path = os.path.join(data_dir,img_name)

    props=[]

    if 'none' not in props_index_str_s:
        for s_index in props_index_str_s.split('-'):
            s_props = np.load(orig_img_path.replace(class_name,f'prompt/{PROPS_SPATIAL_TYPES[int(s_index)]}').replace('png','npy').replace('PNG','npy'))
            s_props_stat = np.load(os.path.join(data_dir,'prompt',f'{PROPS_SPATIAL_TYPES[int(s_index)]}-statistics.npy'),allow_pickle=True)
            if norm: s_props = normalize_mean_std(pad(s_props),s_props_stat.item().get('mean'),s_props_stat.item().get('std'))
            s_props=pad_and_sampler(s_props,140,int(len(pad(s_props))/10))
            props.append(s_props.reshape(1,10))
    if 'none' not in props_index_str_m:
        for m_index in props_index_str_m.split('-'):
            m_props = np.load(orig_img_path.replace(class_name,f'prompt/{PROPS_MORPH_TYPES[int(m_index)]}').replace('png','npy').replace('PNG','npy'))
            m_props_stat=np.load(os.path.join(data_dir,'prompt',f'{PROPS_MORPH_TYPES[int(m_index)]}-statistics.npy'),allow_pickle=True)
            if norm: m_props = normalize_mean_std(m_props,m_props_stat.item().get('mean'),m_props_stat.item().get('std'))
            props.append(m_props.reshape(1,10))

    prompt = torch.from_numpy(np.concatenate(props,axis=0))

    return prompt



def load_and_norm_prompt_slide(data_dir,prior_data_dir,img_name,props_index_str_s,props_index_str_m,norm=1):

    props=[]

    if 'none' not in props_index_str_s:
        for s_index in props_index_str_s.split('-'):
            s_props = np.load(os.path.join(prior_data_dir,f'prompt/{PROPS_SPATIAL_TYPES[int(s_index)]}',img_name.replace('png','npy').replace('PNG','npy')))
            if norm: s_props_stat = np.load(os.path.join(prior_data_dir,'prompt',f'{PROPS_SPATIAL_TYPES[int(s_index)]}-statistics.npy'),allow_pickle=True)
            if norm: s_props = normalize_mean_std(pad(s_props),s_props_stat.item().get('mean'),s_props_stat.item().get('std'))
            s_props=pad_and_sampler(s_props,140,int(len(pad(s_props))/10))
            props.append(s_props.reshape(1,10))
    if 'none' not in props_index_str_m:
        for m_index in props_index_str_m.split('-'):
            m_props = np.load(os.path.join(prior_data_dir,f'prompt/{PROPS_MORPH_TYPES[int(m_index)]}',img_name.replace('png','npy').replace('PNG','npy')))
            if norm: m_props_stat=np.load(os.path.join(prior_data_dir,'prompt',f'{PROPS_MORPH_TYPES[int(m_index)]}-statistics.npy'),allow_pickle=True)
            if norm: m_props = normalize_mean_std(m_props,m_props_stat.item().get('mean'),m_props_stat.item().get('std'))
            props.append(m_props.reshape(1,10))

    prompt = torch.from_numpy(np.concatenate(props,axis=0))

    return prompt



def _read_array_image(path,resize,rgb=False):
    img=cv2.imread(path,-1)
    img=img.astype('float32')
    img=cv2.resize(img,(resize[0],resize[1]))
    if rgb:
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _augment_image(img):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    augseq=iaa.Sequential([
        sometimes(iaa.Affine(
            scale={'x':(0.9,1.01),'y':(0.9,1.1)},
            translate_percent={'x':(-0.07,0.07),'y':(-0.07,0.07)},
            rotate=(-5,5),
            shear=(-5,5),
            )),
        sometimes(iaa.Fliplr(0.5)),
        ],random_order=False)
    seq_det = augseq.to_deterministic()
    img_aug=seq_det.augment_image(img).copy()
    return img_aug


def calcuate_props_of_image(grayI,nucleiM,whiteM,props_index_str_s,props_index_str_m,kfunc_version='S0V0',sample_num=10):
    # READ NULCEI/WHITE MASK
    # whiteM = 1-np.all(whiteI== [0, 0, 0], axis=-1)
    # nucleiM = 1-np.all(nucleiI== [0, 0, 0], axis=-1)

    # SPATIAL STATISTICS
    coordsNuclei = mask2points(nucleiM)
    coordsWhite = mask2points(whiteM)

    props=[]

    spatial_props=[]
    _, s_props = calculate_k_function(coordsNuclei,version=kfunc_version)
    s_props=pad_and_sampler(s_props,140,int(len(pad(s_props,140))/sample_num))
    spatial_props.append(s_props.reshape(1,sample_num))
    _, s_props = calculate_k_function(coordsWhite,version=kfunc_version)
    s_props=pad_and_sampler(s_props,140,int(len(pad(s_props,140))/sample_num))
    spatial_props.append(s_props.reshape(1,sample_num))
    _, s_props = calculate_cross_k_function(coordsNuclei,coordsWhite,version=kfunc_version)
    s_props=pad_and_sampler(s_props,140,int(len(pad(s_props,140))/sample_num))
    spatial_props.append(s_props.reshape(1,sample_num))
    _, s_props = calculate_cross_k_function(coordsWhite,coordsNuclei,version=kfunc_version)
    s_props=pad_and_sampler(s_props,140,int(len(pad(s_props,140))/sample_num))
    spatial_props.append(s_props.reshape(1,sample_num))

    whiteProps=calculate_segmentation_properties_histogram(whiteM,grayI)

    if 'none' not in props_index_str_s:
        for s_index in props_index_str_s.split('-'):
            props.append(spatial_props[int(s_index)])
            
    if 'none' not in props_index_str_m:
        for m_index in props_index_str_m.split('-'):
            m_props,bin_edges,_=plt.hist(whiteProps[int(m_index)],bins=[step*PROPS_RANGE[PROPS_MORPH_TYPES[int(m_index)]]/sample_num for step in range(sample_num+1)],density=False)
            props.append(m_props.reshape(1,sample_num))

    prompt = torch.from_numpy(np.concatenate(props,axis=0))

    return prompt
        

def generate_prior_and_calculate_props(data_dir,img_name,props_index_str_s,props_index_str_m,args):

    img_path=os.path.join(data_dir,img_name)
    nuclei_path = img_path.replace(img_name.split('/')[-2],args.nuclei_folder)
    white_path = img_path.replace(img_name.split('/')[-2],args.white_folder)
    grayI = cv2.imread(img_path,0)
    rgbI = cv2.imread(img_path)[:,:,::-1]
    bgrI = cv2.imread(img_path)

    if args.prior_nuclei:
        nucleiM,_=my_nuclei_extractor.process(rgbI)
        nuclei_segment = bgrI.copy()
        nuclei_segment[nucleiM==0]=[0,0,0]
    else:
        nuclei_segment = cv2.imread(nuclei_path)
        # READ NULCEI/WHITE MASK
        nucleiM = 1-np.all(nuclei_segment== [0, 0, 0], axis=-1)

    if args.prior_white:
        whiteM=white_region_extractor(bgrI,dilate=args.dilate,se_kernel=args.se_kernel,area_thd=args.area_thd,convex_hull=args.convex_hull,clear_border=args.clear_border)
        white_segment = bgrI.copy()
        white_segment[whiteM==0]=[0,0,0]
    else:
        white_segment = cv2.imread(white_path)
        whiteM = 1-np.all(white_segment== [0, 0, 0], axis=-1)

    # whiteM=white_region_extractor(bgrI,dilate=args.dilate,area_thd=args.area_thd,convex_hull=args.convex_hull,clear_border=args.clear_border)
    # white_segment = bgrI.copy()
    # white_segment[whiteM==0]=[0,0,0]

    # load_white_segment = cv2.imread(white_path)
    # load_whiteM = 1-np.all(load_white_segment== [0, 0, 0], axis=-1)

    prompt=calcuate_props_of_image(grayI,nucleiM,whiteM,props_index_str_s,props_index_str_m,kfunc_version=args.kfunc_version,sample_num=args.sample_number)

    # load_prompt=calcuate_props_of_image(grayI,nucleiM,load_whiteM,props_index_str_s,props_index_str_m,kfunc_version=args.kfunc_version)
    # st()

    # aa=0

    return prompt

def generate_prior_and_calculate_props_slide(data_dir,prior_data_dir,img_name,props_index_str_s,props_index_str_m,args):

    img_path=os.path.join(data_dir,img_name)
    nuclei_path = os.path.join(prior_data_dir,args.nuclei_folder,img_name)
    white_path = os.path.join(prior_data_dir,args.white_folder,img_name)
    grayI = cv2.imread(img_path,0)
    rgbI = cv2.imread(img_path)[:,:,::-1]
    bgrI = cv2.imread(img_path)

    if args.prior_nuclei:
        nucleiM,_=my_nuclei_extractor.process(rgbI)
        nuclei_segment = bgrI.copy()
        nuclei_segment[nucleiM==0]=[0,0,0]
    else:
        nuclei_segment = cv2.imread(nuclei_path)
        # READ NULCEI/WHITE MASK
        nucleiM = 1-np.all(nuclei_segment== [0, 0, 0], axis=-1)

    if args.prior_white:
        whiteM=white_region_extractor(bgrI,dilate=args.dilate,se_kernel=args.se_kernel,area_thd=args.area_thd,convex_hull=args.convex_hull,clear_border=args.clear_border)
        white_segment = bgrI.copy()
        white_segment[whiteM==0]=[0,0,0]
    else:
        white_segment = cv2.imread(white_path)
        whiteM = 1-np.all(white_segment== [0, 0, 0], axis=-1)

    prompt=calcuate_props_of_image(grayI,nucleiM,whiteM,props_index_str_s,props_index_str_m,kfunc_version=args.kfunc_version,sample_num=args.sample_number)
   

    return prompt


def load_or_generate_props(data_dir,img_name,props_index_str_s,props_index_str_m,args):
    if args.generate_props:
        prompt=generate_prior_and_calculate_props(data_dir,img_name,props_index_str_s,props_index_str_m,args)
    else:
        prompt=load_props(data_dir,img_name,props_index_str_s,props_index_str_m)
        
    return prompt


def load_or_generate_props_slide(data_dir,prior_data_dir,img_name,props_index_str_s,props_index_str_m,args):

    if args.generate_props:
        prompt=generate_prior_and_calculate_props_slide(data_dir,prior_data_dir,img_name,props_index_str_s,props_index_str_m,args)
    else:
        prompt=load_props_slide(prior_data_dir,img_name,props_index_str_s,props_index_str_m)
        
    return prompt



