import numpy as np
import random
import os
import sys
import math
import cv2
import skimage.io as io
from skimage import color
from skimage.measure import label
import glob
from sklearn.utils import resample
from skimage import measure
from scipy import ndimage
import pandas as pd
import anndata as ad
from scipy.spatial.distance import cdist
from skimage import io
from PIL import Image

from pdb import set_trace as st

from skimage import data,filters,segmentation,measure,morphology,color

import matplotlib.pyplot as plt

from histocartography.preprocessing import NucleiExtractor, NucleiConceptExtractor


# NUCLEI SEGMENTATION
def nuclei_extractor(x,mode='pannuke',batch_size=None):
    # based on HoverNet 
    if mode == 'pannuke':
        model = NucleiExtractor(pretrained_data='pannuke',batch_size=batch_size)
        nuclei_map, _ = model.process(x) # RGB
    elif mode == 'monusac':
        model = NucleiExtractor(pretrained_data='monusac',batch_size=batch_size)
        nuclei_map, _ = model.process(x) # RGB 
    
    return nuclei_map


# WHITE REGION SEGMENTATION
def white_region_extractor(img,dilate=0,se_kernel=11,area_thd=0,convex_hull=0,clear_border=0):

    lower=np.array([0,0,200],dtype='uint8')
    upper=np.array([180,30,255],dtype='uint8')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # attention: bgr order
    mask=cv2.inRange(hsv,lower,upper)
    kernel = np.ones((4,4),np.uint8)
    if dilate: mask = cv2.dilate(mask,kernel,iterations=dilate)
    se = np.ones((se_kernel,se_kernel), dtype='uint8')
    mask_close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se)
    cnt,hierachy = cv2.findContours(mask_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ttmask = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(ttmask, cnt, -1, 255, -1)
    if area_thd: ttmask=morphology.remove_small_objects(ttmask>250, min_size=area_thd, connectivity=5)
    if convex_hull: ttmask=morphology.convex_hull_object(ttmask>0, connectivity=2)
    if clear_border: ttmask=segmentation.clear_border(ttmask)
    output=(ttmask>0)*1  
    return output


def refine_binary_mask(mask,dilate=0,area_thd=0,convex_hull=0,clear_border=0):
    mask=mask.astype(np.uint8)
    kernel = np.ones((4,4),np.uint8)
    if dilate: mask = cv2.dilate(mask,kernel,iterations=dilate)
    se = np.ones((11,11), dtype='uint8')
    mask_close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se)
    cnt,hierachy = cv2.findContours(mask_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ttmask = np.zeros(mask.shape[:2], np.uint8)
    cv2.drawContours(ttmask, cnt, -1, 255, -1)
    if area_thd: ttmask=morphology.remove_small_objects(ttmask>250, min_size=area_thd, connectivity=5)
    if convex_hull: ttmask=morphology.convex_hull_object(ttmask>0, connectivity=2)
    if clear_border: ttmask=segmentation.clear_border(ttmask)
    output=(ttmask>0)*1  
    return output



