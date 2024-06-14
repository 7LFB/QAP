import numpy as np
import os
import sys
import math
import skimage.io as io
from skimage.measure import label, regionprops
import glob
from skimage import measure
from scipy import ndimage
import pandas as pd
import anndata as ad
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from skimage.measure import shannon_entropy
import cv2
import pyfeats
import re

from utils.utils import *
from utils.histo_lib import *

from pdb import set_trace as st


PROPS_RANGE={'white-props-areas':25000,'white-props-eccs':1.0,'white-props-circus':1,'white-props-intens':255,'white-props-entropys':7,'white-props-shapes':10,'white-props-extents':1,'white-props-perimeters':1500,'white-props-percents':1}

PROPS_SPATIAL_TYPES=['nuclei-kfunc','white-kfunc','nuclei2white-kfunc','white2nuclei-kfunc']
PROPS_MORPH_TYPES=['white-props-areas', 'white-props-eccs', 'white-props-circus', 'white-props-intens', 'white-props-entropys', 'white-props-shapes', 'white-props-extents', 'white-props-perimeters', 'white-props-percents']


def mkdirs_if_need(x):
    folder=os.path.dirname(x)
    if not os.path.exists(folder):
        os.makedirs(folder)

# differnet edge correction methods
def Kfunc(dist,n=None,m=1,version=0):
    if version==0:
        r = np.arange(0, np.max(dist), 1)
        K = np.zeros_like(r)
        for i in range(len(r)):
            K[i] = (np.sum(dist < r[i]) / n / m )** 2 * np.pi * r[i] ** 2
    elif version==1:
        r = np.arange(0, np.max(dist), 1)
        K = np.zeros_like(r)
        for i in range(len(r)):
            K[i] = np.sum(dist < r[i])
    elif version==2:
        r = np.arange(0, np.max(dist), 1)
        K = np.zeros_like(r)
        for i in range(len(r)):
            K[i] = np.sum(dist < r[i])/(n*m* np.pi * r[i] ** 2 + 1e-16)
    elif version==3:
        r = np.arange(0, np.max(dist), 1)
        K = np.zeros_like(r)
        for i in range(len(r)):
            K[i] = np.sum(dist < r[i])/(n*m + 1e-16)
    elif version==4:
        r = np.arange(0, np.max(dist), 1)
        K = np.zeros_like(r)
        for i in range(len(r)):
            K[i] = np.sum(dist < r[i])/(n + 1e-16)
    elif version==5:
        r = np.arange(0, np.max(dist), 1)
        K = np.zeros_like(r)
        for i in range(len(r)):
            weights = weights = 1 - (np.pi * r[i]**2) / (4 * np.max(dist)**2)
            K[i] = np.sum(dist < r[i])/(n*m*weights + 1e-16)
    elif version==6:
        r = np.arange(0, np.max(dist), 1)
        K = np.zeros_like(r)
        for i in range(len(r)):
            weights = weights = 1 - r[i] / np.max(dist)
            K[i] = np.sum(dist < r[i])/(n*m*weights + 1e-16)

    return r, K




def calculate_cross_k_function(pointS, pointT, version='S0V0'):

    digits=[int(x) for x in re.findall(r'\d+',version)]

    if len(pointT)<=0 or len(pointS)<=0:
        r=np.arange(0,4,1)
        K=np.zeros_like(r)
        return r,K

    dist = cdist(pointS, pointT)
    # dist(s,t) symmetric or not
    if not digits[0]:
        dist = np.min(dist, axis=1)
    # Calculate K function
    r,K = Kfunc(dist,n=len(pointS),m=len(pointT),version=digits[1])

    return r, K

def calculate_k_function(points,version='S0V0'):
    digits=[int(x) for x in re.findall(r'\d+',version)]

    if len(points)<=1:
        r=np.arange(0,4,1)
        K=np.zeros_like(r)
        return r,K
    
    dist = cdist(points, points)
    # dist(s,t) symmetric or not
    if not digits[0]:
        ind = np.diag_indices_from(dist)
        dist[ind]= math.inf
        dist = np.min(dist, axis=1)
    # Calculate K function
    r,K = Kfunc(dist,n=len(points),version=digits[1])

    return r, K


def calculate_density_map(mask,sigma=10):

    # Calculate distance transform of mask
    dist = ndimage.distance_transform_edt(mask)
    # add by chong
    dist = dist.max()-dist

    density = np.exp(-(dist ** 2) / (2 * sigma ** 2))

    return density


def check_inf_nan(x):
    new_x=[]
    for item in x:
        if math.isinf(item) or math.isnan(item):
            new_x.append(0.0)
        else:
            new_x.append(item)
    return new_x


def circularity_of_region(prop):
    circus=(4 * np.pi * prop.area) / (prop.perimeter ** 2 + 1e-6)
    return circus


def calculate_segmentation_properties(mask,img=None,max=1,filter=0):


    # Calculate segmentation properties
    label = measure.label(mask,connectivity=2)
    props = measure.regionprops(label,intensity_image=img)
    props=sorted(props,key=lambda r: r.area, reverse=False)
    if max:
        props=sorted(props,key=lambda r: r.area, reverse=True) # descending 
        props=props[:max]

    if filter:
        props=[prop for prop in props if circularity_of_region(prop)>filter]

    # Extract area, eccentricity, and shape for each object
    areas = [prop.area for prop in props]
    areas = check_inf_nan(areas)
    eccs = [prop.eccentricity  for prop in props]
    eccs = check_inf_nan(eccs)
    circus = [(4 * np.pi * prop.area) / (prop.perimeter ** 2 + 1e-6) for prop in props]
    circus = check_inf_nan(circus)
    intens =[prop.intensity_mean for prop in props]
    intens = check_inf_nan(intens)
    shapes = [prop.euler_number for prop in props]
    shapes = check_inf_nan(shapes)
    extents = [prop.extent for prop in props]
    extents = check_inf_nan(extents)
    perimeters = [prop.perimeter for prop in props]
    perimeters = check_inf_nan(perimeters)
    percents = [prop.area/(mask.shape[0]*mask.shape[1]+1e-6) for prop in props]
    percents = check_inf_nan(percents)
    entropys = [shannon_entropy(prop.image_intensity) for prop in props]
    entropys = check_inf_nan(entropys)
    if len(areas)==0: areas=[0]
    if len(eccs)==0: eccs=[0]
    if len(circus)==0: circus=[0]
    if len(intens)==0: intens=[0]
    if len(entropys)==0: entropys=[0]
    if len(shapes)==0: shapes=[0]
    if len(extents)==0: extents=[0]
    if len(perimeters)==0: perimeters=[0]
    if len(percents)==0: percents=[0]

    return areas, eccs, circus, intens, entropys, shapes, extents, perimeters, percents 

def mask2anndata(mask):
    # Calculate segmentation properties
    label = measure.label(mask,connectivity=2)
    props = measure.regionprops(label)
    centroids = [prop.centroid for prop in props] #(y,x)
    data = np.array(centroids)
    obs = pd.DataFrame({'cell1':[str(ii) for ii in range(len(centroids))]})
    var = pd.DataFrame({'gene1':['Y','X']})
    adata = ad.AnnData(data, dtype=np.float32, obs=obs, var=var)

    return adata


def calculate_segmentation_properties_histogram(mask,img=None,max=0,filter=0):


    # Calculate segmentation properties
    label = measure.label(mask,connectivity=2)
    props = measure.regionprops(label,intensity_image=img)
    props=sorted(props,key=lambda r: r.area, reverse=False)
    if max:
        props=sorted(props,key=lambda r: r.area, reverse=True) # descending 
        props=props[:max]

    if filter:
        props=[prop for prop in props if circularity_of_region(prop)>filter]
    # Extract area, eccentricity, and shape for each object
    areas = [prop.area for prop in props]
    areas = check_inf_nan(areas)
    eccs = [prop.eccentricity  for prop in props]
    eccs = check_inf_nan(eccs)
    circus = [(4 * np.pi * prop.area) / (prop.perimeter ** 2 + 1e-6) for prop in props]
    circus = check_inf_nan(circus)
    intens =[prop.intensity_mean for prop in props]
    intens = check_inf_nan(intens)
    shapes = [prop.euler_number for prop in props]
    shapes = check_inf_nan(shapes)
    extents = [prop.extent for prop in props]
    extents = check_inf_nan(extents)
    perimeters = [prop.perimeter for prop in props]
    perimeters = check_inf_nan(perimeters)
    percents = [prop.area/(mask.shape[0]*mask.shape[1]+1e-6) for prop in props]
    percents = check_inf_nan(percents)
    entropys = [shannon_entropy(prop.image_intensity) for prop in props]
    entropys = check_inf_nan(entropys)
    if len(areas)==0: areas=[0]
    if len(eccs)==0: eccs=[0]
    if len(circus)==0: circus=[0]
    if len(intens)==0: intens=[0]
    if len(entropys)==0: entropys=[0]
    if len(shapes)==0: shapes=[0]
    if len(extents)==0: extents=[0]
    if len(perimeters)==0: perimeters=[0]
    if len(percents)==0: percents=[0]

    return areas, eccs, circus, intens, entropys, shapes, extents, perimeters, percents 


def mask2points(mask):
    # Calculate segmentation properties
    label = measure.label(mask,connectivity=2)
    props = measure.regionprops(label)
    centroids = [prop.centroid for prop in props] #(y,x)

    return centroids



def pad_and_sampler(x,L=140,step=14):
    x=x.astype(np.float32) # some integers would not change into float automatically, raising error when send to batch
    if len(x)<140:
        x=np.pad(x,L-len(x),'edge')
    x = x[:L]
    return x[::step]

def pad(x,L=140):
    if len(x)<140:
        x=np.pad(x,L-len(x),'edge')
    x = x[:L]
    return x


def sampler(x,step):
    return x[::step]


def normalize_min_max(x):
    x= 2*((x-x.min())/(x.max()-x.min()+1e-10)-0.5)
    return x

def normalize_mean_std(x,x_mean,x_std):
    nx = (x-x_mean)/(x_std+1e-12)
    return nx


def pad_sampler_norm_kfunc(x):
    x=pad_and_sampler(x,L=140,step=20)
    x=normalize_min_max(x)
    return x


def calcuate_norm_k_function_from_seg(whiteI,nucleiI):

    # READ NULCEI/WHITE MASK
    whiteM = 1-np.all(whiteI== [0, 0, 0], axis=-1)
    nucleiM = 1-np.all(nucleiI== [0, 0, 0], axis=-1)
    coordsNuclei = mask2points(nucleiM)
    coordsWhite = mask2points(whiteM)

    _, nK = calculate_k_function(coordsNuclei)
    nK = pad_sampler_norm_kfunc(nK)
 
    _, wK = calculate_k_function(coordsWhite)
    wK = pad_sampler_norm_kfunc(wK)
        
    _, n2wK = calculate_cross_k_function(coordsNuclei,coordsWhite)
    n2wK = pad_sampler_norm_kfunc(n2wK)
        
    _, w2nK = calculate_cross_k_function(coordsWhite,coordsNuclei)
    w2nK = pad_sampler_norm_kfunc(w2nK)

    return nK, wK, n2wK, w2nK


def calculate_general_k_function(mask,max=0):

    # Calculate segmentation properties
    label = measure.label(mask,connectivity=2)
    props = measure.regionprops(label)
    props=sorted(props,key=lambda r: r.area, reverse=True)
    if max:
        props=props[:max]

    # Extract area, eccentricity, and shape for each object
    areas = [prop.area for prop in props]
    eccs = [prop.eccentricity for prop in props]
    shapes = [prop.euler_number for prop in props]
    extents = [prop.extent for prop in props]
    perimeters = [prop.perimeter for prop in props]
    percents = [prop.area/(mask.shape[0]*mask.shape[1]+1e-6) for prop in props]
    if len(areas)==0: areas=[0]
    if len(eccs)==0: eccs=[0]
    if len(shapes)==0: shapes=[0]
    if len(extents)==0: extents=[0]
    if len(perimeters)==0: perimeters=[0]
    if len(percents)==0: percents=[0]

    # areas: [0,40000]
    # eccs, extents, percents: [0,1]
    # perimeters: [0,2000]
    Rs ={'areas':np.arange(0, 40000, 100),'eccs':np.arange(0, 1, 0.01),'extents':np.arange(0, 1, 0.01),'percents':np.arange(0, 1, 0.01),'perimeters':np.arange(0, 2000, 10)}
    rawKfuncs={'areas':areas,'eccs':eccs,'extents':extents,'perimeters':perimeters,'percents':percents}
    itemKfuncs={'areas':{'K':None,'r':None},'eccs':{'K':None,'r':None},'extents':{'K':None,'r':None},'perimeters':{'K':None,'r':None},'percents':{'K':None,'r':None}}
    for key, item in rawKfuncs.items():
        r = Rs[key]
        K = np.zeros_like(r)
        for i in range(len(r)):
            K[i] = ((item < r[i]).sum() / len(item) ) ** 2 * np.pi
        itemKfuncs[key]['K']=K
        itemKfuncs[key]['r']=r

    return itemKfuncs, Rs



def build_prompt(datadir,cvfolder,args=None):
    PRIORS=['white_segment','nuclei_segment']
    TYPES=['none','inflammation','ballooning','steatosis']
    dataD = pd.read_excel(os.path.join(cvfolder,'data.xlsx'))

    for ii in range(len(dataD)):

        opath=dataD['path'][ii]
        label=dataD['label'][ii]

        fold = dataD['fold'][ii]

        if fold != -2: continue

        filename=os.path.basename(opath)

        print(f'{ii}-{filename}-{label}')

        # READ ORIGINAL IMAGE AND PRIOR SEGMENT

        imgIPath = os.path.join(datadir,TYPES[label],filename)

        nucleiIPath=os.path.join(datadir,PRIORS[1],filename)
                                 
        whiteIPath=os.path.join(datadir,PRIORS[0],filename)

        grayI = cv2.imread(imgIPath,0)
        nucleiI=cv2.imread(nucleiIPath)
        whiteI=cv2.imread(whiteIPath)
        # # change white background to black
        # mask = np.all(nucleiI == [255, 255, 255], axis=-1)
        # nucleiI[mask]=[0,0,0]

        # READ NULCEI/WHITE MASK
        whiteM = 1-np.all(whiteI== [0, 0, 0], axis=-1)
        nucleiM = 1-np.all(nucleiI== [0, 0, 0], axis=-1)

        # SPATIAL STATISTICS

        coordsNuclei = mask2points(nucleiM)
        coordsWhite = mask2points(whiteM)

        _, K = calculate_k_function(coordsNuclei)
        newfile=os.path.join(datadir,'prompt','nuclei-kfunc',filename.replace('png','npy').replace('PNG','npy'))
        mkdirs_if_need(newfile)
        np.save(newfile,K)

        _, K = calculate_k_function(coordsWhite)
        newfile=os.path.join(datadir,'prompt','white-kfunc',filename.replace('png','npy').replace('PNG','npy'))
        mkdirs_if_need(newfile)
        np.save(newfile,K)

        _, K = calculate_cross_k_function(coordsNuclei,coordsWhite)
        newfile=os.path.join(datadir,'prompt','nuclei2white-kfunc',filename.replace('png','npy').replace('PNG','npy'))
        mkdirs_if_need(newfile)
        np.save(newfile,K)

        _, K = calculate_cross_k_function(coordsWhite,coordsNuclei)
        newfile=os.path.join(datadir,'prompt','white2nuclei-kfunc',filename.replace('png','npy').replace('PNG','npy'))
        mkdirs_if_need(newfile)
        np.save(newfile,K)


        whiteProps=calculate_segmentation_properties_histogram(whiteM,grayI,max=args.max,filter=args.filter)
        assert len(whiteProps) == len(PROPS_MORPH_TYPES)
        for kk,item in enumerate(PROPS_MORPH_TYPES):
            counts,bin_edges,_=plt.hist(whiteProps[kk],bins=[step*PROPS_RANGE[item]/10 for step in range(11)],density=False)
            newfile=os.path.join(datadir,'prompt',f'{item}',filename.replace('png','npy').replace('PNG','npy'))
            mkdirs_if_need(newfile)
            np.save(newfile,counts)


        #
        # whitePropsKFunc, _ = calculate_general_k_function(whiteM)
        # newfile=os.path.join(datadir,'prompt','white-props-kfunc',filename.replace('png','npy').replace('PNG','npy'))
        # mkdirs_if_need(newfile)
        # np.save(newfile,whitePropsKFunc)



    print(toRed('Build prompt done!'))


def build_prompt_slide(labno,slideDir,priorDir):

    tiles = glob.glob(os.path.join(slideDir,labno,'tile_*.png'))

    tic = time.time()
    for ii, tilefullname in enumerate(tiles):
        # READ ORIGINAL IMAGE AND PRIOR SEGMENT
        tilename = os.path.basename(tilefullname)

        if os.path.exists(os.path.join(priorDir,'prompt','white-props-percents',labno,tilename.replace('png','npy').replace('PNG','npy'))):
            continue
        
        imgIPath = os.path.join(slideDir,labno,tilename)

        nucleiIPath=os.path.join(priorDir,'nuclei_segment',labno,tilename)
                                 
        whiteIPath=os.path.join(priorDir,'white_segment',labno,tilename)

        grayI = cv2.imread(imgIPath,0)
        nucleiI=cv2.imread(nucleiIPath)
        whiteI=cv2.imread(whiteIPath)
        # # change white background to black
        # mask = np.all(nucleiI == [255, 255, 255], axis=-1)
        # nucleiI[mask]=[0,0,0]

        # READ NULCEI/WHITE MASK
        whiteM = 1-np.all(whiteI== [0, 0, 0], axis=-1)
        nucleiM = 1-np.all(nucleiI== [0, 0, 0], axis=-1)

        # SPATIAL STATISTICS

        coordsNuclei = mask2points(nucleiM)
        coordsWhite = mask2points(whiteM)

        _, K = calculate_k_function(coordsNuclei)
        newfile=os.path.join(priorDir,'prompt','nuclei-kfunc',labno,tilename.replace('png','npy').replace('PNG','npy'))
        mkdirs_if_need(newfile)
        np.save(newfile,K)

        _, K = calculate_k_function(coordsWhite)
        newfile=os.path.join(priorDir,'prompt','white-kfunc',labno,tilename.replace('png','npy').replace('PNG','npy'))
        mkdirs_if_need(newfile)
        np.save(newfile,K)

        _, K = calculate_cross_k_function(coordsNuclei,coordsWhite)
        newfile=os.path.join(priorDir,'prompt','nuclei2white-kfunc',labno,tilename.replace('png','npy').replace('PNG','npy'))
        mkdirs_if_need(newfile)
        np.save(newfile,K)

        _, K = calculate_cross_k_function(coordsWhite,coordsNuclei)
        newfile=os.path.join(priorDir,'prompt','white2nuclei-kfunc',labno,tilename.replace('png','npy').replace('PNG','npy'))
        mkdirs_if_need(newfile)
        np.save(newfile,K)


        whiteProps=calculate_segmentation_properties_histogram(whiteM,grayI)
        assert len(whiteProps) == len(PROPS_MORPH_TYPES)
        for kk,item in enumerate(PROPS_MORPH_TYPES):
            counts,bin_edges,_=plt.hist(whiteProps[kk],bins=[step*PROPS_RANGE[item]/10 for step in range(11)],density=False)
            newfile=os.path.join(priorDir,'prompt',f'{item}',labno,tilename.replace('png','npy').replace('PNG','npy'))
            mkdirs_if_need(newfile)
            np.save(newfile,counts)
        avgtime=(time.time()-tic)/(ii+1)
        print('\r',f'{labno}:{ii}/{len(tiles)}: AvgTime: {avgtime:.3f}/tile, ETD: {avgtime*(len(tiles)-ii):.3f}',end='')

        del grayI, nucleiI, whiteI, nucleiM, whiteM

    print('\t')
    print('Build prompt done!')


def build_pyfeats_prompt(datadir,cvfolder,args=None):
    PRIORS=['white_segment','nuclei_segment']
    TYPES=['none','inflammation','ballooning','steatosis']
    dataD = pd.read_csv(os.path.join(cvfolder,'data.csv'))

    for ii in range(len(dataD)):

        opath=dataD['path'][ii]
        label=dataD['label'][ii]

        filename=os.path.basename(opath)

        print(f'{ii}-{filename}-{label}')

        # READ ORIGINAL IMAGE AND PRIOR SEGMENT

        imgIPath = os.path.join(datadir,TYPES[label],filename)

        nucleiIPath=os.path.join(datadir,PRIORS[1],filename)
                                 
        whiteIPath=os.path.join(datadir,PRIORS[0],filename)

        grayImg = cv2.imread(imgIPath,0)
        nucleiI=cv2.imread(nucleiIPath)
        whiteI=cv2.imread(whiteIPath)
        # # change white background to black
        # mask = np.all(nucleiI == [255, 255, 255], axis=-1)
        # nucleiI[mask]=[0,0,0]

        # READ NULCEI/WHITE MASK
        whiteM = 1-np.all(whiteI== [0, 0, 0], axis=-1)
        nucleiM = 1-np.all(nucleiI== [0, 0, 0], axis=-1)

        # SPATIAL STATISTICS

        coordsNuclei = mask2points(nucleiM)
        coordsWhite = mask2points(whiteM)

        if args.region_type=='white':
            mask=whiteM
        else:
            mask=nucleiM

        if args.feats_type=='DWT':
            features,labels=pyfeats.dwt_features(grayImg,mask,wavelet='bior3.3', levels=3)
        elif args.feats_type=='SWT':
            features, labels = pyfeats.swt_features(grayImg, mask, wavelet='bior3.3', levels=3)
        elif args.feats_type=='morph':
            features, cdf = pyfeats.grayscale_morphology_features(grayImg, 25)
        elif args.feats_type=='lte':
            features, labels = pyfeats.lte_measures(grayImg, mask, l=7)
        elif args.feats_type=='fdta':
            features, labels = pyfeats.fdta(grayImg, mask, s=3)
        elif args.feats_type=='glrlm':
            features, labels = pyfeats.glrlm_features(grayImg, mask, Ng=256)
        elif args.feats_type=='fps':
            features, labels = pyfeats.fps(grayImg, mask)
        elif args.feats_type=='multi_histogram':        
            features, labels = pyfeats.multiregion_histogram(grayImg, mask, bins=10, num_eros=3, square_size=3)


        newfile=os.path.join(datadir,'prompt',f'{args.region_type}-pyfeats-{args.feats_type}',filename.replace('png','npy').replace('PNG','npy'))
        mkdirs_if_need(newfile)
        np.save(newfile,features)


      



    print(toRed('Build prompt done!'))






