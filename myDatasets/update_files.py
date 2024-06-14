import os
import numpy
import glob
import cv2
import pandas as pd
import argparse
import termcolor
from pdb import set_trace as st


# convert to colored strings
def toRed(content): return termcolor.colored(content,"red",attrs=["bold"])
def toGreen(content): return termcolor.colored(content,"green",attrs=["bold"])
def toBlue(content): return termcolor.colored(content,"blue",attrs=["bold"])
def toCyan(content): return termcolor.colored(content,"cyan",attrs=["bold"])
def toYellow(content): return termcolor.colored(content,"yellow",attrs=["bold"])
def toMagenta(content): return termcolor.colored(content,"magenta",attrs=["bold"])


ROOT_DIR='/home/comp/chongyin/DataSets/Liver-NASH/PatchOf224X224Percent40StrongIsolation/Box-22'
OUT_DIR='/home/comp/chongyin/DataSets/'

def parse_args():
    parser = argparse.ArgumentParser(description='Train network')
    
    parser.add_argument('--subfolder', default='clean_structures/nuclei_segment', type=str)
    
    args = parser.parse_args()

    return args


def update_file_list():
    pass

def update_image_files(dataD,subfolder):
    paths=dataD['path'].tolist()
    paths=[x.split('/')[-1].split('.')[0] for x in paths]
    paths=set(paths)
    txt_lists=[]
    for fold in range(5):
        for term in ['train','val','test']:
            f=open(os.path.join(ROOT_DIR,'CVFold',f'{term}_fold_{fold}.txt'),'r')
            lines=f.readlines()
            lists=[x.strip().split(' ')[0].split('/')[1].split('.')[0] for x in lines]
            txt_lists+=lists

    txt_lists=set(txt_lists)
    files = glob.glob(os.path.join(ROOT_DIR,subfolder,'*S-*'))

    for item in files:
        if os.path.basename(item).split('.')[0] not in txt_lists:
            print(item)
            os.remove(item)


def main():

    args=parse_args()

    dataD=pd.read_excel(os.path.join(ROOT_DIR,'CVFold','data.xlsx'))
    files1=[]
    files2=[]
    for ii in range(len(dataD)):
        if dataD['fold'][ii]==-1:
            files1.append(dataD['path'][ii])
        if dataD['fold'][ii]==-2:
            files2.append(dataD['path'][ii])

    files1=[x.replace('clean_structures/','') for x in files1]
    files2=[x.replace('clean_structures/','') for x in files2]
    assert len(files1) == len(files2)

    print(files1)
    print('--'*5)
    print(files2)

    for fold in range(5):
        for term in ['test','val','train']:
            f=open(os.path.join(ROOT_DIR,'CVFold','old_cv',f'{term}_fold_{fold}.txt'),'r')
            lines=f.readlines()
            f.close()
            fwrite=open(os.path.join(OUT_DIR,f'{term}_fold_{fold}.txt'),'w')
            for line in lines:
                ss=line.strip().split(' ')
                if ss[0] in files1:
                    index=files1.index(ss[0])
                    newline=f'{files2[index]} {ss[1]}\n'
                    print(toRed(f'{term}_fold_{fold}.txt'))
                    print(toGreen(f'{line}'))
                    print(toGreen(f'{newline}'))
                else:
                    newline=line
                fwrite.write(newline)
    
            fwrite.close()
            print(os.path.join(OUT_DIR,f'{term}_fold_{fold}.txt'))


    



if __name__ == '__main__':
    main()