import os
import numpy
import glob
import cv2
from pdb import set_trace as st


IMG_DIR='/home/comp/chongyin/DataSets/Liver-NASH/PatchOf224X224Percent40StrongIsolation/Box-22/clean_structures/nuclei_segment'

def main():
    files = glob.glob(os.path.join(IMG_DIR,'*.png'))
    for ii,path in enumerate(files):
        print(f'{ii}/{len(files)}')
        img = cv2.imread(path)
        mask=(img[:,:,0]==255)*(img[:,:,1]==255)*(img[:,:,2]==255)
        img[mask]=[0,0,0]
        cv2.imwrite(path,img)


if __name__ == '__main__':
    main()