# Data Instruction

 - MICCAI'21 LiverNASTiles
 - LiverNASTiles-20230729
 - NAFLD-2CLASS
 - Liver-NAS-WSI: store tissue tiles into a list/numpy
    - 06S19236-1.txt (FILE)
        - tile-29-x-y.png
        - tile-30-x-y.png
        - tile-31-x-y.png
    OR
    - 06S19236-1 (FOLDER)
        - tile-29-x-y.png
        - tile-30-x-y.png
        - tile-31-x-y.png

 # Dataloader
  - myTileDataset.py
  - myPromptTileDataset.py
  - mySlideDataset.py
  - myPromptSlide.py



# About properties

 - PROPS_SPATIAL_TYPES=['nuclei-kfunc','white-kfunc','nuclei2white-kfunc','white2nuclei-kfunc']
 - PROPS_MORPH_TYPES=['white-props-areas', 'white-props-eccs', 'white-props-circus', 'white-props-intens', 'white-props-entropys', 'white-props-shapes', 'white-props-extents', 'white-props-perimeters', 'white-props-percents']
 - Combinations
    - Spatial
        - C43: '1-2-3' '0-2-3' '0-1-3' '0-1-2'
        - C44: '0-1-2-3'
    - Morphological
        - C53: '0-1-2' '0-1-6' '0-1-7' '0-2-6' '0-2-7' '0-6-7' '1-2-6' '1-2-7' '1-6-7' '2-6-7'
        - C54: '0-1-2-6' '0-1-2-7' '0-1-6-7' '0-2-6-7' '1-2-6-7'

# Data pre-process and post-process
 - Pre-process
    - inColorRange to segment white regions and nuclei
    - HoverNet for nuclei, inColorRange+drawContour + D0T0H0B0 for white regions (clean-allPrior)
    - 4 image tiles in inflammation
    - how to calculate KFunction (reCalKfunc)