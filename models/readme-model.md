# datasets introduction
- **myPromptAugTileDataset.py**: for loading tiles
- **myPromptSlideDataset.py**: for loading wsi

## Ours (prompts and confounders p(y|do(x)))
- **Vision prompts based on prior image**
    - **G(Q(prior)):** learn prompts from quantitative attributes extracted from prior segmentation image
        - `**model_v36**`: using self attention (encoder+decoder) to generate prompts. (BEST)


## Key libs
- **promptEncoder.py**: custom prompt generator
- **vision_transformer.py**: the lib to build VPT
- **vqt_vit.py**: the lib to build VQT
- **custom_vqt_vit.py**: develop based on 'vqt_vit.py', to implement VQT + quantitative attribtues
- **vision_transformer_custom.py** develop based on 'vision_transformer.py', to implement VQT + quantitative attributes


# Key function operators
## ~/utils/
    - **spatial_statistics.py**
        - calculate_segmentation_properties(): only return properties of max area region
        - calculate_segmentation_properties_histogram(): return 10 bins histogram
        - calculate_k_function(): return 140 length K(r)

    - **generate_props.py**
    - **analysis_props.py**



## ~/models/promptEncoders.py
    - promptSAT: using official transformerencoderlayer
    - promptEncoderDecoder: using decoder to generate prompts
    



 
