<div align="center">

<!-- <h1>QAP </h1> -->
<h2 class="papername"> Prompting Vision Foundation Models for Pathology Image Analysis </h2>
<div>
    <a target="_blank">Chong Yin</a>,
    <a target="_blank">Siqi Liu</a>,
    <a target="_blank">Kaiyang Zhou</a>,
    <a target="_blank">Vincent Wai-Sun Wong</a>,
    <a target="_blank">Pong C. Yuen</a>
</div>

<div>
IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2024
</div>

<br>
  
<img src='assets/idea-results.png' width='90%'>

</div>

## If you find this work useful for your research, please kindly cite our paper and star our repo.

## Updates
- [03/2024] Code will be released soon.

## Introduction

This is the github repository of *Prompting Vision Foundation Models for Pathology Image Analysis*. In this paper, we propose Quantitative Attribute-based Prompting (QAP), a novel prompting method specifically for liver pathology image analysis. QAP is based on two quantitative attributes, namely K-function-based spatial attributes and histogram-based morphological attributes, which are aimed for quantitative assessment of tissue states. Moreover, a conditional prompt generator is designed to turn these instance-specific attributes into visual prompts.

The framework of the proposed QAP model:

<div align="center">
<img src='./assets/framework' width='100%'>
</div>

## Enhanced Interpretability

 Image samples with its attention map and attribute significance histogram when identifying specific histological findings. Our method enhances interpretability by visually representing the decision-making process through attention maps and attribute significance histograms. a. The structures the model focuses on; b. The attributes of structures the model focus on.

![Score](assets/interpretation.png)

## Boost Diagnosis Performance
We further explore learning various prompts. The tissue structure segments provide more informative cues compared to task-agnostic visual prompts learned from randomly initialized vectors. Additionally, the quantitative attributes obtained from summarizing the statistical information about tissue structures are more explicit. Using prompts learned conditioned on explicit cues can enhance the learning process and improve performance.

![Score](assets/explicitness.png)


<!-- ## Citation

If you find this work useful for your research, please kindly cite our paper:
```
@inproceedings{chen2024lion,
    title={Prompting Vision Foundation Models for Pathology Image Analysis}, 
    author={Chong Yin, Siqi Liu, Kaiyang Zhou, Vincent Wai-Sun Wong, Pong C. Yuen},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2024}
}
``` -->
