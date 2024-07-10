# Random color transformation for single domain generalized retinal image segmentation
Please read our [paper](https://doi.org/10.1016/j.engappai.2024.108907) for more details!

## Introduction
Fundus examination, conducted through the analysis of retinal images, plays a pivotal role in aiding the diagnosis of ophthalmic diseases. Currently, deep learning models have been widely applied to retinal image analysis, specifically retinal image segmentation, including vessels, optic cups, lesions, etc., yielding promising outcomes. Nonetheless, the performance of these deep learning models experiences substantial degradation due to the domain shift between the distribution of the training images and unseen test images. In this paper, we focus on the challenging single-domain generalization (SDG), which aims to learn a generalized model on only one source domain, with the expectation that it performs well on unseen test domains. Our work is motivated by the observation that the main differences in retinal images from different domains primarily reside in their color variations, rather than changes in the shapes of objects. To this end, we present random color transformation (RCT) for SDG. RCT performs random linear transformations to each color channel of the training image. Through this approach, RCT can generate training images with rich and vibrant color representations, while preserving the structural information of objects in the images. Experiments are conducted over optic cup segmentation, retinal vessel segmentation, and diabetic retinopathy multi-lesion segmentation tasks, involving eight publicly available datasets. Experimental results show that the proposed RCT outperforms comparison SDG methods, achieving improvements of 5.9%, 1.3%, and 3.6% compared to the second-best method on the optic cup, vessel, and lesion segmentation tasks, respectively. The source code will be available at https://github.com/guomugong/RCT.


## Our pretrained model on the DRIVE dataset is at snapshot/pretrain_drive.pth
1) Testing the pretrained model on the DRIVE dataset
```
python3 predict.py drive drive 1
```
2) Testing the pretrained model on the STARE dataset
```
python3 predict.py drive stare 1
```
3) Testing the pretrained model on the CHASE dataset
```
python3 predict.py drive chase 1
```

## Training with RCT
1) Training a Unet on the DRIVE dataset (modify train.py and utils/dataset.py if necessary)
```
python3 train.py
```
2) Testing a well-trained Unet on the STARE dataset
```
Modify predict.py to load your checkpoint
python3 predict.py drive stare 1
```

## Training with Ensemble-RCT
You can modify utils/dataset.py to adjust the transformation strength of RCT.
Run train.py
Repeat the procedure for five runs.

## License
This code can be utilized for academic research.
