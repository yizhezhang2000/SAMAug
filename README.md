# SAMAug: Augmenting Medical Images with Segmentation Foundation Model

## Oct. 5th 2023 Update:

Uploaded python scripts (model training and testing with SAMAug) for the polyp segmentation experiments.

## 
We introduce SAMAug, an efficient method that utilizes a segmentation foundation model (SAM) for improving medical image segmentation. SAMAug utilizes a segmentation foundation model (SAM) to augment medical images. The augmented images thus generated are then used for training and testing a task-specific medical image segmentation model (e.g., a U-Net model for cell segmentation). SAMAug does not require fine-tuning on the foundation model. Please see below for an overview of the proposed method.


<img src="https://github.com/yizhezhang2000/SAMAug/blob/main/SAMAug_overview.png" width="95%" height="95%" />

Examples of the SAM-augmented images:

<img src="https://github.com/yizhezhang2000/SAMAug/blob/main/examples.png" width="95%" height="95%" />

More technical details can be found in this technical report: 

Yizhe Zhang, Tao Zhou, Peixian Liang, Danny Z. Chen, Input Augmentation with SAM: Boosting Medical Image Segmentation with Segmentation Foundation Model, arXiv preprint arXiv:2304.11332. 

Link: https://arxiv.org/abs/2304.11332

Below we highlight some experimental results.

## Experiments and Results

### Polyp Segmentation in Endoscopic Images:
(https://github.com/DengPingFan/PraNet)

CVC-ClinicDB:
Model | SAMAug | meanDic | meanIoU | Sm | 
--- | :---: | :---: | :---: | :---: | 
PraNet[1] | &#10007; | 85.8 | 80.0 | 90.6 |
PraNet[1] | &#10003; | 89.1 | 83.9 | 93.1 | 

CVC-300:
Model | SAMAug | meanDic | meanIoU | Sm | 
--- | :---: | :---: | :---: | :---: |
PraNet[1] | &#10007; | 87.7 | 80.2 | 92.6 |
PraNet[1] | &#10003; | 87.9 | 80.6 | 92.8 | 

CVC-ColonDB:
Model | SAMAug | meanDic | meanIoU | Sm |
--- | :---: | :---: | :---: | :---: |
PraNet[1] | &#10007; | 67.3 | 59.8 | 79.4 |
PraNet[1] | &#10003; | 70.6 | 63.2 | 81.9 | 

ETIS-LaribPolypDB:
Model | SAMAug | meanDic | meanIoU | Sm | 
--- | :---: | :---: | :---: | :---: |
PraNet[1] | &#10007; | 57.6 | 50.8 | 76.1 |
PraNet[1] | &#10003; | 64.0 | 57.2 | 79.4 |

Kvasir:
Model | SAMAug | meanDic | meanIoU |  Sm | 
--- | :---: | :---: | :---: | :---: |
PraNet[1] | &#10007; | 85.4 | 78.8 | 88.0 |
PraNet[1] | &#10003; | 89.7 | 83.7 | 91.2 |

### Cell Segmentation in Histology Images:
MoNuSeg (https://monuseg.grand-challenge.org/):
Model | SAMAug | AJI | Pixel F-score |
--- | :---: | :---: | :---: |
U-Net[2] | &#10007; | 58.36 | 75.70 | 
U-Net[2] | &#10003; | 64.30 | 82.56 | 
P-Net[3] | &#10007; | 59.46 | 77.09 | 
P-Net[3] | &#10003; | 63.98 | 82.56 | 
Attention Net[4] | &#10007; | 58.76 | 75.43 | 
Attention Net[4] | &#10003; | 63.15 | 81.49 | 

[1] Fan, Deng-Ping, et al. "Pranet: Parallel reverse attention network for polyp segmentation." MICCAI, 2020.

[2] Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: MICCAI, 2015.

[3] Wang, Guotai, et al. "DeepIGeoS: a deep interactive geodesic framework for medical image segmentation." IEEE-TPAMI, 2018.

[4] Oktay, Ozan, et al. "Attention U-Net: Learning Where to Look for the Pancreas." Medical Imaging with Deep Learning, 2018.

## Pre-computed SAM-Augmented Images

The SAM-augmented images used in the polyp segmentation (CVC-ClinicDB, CVC-300, CVC-ColonDB, ETIS-LaribPolypDB, and Kvasir datasets) can be downloaded at the link below. One may consider using these augmented data to train and test new polyp segmentation models.

https://drive.google.com/drive/folders/1q6Ics1OuKVv0c1xGddUrQTx5QZhvB2iS?usp=share_link

In the near future, we will share more SAM-augmented images for more datasets. 

You can also refer the script in [SAMAug.py](https://github.com/yizhezhang2000/SAMAug/blob/main/SAMAug.py) for generating SAM-augmented images for your own medical image data.

##
Questions and comments are welcome! We believe there is room for further improvement. Please consider sharing your experience in using SAMAug. Thank you.




