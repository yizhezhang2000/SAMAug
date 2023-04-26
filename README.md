# SAMAug: Boosting Medical Image Segmentation with Segmentation Foundation Model

We introduce SAMAug, an efficient method that utilizes a segmentation foundation model (e.g., SAM) for improving medical image segmentation. SAMAug utilizes a segmentation foundation model (SAM) to augment image inputs for medical images. The augmented images thus generated are then used for training and testing a task-specific medical image segmentation model (e.g., a U-Net model). SAMAug does not require fine-tuning on the foundation model. See below for an overview of the proposed method.


<img src="https://github.com/yizhezhang2000/SAMAug/blob/main/SAMAug_overview.png" width="80%" height="80%" />

Examples of the SAM-augmented images:

<img src="https://github.com/yizhezhang2000/SAMAug/blob/main/examples.png" width="80%" height="80%" />

More technical details can be found in this technical report: 

Yizhe Zhang, Tao Zhou, Peixian Liang, Danny Z. Chen, Input Augmentation with SAM: Boosting Medical Image Segmentation with Segmentation Foundation Model, arXiv preprint arXiv:2304.11332. Link: https://arxiv.org/abs/2304.11332

## Experiments and Results

### Polyp Segmentation in Endoscopic Images:
(https://github.com/DengPingFan/PraNet)

CVC-ClinicDB:
Model | SAMAug | meanDic | meanIoU |
--- | --- | --- | --- |
PraNet[1] | no | 85.8 | 80.0 | 
PraNet[1] | yes | 89.1 | 83.9 | 

CVC-300:
Model | SAMAug | meanDic | meanIoU |
--- | --- | --- | --- |
PraNet[1] | no | 87.7 | 80.2 | 
PraNet[1] | yes | 87.9 | 80.6 | 

CVC-ColonDB:
Model | SAMAug | meanDic | meanIoU |
--- | --- | --- | --- |
PraNet[1] | no | 67.3 | 59.8 | 
PraNet[1] | yes | 70.6 | 63.2 | 

ETIS-LaribPolypDB:
Model | SAMAug | meanDic | meanIoU |
--- | --- | --- | --- |
PraNet[1] | no | 57.6 | 50.8 | 
PraNet[1] | yes | 64.0 | 57.2 | 

Kvasir:
Model | SAMAug | meanDic | meanIoU |
--- | --- | --- | --- |
PraNet[1] | no | 85.4 | 78.8 | 
PraNet[1] | yes | 89.7 | 83.7 | 

[1] Fan, Deng-Ping, et al. "Pranet: Parallel reverse attention network for polyp segmentation." Medical Image Computing and Computer Assisted Intervention–MICCAI 2020: 23rd International Conference, Lima, Peru, October 4–8, 2020, Proceedings, Part VI 23. Springer International Publishing, 2020.

### Cell Segmentation in Histology Images:
MoNuSeg (https://monuseg.grand-challenge.org/):
Model | SAMAug | AJI | Pixel F-score |
--- | --- | --- | --- |
U-Net[2] | no | 58.36 | 75.70 | 
U-Net[2] | yes | 64.30 | 82.56 | 
P-Net[3] | no | 59.46 | 77.09 | 
P-Net[3] | yes | 63.98 | 82.56 | 
Attention Net[4] | no | 58.76 | 75.43 | 
Attention Net[4] | yes | 63.15 | 81.49 | 

[2] Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: MICCAI 2015.

[3] Wang, Guotai, et al. "DeepIGeoS: a deep interactive geodesic framework for medical image segmentation." IEEE-TPAMI(2018).

[4] Oktay, Ozan, et al. "Attention U-Net: Learning Where to Look for the Pancreas." Medical Imaging with Deep Learning, 2018.

## Pre-computed SAM-Augmented Images

The SAM-augmented images for the CVC-ClinicDB, CVC-300, CVC-ColonDB, ETIS-LaribPolypDB, and Kvasir datasets can be obtained at the link below.


In the near future, we will release more SAM-augmented images for more datasets. 

You can also use the script in SAMAug_batch.py for generating SAM-augmented images for your own medical image dataset.


Questions and comments are welcome! Thank you!




