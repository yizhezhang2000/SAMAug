# SAMAug: Boosting Medical Image Segmentation with Segmentation Foundation Model

We introduce an efficient method (SAMAug) for utilizing segmentation foundation model (e.g., SAM) for boosting medical image segmentation. SAMAug utilizes segmentation foundation model (SAM) to augment image inputs for medical images. The augmented images thus generated are used for training and testing a task-specific medical image segmentation model (e.g., a U-Net model). SAMAug does not require fine-tuning on the foundation model. See below for an overview of the proposed method.




More technical details can be found in this technical report: https://arxiv.org/abs/2304.11332



## Experiments and Results:

### cell segmentation in H&E stained images:
MoNuSeg (https://monuseg.grand-challenge.org/):
Model | SAMAug | AJI | Pixel F-score |
--- | --- | --- | --- |
U-Net | no | 58.36 | 75.70 | 
U-Net | yes | 64.30 | 82.56 | 
P-Net | no | 59.46 | 77.09 | 
P-Net | yes | 63.98 | 82.56 | 
Attention Net | no | 58.76 | 75.43 | 
Attention Net | yes | 63.15 | 81.49 | 


### polyp segmentation:
(https://github.com/DengPingFan/PraNet)
CVC-ClinicDB:
Model | SAMAug | meanDic | meanIoU |
--- | --- | --- | --- |
PraNet | no | 85.8 | 80.0 | 
PraNet | yes | 89.1 | 83.9 | 

CVC-300:
Model | SAMAug | meanDic | meanIoU |
--- | --- | --- | --- |
PraNet | no | 87.7 | 80.2 | 
PraNet | yes | 87.9 | 80.6 | 

CVC-ColonDB:
Model | SAMAug | meanDic | meanIoU |
--- | --- | --- | --- |
PraNet | no | 67.3 | 59.8 | 
PraNet | yes | 70.6 | 63.2 | 

ETIS-LaribPolypDB:
Model | SAMAug | meanDic | meanIoU |
--- | --- | --- | --- |
PraNet | no | 57.6 | 50.8 | 
PraNet | yes | 64.0 | 57.2 | 

Kvasir:
Model | SAMAug | meanDic | meanIoU |
--- | --- | --- | --- |
PraNet | no | 85.4 | 78.8 | 
PraNet | yes | 89.7 | 83.7 | 

## Pre-computed SAM Augmented Images

The SAM augmented images for the CVC-ClinicDB, CVC-300, CVC-ColonDB, ETIS-LaribPolypDB and Kvasir datasets can be obtained at the link below.



In the near future, we will release more SAM augmented images for more datasets. 

You can also use the script in SAMAug_batch.py for generating SAM augmented images for your own medical image dataset.


Please let us know your thoughts on this method. Thank you! 





