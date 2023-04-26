# SAMAug: Boosting Medical Image Segmentation with Segmentation Foundation Model

We introduce an efficient method (SAMAug) for utilizing segmentation foundation model (e.g., SAM) for boosting medical image segmentation. SAMAug utilizes segmentation foundation model (SAM) to augment image inputs for medical image. The augmented images thus generated are used for training and testing a task-specific medical image segmentation model. SAMAug does not require fine-tuning on the foundation model. An overview of the proposed method can be seen below.




More technical details can be found in this technical report: https://arxiv.org/abs/2304.11332




## Experiments and Results:

### cell segmentation in H&E stained images:


### polyp segmentation:

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
PraNet | yes | 64.5 | 57.1 | 

Kvasir:
Model | SAMAug | meanDic | meanIoU |
--- | --- | --- | --- |
PraNet | no | 85.4 | 78.8 | 
PraNet | yes | 89.7 | 83.7 | 

## Pre-computed SAM Augmented Images

The SAM augmented images for the CVC-ClinicDB, CVC-300, CVC-ColonDB, ETIS-LaribPolypDB and Kvasir datasets can be obtained at the link below.



In the near future, we will release more SAM augmented images for more datasets. To generate SAM augmented images, please take a look at the link below.



Please let us know your thoughts on this method. Thank you! 





