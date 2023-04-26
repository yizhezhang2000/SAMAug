# SAMAug - Boosting Medical Image Segmentation with Segmentation Foundation Model

We introduce an efficient method (SAMAug) for utilizing segmentation foundation model (e.g., SAM) for boosting medical image segmentation. SAMAug utilizes segmentation foundation model (SAM) to augment image inputs for medical image segmentation. The augmented images thus generated are used for training and testing a task-specific medical image segmentation model. SAMAug does not require fine-tuning on the foundation model. An overview of the proposed method can be seen below.




More technical details can be found in this technical report: https://arxiv.org/abs/2304.11332



## Pre-computed SAM Augmented Images

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
PraNet | yes | 69.7 | 62.2 | 

ETIS-LaribPolypDB:
Model | SAMAug | meanDic | meanIoU |
--- | --- | --- | --- |
PraNet | no | 57.6 | 50.8 | 
PraNet | yes | 64.5 | 57.1 | 

Kvasir:
Model | SAMAug | meanDic | meanIoU |
--- | --- | --- | --- |
PraNet | no | 85.4 | 78.8 | 
PraNet | yes | 88.4 | 81.6 | 

The SAM augmented images for the polyp segmentation experiments can be found here. 


To generate SAM augmented images yourself, please take a look:

In the near future, we will release more SAM augmented images for more datasets.

Please let us know your thoughts on this method. Thank you! 





