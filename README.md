# SAMAug

We introduce a simple yet effective method (SAMAug) for utilizing segmentation foundation model (e.g., SAM) for improving medical image segmentation. SAMAug utilizes segmentation foundation model (SAM) to augment an image input for medical image segmentation. This method does not require fine-tuning on the foundation model, and expects to have the foundation model (SAM) only being applied once per image in model training. An overview of the proposed method can be seen below.



Experiments and Results:
on cell segmentation in H&E stained images:


on polyp segmentation:

CVC-300:
Model | SAMAug | meanDic | meanIoU |
--- | --- | --- | --- |
PraNet | no | 87.7 | 80.2 | 
--- | --- | --- | --- |
PraNet | yes | 87.9 | 80.6 | 


The SAM augmented images for the polyp segmentation experiments can be found here. 


To generate SAM augmented images yourself, please take a look:

In the near future, we will release more SAM augmented images for more datasets.

Please let us know your thoughts on this method. Thank you! 





