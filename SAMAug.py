import numpy as np
import skimage
from skimage.segmentation import find_boundaries
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
device='cuda'
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam,crop_nms_thresh=0.5,box_nms_thresh=0.5,pred_iou_thresh=0.5)

def SAMAug(tI,mask_generator):
    masks = mask_generator.generate(tI)
    tI=skimage.img_as_float(tI)  
    SegPrior=np.zeros((tI.shape[0],tI.shape[1]))
    BoundaryPrior=np.zeros((tI.shape[0],tI.shape[1]))
    for maskindex in range(len(masks)):
        thismask=masks[maskindex]['segmentation']
        stability_score =masks[maskindex]['stability_score']
        thismask_=np.zeros((thismask.shape))
        thismask_[np.where(thismask==True)]=1
        SegPrior[np.where(thismask_==1)]=SegPrior[np.where(thismask_==1)]+stability_score
        BoundaryPrior=BoundaryPrior+find_boundaries(thismask_,mode='thick') 
        BoundaryPrior[np.where(BoundaryPrior>0)]=1
        tI[:,:,1]=tI[:,:,1]+SegPrior
        tI[:,:,2]=tI[:,:,2]+BoundaryPrior
    return tI
