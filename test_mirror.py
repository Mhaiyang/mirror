"""
  @Time    : 2018-05-07
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com

"""

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import mrcnn.utils as utils
import mrcnn.model as modellib
import mrcnn.visualize as visualize
from mrcnn.config import Config

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
MIRROR_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_mirror_heads.h5")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "data", "test", "image")
OUTPUT_PATH = os.path.join(ROOT_DIR, 'data', 'test', "output")


## Configurations
class MirrorConfig(Config):
    """Derives from the base Config class and overrides values specific to the Mirror dataset"""
    NAME = "Mirror"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1 # Mirror has only one class (mirror).
    RPN_ANCHOR_SCALES = (16, 32, 128, 32, 16)  # anchor side in pixels
    DETECTION_MIN_CONFIDENCE = 0.5


class InferenceConfig(MirrorConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    iou_threshold = 0.5


config = InferenceConfig()
config.display()


# ## Create Model and Load Trained Weights

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)

# Load weights trained on MS-COCO
model.load_weights(MIRROR_MODEL_PATH, by_name=True)


# MIRROR Class Names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'Mirror']


# ## Run Object Detection
imglist = os.listdir(IMAGE_DIR)
print("Total {} test images".format(len(imglist)))
for imgname in imglist:

    i = 0
    mAPs = []
    mAPs_range = []

    image = skimage.io.imread(os.path.join(IMAGE_DIR, imgname))
    # Run detection
    results = model.detect(imgname, [image], verbose=1)
    # Visualize results
    # As detect function returns a list of dict, one dict per image,
    # and each call detect function only feed one image, r =results[0]
    r = results[0]
    visualize.display_instances_and_save_image(imgname, OUTPUT_PATH, image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])

    ###########################################################################
    ################  Quantitative Evaluation for Single Image ################
    ###########################################################################
    # gt_box = c
    # gt_class_id = b
    # gt_mask = c
    # pred_box = r['rois']
    # pred_class_id = r['class_ids']
    # pred_score = r['scores']
    # pred_mask = r['masks']
    #
    # # mAP for a certain IoU threshold
    # mAP, precisions, recalls, overlaps = utils.compute_ap(gt_box, gt_class_id, gt_mask,
    #                                                 pred_box, pred_class_id, pred_score, pred_mask,
    #                                                 iou_threshold = InferenceConfig.iou_threshold)
    # mAPs[i] = mAP
    # print("mAP is : {}".format(mAP))
    #
    # # mAP over range of IoU thresholds
    # AP = utils.compute_ap_range(gt_box, gt_class_id, gt_mask,
    #                             pred_box, pred_class_id, pred_score, pred_mask,
    #                             iou_thresholds=None, verbose=1)
    # mAPs_range[i] = AP
    # print("mAP over range of IoU thresholds is : {}".format(AP))
    #
    # ###########################################################################
    # ################  Quantitative Evaluation for All Image ################
    # ###########################################################################
    # mean_mAP = sum(mAPs)/len(mAPs)
    # mean_mAP_range = sum(mAPs_range)/len(mAPs_range)
    # print("For test data set, \n mean_mAP is : {} \n mean_mAP_range is : {}"
    #       .format(mean_mAP, mean_mAP_range))





