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
import yaml

import mrcnn.utils as utils
import mrcnn.model as modellib
import mrcnn.visualize as visualize
from mrcnn.config import Config
from PIL import Image


# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
MIRROR_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_mirror_heads.h5")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "data", "test", "image")
OUTPUT_PATH = os.path.join(ROOT_DIR, 'data', 'test', "output_all")
if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)


## Configurations
class MirrorConfig(Config):
    """Derives from the base Config class and overrides values specific to the Mirror dataset"""
    NAME = "Mirror"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1 # Mirror has only one class (mirror).
    RPN_ANCHOR_SCALES = (256, 128, 64, 32, 16)  # anchor side in pixels
    DETECTION_MIN_CONFIDENCE = 0.7


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

# Written by TaylorMei
def get_mask(imgname):
    """Get mask by specified single image name"""
    filestr = imgname.split(".")[0]
    mask_folder = os.path.join(ROOT_DIR, "data", "test", "mask")
    mask_path = mask_folder + "/" + filestr + "_json/label8.png"
    if not os.path.exists(mask_path):
        print("{} has no label8.png")
    mask = Image.open(mask_path)
    width, height = mask.size
    num_obj = np.max(mask)

    gt_mask = np.zeros([height, width, num_obj], dtype=np.uint8)
    for index in range(num_obj):
        """j is row and i is colum"""
        for i in range(width):
            for j in range(height):
                at_pixel = mask.getpixel((i,j))
                if at_pixel == index + 1:
                    gt_mask[j, i, index] = 1
    return gt_mask

def get_class_ids(imgname):
    """Get class_id by specified single image name"""
    filestr = imgname.split(".")[0]
    labels = []
    class_folder = os.path.join(ROOT_DIR, "data", "test", "mask")
    class_path = class_folder + "/" + filestr + "_json/info.yaml"
    with open(class_path) as f:
        temp = yaml.load(f.read())
        labels = temp['label_names']
        del labels[0]
    labels_form = []
    for i in range(len(labels)):
        if labels[i].find("mirror")!=-1:
            labels_form.append("mirror")
    num = len(labels_form)
    class_ids = np.ones([num], dtype=int)
    return class_ids


# ## Run Object Detection
imglist = os.listdir(IMAGE_DIR)
print("Total {} test images".format(len(imglist)))

i = 0
mAPs = []
mAPs_range = []

for imgname in imglist:

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
    gt_mask = get_mask(imgname)
    gt_box = utils.extract_bboxes(gt_mask)
    gt_class_id = get_class_ids(imgname)
    pred_box = r['rois']
    pred_class_id = r['class_ids']
    pred_score = r['scores']
    pred_mask = r['masks']

    N = pred_box.shape[0]
    if N:
        # mAP for a certain IoU threshold
        mAP, precisions, recalls, overlaps = utils.compute_ap(gt_box, gt_class_id, gt_mask,
                                                        pred_box, pred_class_id, pred_score, pred_mask,
                                                        iou_threshold = InferenceConfig.iou_threshold)
        # mAP over range of IoU thresholds
        # Default range is 0.5---0.95, interval is 0.05
        print("Precisions is {} \n Recalls is {} \n Overlaps is {}".format(precisions, recalls, overlaps))
        AP = utils.compute_ap_range(gt_box, gt_class_id, gt_mask,
                                    pred_box, pred_class_id, pred_score, pred_mask,
                                    iou_thresholds=None, verbose=1)
    else:
        mAP = 0
        AP = 0

    mAPs.append(mAP)
    print("mAP is : {}".format(mAP))
    mAPs_range.append(AP)
    print("mAP over range of IoU thresholds is : {}".format(AP))

    i = i + 1

###########################################################################
################  Quantitative Evaluation for All Image ################
###########################################################################
mean_mAP = sum(mAPs)/len(mAPs)
mean_mAP_range = sum(mAPs_range)/len(mAPs_range)
print("For test data set, \n mean_mAP is : {} \n mean_mAP_range is : {}"
      .format(mean_mAP, mean_mAP_range))





