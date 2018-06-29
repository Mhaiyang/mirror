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

import mrcnn.utils
import mrcnn.model as modellib
import mrcnn.visualize as visualize
from mrcnn.config import Config

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
MIRROR_MODEL_PATH = os.path.join(MODEL_DIR, "mirror20180605T2258", "mask_rcnn_mirror_0100.h5")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "data", "test", "image")
OUTPUT_PATH = os.path.join(ROOT_DIR, 'data', 'test', "output")


## Configurations
class MirrorConfig(Config):
    """Derives from the base Config class and overrides values specific to the Mirror dataset"""
    NAME = "Mirror"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1 # Mirror has only one class (mirror).
    DETECTION_MIN_CONFIDENCE = 0.9


class InferenceConfig(MirrorConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


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
    image = skimage.io.imread(os.path.join(IMAGE_DIR, imgname))
    # Run detection
    results = model.detect(imgname, [image], verbose=1)
    # Visualize results
    # As detect function returns a list of dict, one dict per image,
    # and each call detect function only feed one image, r =results[0]
    r = results[0]
    visualize.display_instances_and_save_image(imgname, OUTPUT_PATH, image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])

