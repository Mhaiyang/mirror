"""
  @Time    : 2018-9-8 03:40
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com
  
  @Project : mirror
  @File    : demo.py.py
  @Function: run detection and segmentation.
  
"""

import os
import numpy as np
from skimage import io, transform
import mrcnn.utils as utils
import mrcnn.visualize as visualize
from mirror import MirrorConfig
import mrcnn.fusion_context_guided_decoder as modellib

version = "fusion_context_guided_decoder"
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs_" + version)
MODEL_PATH = os.path.join(MODEL_DIR, "mirror_all.h5")
INPUT_DIR = os.path.join(ROOT_DIR, "demo_input")
OUTPUT_DIR = os.path.join(ROOT_DIR, "demo_output")
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)


# Configuration
class DemoConfig(MirrorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    DETECTION_MIN_CONFIDENCE = 0.7


config = DemoConfig()
config.display()

model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)
model.load_weights(MODEL_PATH, by_name=True)
if version == "fusion_context_guided_decoder":
    print("load additional weights.")
    mapping = dict()
    mapping["fusion_class_conv1_second"] = "fusion_class_conv1"
    mapping["fusion_class_conv2_second"] = "fusion_class_conv2"
    mapping["fusion_class_conv3_second"] = "fusion_class_conv3"
    mapping["fusion_class_conv4_second"] = "fusion_class_conv4"
    for layer in model.keras_model.layers:
        if layer.name in mapping:
            weight_name = mapping[layer.name]
            layer.set_weights(model.keras_model.get_layer(weight_name).get_weights())

class_names = ["BG", "Mirror"]

imglist = os.listdir(INPUT_DIR)

for imgname in imglist:
    print("Image {}".format(imgname))
    raw_image = io.imread(os.path.join(INPUT_DIR, imgname))
    h, w, c = raw_image.shape
    if h > w:
        image = transform.resize(raw_image, (640, 512, 3), order=3)
    else:
        image = transform.resize(raw_image, (512, 640, 3), order=3)
    io.imsave(os.path.join(OUTPUT_DIR, imgname), image)
    img = io.imread(os.path.join(OUTPUT_DIR, imgname))
    results = model.detect(imgname, [img], verbose=1)
    r = results[0]
    visualize.display_instances_and_save_image(imgname, OUTPUT_DIR, img, r['rois'], r['masks'], r['class_ids'],
                                               class_names, r['scores'])




