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
import mrcnn.attention3 as modellib
from keras.models import Model
import matplotlib.image

version = "attention3"
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs_" + version)
MODEL_PATH = os.path.join(MODEL_DIR, "mirror_all_35.h5")
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
if version == "attention3":
    print("load additional weights.")
    mapping = dict()
    mapping["fusion_attention_pooling_second"] = "fusion_attention_pooling"
    mapping["fusion_attention_weights_second"] = "fusion_attention_weights"
    mapping["fusion_class_conv1_second"] = "fusion_class_conv1"
    mapping["fusion_class_conv2_second"] = "fusion_class_conv2"
    for layer in model.keras_model.layers:
        if layer.name in mapping:
            print(layer.name)
            weight_name = mapping[layer.name]
            layer.set_weights(model.keras_model.get_layer(weight_name).get_weights())

class_names = ["BG", "Mirror"]

imglist = os.listdir(INPUT_DIR)

# # full model.
# for imgname in imglist:
#     print("Image {}".format(imgname))
#     raw_image = io.imread(os.path.join(INPUT_DIR, imgname))
#     h, w, c = raw_image.shape
#     if h > w:
#         image = transform.resize(raw_image, (640, 512, 3), order=3)
#     else:
#         image = transform.resize(raw_image, (512, 640, 3), order=3)
#     io.imsave(os.path.join(OUTPUT_DIR, imgname), image)
#     img = io.imread(os.path.join(OUTPUT_DIR, imgname))
#     results = model.detect(imgname, [img], verbose=1)
#     r = results[0]
#     visualize.display_instances_and_save_image(imgname, OUTPUT_DIR, img, r['rois'], r['masks'], r['class_ids'],
#                                                class_names, r['scores'])
# Visualize feature maps.
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

    feature_model = Model(inputs=model.keras_model.input, outputs=model.keras_model.get_layer("decoder_mask_p2pooled").output)
    assert model.mode == "inference", "Create model in inference mode."
    assert len(
        [img]) == model.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

    # Mold inputs to format expected by the neural network
    # images is a list which has only one image.
    molded_images, image_metas, windows = model.mold_inputs([img])

    # Validate image sizes
    # All images in a batch MUST be of the same size
    image_shape = molded_images[0].shape
    for g in molded_images[1:]:
        assert g.shape == image_shape, \
            "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

    # Anchors
    anchors = model.get_anchors(image_shape)
    # Duplicate across the batch dimension because Keras requires it
    # Each batch has the same anchors.
    # TODO: can this be optimized to avoid duplicating the anchors?
    anchors = np.broadcast_to(anchors, (model.config.BATCH_SIZE,) + anchors.shape)

    # Run object detection
    p2_features = \
        feature_model.predict([molded_images, image_metas, anchors])
    p2_feature = p2_features[0, 0, ...]
    for i in range(p2_feature.shape[-1]):
        output_name = OUTPUT_DIR + '/' + str(i) + ".png"
        matplotlib.image.imsave(output_name, p2_feature[:,:,i])
        print(output_name)





