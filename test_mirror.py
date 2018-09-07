"""
  @Time    : 2018-05-07
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com

"""
import os
import skimage.io
import numpy as np
import mrcnn.utils as utils
import mrcnn.visualize as visualize
import evaluate
from mirror import MirrorConfig
# Important, need change when test different models.
import mrcnn.fusion as modellib

# Directories of the project
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs_fusion/mirror20180903T2103")
MIRROR_MODEL_PATH = os.path.join(MODEL_DIR, "mirror_0030.h5")
IMAGE_DIR = os.path.join(ROOT_DIR, "augmentation", "test", "image")
MASK_DIR = os.path.join(ROOT_DIR, "augmentation", "test", "mask")
OUTPUT_PATH = os.path.join(ROOT_DIR, 'augmentation', 'test', "output_fusion")
if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)


# Configurations
class InferenceConfig(MirrorConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    # Up to now, batch size must be one.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    # These two must be same when test. As we use shared connecting the detection and segmentation.
    DETECTION_MAX_INSTANCES = 100

    DETECTION_MIN_CONFIDENCE = 0.7
    # Important. If Iou greater than this threshold, this prediction will be considered as true.
    bbox_iou_threshold = 0.5
    mask_iou_threshold = 0.5


config = InferenceConfig()
config.display()

# ## Create Model and Load Trained Weights
model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)
# ## Load weights
model.load_weights(MIRROR_MODEL_PATH, by_name=True)
# ## For fusion_context_guided_decoder
# mapping = dict()
# mapping["fusion_class_conv1_second"] = "fusion_class_conv1"
# mapping["fusion_class_conv2_second"] = "fusion_class_conv2"
# mapping["fusion_class_conv3_second"] = "fusion_class_conv3"
# mapping["fusion_class_conv4_second"] = "fusion_class_conv4"
# for layer in model.keras_model.layers:
#     if layer.name in mapping:
#         weight_name = mapping[layer.name]
#         layer.set_weights(model.keras_model.get_layer(weight_name).get_weights())
# for layer in model.keras_model.layers:
#     print(layer.name)
#     if layer.name == "fusion_class_conv2" or layer.name == "fusion_class_conv2_second":
#         print(layer.get_weights())

# MIRROR Class Names
class_names = ['BG', 'Mirror']

# ## Run Object Detection
imglist = os.listdir(IMAGE_DIR)
print("Total {} test images".format(len(imglist)))

i = 0
mAPs_box = []
mAPs_mask = []
mAPs_range_mask = []

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
    num_obj, gt_mask = evaluate.get_mask(imgname, MASK_DIR)
    gt_box = utils.extract_bboxes(gt_mask)
    gt_class_id = evaluate.get_class_ids(imgname, MASK_DIR)
    pred_box = r['rois']
    pred_class_id = r['class_ids']
    pred_score = r['scores']
    pred_mask = r['masks']

    N = pred_box.shape[0]
    if N:
        # mAP of box for a certaion IoU threshold
        mAP_box, precisions_box, recalls_box, overlaps_box = utils.compute_ap_box(pred_box, gt_box,
                                                                                  InferenceConfig.bbox_iou_threshold)
        # mAP of mask for a certain IoU threshold
        mAP_mask, precisions_mask, recalls_mask, overlaps_mask = utils.compute_ap_mask(gt_box, gt_class_id, gt_mask,
                                                        pred_box, pred_class_id, pred_score, pred_mask,
                                                        iou_threshold = InferenceConfig.mask_iou_threshold)

        # mAP of mask over range of IoU thresholds. Default range is 0.5---0.95, interval is 0.05
        AP_mask = utils.compute_ap_range(gt_box, gt_class_id, gt_mask,
                                    pred_box, pred_class_id, pred_score, pred_mask,
                                    iou_thresholds=None, verbose=1)

        print("{:15} {} \n{:15} {} \n{:15} {}".format("Box  Precisions", precisions_box,
                                                      "Box  Recalls", recalls_box, "Box  Overlaps", overlaps_box))
        print("{:15} {} \n{:15} {} \n{:15} {}".format("Mask Precisions", precisions_mask,
                                                      "Mask Recalls", recalls_mask, "Mask Overlaps", overlaps_mask))
    else:
        mAP_box = 0
        mAP_mask = 0
        AP_mask = 0

    mAPs_box.append(mAP_box)
    mAPs_mask.append(mAP_mask)
    mAPs_range_mask.append(AP_mask)
    print("{:35} {} \n{:35} {} \n{:35} {}".format("mAP_box", mAP_box, "mAP_mask", mAP_mask, "mAP_range_mask", AP_mask))

    i = i + 1

################################################################################################
############  Quantitative Evaluation for All Image   ##########################################
mean_mAP_box = sum(mAPs_box)/len(mAPs_box)
mean_mAP_mask = sum(mAPs_mask)/len(mAPs_mask)
mean_mAP_range_mask = sum(mAPs_range_mask)/len(mAPs_range_mask)
print("For Test Data Set, \n{:20} {} \n{:20} {} \n{:20} {}"
      .format("mean_mAP_box", mean_mAP_box, "mean_mAP_mask", mean_mAP_mask, "mean_mAP_range_mask", mean_mAP_range_mask))





