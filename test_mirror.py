"""
  @Time    : 2018-05-07
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com

"""
import os
import skimage.io
import numpy as np
import mhy.utils as utils
import mhy.visualize as visualize
import evaluate
from mirror import MirrorConfig
# Important, need change when test different models.
import mhy.c26dmde as modellib

# Directories of the project
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "log_123", "c26dmde")
MIRROR_MODEL_PATH = os.path.join(MODEL_DIR, "mirror_c26dmde_all.h5")
IMAGE_DIR = os.path.join(ROOT_DIR, "augmentation", "test", "image")
MASK_DIR = os.path.join(ROOT_DIR, "augmentation", "test", "mask")
OUTPUT_PATH = os.path.join(ROOT_DIR, 'augmentation', 'test', "output_c26dmde")
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
    DETECTION_MAX_INSTANCES = 60

    DETECTION_MIN_CONFIDENCE = 0.7
    # Important. If Iou greater than this threshold, this prediction will be considered as true.
    bbox_iou_threshold1 = 0.5
    bbox_iou_threshold2 = 0.75
    bbox_iou_threshold3 = 0.85
    mask_iou_threshold1 = 0.5
    mask_iou_threshold2 = 0.75
    mask_iou_threshold3 = 0.85


config = InferenceConfig()
config.display()

# ## Create Model and Load Trained Weights
model = modellib.C26DMDE(mode="inference", config=config, model_dir=MODEL_DIR)
# ## Load weights
model.load_weights(MIRROR_MODEL_PATH, by_name=True)
# For fusion_context_guided_decoder.py  p1.py  path_full.py  post_relu.py
# mapping = dict()
# mapping["fusion_class_conv1_second"] = "fusion_class_conv1"
# mapping["fusion_class_conv2_second"] = "fusion_class_conv2"
# mapping["fusion_class_conv3_second"] = "fusion_class_conv3"
# mapping["fusion_class_conv4_second"] = "fusion_class_conv4"
## attention.py, attention2.py #####
# mapping["fusion_attention_short_second"] = "fusion_attention_short"
## attention3.py
# mapping["fusion_attention_pooling_second"] = "fusion_attention_pooling"
# mapping["fusion_attention_weights_second"] = "fusion_attention_weights"
# mapping["fusion_class_conv1_second"] = "fusion_class_conv1"
# mapping["fusion_class_conv2_second"] = "fusion_class_conv2"
# ## ad.py
# mapping["edge_class_conv1_second"] = "edge_class_conv1"
# mapping["edge_class_conv2_second"] = "edge_class_conv2"
# mapping["edge_class_conv3_second"] = "edge_class_conv3"
# mapping["edge_class_conv4_second"] = "edge_class_conv4"
#
# mapping["content_class_conv1_second"] = "content_class_conv1"
# mapping["content_class_conv2_second"] = "content_class_conv2"
# mapping["content_class_conv3_second"] = "content_class_conv3"
# mapping["content_class_conv4_second"] = "content_class_conv4"
#
# mapping["context_class_conv1_second"] = "context_class_conv1"
# mapping["context_class_conv2_second"] = "context_class_conv2"
# mapping["context_class_conv3_second"] = "context_class_conv3"
# mapping["context_class_conv4_second"] = "context_class_conv4"
#
# # mapping["aggregation_conv_second"] = "aggregation_conv"
#
# for layer in model.keras_model.layers:
#     if layer.name in mapping:
#         print(layer.name)
#         weight_name = mapping[layer.name]
#         layer.set_weights(model.keras_model.get_layer(weight_name).get_weights())
# print("Additional weights have been loaded.")
# For validate.
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
mAPs_box_1 = []
mAPs_box_2 = []
mAPs_box_3 = []
mAPs_mask_1 = []
mAPs_mask_2 = []
mAPs_mask_3 = []
mAPs_range_mask = []

for imgname in imglist:

    print("###############  {}   ###############".format(i))
    image = skimage.io.imread(os.path.join(IMAGE_DIR, imgname))
    # Run detection
    results = model.detect(imgname, [image], verbose=1)
    # Visualize results
    # As detect function returns a list of dict, one dict per image,
    # and each call detect function only feed one image, r =results[0]
    r = results[0]
    visualize.display_instances_and_save_image(imgname, image, r['rois'], r['masks'], r['class_ids'],
                                class_names, False, OUTPUT_PATH, r['scores'])
    skimage.io.imsave(os.path.join(OUTPUT_PATH, str(imgname[:-4] ) + "_c26dmde.jpg"), 255*r['edges'].astype(np.uint8))

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
        mAP_box1, precisions_box1, recalls_box1, overlaps_box1 = utils.compute_ap_box(pred_box, gt_box,
                                                                                  InferenceConfig.bbox_iou_threshold1)

        mAP_box2, precisions_box2, recalls_box2, overlaps_box2 = utils.compute_ap_box(pred_box, gt_box,
                                                                                  InferenceConfig.bbox_iou_threshold2)

        mAP_box3, precisions_box3, recalls_box3, overlaps_box3 = utils.compute_ap_box(pred_box, gt_box,
                                                                                  InferenceConfig.bbox_iou_threshold3)

        print("Threshold = 0.5 \n{:15} {} \n{:15} {} \n{:15} {}".format("Box  Precisions", precisions_box1,
                                                      "Box  Recalls", recalls_box1, "Box  Overlaps", overlaps_box1))
        print("Threshold = 0.75\n{:15} {} \n{:15} {} \n{:15} {}".format("Box  Precisions", precisions_box2,
                                                      "Box  Recalls", recalls_box2, "Box  Overlaps", overlaps_box2))
        print("Threshold = 0.85\n{:15} {} \n{:15} {} \n{:15} {}".format("Box  Precisions", precisions_box3,
                                                      "Box  Recalls", recalls_box3, "Box  Overlaps", overlaps_box3))

        # mAP of mask for a certain IoU threshold
        mAP_mask1, precisions_mask1, recalls_mask1, overlaps_mask1 = utils.compute_ap_mask(gt_box, gt_class_id, gt_mask,
                                                        pred_box, pred_class_id, pred_score, pred_mask,
                                                        iou_threshold = InferenceConfig.mask_iou_threshold1)
        mAP_mask2, precisions_mask2, recalls_mask2, overlaps_mask2 = utils.compute_ap_mask(gt_box, gt_class_id, gt_mask,
                                                        pred_box, pred_class_id, pred_score, pred_mask,
                                                        iou_threshold = InferenceConfig.mask_iou_threshold2)
        mAP_mask3, precisions_mask3, recalls_mask3, overlaps_mask3 = utils.compute_ap_mask(gt_box, gt_class_id, gt_mask,
                                                        pred_box, pred_class_id, pred_score, pred_mask,
                                                        iou_threshold = InferenceConfig.mask_iou_threshold3)

        print("Threshold = 0.5\n{:15} {} \n{:15} {} \n{:15} {}".format("Mask Precisions", precisions_mask1,
                                                      "Mask Recalls", recalls_mask1, "Mask Overlaps", overlaps_mask1))
        print("Threshold = 0.75\n{:15} {} \n{:15} {} \n{:15} {}".format("Mask Precisions", precisions_mask2,
                                                      "Mask Recalls", recalls_mask2, "Mask Overlaps", overlaps_mask2))
        print("Threshold = 0.85\n{:15} {} \n{:15} {} \n{:15} {}".format("Mask Precisions", precisions_mask3,
                                                      "Mask Recalls", recalls_mask3, "Mask Overlaps", overlaps_mask3))

        # mAP of mask over range of IoU thresholds. Default range is 0.5---0.95, interval is 0.05
        AP_mask = utils.compute_ap_range(gt_box, gt_class_id, gt_mask,
                                    pred_box, pred_class_id, pred_score, pred_mask,
                                    iou_thresholds=None, verbose=1)

    else:
        mAP_box1, mAP_box2, mAP_box3 = 0, 0, 0
        mAP_mask1, mAP_mask2, mAP_mask3 = 0, 0, 0
        AP_mask = 0

    mAPs_box_1.append(mAP_box1)
    mAPs_box_2.append(mAP_box2)
    mAPs_box_3.append(mAP_box3)
    mAPs_mask_1.append(mAP_mask1)
    mAPs_mask_2.append(mAP_mask2)
    mAPs_mask_3.append(mAP_mask3)
    mAPs_range_mask.append(AP_mask)
    print("{:35} {} {} {} \n{:35} {} {} {} \n{:35} {}".format("mAP_box", mAP_box1, mAP_box2, mAP_box3,
                                                              "mAP_mask", mAP_mask1, mAP_mask2, mAP_mask3,
                                                              "mAP_range_mask", AP_mask))

    i = i + 1

################################################################################################
############  Quantitative Evaluation for All Image   ##########################################
mean_mAP_box_1 = sum(mAPs_box_1)/len(mAPs_box_1)
mean_mAP_box_2 = sum(mAPs_box_2)/len(mAPs_box_2)
mean_mAP_box_3 = sum(mAPs_box_3)/len(mAPs_box_3)
mean_mAP_mask_1 = sum(mAPs_mask_1)/len(mAPs_mask_1)
mean_mAP_mask_2 = sum(mAPs_mask_2)/len(mAPs_mask_2)
mean_mAP_mask_3 = sum(mAPs_mask_3)/len(mAPs_mask_3)
mean_mAP_range_mask = sum(mAPs_range_mask)/len(mAPs_range_mask)
print("For Test Data Set, \n{:20} {} \n{:20} {} \n{:20} {} \n{:20} {} \n{:20} {} \n{:20} {} \n{:20} {}"
      .format("mean_mAP_box_50", mean_mAP_box_1, "mean_mAP_box_75", mean_mAP_box_2, "mean_mAP_box_85", mean_mAP_box_3,
              "mean_mAP_mask_50", mean_mAP_mask_1, "mean_mAP_mask_75", mean_mAP_mask_2, "mean_mAP_mask_85", mean_mAP_mask_3,
              "mean_mAP_range_mask", mean_mAP_range_mask))





