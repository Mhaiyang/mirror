"""
  @Time    : 2018-05-07
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com

"""
import os
import skimage.io
import mrcnn.utils as utils
import mrcnn.model as modellib
import mrcnn.visualize as visualize
import evaluate
from mrcnn.config import Config

# Directories of the project
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs/3")
MIRROR_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_mirror_all.h5")
IMAGE_DIR = os.path.join(ROOT_DIR, "augmentation", "test", "image")
MASK_DIR = os.path.join(ROOT_DIR, "augmentation", "test", "mask")
OUTPUT_PATH = os.path.join(ROOT_DIR, 'augmentation', 'test', "output_all")
if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

## Configurations
class MirrorConfig(Config):
    NAME = "Mirror"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1     # Mirror has only one class (mirror).
    RPN_ANCHOR_SCALES = (256, 128, 64, 32, 16)  # anchor side in pixels
    DETECTION_MIN_CONFIDENCE = 0.7

class InferenceConfig(MirrorConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Important. If Iou greater than this threshold, this prediction will be considered as true.
    bbox_iou_threshold = 0.5
    mask_iou_threshold = 0.5

config = InferenceConfig()
config.display()

# ## Create Model and Load Trained Weights
model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)
model.load_weights(MIRROR_MODEL_PATH, by_name=True)

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





