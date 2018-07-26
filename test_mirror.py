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
MODEL_DIR = os.path.join(ROOT_DIR, "logs/2")
MIRROR_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_mirror_all.h5")
IMAGE_DIR = os.path.join(ROOT_DIR, "data", "test", "image")
OUTPUT_PATH = os.path.join(ROOT_DIR, 'data', 'test', "output_all")
if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

## Configurations
class MirrorConfig(Config):
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
model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)
model.load_weights(MIRROR_MODEL_PATH, by_name=True)

# MIRROR Class Names
class_names = ['BG', 'Mirror']

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
    gt_mask = evaluate.get_mask(imgname)
    gt_box = utils.extract_bboxes(gt_mask)
    gt_class_id = evaluate.get_class_ids(imgname)
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
        # mAP over range of IoU thresholds. Default range is 0.5---0.95, interval is 0.05
        print("{:15} {} \n{:15} {} \n{:15} {}".format("Precisions", precisions, "Recalls", recalls, "Overlaps", overlaps))
        AP = utils.compute_ap_range(gt_box, gt_class_id, gt_mask,
                                    pred_box, pred_class_id, pred_score, pred_mask,
                                    iou_thresholds=None, verbose=1)
    else:
        mAP = 0
        AP = 0

    mAPs.append(mAP)
    mAPs_range.append(AP)
    print("{:35} {} \n{:35} {}".format("mAP", mAP, "mAP over range of IoU thresholds", AP))

    i = i + 1

################################################################################################
############  Quantitative Evaluation for All Image   ##########################################
mean_mAP = sum(mAPs)/len(mAPs)
mean_mAP_range = sum(mAPs_range)/len(mAPs_range)
print("For Test Data Set, \n{:20} {} \n{:20} {}"
      .format("mean_mAP", mean_mAP, "mean mAP range", mean_mAP_range))





