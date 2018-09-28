"""
  @Time    : 2018-05-07
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com

"""
import os
import skimage.io
import mhy.utils as utils
import mhy.visualize as visualize
import evaluate
from mirror import MirrorConfig
# Important, need change when test different models.
import mhy.context as modellib

# Directories of the project
ROOT_DIR = os.getcwd()
print(ROOT_DIR)
MODEL_DIR = os.path.join(ROOT_DIR, "log", "context")
MIRROR_MODEL_PATH = os.path.join(MODEL_DIR, "mirror_context_heads.h5")
IMAGE_DIR = os.path.join(ROOT_DIR, "augmentation", "test", "image")
MASK_DIR = os.path.join(ROOT_DIR, "augmentation", "test", "mask")
OUTPUT_PATH = os.path.join(ROOT_DIR, 'augmentation', 'test', "output_context_bbox")
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
    bbox_iou_threshold1 = 0.5
    bbox_iou_threshold2 = 0.75
    bbox_iou_threshold3 = 0.85


config = InferenceConfig()
config.display()

# ## Create Model and Load Trained Weights
model = modellib.Context(mode="inference", config=config, model_dir=MODEL_DIR)
# ## Load weights
model.load_weights(MIRROR_MODEL_PATH, by_name=True)

# MIRROR Class Names
class_names = ['BG', 'Mirror']

# ## Run Object Detection
imglist = os.listdir(IMAGE_DIR)
print("Total {} test images".format(len(imglist)))

i = 0
mAPs_box_1 = []
mAPs_box_2 = []
mAPs_box_3 = []

for imgname in imglist:

    print("###############  {}   ###############".format(i))
    image = skimage.io.imread(os.path.join(IMAGE_DIR, imgname))
    # Run detection
    results = model.detect(imgname, [image], verbose=1)
    # Visualize results
    # As detect function returns a list of dict, one dict per image,
    # and each call detect function only feed one image, r =results[0]
    r = results[0]
    visualize.display_instances_and_save_image(imgname, image, r['rois'], r['class_ids'],
                                class_names, True, OUTPUT_PATH, r['scores'])

    ###########################################################################
    ################  Quantitative Evaluation for Single Image ################
    ###########################################################################
    num_obj, gt_mask = evaluate.get_mask(imgname, MASK_DIR)
    gt_box = utils.extract_bboxes(gt_mask)
    gt_class_id = evaluate.get_class_ids(imgname, MASK_DIR)
    pred_box = r['rois']
    pred_class_id = r['class_ids']
    pred_score = r['scores']

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

    else:
        mAP_box1, mAP_box2, mAP_box3 = 0, 0, 0

    mAPs_box_1.append(mAP_box1)
    mAPs_box_2.append(mAP_box2)
    mAPs_box_3.append(mAP_box3)
    print("{:35} {} {} {}".format("mAP_box", mAP_box1, mAP_box2, mAP_box3))

    i = i + 1

################################################################################################
############  Quantitative Evaluation for All Image   ##########################################
mean_mAP_box_1 = sum(mAPs_box_1)/len(mAPs_box_1)
mean_mAP_box_2 = sum(mAPs_box_2)/len(mAPs_box_2)
mean_mAP_box_3 = sum(mAPs_box_3)/len(mAPs_box_3)
print("For Test Data Set, \n{:20} {} \n{:20} {} \n{:20} {}" .format("mean_mAP_box_50", mean_mAP_box_1,
                                                                    "mean_mAP_box_75", mean_mAP_box_2,
                                                                    "mean_mAP_box_85", mean_mAP_box_3))
