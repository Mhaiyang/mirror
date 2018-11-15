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
import time
from mirror import MirrorConfig
# Important, need change when test different models.
import mrcnn.model as modellib

# Directories of the project
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "log", "mask_rcnn")
MIRROR_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_mirror_0030.h5")
IMAGE_DIR = "/home/taylor/Mirror-Segmentation/data_640/test3/image"
MASK_DIR = "/home/taylor/Mirror-Segmentation/data_640/test3/mask"
OUTPUT_PATH = "/home/taylor/Mirror-Segmentation/data_640/test3/mask_rcnn"
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


config = InferenceConfig()
config.display()

# ## Create Model and Load Trained Weights
model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)
# ## Load weights
model.load_weights(MIRROR_MODEL_PATH, by_name=True)

# ## Run Object Detection
imglist = os.listdir(IMAGE_DIR)
print("Total {} test images".format(len(imglist)))

IOU = []
ACC_all = []
ACC_mirror = []
BER = []
start = time.time()
for i, imgname in enumerate(imglist):

    print("###############  {}   ###############".format(i+1))
    image = skimage.io.imread(os.path.join(IMAGE_DIR, imgname))
    # Run detection
    results = model.detect(imgname, [image], verbose=1)
    r = results[0]
    predict_mask = r["masks"]
    N = np.shape(predict_mask)[-1]

    height = np.shape(image)[0]
    width = np.shape(image)[1]
    mask = np.zeros([height, width], dtype=np.uint8)
    print(N)
    print(mask.shape)

    if N != 0:
        for channel in range(N):
            for y in range(height):
                for x in range(width):
                    if predict_mask[y, x, channel]:
                        mask[y, x] = 1

    visualize.save_mask_and_masked_image(imgname, image, mask, OUTPUT_PATH)
    gt_mask = evaluate.get_mask(imgname, MASK_DIR)

    print(gt_mask.shape)

    iou = evaluate.iou(mask, gt_mask)
    acc_all = evaluate.accuracy_all(mask, gt_mask)
    acc_mirror = evaluate.accuracy_mirror(mask, gt_mask)
    ber = evaluate.ber(mask, gt_mask)

    print("iou : {}".format(iou))
    print("acc_all : {}".format(acc_all))
    print("acc_mirror : {}".format(acc_mirror))
    print("ber : {}".format(ber))
    IOU.append(iou)
    ACC_all.append(acc_all)
    ACC_mirror.append(acc_mirror)
    BER.append(ber)

end = time.time()

mean_IOU = 100 * sum(IOU) / len(IOU)
mean_ACC_all = 100 * sum(ACC_all) / len(ACC_all)
mean_ACC_mirror = 100 * sum(ACC_mirror) / len(ACC_mirror)
mean_BER = 100 * sum(BER) / len(BER)

print("Time is : {}".format(end - start))
print(len(IOU))
print(len(ACC_all))
print(len(ACC_mirror))
print(len(BER))

print("For Test Data Set, \n{:20} {:.2f} \n{:20} {:.2f} \n{:20} {:.2f} \n{:20} {:.2f}".
      format("mean_IOU", mean_IOU, "mean_ACC_all", mean_ACC_all, "mean_ACC_mirror", mean_ACC_mirror, "mean_BER", mean_BER))





