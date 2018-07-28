"""
  @Time    : 2018-7-28 04:56
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com
  
  @Project : mirror
  @File    : augmentation.py
  @Function: data augmentation
  
"""
import os
import sys
import random
import yaml
import numpy as np
from PIL import Image
sys.path.append("../mrcnn")
import mrcnn.utils as utils
import skimage.io

DATA_DIR = os.path.abspath(os.path.join(os.getcwd(), "../data", "train"))
IMAGE_DIR = os.path.join(DATA_DIR, "image")
MASK_DIR = os.path.join(DATA_DIR, "mask")
OUTPUT_DIR = os.path.join(DATA_DIR, "augmentation", "train")
iou_threshold = 0.7

def crop(y1, x1, y2, x2, image, mask):
    cropped_image = image[y1:y2, x1:x2, :]
    cropped_mask = mask[y1:y2, x1:x2, :]

    return cropped_image, cropped_mask

imglist = os.listdir(IMAGE_DIR)
print("Total {} images will be augmented!")

for imgname in imglist:

    filestr = imgname.split(".")[0]
    image_path = IMAGE_DIR + "/" + imgname
    mask_path = MASK_DIR + "/" + filestr + "_json/label8.png"
    if not os.path.exists(mask_path):
        print("{} has no label8.png")
    mask = Image.open(mask_path)
    num_obj = np.max(mask)

    # Get image
    image = skimage.io.imread(image_path)

    # Get gt_mask (binary map)
    width, height = mask.size
    gt_mask = np.zeros([height, width, num_obj], dtype=np.uint8)
    for index in range(num_obj):
        for i in range(width):
            for j in range(height):
                at_pixel = mask.getpixel((i, j))
                if at_pixel == index + 1:
                    gt_mask[j, i, index] = 1

    # Get label info (yaml file)
    with open(MASK_DIR + "/" + filestr + "_json/info.yaml") as f:
        temp = yaml.load(f.read())
        labels = temp['label_names']

    # crop operation
    ratios = [(160, 128), (320, 256), (480, 384), (640, 512)]
    shapes = ["wide", "square", "high"]
    for i, ratio in enumerate(ratios):
        shape = random.sample(shapes, 1)[0]
        iou = np.zeros([num_obj])
        if shape == "wide":
            while iou < iou_threshold:
                y1 = random.randint(1, height - ratio[1] + 1)
                x1 = random.randint(1, width - ratio[0] + 1)
                y2 = y1 + 896
                x2 = x1 + 1120
                box = np.array([y1, x1, y2, x2])
                box_area = 1120.0 * 896.0
                boxes = utils.extract_bboxes(gt_mask)
                boxes_area = np.zeros([boxes.shape[0]], dtype=float)
                for i in range(len(boxes_area)):
                    boxes_area[i] = (boxes[2] - boxes[0]) * (boxes[3] - boxes[1])
                iou = utils.compute_iou(box, boxes, box_area, boxes_area)
                if len(np.where(iou >= iou_threshold)[0]):
                    # Handle image, label, and mask
                    new_image = image[y1:y2, x1:x2, :]
                    skimage.io.imsave(new_image, OUTPUT_DIR + "/image/" + filestr + "_" + shape + "_" + str(i) + ".jpg")
                    new_mask = mask.crop((x1, y1, x2, y2))
                    new_mask.save(OUTPUT_DIR + "/mask/" + filestr + "_" + shape + "_" + str(i) + "_json/label8.png")
                    new_label = labels[:len(np.where(iou > iou_threshold)[0])+1]









