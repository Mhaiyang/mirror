"""
  @Time    : 2018-7-28 04:56
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com
  
  @Project : mirror
  @File    : augmentation.py
  @Function: data augmentation

  @Last modified: 2018-8-28
  
"""
import os
import sys
import random
import yaml
import numpy as np
from PIL import Image
sys.path.append("../mrcnn")
import mrcnn.utils as utils
from skimage import io, transform


DATA_DIR = os.path.abspath(os.path.join(os.getcwd(), "../data", "val"))
IMAGE_DIR = os.path.join(DATA_DIR, "image")
if not os.path.exists(IMAGE_DIR):
    os.mkdir(IMAGE_DIR)
MASK_DIR = os.path.join(DATA_DIR, "mask")
if not os.path.exists(MASK_DIR):
    os.mkdir(MASK_DIR)
OUTPUT_DIR = os.path.join(DATA_DIR, "../../augmentation", "val")
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
    os.mkdir(os.path.join(OUTPUT_DIR, "image"))
    os.mkdir(os.path.join(OUTPUT_DIR, "mask"))

iou_initial = 0.7

imglist = os.listdir(IMAGE_DIR)
print("Total {} images will be augmented!".format(len(imglist)))

for imgname in imglist:
    print(imgname)
    filestr = imgname.split(".")[0]
    image_path = IMAGE_DIR + "/" + imgname
    mask_path = MASK_DIR + "/" + filestr + "_json/label8.png"
    if not os.path.exists(mask_path):
        print("{} has no label8.png")
    mask = Image.open(mask_path)
    num_obj = np.max(mask)

    # Get image
    image = io.imread(image_path)

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

    # Resize to a fixed size (620x512 or 512x620)
    if width > height:
        fixed_size = (640, 512)
    else:
        fixed_size = (512, 640)
    fixed_image = transform.resize(image, (fixed_size[1], fixed_size[0]), order=3)
    io.imsave(OUTPUT_DIR + "/image/" + filestr + "_" + str(width) + "x" + str(height) + ".jpg", fixed_image)
    fixed_mask = mask.resize(fixed_size, Image.BICUBIC)
    fixed_mask.save(OUTPUT_DIR + "/mask/" + filestr + "_" + str(width) + "x" + str(height) + "_json/label8.png")
    with open(OUTPUT_DIR + "/mask/" + filestr + "_" + str(width) + "x" + str(height) + "_json/info.yaml", "w") as f:
        yaml.dump(labels, f)

    # crop operation. (width, height)
    scales_1 = [(960, 768), (960, 960), (768, 960)]
    scales_2 = [(640, 512), (640, 640), (512, 640)]
    ratio_1 = random.sample(scales_1, 1)[0]
    ratio_2 = random.sample(scales_2, 1)[0]
    ratios = []
    ratios.append(ratio_1)
    ratios.append(ratio_2)

    for x, ratio in enumerate(ratios):
        print(ratio)
        if ratio[0] > ratio[1]:
            final_size = (640, 512)
        elif ratio[0] == ratio[1]:
            final_size = (640, 640)
        else:
            final_size = (512, 640)
        iou_threshold = iou_initial
        iou = np.zeros([num_obj])
        iteration = 0
        while not len(np.where(iou >= iou_threshold)[0]):
            iteration += 1
            if iteration > 100:
                iou_threshold -= 0.001
            # if iou_threshold < IoU_MIN:
            #     print("NO SUITABLE CROPPING")
            #     break
            y1 = random.randint(0, height - ratio[1] - 1)
            x1 = random.randint(0, width - ratio[0] - 1)
            y2 = y1 + ratio[1]
            x2 = x1 + ratio[0]
            box = np.array([y1, x1, y2, x2])
            box_area = float(ratio[0]*ratio[1])
            boxes = utils.extract_bboxes(gt_mask)
            boxes_area = np.zeros([boxes.shape[0]], dtype=float)
            for i in range(len(boxes_area)):
                boxes_area[i] = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
            iou = utils.compute_iou(box, boxes, box_area, boxes_area)
            if len(np.where(iou >= iou_threshold)[0]):
                # have found suitable box
                print("iou : {}".format(iou))
                # Handle image
                if not os.path.exists(OUTPUT_DIR + "/mask/" + filestr + "_" + str(ratio[0]) + "x" + str(ratio[1]) + "_json"):
                    os.mkdir(OUTPUT_DIR + "/mask/" + filestr + "_" + str(ratio[0]) + "x" + str(ratio[1]) + "_json")
                new_image = image[y1:y2, x1:x2, :]
                if x == 0:
                    new_image = transform.resize(new_image, (final_size[1], final_size[0]), order=3)
                io.imsave(OUTPUT_DIR + "/image/" + filestr + "_" + str(ratio[0]) + "x" + str(ratio[1]) + ".jpg", new_image)
                # Handle mask
                new_mask = mask.crop((x1, y1, x2, y2))
                if x == 0:
                    new_mask = new_mask.resize(final_size, Image.BICUBIC)
                new_mask.save(OUTPUT_DIR + "/mask/" + filestr + "_" + str(ratio[0]) + "x" + str(ratio[1]) + "_json/label8.png")
                max_value = 0
                for column in range(ratio[0]):
                    for row in range(ratio[1]):
                        pixel = new_mask.getpixel((column, row))
                        if max_value <= pixel:
                            max_value = pixel
                new_label = labels[:max_value + 1]
                temp["label_names"] = new_label
                with open(OUTPUT_DIR + "/mask/" + filestr + "_" + str(ratio[0]) + "x" + str(ratio[1]) + "_json/info.yaml", "w") as f:
                    yaml.dump(temp, f)
                print("##########  Okay!  ###########")





