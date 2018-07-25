"""
  @Time    : 2018-7-26 05:42
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com
  
  @Project : mirror
  @File    : evaluate.py
  @Function: evaluate our model
  
"""
import numpy as np
import os
import yaml
from PIL import Image

ROOT_DIR = os.getcwd()

def get_mask(imgname):
    """Get mask by specified single image name"""
    filestr = imgname.split(".")[0]
    mask_folder = os.path.join(ROOT_DIR, "data", "test", "mask")
    mask_path = mask_folder + "/" + filestr + "_json/label8.png"
    if not os.path.exists(mask_path):
        print("{} has no label8.png")
    mask = Image.open(mask_path)
    width, height = mask.size
    num_obj = np.max(mask)

    gt_mask = np.zeros([height, width, num_obj], dtype=np.uint8)
    for index in range(num_obj):
        """j is row and i is colum"""
        for i in range(width):
            for j in range(height):
                at_pixel = mask.getpixel((i,j))
                if at_pixel == index + 1:
                    gt_mask[j, i, index] = 1
    return gt_mask

def get_class_ids(imgname):
    """Get class_id by specified single image name"""
    filestr = imgname.split(".")[0]
    labels = []
    class_folder = os.path.join(ROOT_DIR, "data", "test", "mask")
    class_path = class_folder + "/" + filestr + "_json/info.yaml"
    with open(class_path) as f:
        temp = yaml.load(f.read())
        labels = temp['label_names']
        del labels[0]
    labels_form = []
    for i in range(len(labels)):
        if labels[i].find("mirror")!=-1:
            labels_form.append("mirror")
    num = len(labels_form)
    class_ids = np.ones([num], dtype=int)
    return class_ids

####################### Another method to measure our model ##############################
# Compute f1_measure
def compute_f1_measure(pred_mask, gt_mask):
    overlap_area = 0
    mask_area = 0
    FP = 0
    FN = 0
    height, width = gt_mask.shape[0], gt_mask.shape[1]
    for i in range(height):
        for j in range(width):
            if gt_mask[i][j]:
                mask_area += 1
            for k in range(pred_mask.shape[2]):
                if pred_mask[i][j][k] and pred_mask[i][j][k] == gt_mask[i][j]:
                    overlap_area += 1
                if pred_mask[i][j][k] and pred_mask[i][j][k] != gt_mask[i][j]:
                    FP += 1
                if not pred_mask[i][j][k] and pred_mask[i][j][k] != gt_mask[i][j]:
                    FN += 1
    print("overlap_area", overlap_area)
    print("mask_area:", mask_area)
    TP = overlap_area
    P = TP/(TP+FP)
    R = TP/(TP+FN)
    f1_measure = 2*P*R/(P+R)
    return f1_measure

def compute_mAP(pred_mask, gt_mask):
    overlap_area = 0
    mask_area = 0
    FP = 0
    FN = 0
    height, width = gt_mask.shape[0], gt_mask.shape[1]
    for i in height:
        for j in width:
            if gt_mask[i][j]:
                mask_area += 1
            for k in range(pred_mask.shape[2]):
                if pred_mask[i][j][k] and pred_mask[i][j][k] == gt_mask[i][j]:
                    overlap_area += 1
                if pred_mask[i][j][k] and pred_mask[i][j][k] != gt_mask[i][j]:
                    FP += 1
                if not pred_mask[i][j][k] and pred_mask[i][j][k] != gt_mask[i][j]:
                    FN += 1
    print ("overlap_area", overlap_area)
    print ("mask_area:", mask_area)
    TP = overlap_area
    P = TP/(TP+FP)
    return P