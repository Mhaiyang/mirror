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

def get_mask(imgname, MASK_DIR):
    """Get mask by specified single image name"""
    filestr = imgname.split(".")[0]
    mask_folder = MASK_DIR
    mask_path = mask_folder + "/" + filestr + "_json/label8.png"
    if not os.path.exists(mask_path):
        print("{} has no label8.png")
    mask = Image.open(mask_path)
    width, height = mask.size
    num_obj = np.max(mask)

    gt_mask = np.zeros([height, width, num_obj], dtype=np.uint8)
    for index in range(num_obj):
        """j is row and i is column"""
        for i in range(width):
            for j in range(height):
                at_pixel = mask.getpixel((i,j))
                if at_pixel == index + 1:
                    gt_mask[j, i, index] = 1
    return num_obj, gt_mask

def get_class_ids(imgname, MASK_DIR):
    """Get class_id by specified single image name"""
    filestr = imgname.split(".")[0]
    labels = []
    class_folder = MASK_DIR
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
