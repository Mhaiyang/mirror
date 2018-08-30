"""
  @Time    : 2018-8-29 00:00
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com

  @Project : mirror
  @File    : inspect_mask.py
  @Function: inspect if training data contains null mask.

"""
import os
import numpy as np
from PIL import Image

mask_folder = '/home/iccd/mirror/augmentation/train/mask/'
lists = os.listdir(mask_folder)
# print(len(list))
for list in lists:
    mask_path = os.path.join(mask_folder, list, "label8.png")
    mask = Image.open(mask_path)
    n = np.max(mask)
    if n == 0:
        print(list)