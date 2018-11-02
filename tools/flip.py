"""
  @Time    : 2018-10-16 06:46
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com
  
  @Project : mirror
  @File    : flip.py
  @Function: 
  
"""
import os
import numpy as np
import cv2
from PIL import Image

DATA_DIR = "/home/taylor/mirror/augmentation/train"
IMAGE_DIR = os.path.join(DATA_DIR, "image")
MASK_DIR = os.path.join(DATA_DIR, "mask")

imglist = os.listdir(IMAGE_DIR)
print("Total {} masks will be computed!".format(len(imglist)))

n = 0
for imgname in imglist:
    n += 1
    filestr = str(imgname.split(".")[0])
    img_path = IMAGE_DIR + "/" + imgname
    mask_path = MASK_DIR + "/" + filestr + "_json/label8.png"
    output_path = DATA_DIR + "/mask/" + filestr + "_json/"

    image = Image.open(img_path)
    gray = image.convert("L")
    gray.save(output_path+"image.png")
    gray_np = np.array(gray)

    mask = Image.open(mask_path)
    num_obj = np.max(mask)
    width, height = mask.size
    gt_mask = np.zeros([height, width, num_obj], dtype=np.uint8)
    for index in range(num_obj):
        for i in range(width):
            for j in range(height):
                at_pixel = mask.getpixel((i, j))
                if at_pixel == index + 1:
                    gt_mask[j, i, index] = 1
    np.save(output_path, gt_mask*255)

    only_mirror = np.zeros([height, width, num_obj], dtype=np.uint8)
    for c in range(num_obj):
        only_mirror[:, :, c] = gray_np[:, :] * gt_mask[:, :, c]

    # flip = only_mirror.transpose(Image.FLIP_LEFT_RIGHT)

    flip = np.zeros([height, width, num_obj], dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            flip[y, width-1-x, :] = only_mirror[y, x, :]

    # Save
    for z in range(num_obj):
        cv2.imwrite(output_path + "original" + str(z) + ".png", only_mirror[:, :, z])
        cv2.imwrite(output_path + "flip" + str(z) + ".png", flip[:, :, z])

    np.save(output_path + "flip.npy", flip)

    print("{}  {}".format(n, output_path + "flip.npy"))
