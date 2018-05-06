import numpy as np
from PIL import Image

image = Image.open("/home/taylor/mirror/data/val/json/0036_json/label8.png")
print(np.max(image))
mask = np.zeros([200, 300, 2], dtype=np.uint8)
for index in range(2):
    """j is row and i is column"""
    for i in range(300):
        for j in range(200):
            at_pixel = image.getpixel((i, j))
            if at_pixel == index + 1:
                mask[j, i, index] = 1  # [row column channel]

print(mask[:,:,0])