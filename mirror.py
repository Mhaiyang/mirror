import os
import numpy as np
from PIL import Image
from mrcnn.config import Config
import mrcnn.utils as utils
import yaml


### Configurations
class MirrorConfig(Config):
    """Configuration for training on the mirror dataset.
    Derives from the base Config class and overrides values specific
    to the mirror dataset.
    """
    # Give the configuration a recognizable name
    NAME = "Mirror"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 mirror

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1280

    BACKBONE_STRIDES = [4, 8, 16, 32, 64]   # for compute pyramid feature size
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)  # anchor side in pixels
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)

    # Shape of output mask
    # To change this you also need to change the neural network mask branch
    MASK_SHAPE = [28, 28]

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 100

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 3465

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 247

    # skip detection with <x% confidence
    DETECTION_MIN_CONFIDENCE = 0.7

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14

    # Learning rate
    LEARNING_RATE = 0.001



### Dataset
class MirrorDataset(utils.Dataset):

    def get_obj_index(self, image):
        """Get the number of instance in the image
        """
        n = np.max(image)
        return n

    def from_yaml_get_class(self,image_id):
        """Translate the yaml file to get label """
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read())
            labels = temp['label_names']
            del labels[0]
        return labels

    def draw_mask(self, num_obj, mask, image, image_id):
        info = self.image_info[image_id]
        for index in range(num_obj):
            """j is row and i is column"""
            for i in range(info['width']):
                for j in range(info['height']):
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1   # [height width channel] i.e. [h, w, c]
        return mask

    def load_mirror(self, count, img_folder, mask_folder, imglist):
        self.add_class("Mirror", 1, "mirror")
        # self.add_class("Mirror", 2, "reflection")
        for i in range(count):
            filestr = imglist[i].split(".")[0]  # 10.jpg for example
            mask_path = mask_folder + "/" + filestr + "_json/label8.png"
            yaml_path = mask_folder + "/" + filestr + "_json/info.yaml"
            if not os.path.exists(mask_path):
                print("{} is incorrect".format(filestr))
                continue
            img = Image.open(mask_path)
            width, height = img.size
            self.add_image("Mirror", image_id=i, path=img_folder + "/" + imglist[i],
                           width=width, height=height, mask_path=mask_path, yaml_path=yaml_path)

    def load_mask(self, image_id):
        global iter_num
        info = self.image_info[image_id]
        count = 1
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img, image_id)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        labels=[]
        labels=self.from_yaml_get_class(image_id)
        labels_form=[]
        for i in range(len(labels)):
            if labels[i].find("mirror")!=-1:
                #print "box"
                labels_form.append("mirror")
        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)



