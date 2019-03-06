"""
  @Time    : 2018-8-29 00:00
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com

  @Project : mirror
  @File    : train_mirror.py
  @Function: train code.

"""
import os
import mirror
# Need modify:
import mhy.model as modellib

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "log", "mask_rcnn_msd9")

# Local path to trained weights file
# COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "resnet101_weights_tf.h5")

config = mirror.MirrorConfig()
config.display()

# Configuration
# dataset_root_path = os.path.abspath(os.path.join(ROOT_DIR, "./MSD9"))
dataset_root_path = "/home/iccd/data/msd9"
train_folder = dataset_root_path + "/train"
val_folder = dataset_root_path + "/test"
train_image_folder = train_folder + "/image"
train_mask_folder = train_folder + "/mask_json"
val_image_folder = val_folder + "/image"
val_mask_folder = val_folder + "/mask_json"
train_imglist = os.listdir(train_image_folder)
train_count = len(train_imglist)
val_imglist = os.listdir(val_image_folder)
val_count = len(val_imglist)
print("Train Image Count : {} \nValidation Image Count : {}".format(train_count, val_count))

# Training dataset
dataset_train = mirror.MirrorDataset()
dataset_train.load_mirror(train_count, train_image_folder,
                          train_mask_folder, train_imglist)     # add class and add image.
dataset_train.prepare("train")

# Validation dataset
dataset_val = mirror.MirrorDataset()
dataset_val.load_mirror(val_count, val_image_folder,
                        val_mask_folder, val_imglist)      # add class and add image

# Load and display random samples
# image_ids = np.random.choice(dataset_train.image_ids, 5)
# for image_id in image_ids:
#     image = dataset_train.load_image(image_id)
#     mask, class_ids = dataset_train.load_mask(image_id)
#     visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

### Create Model  ###
model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)

# ## Training

# 1. Train the head branches
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=30,
            layers='heads', save_model_each_epoch=False)
model_path = os.path.join(MODEL_DIR, "mirror_msd9_heads_30.h5")
model.keras_model.save_weights(model_path)

# 2. Fine tune all layers
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=40,
            layers="all", save_model_each_epoch=False)
model_path = os.path.join(MODEL_DIR, "mirror_msd9_all_40.h5")
model.keras_model.save_weights(model_path)
