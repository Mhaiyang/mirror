import shutil
import os

source_path = "/media/taylor/mhy/mirror/take_photo4/"
destination_path = "/home/taylor/mirror/data/"
if not os.path.exists(destination_path + "train/"):
    os.mkdir(destination_path + "train/")
    os.mkdir(destination_path + "train/image/")
    os.mkdir(destination_path + "train/mask/")
if not os.path.exists(destination_path + "val/"):
    os.mkdir(destination_path + "val/")
    os.mkdir(destination_path + "val/image")
    os.mkdir(destination_path + "val/mask/")
if not os.path.exists(destination_path + "test/"):
    os.mkdir(destination_path + "test/")
    os.mkdir(destination_path + "test/image/")
    os.mkdir(destination_path + "test/mask/")
    os.mkdir(destination_path + "test/output/")

list = os.listdir(source_path + "mask")
print("Total {} images will be processed!".format(len(list)))

train_order, val_order, test_order = 1643, 118, 585

for i, name in enumerate(list):
    # The ratio of train:validation:test is equal to 0.9:0.004:0.006
    # Train data
    if i < len(list)*0.7:
        order = list[i][:-5]
        # copy mask json file
        old_json = source_path + "mask/" + order + ".json"
        new_json = destination_path + "train/" + "mask/" + str(train_order) + ".json"
        shutil.copy(old_json, new_json)
        # copy image
        old_image = source_path + "image/" + order + ".jpg"
        new_image = destination_path + "train/" + "image/" + str(train_order) + ".jpg"
        shutil.copy(old_image, new_image)
        print("{} Processing Train {}.jpg".format(train_order, order))
        train_order += 1
    # Validation data
    elif i < len(list)*0.75:
        order = list[i][:-5]
        # copy mask json file
        old_json = source_path + "mask/" + order + ".json"
        new_json = destination_path + "val/" + "mask/" + str(val_order) + ".json"
        shutil.copy(old_json, new_json)
        # copy image
        old_image = source_path + "image/" + order + ".jpg"
        new_image = destination_path + "val/" + "image/" + str(val_order) + ".jpg"
        shutil.copy(old_image, new_image)
        print("{} Processing Val {}.jpg".format(val_order, order))
        val_order += 1
    # Test data
    else:
        order = list[i][:-5]
        # copy mask json file
        old_json = source_path + "mask/" + order + ".json"
        new_json = destination_path + "test/" + "mask/" + str(test_order) + ".json"
        shutil.copy(old_json, new_json)
        # copy image
        old_image = source_path + "image/" + order + ".jpg"
        new_image = destination_path + "test/" + "image/" + str(test_order) + ".jpg"
        shutil.copy(old_image, new_image)
        print("{} Processing Test {}.jpg".format(test_order, order))
        test_order += 1
print("Total {} images have been processed!".format(len(list)))
print("Train count: {} \nValidation count: {}\nTest count: {}".format(train_order-1, val_order-1, test_order-1))

