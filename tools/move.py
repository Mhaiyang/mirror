import shutil
import os

source_path = "/home/taylor/Desktop/"
destination_path = "/home/taylor/mirror/data/"

list = os.listdir(source_path + "newmask")
print("Total {} images!".format(len(list)))
train_order, val_order, test_order = 0, 0, 0
for i, name in enumerate(list):
    if i < len(list)*0.8:
        order = list[i][:-5]
        # copy mask json file
        old_json = source_path + "newmask/" + order + ".json"
        new_json = destination_path + "train/" + "json/" + format(str(train_order), '0>4s') + ".json"
        shutil.copy(old_json, new_json)
        # copy image
        old_image = source_path + "newdata/" + order + ".jpg"
        new_image = destination_path + "train/" + "image/" + format(str(train_order), '0>4s') + ".jpg"
        shutil.copy(old_image, new_image)
        train_order += 1
    elif i < len(list)*0.9:
        order = list[i][:-5]
        # copy mask json file
        old_json = source_path + "newmask/" + order + ".json"
        new_json = destination_path + "val/" + "json/" + format(str(val_order), '0>4s') + ".json"
        shutil.copy(old_json, new_json)
        # copy image
        old_image = source_path + "newdata/" + order + ".jpg"
        new_image = destination_path + "val/" + "image/" + format(str(val_order), '0>4s') + ".jpg"
        shutil.copy(old_image, new_image)
        val_order += 1
    else:
        order = list[i][:-5]
        # copy mask json file
        old_json = source_path + "newmask/" + order + ".json"
        new_json = destination_path + "test/" + "json/" + format(str(test_order), '0>4s') + ".json"
        shutil.copy(old_json, new_json)
        # copy image
        old_image = source_path + "newdata/" + order + ".jpg"
        new_image = destination_path + "test/" + "image/" + format(str(test_order), '0>4s') + ".jpg"
        shutil.copy(old_image, new_image)
        test_order += 1
print("Train count: {} \nVal count: {}\nTest_count: {}".format(train_order, val_order, test_order))

