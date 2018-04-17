#!/bin/bash
image="/home/taylor/Mask_RCNN/dataset/mask/00"
format=".json"
for((i=10;i<40;i++))
    do
        num=${i}
        labelme_json_to_dataset ${image}${num}${format}
    done
