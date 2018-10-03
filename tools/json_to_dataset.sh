#!/bin/bash
image="/home/taylor/Desktop/image/"
format=".json"
for((i=168;i<=168;i++))
    do
        num=${i}
        labelme_json_to_dataset ${image}${num}${format}
    done
