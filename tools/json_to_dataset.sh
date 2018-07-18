#!/bin/bash
image="/home/taylor/mirror/data/test/mask/"
format=".json"
for((i=1;i<=139;i++))
    do
        num=${i}
        labelme_json_to_dataset ${image}${num}${format}
    done
