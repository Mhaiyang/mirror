#!/bin/bash
image="/home/taylor/mirror/data/test/json/000"
format=".json"
for((i=0;i<10;i++))
    do
        num=${i}
        labelme_json_to_dataset ${image}${num}${format}
    done
