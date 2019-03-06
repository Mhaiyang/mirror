#!/bin/bash
image="/home/iccd/data/2019beforetrue/ylt/"
format=".json"
for((i=1;i<=5000;i++))
    do
        num=${i}
        labelme_json_to_dataset ${image}${num}${format}
    done
