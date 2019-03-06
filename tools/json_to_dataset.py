"""
 @Time    : 202/17/19 14:22
 @Author  : TaylorMei
 @Email   : mhy845879017@gmail.com
 
 @Project : mirror
 @File    : json_to_dataset.py
 @Function:
 
"""
import os

json_path = '/home/iccd/data/2019beforetrue/ylt/'

json_list = os.listdir(json_path)

for i, json_name in enumerate(json_list):
    print(i, json_name)

    full_path = json_path + json_name

    os.system('labelme_json_to_dataset %s'%(full_path))
