"""
  @Time    : 2018-5-7 23:23
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com
  
  @Project : mirror
  @File    : test.py
  @Function: for test code
  
"""
import yaml

with open("/home/taylor/mirror/data/train/mask/1_json/info.yaml") as f:
    temp = yaml.load(f.read())
    labels = temp['label_names']
    a = 2
    print(labels[:a])