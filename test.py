"""
  @Time    : 2018-5-7 23:23
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com
  
  @Project : mirror
  @File    : test.py
  @Function: for test code
  
"""
import numpy as np

a = np.ones([3, 3])
b = np.pad(a, [(1, 1), (0, 0)], mode="constant")
print(b)
print(a.shape[0])



