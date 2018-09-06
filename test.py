"""
  @Time    : 2018-5-7 23:23
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com
  
  @Project : mirror
  @File    : test.py
  @Function: for test code
  
"""
import numpy as np

a = np.array([[1,2,3,2],[2,1,3,2]])
b = np.where(a[:,3] == 2)
print(a[:, 3])
print(b[0])
