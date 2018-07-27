"""
  @Time    : 2018-5-7 23:23
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com
  
  @Project : mirror
  @File    : test.py
  @Function: for test code
  
"""
import numpy as np

a = np.array([3, 4, 5])
b = np.arange(3) + 1
test = np.cumsum(a > 3)
print(a)
print(b)
print(test)
print(test/b)
