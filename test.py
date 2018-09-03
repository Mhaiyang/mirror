"""
  @Time    : 2018-5-7 23:23
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com
  
  @Project : mirror
  @File    : test.py
  @Function: for test code
  
"""
import numpy as np
import tensorflow as tf

a = np.array(list(range(100)))
a = a.reshape([10,10])
b = tf.constant(a)
print(b)
index = tf.where(b>50)
print(index)
c = tf.gather_nd(b,index)
print(c)
c = tf.gather(b,index)
print(c)