"""
  @Time    : 2018-5-7 23:23
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com
  
  @Project : mirror
  @File    : test.py
  @Function: for test code
  
"""
import tensorflow as tf

# 两个矩阵相乘
x = tf.constant([[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]])
y = tf.constant([[[0, 0, 1.0], [0, 0, 1.0], [0, 0, 1.0]], [[0, 1.0, 0], [0, 1.0, 0], [0, 1.0, 0]]])
# 注意这里这里x,y要有相同的数据类型，不然就会因为数据类型不匹配而出错
z = tf.multiply(x, y)

with tf.Session() as sess:
    print(sess.run(x))
    print(sess.run(y))
    print(sess.run(z))
