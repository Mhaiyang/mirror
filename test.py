"""
  @Time    : 2018-5-7 23:23
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com
  
  @Project : mirror
  @File    : test.py
  @Function: for test code
  
"""
import tensorflow as tf

a = tf.ones([2,2])
b = tf.where(tf.equal(a, 1))

with tf.Session() as sess:
    print(sess.run(b))
    print(sess.run(b[:,0]))


