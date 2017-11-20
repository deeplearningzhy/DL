#coding=utf-8
# from __future__ import print_function#在这里这句话可以不要

import tensorflow as tf
#1创建一个常量操作
hello = tf.constant('Hello, TensorFlow!')
#2开始会话
sess = tf.Session()
#3Run the op
result= sess.run(hello)
#4打印
print(result)