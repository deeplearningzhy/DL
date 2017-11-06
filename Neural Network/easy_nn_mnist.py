#!/usr/bin/env python
#coding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#加载数据，如果不存在他会首先自动下载数据到指定的目录
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# X是一个占位符 ,这个值后续再放入让TF计算，这里是一个784维，但是训练数量不确定的（用None表示）的浮点值
x = tf.placeholder(tf.float32, [None, 784])
# 设置对应的权值和偏置的表示，Variable代表一个变量，会随着程序的生命周期做一个改变
# 需要给一个初始的值，这里都全部表示为0
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# 这里构造NN模型，y是模型的输出，matmul是矩阵乘法
y = tf.nn.softmax(tf.matmul(x, W) + b)
# 这里是保存真实的Label，同样是占位符，原理同X
y_ = tf.placeholder(tf.float32, [None, 10])
# 在机器学习的模型中，我们需要定义一个衡量模型好坏的方式，称为代价函数（Cost Loss），这里使用了交叉熵去衡量 reduce_sum 累加
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#训练的步骤，告诉tf，用梯度下降法去优化，学习率是0.5，目的是最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 到目前为止，我们已经定义完了所有的步骤，下面就需要初始化这个训练步骤了，首先初始化所有变量（之前定义的变量）
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#运行步骤1000次
for i in range(1000):
  #载入数据
  batch_xs, batch_ys = mnist.train.next_batch(100)
  #执行步骤，但是需要填入placeholder的值
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
#进行模型的的衡量
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#理论上应该是92%附近
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
