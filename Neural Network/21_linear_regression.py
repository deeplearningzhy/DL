#coding=utf-8
'''
线性回归算法
'''
# from __future__ import print_function
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random

# Parameters模型参数
learning_rate = 0.01#学习率
training_epochs = 1000#训练轮数
display_step = 50#训练多少次显示一次准确率

# Training Data训练数据点是由numpy产生的随机值
train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]#这个表示什么？ n_samples=17,shape[0]用来算出一个数组有多少列

# tf Graph Inputtf图的输入：用占位符表示
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights设置model的权重和偏置
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model构造一个线性模型，输出值就是预测值logits
pred = tf.add(tf.multiply(X, W), b)
loss = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)#损失函数
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)#当然是最小化loss

# Mean squared error（均方误差，也就是平方代价函数）
# Gradient descent梯度下降
#  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
# 注意，最小化（）知道修改W和b，因为变量对象默认是可训练的

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training（记住步骤）
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    # Fit all training data#这个training_epoch其实就是训练步数step吧
    for epoch in range(training_epochs):
        # for (x, y) in zip(train_X, train_Y):#？？
            # sess.run(optimizer, feed_dict={X: x, Y: y})#(x,y)自动匹配train_x,train_Y,得到一个坐标点
        sess.run(optimizer, feed_dict={X:train_X , Y:train_Y })
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:#为什么要用epoch+1？
            c = sess.run(loss, feed_dict={X: train_X, Y:train_Y})#为什么这里不是用的上面的x，y？，放x，y会出现没有定义的显示
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")

    training_loss = sess.run(loss, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_loss, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

    # Testing example, as requested (Issue #2)
    test_X = numpy.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_Y = numpy.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

    print("Testing... (Mean square loss Comparison)")
    testing_loss = sess.run(tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print("Testing cost=", testing_loss)
    print("Absolute mean square loss difference:", abs(training_loss - testing_loss))

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()