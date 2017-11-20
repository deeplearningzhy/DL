#coding=utf-8
from __future__ import print_function#原来出错的原因是没有加上这个
import tensorflow as tf
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_DATA', one_hot=True)#这个/tmp/data/路径是
'''
A logistic regression learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
线性回归与逻辑回归有什么区别？
'''
# Parameters网络参数(旋钮可调节)
learning_rate = 0.01#学习率
training_epochs = 25#训练轮数
batch_size = 100#批次
display_step = 1#每隔1步显示一次准确率，损失等训练信息

"""线性回归这里会有这样一串代码
# Training Data训练数据点是由numpy产生的随机值
train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]
然后，后面的与逻辑回归基本一致
"""

# tf Graph Input tf图的输入：占位符（图片与标签）
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

# Set model weights设置模型的权重和偏置
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model构造模型
"""
逻辑回归用pred = tf.nn.softmax(tf.matmul(x, W) + b)
线性回归用pred = tf.add(tf.multiply(X, W), b)#multiply与matmul的区别
"""
logits = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

"""
线性回归用的是平方差损失函数
loss = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)#损失函数
逻辑回归用的是交叉熵损失函数
"""
# Minimize error using cross entropy，这里的reduction_indices=1是什么意思？笔记本有记录，每一行求和
# loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
#定义损失
loss=-tf.reduce_sum(y*tf.log(logits))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)#这个最后在训练循环里面要用sess.run([optimizer,loss])
#sess.run它们之后返回什么呢？optimizer是个优化器，不用返回，loss是损失，要返回

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


# Start training正式开始训练

with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):#@@@轮数

        avg_loss = 0. #为什么要设置一个avg_cost?，o，前面算出来的是总的（每一个batch上的loss，故每一次的c都不一样）

        total_batch = int(mnist.train.num_examples/batch_size)#算出训练总批次：总的训练样例/一个批次里面的训练样本

        for i in range(total_batch): #@@@ Loop over all batches在所有的批次上进行循环

            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feed_dict = {x: batch_xs, y: batch_ys}
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, loss], feed_dict=feed_dict)
            # Compute average loss
            avg_loss += c / total_batch
        # Display logs per epoch step（）
        if (epoch+1) % display_step == 0:#注意这里是epoch+1,后面取余也要是epoch+1,每隔多长时间打印一次loss
            #为什么要从epoch+1开始而不是epoch？注意看第一个@@@，若为epoch，那么当epoch=0时，0%1==1,就不能把epoch=0算进去了
            print("Epoch:", '%04d' % (epoch+1), "loss=", "{:.9f}".format(avg_loss))#打印出每一轮的损失

    print("Optimization Finished!")

    # Test model测试模型
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    """
    来好好研究下tf.argmax()笔记本上有记载，是找出指定维度的最大值
    经过softmax之后，logits对应输出的是每一个数字可能对应的概率，当然是概率越大，就越有可能是那个数字，
    这样第一个argmax就找出了预测值的最可能的那个数字
    又因为标签是one hot编码的，因此第二个argmax旧找出了真实的数字，
    两个一比较，返回一个批次的布尔值,经比较预测的=真实的，返回1,否则返回0,
    于是输出一个批次比较过的bool值[1,1,0,1,1,1,0,...]
    """

    # Calculate accuracy这才是真正开始计算准确率
    # tf.cast()是类型转换函数，tf.reduce_mean()是求平均值（一个批次内的）
    #所以accuracy仍是tensor，打印出来需要用sess.run(accuracy),不过这里用了accuracy.eval()是一样的效果
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #给网络喂测试集，算出测试集在模型上的准确率
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

    # Successfully
    # downloaded
    # train - images - idx3 - ubyte.gz
    # 9912422
    # bytes.
    # Extracting
    # MNIST_DATA / train - images - idx3 - ubyte.gz
    # Successfully
    # downloaded
    # train - labels - idx1 - ubyte.gz
    # 28881
    # bytes.
    # Extracting
    # MNIST_DATA / train - labels - idx1 - ubyte.gz
    # Successfully
    # downloaded
    # t10k - images - idx3 - ubyte.gz
    # 1648877
    # bytes.
    # Extracting
    # MNIST_DATA / t10k - images - idx3 - ubyte.gz
    # Successfully
    # downloaded
    # t10k - labels - idx1 - ubyte.gz
    # 4542
    # bytes.
    # Extracting
    # MNIST_DATA / t10k - labels - idx1 - ubyte.gz
    # 2017 - 11 - 20
    # 21:24:54.884923: W
    # tensorflow / core / platform / cpu_feature_guard.cc:45] The
    # TensorFlow
    # library
    # wasn
    # 't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
    # 2017 - 11 - 20
    # 21:24:54.900497: W
    # tensorflow / core / platform / cpu_feature_guard.cc:45] The
    # TensorFlow
    # library
    # wasn
    # 't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
    # 2017 - 11 - 20
    # 21:24:54.900533: W
    # tensorflow / core / platform / cpu_feature_guard.cc:45] The
    # TensorFlow
    # library
    # wasn
    # 't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
    # Epoch: 0001
    # loss = 42.650741650
    # Epoch: 0002
    # loss = 31.656855254
    # Epoch: 0003
    # loss = 30.522625838
    # Epoch: 0004
    # loss = 29.715317721
    # Epoch: 0005
    # loss = 29.417828149
    # Epoch: 0006
    # loss = 28.515295027
    # Epoch: 0007
    # loss = 28.577941169
    # Epoch: 000
    # 8
    # loss = 28.224206013
    # Epoch: 000
    # 9
    # loss = 27.906214855
    # Epoch: 0010
    # loss = 28.023515564
    # Epoch: 0011
    # loss = 27.744919028
    # Epoch: 0012
    # loss = 27.522093164
    # Epoch: 0013
    # loss = 27.472590124
    # Epoch: 0014
    # loss = 27.449517033
    # Epoch: 0015
    # loss = 27.172521281
    # Epoch: 0016
    # loss = 27.283209616
    # Epoch: 0017
    # loss = 27.066133721
    # Epoch: 001
    # 8
    # loss = 26.944942129
    # Epoch: 001
    # 9
    # loss = 26.930433263
    # Epoch: 0020
    # loss = 26.731417178
    # Epoch: 0021
    # loss = 26.823095858
    # Epoch: 0022
    # loss = 26.688549834
    # Epoch: 0023
    # loss = 26.785986948
    # Epoch: 0024
    # loss = 26.637720351
    # Epoch: 0025
    # loss = 26.514974518
    # Optimization
    # Finished!
    # Accuracy: 0.9227
    #
    # Process
    # finished
    # with exit code 0
