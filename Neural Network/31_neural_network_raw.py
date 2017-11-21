#coding=utf-8
from __future__ import print_function
import tensorflow as tf
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
mnist = input_data.read_data_sets("mnist_data", one_hot=True)

""" 
Neural Network.常规前馈神经网络
A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow. This example is using the MNIST database
of handwritten digits 。
双隐层全连接神经网络（又称多层感知器）用TensorFlow实现。 这个例子是使用MNIST数据库的手写数字
"""
# Parameters可调节的模型参数
learning_rate = 0.1#学习率
num_steps = 500#训练步数
batch_size = 128#一个批次有128个样本
display_step = 100#每隔100步显示一下损失信息

# Network Parameters 神经网络的参数
n_hidden_1 = 256 # 1st layer number of neurons（第一隐层的神经元个数）
n_hidden_2 = 256 # 2nd layer number of neurons（第二隐层的神经元个数）
num_input = 784 # MNIST data input (img shape: 28*28)（输入层神经元个数）
num_classes = 10 # MNIST total classes (0-9 digits)（输出层神经元个数，10个类别）

# tf Graph input（tf图的input）X对应数据集里面真实的图片，Y对应相应的标签，None代表一个批次的样本数
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias（将网络的每一层的权重和偏置存在字典里，权重，偏置各对应一个字典）
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


# Create model（开始构造NN模型，这里是用函数封装）
def neural_net(x):
    """
    函数参数：x:代表输入层的特征向量
    返回值:out_layer：代表输出层的特征向量
    """
    # Hidden fully connected layer with 256 neurons(第一层隐藏层)
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons（第二层隐藏层）
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class（输出层）
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model（构造模型，直接调用封装好了的函数模型）
logits = neural_net(X)#将数据集中的样本（X！）输入网络
prediction = tf.nn.softmax(logits)#与逻辑回归类似，还是要用到softmax处理预测结果，把它们转换成概率值，和为1

# Define loss and optimizer（定义损失和优化器）
#损失函数使用交叉熵函数
#优化器使用Adam
#训练操作train_op就是要让optimizer.minimize(loss_op)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model(预测模型)
#correct_pred输出bool值，预测的=真实的，输出1,否则输出0。correct应该也是Tensor
#accuracy算平均值，由于是布尔值，所以可以算出一个批次中预测对的百分比，
# accuracy是tensor!.那如何打印出它里面装载的具体值呢？用sess.run(accuracy)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean( tf.cast(correct_pred, tf.float32) )

#等到把accuracy写完之后，网络的inference差不多就结束了，下面开始初始化我们上面定义的所有变量
#接着开始定义会话，训练模型
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training（训练阶段！！！）
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):#num_steps=500,从第1步打印到500步，这个地方也要比较其他算法
        #开始准备分批次喂入网络中的值
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_x, Y: batch_y}
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict=feed_dict)#特别注意这里的sess.run(前面一个是train_op，与其他算法比较看看)，而且sess.run之后没有返回值
        if step % display_step == 0 or step == 1:#因为step是从1开始，所以后面要有一个or step==1（每隔多少步显示一次）
            # Calculate batch loss and accuracy（开始计算每一批次的loss和accuracy）
            loss, acc = sess.run([loss_op, accuracy], feed_dict=feed_dict)
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")#训练阶段完成

    # Calculate accuracy for MNIST test images
    #开始测试阶段，只是feed_dict到网络的样本不一样
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: mnist.test.images,Y: mnist.test.labels}))

    # IOError: [Errno socket error][Errno101] Network is unreachable
    #???
