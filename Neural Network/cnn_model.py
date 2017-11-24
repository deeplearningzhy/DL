#coding=utf-8
import tensorflow as tf
from PIL import Image
import numpy as np
import myrecord

#定义超参数
batch_size=10
learning_rate=1e-4
num_steps=50
n_classes=2
display_step=1


#实现此卷积神经网络有很多的权重和偏向需要创建，因此我们先定义好初始化函数以便重复使用
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 截断正态分布，此函数原型为尺寸、均值、标准差
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#卷积层，池化层也是接下来要重复使用的，因此也分别为它们创建函数
#32-5×5conv（padding=1,stride=1,RELU）
#64-5*5conv(padding=1,stride=1,RELU)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#Max_pool 3*3(stride=3)
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1],padding='SAME')

#在正式设计卷积神经网络的结构之前，先定义输入的placeholder，x是特征，y_是真实的label
image_holder=tf.placeholder(tf.float32,[batch_size,100,100,1])
label_holder=tf.placeholder(tf.float32,[batch_size,n_classes])
# label_holder = tf.one_hot( label_holder , n_classes )
keep_prob=tf.placeholder(tf.float32)
#接下来我们定义第一个卷积层
W_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])
conv1=tf.nn.relu(conv2d(image_holder,W_conv1)+b_conv1)
pool1=max_pool_2x2(conv1)
norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,beta=0.75, name='norm1')

#现在定义第二个卷积层
W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])
conv2=tf.nn.relu(conv2d(norm1,W_conv2)+b_conv2)
pool2=max_pool_2x2(conv2)
norm2=tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,beta=0.75, name='norm2')
#拆拆拆！！！


#=======================拼接=============================================
#在两个卷积层之后，将使用一个全连接层，这里需要先把前面两个卷积层的输出结果全部flatten
#使用tf.reshape函数将每个样本都变成一维向量。使用get_shape函数，获取数据扁平化之后的长度
#接着使用自定义的初始化函数对全连接层的weight进行初始化。
#一共有3个fc层
reshape=tf.reshape(norm2,[batch_size,-1])
dim=reshape.get_shape()[1].value
#fc1
W_fc1=weight_variable([dim,1024])
b_fc1=bias_variable([1024])
fc1=tf.nn.relu(tf.matmul(reshape,W_fc1)+b_fc1)
#fc2
W_fc2=weight_variable([1024,1024])
b_fc2=bias_variable([1024])
fc2=tf.nn.relu(tf.matmul(fc1,W_fc2)+b_fc2)
#fc3
W_fc3=weight_variable([1024,1024])
b_fc3=bias_variable([1024])
fc3=tf.nn.relu(tf.matmul(fc2,W_fc3)+b_fc3)
#dropout
fc3_dropout=tf.nn.dropout(fc3,keep_prob)

#softmax为最后一层，依然先创建这一层的weight
W_fc4=weight_variable([1024,n_classes])
b_fc4=bias_variable([n_classes])
# softmax = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
logits=tf.nn.softmax(tf.matmul(fc3_dropout,W_fc4)+b_fc4)
# logits:tensor[batch_size,n_classes]
#label_holder:tensor[batch_size]

#定义损失函数为cross_entropy,优化器为Adam
cross_entropy = tf.nn.softmax_cross_entropy_with_logits\
                        (logits=logits, labels=label_holder, name='cross_entropy')
loss = tf.reduce_mean(cross_entropy, name='loss')
train_step=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


# cross_entropy=tf.reduce_mean(-tf.reduce_sum(label_holder*tf.log(logits),reduction_indices=[1]))


#在继续定义评测准确率的操作
# label_holder=tf.cast(label_holder,tf.int64)
# correct_prediction=tf.nn.in_top_k(logits, label_holder, 1)
correct_prediction=tf.equal(tf.argmax(logits,1),tf.argmax(label_holder,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#=========================model finished====================================
#下面开始训练过程，首先依然是初始化所有参数，设置训练时dropout的keep_prob比率为0.5
#然后使用大小为100的mini_batch,一共进行20000次迭代
#参与训练的样本总共为17750,其中每100次训练，我们会对准确率进行一次评测（评测时keep_prob设置为1）
#用以实时监测模型的性能
summary_op = tf.summary.merge_all()
file_dir="/home/user/zhyproject/shiyan/data2/train/"#训练图片的路径
save_dir = "/home/user/zhyproject/shiyan/data2/"
name = 'train'
logs_train_dir = '/home/user/zhyproject/shiyan/log_train/'  # 训练之后模型要保存的地方
tfrecords_file = '/home/user/zhyproject/shiyan/data2/train.tfrecords'

images, labels =myrecord.get_files(file_dir)
myrecord.convert_to_tfrecord(images, labels, save_dir, name)
image_batch, label_batch =myrecord.read_and_decode(tfrecords_file, batch_size=10)
# Initialize the variables (i.e. assign their default value)
# Start Training 开始训练
# Start a new TF session 创建一个tensorflow会话，所有的计算图都要在会话中执行
sess = tf.Session()
train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
saver = tf.train.Saver()

# Run the initializer
init = tf.global_variables_initializer() #初始化所有变量
sess.run(init)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
# Training

for step  in range(1, num_steps+1):
    # Prepare Data准备数据
    # Get the next batch of MNIST data (only images are needed, not labels)
    #取得下一个训练数据样本（由于样本太大，需要分批次训练）
    # batch_x, _ = mnist.train.next_batch(batch_size)
    img_feed, label_feed = sess.run([image_batch, label_batch])
    img_feed=img_feed.reshape([batch_size,100,100,1])
    # label_feed=label_feed.reshape([batch_size,n_classes])还是会报错
    # print img_feed.shape,label_feed.shape
    #修改：这里要喂我们自己的数据
    # Train ，每一批训练完以后，将其他训练批次的样本数据喂给输入
    # feed_dict = {image_holder:img_feed,label_holder:label_feed}
    sess.run(train_step,feed_dict={image_holder:img_feed,label_holder:label_feed,
                                   keep_prob:0.8})#第一次keep_prob
    if step % display_step==0 or step==1:
        #计算每一批次的损失和准确率
        l,acc=sess.run([loss,accuracy],feed_dict={image_holder:img_feed,label_holder:label_feed,
                                                           keep_prob:1.0})#第二次keep_prob

        print "Step: %d,Loss=%.4f,Train_accuracy=%.3f" % (step,l,acc)
print("Optimization Finished!")



