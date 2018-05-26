#coding=utf-8
from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist_data", one_hot=True)

import tensorflow as tf
#######################################
 # Parameters
learning_rate = 0.1
num_steps = 500
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])
is_training=tf.placeholder(tf.bool)

def batch_norm(inputs, is_training,is_conv_out=False,decay = 0.999):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
    if is_training:
        if is_conv_out:
            batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
        else:
            batch_mean, batch_var = tf.nn.moments(inputs,[0])


        train_mean = tf.assign(pop_mean,tf.multiply(pop_mean, decay)+ batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,tf.multiply(pop_var, decay) + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, 0.001)

    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, 0.001)

# Create model
def neural_net(x):
    with tf.variable_scope('neural_network'):
        # Hidden fully connected layer with 256 neurons
        layer1=tf.layers.dense(x,n_hidden_1)
        layer1=batch_norm(layer1,is_training)
        layer1=tf.nn.relu(layer1)

        layer2=tf.layers.dense(layer1,n_hidden_2)
        layer2=batch_norm(layer2,is_training)
        layer2=tf.nn.relu(layer2)
        output=tf.layers.dense(layer2,num_classes)
        return output
#######################################################
# Construct model logits是没有加激活函数的
logits = neural_net(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
#Define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op=optimizer.minimize(loss_op)
#收集需要更新的变量
# net_vars = filter(lambda x: x.name.startswith('ne'), tf.trainable_variables())

# 或者
# net_var=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="neural_network")
# update_op=tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope="neural_network")
# with tf.control_dependencies(update_op):
#     train_op=optimizer.minimize(loss_op,var_list=net_var)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
##############################################################
# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x,
                                      Y: batch_y,
                                      is_training:True
                                      })
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y,
                                                                 is_training:False
                                                                 })
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:",sess.run(accuracy,
                                    feed_dict={X: mnist.test.images,
                                               Y: mnist.test.labels,
                                               is_training:False
                                               }))