#coding=utf-8
from __future__ import division, print_function, absolute_import

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/mnist_data/", one_hot=True)
#######################################################
# Training Parameters
learning_rate = 0.001
num_steps = 500
batch_size = 128
display_step = 10

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)


# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
is_training=tf.placeholder(tf.bool)
###################################################
# Create the neural network
def conv_net(x, n_classes):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet'):

        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, use_bias=False,activation=None)
        conv1=tf.layers.batch_normalization(conv1,training=is_training)
        conv1=tf.nn.relu(conv1)
        # Convolution Layer with 64 filters and a kernel size of 3
        conv2 = tf.layers.conv2d(conv1, 64, 3, use_bias=False,activation=None)
        conv2=tf.layers.batch_normalization(conv2,training=is_training)
        conv2=tf.nn.relu(conv2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024,use_bias=False,activation=None)
        fc1=tf.layers.batch_normalization(fc1,training=is_training)
        fc1=tf.nn.relu(fc1)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)

    return out
################################################################
# Construct model
logits = conv_net(X,num_classes)#这2个参数都需要feed
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

###########使用BN之后的重要步骤（不能丢）#################
conv_var=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="ConvNet")
update_op=tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope="ConvNet")
with tf.control_dependencies(update_op):
    train_op=optimizer.minimize(loss_op,var_list=conv_var)



# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
#######################################################
# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, is_training:True})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y,
                                                                 is_training:False})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 256 MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
                                      Y: mnist.test.labels[:256],
                                      is_training:False}))