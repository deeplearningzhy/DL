#coding=utf-8
import tensorflow as tf
import numpy as np
import os
# import TFRecord
import tfrecord_input
import shiyanmodel

N_CLASSES = 2
IMG_W = 100  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 100
BATCH_SIZE = 10
CAPACITY = 20000
MAX_STEP = 10000  # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.0001

def run_training():
    logs_train_dir = '/home/user/zhyproject/shiyan/data2/log2/'  # 训练之后模型要保存的地方
    tfrecords_filename = '/home/user/zhyproject/shiyan/data2/train.tfrecords'
    img, label = tfrecord_input.read_and_decode(tfrecords_filename,10)

    train_batch, train_label_batch = tf.train.shuffle_batch([img, label],
                                                            batch_size=BATCH_SIZE,
                                                            capacity=CAPACITY,
                                                            min_after_dequeue=9)


    train_logits = shiyanmodel.inference(train_batch, BATCH_SIZE, N_CLASSES)
    train_loss = shiyanmodel.losses(train_logits, train_label_batch)
    train_op = shiyanmodel.trainning(train_loss, learning_rate)
    train__acc = shiyanmodel.evaluation(train_logits, train_label_batch)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()

    # init = tf.initialize_all_variables()
    # sess.run(init)
    init_op = tf.group(tf.initialize_all_variables(),
                       tf.initialize_local_variables())
    sess.run(init_op)

    # sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])

            if step % 50 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

if __name__=="__main__":
    run_training()

