#coding=utf-8
import tensorflow as tf
import numpy as np
import os
import read0
from final_model_dcgan import *
CURRENT_DIR = os.getcwd()

def eval():
    # 用于存放测试过程中生成的图片
    eval_dir =CURRENT_DIR+"/eval/"
    # 从此处加载模型
    checkpoint_dir=CURRENT_DIR+"/dcgan_logs/"#(就是train.py里面的train_dir)
    # 加载随机噪声的placeholder
    noise_input,_=input()
    #生成器生成图像
    G = generator(noise_input)
    #产生随机噪声
    sample_z1 = np.random.uniform(-1, 1, size=[BATCH_SIZE, noise_dim])
    sample_z2 = np.random.uniform(-1, 1, size=[BATCH_SIZE, noise_dim])
    sample_z3 = (sample_z1 + sample_z2) / 2
    sample_z4 = (sample_z1 + sample_z3) / 2
    sample_z5 = (sample_z2 + sample_z3) / 2

    #(1) 读取 ckpt 需要 sess，saver
    print("Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    # (2)saver
    saver = tf.train.Saver(tf.all_variables())

    #(3)#sess
    sess=tf.Session()
    #(3)从保存的模型中恢复变量
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        print('Loading success, global_step is %s' % global_step)
    # (4) 用恢复的变量进行生成器的测试
    eval_sess1 = sess.run(G, feed_dict={noise_input: sample_z1,is_training:False})
    eval_sess2 = sess.run(G, feed_dict={noise_input: sample_z4,is_training:False})
    eval_sess3 = sess.run(G, feed_dict={noise_input: sample_z3,is_training:False})
    eval_sess4 = sess.run(G, feed_dict={noise_input: sample_z5,is_training:False})
    eval_sess5 = sess.run(G, feed_dict={noise_input: sample_z2,is_training:False})

    print(eval_sess3.shape)
    #(5)保存测试的生成器图片到特定文件夹
    save_images(eval_sess1, [10, 10], eval_dir + 'eval_%d.png' % 1)
    save_images(eval_sess2, [10, 10], eval_dir + 'eval_%d.png' % 2)
    save_images(eval_sess3, [10, 10], eval_dir + 'eval_%d.png' % 3)
    save_images(eval_sess4, [10, 10], eval_dir + 'eval_%d.png' % 4)
    save_images(eval_sess5, [10, 10], eval_dir + 'eval_%d.png' % 5)

    sess.close()


if __name__ == '__main__':
    eval()