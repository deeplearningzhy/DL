#coding=utf-8
import tensorflow as tf
import os
import read0
from final_model_dcgan import *

CURRENT_DIR = os.getcwd()

def train():
    #设置global step，用来记录训练过程中的step
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # 训练过程中的日志保存文件
    train_dir =CURRENT_DIR+ '/dcgan_logs/'
    img_dir = "/home/lu/cs/DCASE_NEW/DATASET/1/"

    #1. 放置网络输入
    noise_input,real_image_input=input()
    #2. 构建模型
    #(1) Build Generator Network 由生成器生成图像G
    generator_sample = generator(noise_input)
    #(2) Build 2 Discriminator Networks (one from real_image input,one from generator image)
    # （a）真实图像送人判别器
    disc_real = discriminator(real_image_input)
    # （b）生成图像送入判别器
    disc_fake = discriminator(generator_sample, reuse=True)  # 注意这里重用了


    #3. 计算loss
    # Build Loss(Lables for real image:1,for fake image:0)
    #(1) Discriminator loss for real and fake samples
    disc_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits
                                    (logits=disc_real,  # 真实图像送入判别器，D应该输出1
                                     labels=tf.ones([BATCH_SIZE], dtype=tf.int32)  # 1
                                     )
                                    )
    disc_loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits
                                    (logits=disc_fake,  # 生成图像送入判别器，D应输出0
                                     labels=tf.zeros([BATCH_SIZE], dtype=tf.int32)  # 0
                                     )
                                    )

    # Sum both loss
    disc_loss = disc_loss_real + disc_loss_fake

    #(2) Generator loss(the generator trise to fool the discriminator,thus labels are 1)
    gene_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits
                               (logits=disc_fake,  # 生成图像送入判别器，去骗D，让它输出1
                                labels=tf.ones([BATCH_SIZE], dtype=tf.int32)))


    #4.(1) 总结操作
    noise_sum = tf.summary.histogram('noise', noise_input)
    disc_real_sum= tf.summary.histogram('disc_real',disc_real )#真
    disc_fake_sum = tf.summary.histogram('disc_fake', disc_fake)#假
    gene_sum = tf.summary.image('gene', generator_sample)

    d_loss_real_sum = tf.summary.scalar('d_loss_real', disc_loss_real)#真
    d_loss_fake_sum = tf.summary.scalar('d_loss_fake', disc_loss_fake)#假
    d_loss_sum = tf.summary.scalar('d_loss', disc_loss)#和
    g_loss_sum = tf.summary.scalar('g_loss', gene_loss)
    ##4. (2)合并各自的总结
    g_sum = tf.summary.merge([noise_sum, disc_fake_sum, gene_sum, d_loss_fake_sum, g_loss_sum])
    d_sum = tf.summary.merge([noise_sum, disc_real_sum, d_loss_real_sum, d_loss_sum])

    #5. 生成器和判别器需要更新的变量，用于tf.train.Optimizer的var_list
    # Build Optimizers
    optimizer_gene = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
    optimizer_disc = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
    # Training Variables for each optimizer
    # By default in TF,all variables are updated by each optimizer,
    # So we need to precise for each one of them the specific variables to update
    # (1)收集Generator Network Variables，返回list
    gene_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Generator")
    # (2)收集Discriminator Network Variables，返回list
    disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Discriminator")

    # Creating training operations
    # TF UPDATE_OPS collection holds all batch norm operation to update the moving mean/stddev()
    # TF的UPDATE_OPS集合包含了所有批归一化操作中的滑动平均值
    # (1)更新Generator Network Variables(先更新后反向传播)
    gene_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="Generator")

    with tf.control_dependencies(gene_update_ops):
        train_gene = optimizer_gene.minimize(gene_loss, var_list=gene_vars,global_step=global_step)
    # (2)更新Discriminator Network Variables
    disc_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="Discriminator")

    with tf.control_dependencies(disc_update_ops):
        train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars,global_step=global_step)
####################################################
    # 7.# 读取batch张图片
    img_list, lab_list = read0.read_images(img_dir)
    img_batch, lab_batch = read0.get_batch(img_list, lab_list, batch_size=BATCH_SIZE)
########################################
    #6. 开启会话
    #（1）
    saver = tf.train.Saver()
    #（2）
    init = tf.global_variables_initializer()
    sess=tf.Session()
    sess.run(init)
    #（3）
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #（4）
    writer = tf.summary.FileWriter(train_dir, sess.graph)


    import numpy as np
    #8. 取随机噪声
    sample_z = np.random.uniform(-1., 1., size=[BATCH_SIZE, noise_dim])
    fake_images = generator(noise_input, reuse=True)
    #9.Training 循环epoch
    for epoch in range(1, EPOCHS + 1):
        batches = int(len(img_list) / BATCH_SIZE)  # 1类才24个batch
        for batch_i in range(1, batches + 1):
            batch_images = sess.run(img_batch)  # 把tensor（100,100,100,1）变成了numpy数组
            # Rescale to [-1,1],the input range of the discriminator
            batch_images = batch_images * 2. - 1.
            #Generate noise to feed to the generator
            batch_noise = np.random.uniform(-1, 1, size=[BATCH_SIZE, noise_dim])

            # (1)Discriminator Training(更新D的参数)
            _, dl,summary_str = sess.run([train_disc, disc_loss,d_sum],
                             feed_dict={real_image_input: batch_images,
                                        noise_input: batch_noise,
                                        is_training: True  # 训练的时候BN的is_training打开，用来保存滑动平均值
                                        }
                             )
            writer.add_summary(summary_str,batch_i)

            # (2)Generator Training(更新G的参数)
            _, gl,summary_str = sess.run([train_gene, gene_loss,g_sum],
                             feed_dict={noise_input: batch_noise,
                                        is_training: True
                                        }
                             )
            writer.add_summary(summary_str,batch_i)

            #打印损失
            if batch_i % 1 == 0:
                print("EPOCH: [%2d] [%4d/%4d] d_loss:%.8f,g_loss:%.8f"
                      % (epoch, batch_i, batches, dl, gl))


            # Testing:generator images from noise,using the generator network
            #训练过程中，用采样器采样，并且保存采样的图片(注意下他的sess.run和noise_input)
            if batch_i % 5 == 0:#这里有改动
                # for test（测试的时候，就直接训练好了的生成器拿出来，故reuse=True，给他有约束条件的噪声，让它产生图片）

                #注意这里的z应该是epoch外面的z，如果有约束的话，约束也应该是外面的约束
                sample = sess.run(fake_images, feed_dict={noise_input: sample_z,
                                                          is_training: False})
                samples_path = '/home/lu/cs/DCASE_NEW/mycode/fakeimg/'
                save_images(sample, [10, 10],
                            samples_path + 'sample_%d_epoch_%d.jpg' % (epoch, batch_i))
                print('save down')

            #每过多少次迭代，保存一次模型#(别人是隔多少个batch，我是隔多少个epoch)
            if batch_i % 10 ==0:
                checkpoint_path=os.path.join(train_dir,"DCGAN_model.ckpt")
                saver.save(sess,checkpoint_path,global_step=batch_i)
                print("model saved")

    coord.request_stop()
    coord.join(threads)
    sess.close()

if __name__=="__main__":
    train()