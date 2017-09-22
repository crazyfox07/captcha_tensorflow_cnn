# -*- coding:utf-8 -*-
"""
File Name: crack_cnn_tensorflow
Version:
Description:
Author: liuxuewen
Date: 2017/9/20 16:49
"""
import tensorflow as tf
import numpy as np
import re
import time
from util import IMAGE_HEIGHT, IMAGE_WIDTH, LEN_CAPTCHA, LEN_CHAR_SET, get_next_batch, vec2text, get_img, CHARS

X = tf.placeholder(tf.float32,[None, IMAGE_HEIGHT * IMAGE_WIDTH])
Y = tf.placeholder(tf.float32,[None, LEN_CAPTCHA*LEN_CHAR_SET])
keep_prob = tf.placeholder(tf.float32)  # dropout
w_alpha = 0.1
b_alpha = 0.1


def add_layer(input=None, w_shape=None, b_shape=None, conv2d=False, active_func=None, name=None):
    # 前两维是patch的大小，第三维时输入通道的数目，最后一维是输出通道的数目。我们对每个输出通道加上了偏置(bias)
    w = tf.Variable(w_alpha * tf.random_normal(w_shape))
    b = tf.Variable(b_alpha * tf.random_normal(b_shape))
    # 卷基层与池化层
    if conv2d and active_func:
        conv = active_func(tf.nn.bias_add(tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding='SAME'), b))
        conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv = tf.nn.dropout(conv, keep_prob=keep_prob)
        return conv
    # 全连接层
    elif active_func:
        dense = tf.reshape(input, [-1, w_shape[0]])
        dense = active_func(tf.add(tf.matmul(dense, w), b))
        dense = tf.nn.dropout(dense, keep_prob=keep_prob)
        return dense
    # 输出层
    else:
        out = tf.add(tf.matmul(input, w), b)
        return out


# 定义模型
def model():
    # 为了使得图片与计算层匹配，我们首先reshape输入图像x为4维的tensor，第2、3维对应图片的宽和高，最后一维对应颜色通道的数目。
    x = tf.reshape(X, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    # 第一层
    layer1 = add_layer(input=x, w_shape=[3, 3, 1, 32], b_shape=[32], conv2d=True, active_func=tf.nn.relu, name='layer1')
    # 第二层
    layer2 = add_layer(input=layer1, w_shape=[3, 3, 32, 64], b_shape=[64], conv2d=True, active_func=tf.nn.relu,
                       name='layer2')
    # 第三层
    layer3 = add_layer(input=layer2, w_shape=[3, 3, 64, 64], b_shape=[64], conv2d=True, active_func=tf.nn.relu,
                       name='layer2')
    # 全连接层
    layer_full = add_layer(input=layer3, w_shape=[8 * 20 * 64, 1024], b_shape=[1024], active_func=tf.nn.relu,
                           name='layer_full')
    # 输出层
    layer_out = add_layer(input=layer_full, w_shape=[1024, LEN_CHAR_SET * LEN_CAPTCHA],
                          b_shape=[LEN_CHAR_SET * LEN_CAPTCHA], name='layer_out')
    return layer_out


# 训练
def train_model():
    output = model()
    # loss 损失数值
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Y))
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    # optimizer 为了加快训练 learning_rate 应该开始大，然后慢慢衰
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    predict = tf.reshape(output, [-1, LEN_CAPTCHA, LEN_CHAR_SET])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, LEN_CAPTCHA, LEN_CHAR_SET]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # tmp = tf.train.latest_checkpoint('model/')
        # saver.restore(sess, tmp)#从模型中读取数据，可以充分利用之前的经验
        # print(tmp)

        #当直接读取模型时，需要把变量初始化去掉
        sess.run(tf.global_variables_initializer())

        step = 0
        batch_id=0
        while True:
            batch_x, batch_y = get_next_batch(64,batch_id)
            if batch_x is None:
                batch_id = 0
                batch_x, batch_y = get_next_batch(64, batch_id)
            batch_id+=1

            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
            #print(step, loss_)

            # 每100 step计算一次准确率
            if step % 100 == 0:
                batch_x_test, batch_y_test = get_next_batch(64)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.0})
                print(step, acc)
                # 如果准确率大于99%,保存模型,完成训练
                if acc > 0.9:
                    saver.save(sess, "./model/crack_capcha.model2", global_step=step)
                    break
            step += 1


def crack_captcha(captcha_images):

    output = model()
    #saver = tf.train.Saver('./model/')
    tmp = tf.train.latest_checkpoint('model/')
    print(tmp)
    saver = tf.train.Saver()
    text_list=list()
    with tf.Session() as sess:
        #saver.restore(sess, tf.train.latest_checkpoint('.'))
        # saver = tf.train.import_meta_graph('./model/crack_capcha.model-3600.meta')
        # saver.restore(sess, tf.train.latest_checkpoint('./'))
        saver.restore(sess, tmp)

        for captcha_image in captcha_images:
            predict = tf.argmax(tf.reshape(output, [-1, LEN_CAPTCHA, LEN_CHAR_SET]), 2)
            text = sess.run(predict, feed_dict={X: captcha_image, keep_prob: 1})
            text = list(text[0])
            r=list()
            for index in text:
                r.append(CHARS[index])
            text_list.append(r)

        #print('output={}'.format(sess.run(output)))


        return text_list


        #return vec2text(vector)


if __name__ == '__main__':
    # begin=time.time()
    # train_model()
    # end=time.time()
    # print('time use: {}'.format(end-begin))
    import os
    dir=r'D:\project\图像识别\image\test'
    #dir=r'D:\project\图像识别\image\tmp'
    list_dir = os.listdir(dir)
    img_path=os.path.join(dir,list_dir[0])
    reals=[re.findall(r'(\w{4})\.png',img_name)[0] for img_name in list_dir[:100]]
    imgs=[get_img(os.path.join(dir,img_name)) for img_name in list_dir[:100]]
    predicts=crack_captcha(imgs)
    #real=re.findall(r'(\w{4})\.png',img_path)[0]
    for r in zip(predicts,reals,list_dir[:100]):
        predict=''.join([str(i) for i in r[0]])
        print('predic={},real={},img_path={}'.format(predict,r[1],r[2]))






