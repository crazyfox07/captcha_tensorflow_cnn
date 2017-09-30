# -*- coding:utf-8 -*-
"""
File Name: lstm_tensorflow
Version:
Description:
Author: liuxuewen
Date: 2017/9/29 16:34
"""
import os

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data
import re
# 设置 GPU 按需增长
from util import get_next_batch, get_img

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# 首先导入数据，看一下数据的形式
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# print(mnist.train.images.shape)

# 首先设置好模型用到的各个超参数
lr = 1e-3
# 在训练和测试的时候，我们想用不同的 batch_size.所以采用占位符的方式
batch_size = 4  # 注意类型必须为 tf.int32
# batch_size = 128

# 每个时刻的输入特征是28维的，就是每个时刻输入一行，一行有 28 个像素
input_size = 16
# 时序持续长度为28，即每做一次预测，需要先输入28行
timestep_size = 16
# 每个隐含层的节点数
hidden_size = 256
# LSTM layer 的层数
layer_num = 1
# 最后输出分类类别数量，如果是回归预测的话应该是 1
class_num = 4

_X = tf.placeholder(tf.float32, [None, input_size * timestep_size])
y = tf.placeholder(tf.float32, [None, class_num])
keep_prob = tf.placeholder(tf.float32)

# 开始搭建 LSTM 模型
# 把784个点的字符信息还原成 28 * 28 的图片
# 下面几个步骤是实现 RNN / LSTM 的关键
####################################################################
# **步骤1：RNN 的输入shape = (batch_size, timestep_size, input_size)
X = tf.reshape(_X, [-1, input_size, timestep_size])


#


# **步骤4：调用 MultiRNNCell 来实现多层 LSTM
# mlstm_cell = rnn.MultiRNNCell([lstm_cell for i in range(layer_num)], state_is_tuple=True)
# a=tf.contrib.rnn.LSTMCell(hidden_size,state_is_tuple=True)
def get_lstm_cell():
    # **步骤2：定义一层 LSTM_cell，只需要说明 hidden_size, 它会自动匹配输入的 X 的维度
    lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
    # **步骤3：添加 dropout layer, 一般只设置 output_keep_prob
    lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
    return lstm_cell


mlstm_cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell() for i in range(layer_num)], state_is_tuple=True)

# **步骤5：用全零来初始化state
init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)


# **步骤6：方法一，调用 dynamic_rnn() 来让我们构建好的网络运行起来
# ** 当 time_major==False 时， outputs.shape = [batch_size, timestep_size, hidden_size]
# ** 所以，可以取 h_state = outputs[:, -1, :] 作为最后输出
# ** state.shape = [layer_num, 2, batch_size, hidden_size],
# ** 或者，可以取 h_state = state[-1][1] 作为最后输出
# ** 最后输出维度是 [batch_size, hidden_size]
# outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=X, initial_state=init_state, time_major=False)
# h_state = outputs[:, -1, :]  # 或者 h_state = state[-1][1]

# *************** 为了更好的理解 LSTM 工作原理，我们把上面 步骤6 中的函数自己来实现 ***************
# 通过查看文档你会发现， RNNCell 都提供了一个 __call__()函数（见最后附），我们可以用它来展开实现LSTM按时间步迭代。
# **步骤6：方法二，按时间步展开计算
def output():
    outputs = list()
    state = init_state
    with tf.variable_scope('RNN'):
        for timestep in range(timestep_size):
            if timestep > 0:
                tf.get_variable_scope().reuse_variables()
            # 这里的state保存了每一层 LSTM 的状态
            (cell_output, state) = mlstm_cell(X[:, timestep, :], state)
            outputs.append(cell_output)
    h_state = outputs[-1]

    # 设置 loss function 和 优化器，展开训练并完成测试
    # 上面 LSTM 部分的输出会是一个 [hidden_size] 的tensor，我们要分类的话，还需要接一个 softmax 层
    # 首先定义 softmax 的连接权重矩阵和偏置
    # out_W = tf.placeholder(tf.float32, [hidden_size, class_num], name='out_Weights')
    # out_bias = tf.placeholder(tf.float32, [class_num], name='out_bias')
    # 开始训练和测试
    W = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32)
    bias = tf.Variable(tf.constant(0.1, shape=[class_num]), dtype=tf.float32)
    y_pre = tf.nn.softmax(tf.matmul(h_state, W) + bias)
    return y_pre


def train():
    y_pre = output()
    # 损失和评估函数
    cross_entropy = -tf.reduce_mean(y * tf.log(y_pre))
    train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print('begin:\n')
    sess.run(tf.global_variables_initializer())
    # saver = tf.train.Saver()
    # tmp = tf.train.latest_checkpoint('model/')
    # saver.restore(sess, tmp)
    for i in range(1000):

        batch_x, batch_y = get_next_batch(batch_size)
        if (i + 1) % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={
                _X: batch_x, y: batch_y, keep_prob: 1.0})
            # 已经迭代完成的 epoch 数: mnist.train.epochs_completed
            print("step %d, training accuracy %g" % ((i + 1), train_accuracy))

        sess.run(train_op, feed_dict={_X: batch_x, y: batch_y, keep_prob: 0.5})

    saver = tf.train.Saver()
    saver.save(sess, "./model/crack_capcha.model6", global_step=1000)


def test():
    y_pre = output()
    img_path = r'D:\project\图像识别\image\chinese\test2'
    # 计算测试数据的准确率
    saver = tf.train.Saver()
    tmp = tf.train.latest_checkpoint('model/')
    saver.restore(sess, tmp)

    imgs=os.listdir(img_path)
    [print(img) for img in imgs]
    imgs=[get_img(os.path.join(img_path,img_name)) for img_name in imgs]
    imgs=np.reshape(imgs,(-1,256))
    predict = sess.run(y_pre, feed_dict={_X: imgs, keep_prob: 1.0})
    predict = sess.run(tf.argmax(predict, 1))

    print(predict)
    #print(real)


#train()
test()
