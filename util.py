# -*- coding:utf-8 -*-
"""
File Name: util
Version:
Description:
Author: liuxuewen
Date: 2017/9/20 18:06
"""
import numpy as np
import string
import os
import re
from PIL import Image


IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
LEN_CAPTCHA = 4
CHARS = string.digits+string.ascii_letters
LEN_CHAR_SET = len(CHARS)

img_path = r'D:\project\图像识别\image'
imgs_train = os.listdir(r'{}\train'.format(img_path))
#imgs_train=os.listdir(r'D:\project\图像识别\image\tmp')
L=len(imgs_train)
print(L)

# 把彩色图像转为灰度图像（色彩对识别验证码没有什么用）
def convert2gray(img):
    if len(img.shape) > 2:
        #print(img.shape)
        gray = np.mean(img, -1)
        return gray
    else:
        return img


# 文本转向量
def text2vec(text):
    v = np.zeros(len(CHARS) * LEN_CAPTCHA)
    for i, num in enumerate(text):
        index = i * LEN_CHAR_SET + CHARS.index(num)
        v[index] = 1
    return v


# 向量转文本
def vec2text(vec):
    text = list()
    for i, j in enumerate(vec):
        if j == 1:
            index = i % 10
            char = CHARS[index]
            text.append(char)
    return ''.join(text)


# 生成一个训练batchv  一个批次为 默认100 张图片 转换为向量
def get_next_batch(batch_size=100,batch_id=0):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, LEN_CAPTCHA * LEN_CHAR_SET])

    if (batch_id+1)*batch_size>L:
        return None,None
    imgs_name=imgs_train[batch_id*batch_size:(batch_id+1)*batch_size]
    for i,img_name in enumerate(imgs_name):
        #img_name = imgs_train[0]
        text = re.findall('_(\d{4})\.png', img_name)[0]
       # print(text)
        # 获取图片，并灰度转换
        img = Image.open(r'{}\train\{}'.format(img_path, img_name))
        img=img.resize((IMAGE_WIDTH,IMAGE_HEIGHT),Image.ANTIALIAS)# w代表宽度，h代表高度，最后一个参数指定采用的算法
        img = np.array(img)
        img = convert2gray(img)

        batch_x[i, :] = img.flatten() / 255
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y

def get_img(img_path):
    img=Image.open(img_path)
    img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)  # w代表宽度，h代表高度，最后一个参数指定采用的算法
    img = np.array(img)
    img = convert2gray(img)
    img=img.flatten() / 255
    return np.reshape(img,(1,img.size))


if __name__ == '__main__':
    x = np.zeros([1, IMAGE_HEIGHT * IMAGE_WIDTH])
    print(len(imgs_train))