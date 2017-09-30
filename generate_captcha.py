# -*- coding:utf-8 -*-
"""
File Name: generate_captcha
Version:
Description:
Author: liuxuewen
Date: 2017/9/20 15:51
"""
import string
import random
from captcha.image import ImageCaptcha
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import time
import hashlib

#from util import CHARS

CHARS=string.digits

def md5_encrypt(text):
    m=hashlib.md5()
    m.update(text.encode('utf-8'))
    return m.hexdigest()

def get_text():
    chars=CHARS
    text=random.sample(chars,1)
    return ''.join(text)

def gen_captcha():
    for i in range(1000):
        m=md5_encrypt(str(time.time()))
        image=ImageCaptcha(width=64,height=64)
        text=get_text()
        image.write(text,r'D:\project\图像识别\image\train2\{}_{}.png'.format(m,text))



def fun():
    gen_captcha()

if __name__ == '__main__':
    fun()
