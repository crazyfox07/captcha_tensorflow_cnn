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

from util import CHARS


def md5_encrypt(text):
    m=hashlib.md5()
    m.update(text.encode('utf-8'))
    return m.hexdigest()

def get_text():
    chars=CHARS
    text=random.sample(chars,4)
    return ''.join(text)

def gen_captcha():
    for i in range(100000):
        m=md5_encrypt(str(time.time()))
        image=ImageCaptcha(width=160,height=60)
        text=get_text()
        image.write(text,r'D:\project\图像识别\image\train\{}_{}.png'.format(m,text))



def fun():
    gen_captcha()

if __name__ == '__main__':
    fun()
