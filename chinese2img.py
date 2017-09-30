#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
File Name : 'chinese2img'.py 
Description:
Author: 'zhengyang' 
Date: '2017/9/26' '18:11'
"""

import os
import sys
import json
import hashlib
import time

from PIL import Image, ImageDraw, ImageFont

with open("common_hanzi_1.json", "r") as f:
    common_hanzi_json = f.read()

#common_hanzi_dict = json.loads(common_hanzi_json)


# font = ImageFont.truetype('C:\Windows\Fonts\msyh.ttf', 13)
def md5_encrypt(text):
    m=hashlib.md5()
    m.update(text.encode('utf-8'))
    return m.hexdigest()

def get_img(hanzi, img_path, font, start_x, start_y, wide, font_size, high=16):

    #assert font_size <= max(wide, high)
    assert start_x < wide / 3
    assert start_y < high / 3
    font = ImageFont.truetype(font, font_size)

    im = Image.new("L", (16, wide), 255)

    draw = ImageDraw.Draw(im)
    # print "start_x:{} start_y:{} font:{}".format(start_x, start_y, font)
    draw.text((start_x, start_y), hanzi, font=font)

    # del draw
    # im.show()
    im.save(img_path, "PNG")


def main():
    common_hanzi_dict={0:'捷',1:'信',2:'金',3:'融'}
    for i, (hanzi_id, hanzi) in enumerate(common_hanzi_dict.items()):
        img_dir = r"D:\project\图像识别\image\chinese\train3"
        if os.path.exists(img_dir):
            pass
        else:
            os.makedirs(img_dir)

        j = 1
        font_list = ['simsun.ttc', 'simhei.ttf']
        for font in font_list:
            for start_x in (0, 1):
                for start_y in (0, 1):
                    for wide in (16, 17):
                        for font_size in (16, 17):
                            m = md5_encrypt(str(time.time()))
                            img_path = os.path.join(img_dir, r'{}_{}.png'.format(m,hanzi_id))
                            get_img(hanzi, img_path, font, start_x, start_y, wide, font_size)
                            j += 1


if __name__ == "__main__":
    main()

