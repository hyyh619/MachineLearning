# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 10:47:08 2017

@author: benethuang
"""

from PIL import Image
import numpy as np
import pickle, glob, os
import re


def get_array(path, fileFilter) :
    arr_x = [[]]
    arr_y = []
    fileSearch = path + fileFilter
    for infile in glob.glob(fileSearch):
        file, ext = os.path.splitext(infile)
        filename = os.path.basename(infile)
        Img = Image.open(infile)
        width = Img.size[0]
        height = Img.size[1]
        print(filename, width, height)
        
    #    if width>height:
    #        tmp = int((width-height)/2)
    #        Img = Img.crop([width-height-tmp,0,width-tmp,height])
    #    if height>width:
    #        tmp = int((height-width)/2)
    #        Img = Img.crop([0,height-width-tmp,width,height-tmp])
    #    size = 32,32
    #    Img.thumbnail(size, Image.ANTIALIAS)
    
    # For RGB
    #    r,g,b = Img.split()
    #    r_array = np.array(r).reshape([1024])
    #    g_array = np.array(g).reshape([1024])
    #    b_array = np.array(b).reshape([1024])
    #    merge_array = np.concatenate((r_array,g_array,b_array))
    #    if arr == [[]]:
    #        arr = [merge_array]
    #        continue
    #    arr = np.concatenate((arr, [merge_array]),axis=0)
    
        # For gray image data
        im_array = np.array(Img)
        gray_array = np.array(im_array).reshape([3200])
        if arr_x == [[]]:
            arr_x = [gray_array]
        else :
            arr_x = np.concatenate((arr_x, [gray_array]),axis=0)
        
        # for labels
        res2 = re.findall(r"\d+", filename)
        arr_y.append(res2[1])
        
    return arr_x, arr_y
    
train_x, train_y = get_array("E:/Development/hy_code/drl_kupao/GameData/KingGlory/1280x720/HeroCharGraySamples/train/", "*.png")
test_x, test_y =  get_array("E:/Development/hy_code/drl_kupao/GameData/KingGlory/1280x720/HeroCharGraySamples/test/", "*.png")   

f = open('./HeroCharGraySamples.pkl','wb')
dic = {'train_x': train_x,
       'train_y': train_y,
       'test_x': test_x,
       'test_y': test_y}
pickle.dump(dic, f, -1)

print("finish")