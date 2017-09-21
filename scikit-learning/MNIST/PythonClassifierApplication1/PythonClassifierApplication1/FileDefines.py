#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 10:22:31 2017

@author: yinghuang
"""

import os
import fnmatch

CSV_PATH            = "D:/Development/hy_code/MachineLearning/Tensorflow/GameCharacters_Tensorflow/Samples/"
DATA_PATH           = "D:/Development/hy_code/MachineLearning/Tensorflow/GameCharacters_Tensorflow/Samples/"
TF_RECORDS_PATH     = "D:/Development/hy_code/MachineLearning/Tensorflow/GameCharacters_Tensorflow/Samples/"
RUN_TF_RECORDS_PATH = "D:/Development/hy_code/MachineLearning/Tensorflow/GameCharacters_Tensorflow/Samples_CoolRunning/"
MODELS_PATH         = "D:/Development/hy_code/MachineLearning/Tensorflow/GameCharacters_Tensorflow/Models/"
TEST_TF_FILE        = TF_RECORDS_PATH + "test*.tfrecords"
TRAIN_TF_FILE       = TF_RECORDS_PATH + "train*.tfrecords"
RUN_TEST_TF_FILE    = RUN_TF_RECORDS_PATH + "test*.tfrecords"
RUN_TRAIN_TF_FILE   = RUN_TF_RECORDS_PATH + "train*.tfrecords"
TEST_CSV_FILE       = CSV_PATH+"test.csv"
TRAIN_CSV_FILE      = CSV_PATH+"train.csv"
MODEL_NAME          = "Num.pb"
FREEZE_MODEL_NAME   = "ModelsNum.pb"

ROWS  = 28
COLS  = 28
DEPTH = 1

# Training for Poker, Poker has 82921 samples in training set and 28767 samples in testing set.
POKER_TRAIN_SAMPLES = 82921
POKER_TEST_SAMPLES  = 28767

# Training for CoolRunning, CoolRunning has 32342 samples in training set
RUN_TRAIN_SAMPLES = 32342
RUN_TEST_SAMPLES  = 7362

def IterFindFiles(path, fnexp):
    for root, dirs, files in os.walk(path):
        for filename in fnmatch.filter(files, fnexp):
            yield os.path.join(root, filename)

def FindTFRecordsFiles(path, file):
    data = []
    fileFilter = file + "*" + ".tfrecords"
    for filename in IterFindFiles(path, fileFilter) :
        print (filename)
        data.append(filename)
    return data

#统计某文件夹下文件的个数　
#ls -l |grep "^-"|wc -l
#
#统计某文件夹下目录的个数　　
#ls -l |grep "^ｄ"|wc -l
#
#统计文件夹下文件的个数，包括子文件夹里的　　
#ls -lR|grep "^-"|wc -l