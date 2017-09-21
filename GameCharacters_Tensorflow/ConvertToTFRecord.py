#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 17:02:26 2017

@author: yinghuang
"""

import tensorflow as tf
from PIL import Image
import FileDefines as fd

def encode_label(label):
    return int(label)

def read_label_file(file):
    f = open(file, "r")
    filepaths = []
    labels = []

    for line in f:
        filepath, label = line.split(",")
        filepaths.append(filepath)
        labels.append(encode_label(label))

    return filepaths, labels

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def WriteTFRecords(path, prefixName, files, labels):
    i = 0
    j = 0

    for imgFile, label in zip(files, labels):
        if i == 0 :
            fileName = path + prefixName + str(j) + '.tfrecords'
            writer = tf.python_io.TFRecordWriter(fileName)

        i = i+1

        img = Image.open(imgFile)
        h, w = img.size[:2]
        d = img.getbands()
        d = len(d)
        img_raw = img.tostring()

        if (h == 28 and w == 28 and d == 1) :
            example = tf.train.Example(features = tf.train.Features(feature = {
                                        'height': _int64_feature(h),
                                        'width': _int64_feature(w),
                                        'depth': _int64_feature(d),
                                        'image_raw': _bytes_feature(img_raw),
                                        'label':_int64_feature(label)}))
            writer.write(example.SerializeToString())
        else :
            print imgFile

        if i == 10000 :
            i = 0
            writer.close()
            j = j+1

    writer.close()

def convert_to(data_path, train_files, test_files, train_labels, test_labels):

    """
    Converts s dataset to tfrecords
    """

    WriteTFRecords(data_path, 'test', test_files, test_labels)
    WriteTFRecords(data_path, 'train', train_files, train_labels)

def main(argv=None):
    # reading labels and file path
    train_filepaths, train_labels = read_label_file(fd.TRAIN_CSV_FILE)
    test_filepaths, test_labels = read_label_file(fd.TEST_CSV_FILE)
    convert_to(fd.DATA_PATH, train_filepaths, test_filepaths, train_labels, test_labels)


if __name__ == '__main__':
    main()