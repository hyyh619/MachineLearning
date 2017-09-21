#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 15:37:56 2017

@author: hy
"""

#-*- coding:utf-8 -*-  
import argparse   
import tensorflow as tf
import FileDefines as fd

MIN_DEQ              = 10000
BATCH_SIZE           = 1
CAPACITY             = MIN_DEQ + 3 * BATCH_SIZE

# Get batch of samples and labels
def GetBatch(path, tfrecords):
    #files = tf.train.match_filenames_once(tfrecords)
    files = fd.FindTFRecordsFiles(path, tfrecords)
    filename_queue = tf.train.string_input_producer(files, shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
            serialized_example,
            features = {
                    'height': tf.FixedLenFeature([], tf.int64),
                    'width': tf.FixedLenFeature([], tf.int64),
                    'depth': tf.FixedLenFeature([], tf.int64),
                    'image_raw': tf.FixedLenFeature([], tf.string),
                    'label':tf.FixedLenFeature([], tf.int64)})
    
    samples = features['image_raw']
    decoded_images  = tf.decode_raw(samples, tf.uint8)
    retyped_images = tf.image.convert_image_dtype(decoded_images, dtype = tf.float32)
    labels  = features['label']

    images  = tf.reshape(retyped_images, [fd.ROWS, fd.COLS, fd.DEPTH])
    example_batch, label_batch = tf.train.shuffle_batch(
            [images, labels], batch_size = BATCH_SIZE, capacity = CAPACITY, min_after_dequeue = MIN_DEQ)
    example_batch = tf.reshape(example_batch, [-1, 28, 28, 1], 'input')
    return example_batch, label_batch
  
def load_graph(frozen_graph_filename):  
    # We parse the graph_def file  
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:  
        graph_def = tf.GraphDef()  
        graph_def.ParseFromString(f.read())  
  
    # We load the graph_def in the default graph  
    with tf.Graph().as_default() as graph:  
        tf.import_graph_def(  
            graph_def,   
            input_map=None,   
            return_elements=None,   
            name="prefix",   
            op_dict=None,   
            producer_op_list=None  
        )  
    return graph  
  
if __name__ == '__main__':  
    #加载已经将参数固化后的图
    x1 = tf.placeholder(tf.float32, name='input', shape=[None, 28, 28, 1])
    graph = load_graph(fd.FREEZE_MODEL_NAME)
    
    testSampleBatch, testLabelBatch = GetBatch(fd.RUN_TF_RECORDS_PATH, "test")
  
    # We can list operations  
    #op.values() gives you a list of tensors it produces  
    #op.name gives you the name  
    #输入,输出结点也是operation,所以,我们可以得到operation的名字  
    for op in graph.get_operations():
        print(op.name,op.values())

    # prefix/Placeholder/inputs_placeholder
    # prefix/Accuracy/predictions
    #操作有:prefix/Placeholder/inputs_placeholder
    #操作有:prefix/Accuracy/predictions
    #为了预测,我们需要找到我们需要feed的tensor,那么就需要该tensor的名字  
    #注意prefix/Placeholder/inputs_placeholder仅仅是操作的名字,prefix/Placeholder/inputs_placeholder:0才是tensor的名字

    x = graph.get_tensor_by_name('prefix/input_2:0')  
    y = graph.get_tensor_by_name('prefix/softmax/predictions:0')  
          
    with tf.Session(graph=graph) as sess:
        inputSamples, inputLabels = sess.run([testSampleBatch, testLabelBatch])
        y_out = sess.run(y, feed_dict={  
            x:inputSamples # < 45  
        })
        print(y_out) # [[ 0.]] Yay!  
    print ("finish")  