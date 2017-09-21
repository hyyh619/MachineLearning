#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 18:40:15 2017

@author: yinghuang
"""
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import FileDefines as fd
import CNN as net
import mnist_inference as mn

# Super Parameters for CNN
MIN_DEQ              = 10000
BATCH_SIZE           = 100
CAPACITY             = MIN_DEQ + 3 * BATCH_SIZE
LEARNING_RATE_BASE   = 0.8
LEARNING_RATE_DECAY  = 0.99
REGULARIZATION_RATE  = 0.0001
EPOCH_NUM            = 10
TRAIN_SAMPLES        = 82921
TRAINING_STEPS       = TRAIN_SAMPLES * EPOCH_NUM / BATCH_SIZE
MOVING_AVERAGE_DECAY = 0.99
CHECK_NUM            = 1000

def GetTFRecords(tfrecords):
    files = tf.train.match_filenames_once(tfrecords)
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
    return features

def Training(train_tfrecords, test_tfrecords):
    train_features = GetTFRecords(train_tfrecords)

    samples = train_features['image_raw']
    decoded_images  = tf.decode_raw(samples, tf.uint8)
    retyped_images = tf.cast(decoded_images, tf.float32)
    labels  = train_features['label']

    images  = tf.reshape(retyped_images, [fd.ROWS, fd.COLS, fd.DEPTH])
    #images = retyped_images
    example_batch, label_batch = tf.train.shuffle_batch(
            [images, labels], batch_size = BATCH_SIZE, capacity = CAPACITY, min_after_dequeue = MIN_DEQ)

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mn.inference(example_batch, regularizer)
    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=label_batch)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        TRAIN_SAMPLES / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

#    saver = tf.train.Saver()
#    with tf.Session() as sess:
#        tf.global_variables_initializer().run()
#
#        for i in range(TRAINING_STEPS):
#            _, loss_value, step = sess.run(train_step)
#            if i % CHECK_NUM == 0:
#                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
#                saver.save(sess, os.path.join(fd.MODELS_PATH, fd.MODEL_NAME, str(i)), global_step=global_step)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 循环的训练神经网络。
        saver = tf.train.Saver()

        for i in range(TRAINING_STEPS):
            sess.run(train_op)

            if i > 0 and i % 100 == 0:
                print("After %d training step(s), loss is %g " % (i, sess.run(loss)))
            if i > 0 and i % CHECK_NUM == 0:
                print("After %d training step(s), loss on training batch is %g." % (i, sess.run(loss)))
                modelName = fd.MODELS_PATH + fd.MODEL_NAME + str(i) + ".model"
                saver.save(sess, modelName, global_step=global_step)

        coord.request_stop()
        coord.join(threads)

#    with tf.Session() as sess:
#        tf.global_variables_initializer().run()
#
#        coord = tf.train.Coordinator()
#        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#
#        for i in range(2):
#            # print sess.run(files)
#            # print sess.run([train_features['height'], train_features['label']])
#            # image, label, height, width, depth = sess.run([images, labels, heights, widths, depths])
#            # print image
#            # print image.shape
#            # print label
#            # print depth
#            # img = tf.reshape(image, [width, height, 1])
#
#            cur_images, cur_labels = sess.run([example_batch, label_batch])
#            # print cur_images, cur_labels
#            print cur_labels
#            # print "test"
#
#        coord.request_stop()
#        coord.join(threads)
    return

def main(argv=None):
    Training(fd.TRAIN_TF_FILE, fd.TEST_TF_FILE)
    return

if __name__ == '__main__':
    tf.app.run()