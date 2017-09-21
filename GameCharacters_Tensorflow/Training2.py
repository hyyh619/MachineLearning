#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 16:21:14 2017

@author: yinghuang
"""
import tensorflow as tf
import FileDefines as fd
import GameCharacterCNN2 as net
import time

# Super Parameters for CNN
MIN_DEQ              = 10000
BATCH_SIZE           = 64
CAPACITY             = MIN_DEQ + 3 * BATCH_SIZE
LEARNING_RATE_BASE   = 0.01
LEARNING_RATE_DECAY  = 0.99
REGULARIZATION_RATE  = 0.0001
EPOCH_NUM            = 10
MOVING_AVERAGE_DECAY = 0.999
CHECK_NUM            = 1000

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

def DefTrainingGraph(sampleBatch, labelBatch, useRegularizer, trainingSteps) :
    if useRegularizer:
        regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
        y, _ = net.inference(sampleBatch, True, regularizer)
    else :
        y, _ = net.inference(sampleBatch, True, None)
        
    globalStep = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, globalStep)
    variablesAveragesOp = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=labelBatch)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    if useRegularizer:
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    else :
        loss = cross_entropy_mean
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        globalStep,
        trainingSteps / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)
    trainStep = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=globalStep)
    
    with tf.control_dependencies([trainStep, variablesAveragesOp]):
        trainOp = tf.no_op(name='train')
    return trainOp, loss, globalStep, variablesAveragesOp

def DefTestGraph(sampleBatch, labelBatch) :
    y, _ = net.inference(sampleBatch, False, None, True)
    correctPred = tf.equal(tf.argmax(y, 1), labelBatch)
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
    return accuracy

def Training(train_tfrecords, test_tfrecords, useRegularizer, trainSampleNum, testSampleNum):
    xTrain = tf.placeholder(tf.float32, name='input', shape=[None, 28, 28, 1])
    xTest = tf.placeholder(tf.float32, name='input', shape=[None, 28, 28, 1])
    trainingSteps = int(trainSampleNum * EPOCH_NUM / BATCH_SIZE)
    testSteps = int(testSampleNum / BATCH_SIZE)
    
    trainSampleBatch, trainLabelBatch = GetBatch(train_tfrecords, "train")
    testSampleBatch, testLabelBatch = GetBatch(test_tfrecords, "test")

    trainOp, loss, globalStep, variablesAveragesOp = DefTrainingGraph(trainSampleBatch, trainLabelBatch, useRegularizer, trainingSteps)
    
    # Test
    accuracy = DefTestGraph(testSampleBatch, testLabelBatch)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 循环的训练神经网络。
        saver = tf.train.Saver()

        start_time = time.time()
        for i in range(trainingSteps):
            trainSamples, trainLabels = sess.run([trainSampleBatch, trainLabelBatch])
            sess.run(trainOp, feed_dict = {xTrain:trainSamples})

            if i % 100 == 0:
                duration = time.time() - start_time
                print("After %d training step(s), loss is %g, time: %.3f" % (i, sess.run(loss), duration))
                start_time = time.time()
                
            if i > 0 and i % CHECK_NUM == 0:
                print("After %d training step(s), loss on training batch is %g." % (i, sess.run(loss)))
                modelName = fd.MODELS_PATH + fd.MODEL_NAME + str(i) + ".model"
                saver.save(sess, modelName, global_step = globalStep)
                
                # Get the accuracy based on test set.
                testAccuracy = 0.0
                for i in range(testSteps):
                    testSamples, testLabels = sess.run([testSampleBatch, testLabelBatch])
                    testAccuracy = testAccuracy + sess.run(accuracy, feed_dict = {xTest:trainSamples})

                testAccuracy = testAccuracy / testSteps;
                print("test accuracy is %g." % (testAccuracy))

        coord.request_stop()
        coord.join(threads)
    return

def main(argv=None):
    # if use regularizer, the best loss is 0.1
    # if not using regularizer, the best loss is 0.00048936
    # Training(fd.TF_RECORDS_PATH, fd.TF_RECORDS_PATH, False, fd.POKER_TRAIN_SAMPLES, fd.POKER_TEST_SAMPLES
    # Training(fd.RUN_TF_RECORDS_PATH, fd.RUN_TF_RECORDS_PATH, False, fd.RUN_TRAIN_SAMPLES, fd.RUN_TEST_SAMPLES)
    
    # Test regular
    Training(fd.RUN_TF_RECORDS_PATH, fd.RUN_TF_RECORDS_PATH, True, fd.RUN_TRAIN_SAMPLES, fd.RUN_TEST_SAMPLES)

    # Just for test
    # Training(fd.RUN_TF_RECORDS_PATH, fd.RUN_TF_RECORDS_PATH, False, 10*BATCH_SIZE, fd.RUN_TEST_SAMPLES)
    return

if __name__ == '__main__':
    tf.app.run()