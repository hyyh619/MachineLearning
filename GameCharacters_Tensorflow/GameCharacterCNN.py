import tensorflow as tf

INPUT_NODE  = 784
OUTPUT_NODE = 10

IMAGE_SIZE   = 28
NUM_CHANNELS = 1
NUM_LABELS   = 10

#######################################################
# Cool Running:
# best cross_entropy: 0.1 with regularizer
#CONV1_DEEP = 10
#CONV1_SIZE = 5
#
#CONV2_DEEP = 20
#CONV2_SIZE = 5
#
#FC_SIZE = 256

# best loss: 0.00048936 without regularizer
#CONV1_DEEP = 10
#CONV1_SIZE = 5
#
#CONV2_DEEP = 20
#CONV2_SIZE = 5
#
#FC_SIZE = 256

# best loss: 0.000565655 without regularizer
#CONV1_DEEP = 32
#CONV1_SIZE = 5
#
#CONV2_DEEP = 64
#CONV2_SIZE = 5
#
#FC_SIZE = 512
#######################################################

#######################################################
# Poker
# best cross_entropy: 0.000167231 without regularizer
#CONV1_DEEP = 32
#CONV1_SIZE = 5
#
#CONV2_DEEP = 64
#CONV2_SIZE = 5
#
#FC_SIZE = 512
#######################################################

CONV1_DEEP = 32
CONV1_SIZE = 5

CONV2_DEEP = 64
CONV2_SIZE = 5

FC_SIZE = 512

def inference(input_tensor, train, regularizer, reuse = False):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    with tf.name_scope('input_layer'):
        input_layer = tf.reshape(input_tensor, [-1, 28, 28, 1], 'input')
    # input_layer = tf.placeholder(tf.float32, name='input', shape=[64, 28, 28, 1])
#
#    with tf.variable_scope('layer1-conv1-classifier', reuse = reuse):
#        conv1_weights = tf.get_variable(
#            "weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
#            initializer=tf.truncated_normal_initializer(stddev=0.1))
#        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
#        conv1 = tf.nn.conv2d(input_classifier, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
#        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.variable_scope('layer1-conv1', reuse = reuse):
        conv1_weights = tf.get_variable(
            "weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1)) 
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_layer, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")

    with tf.variable_scope("layer3-conv2", reuse = reuse):
        conv2_weights = tf.get_variable(
            "weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        pool_shape = pool2.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(pool2, [-1, nodes])

    with tf.variable_scope('layer5-fc1', reuse = reuse):
        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: 
            tf.add_to_collection('losses', regularizer(fc1_weights))
            
        fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train: 
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer6-fc2', reuse = reuse):
        fc2_weights = tf.get_variable("weight", [FC_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: 
            tf.add_to_collection('losses', regularizer(fc2_weights))
            
        fc2_biases = tf.get_variable("bias", [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(logit, name='predictions')

    return logit, prediction