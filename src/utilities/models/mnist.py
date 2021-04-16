import tensorflow as tf

'''
Number of parameters in following model: 3,274,634
'''

def create_plh():
    placeholders = tf.placeholder(tf.float32, [None, 28, 28], name='image'),\
                   tf.placeholder(tf.float32, [None], name='label')
    return placeholders


def create_conv(feature):
    # Reshape feature to 4d tensor with 2nd and 3rd dimensions being
    # image width and height final dimension being the number of color channels.
    layers = tf.layers
    feature = tf.reshape(feature, [-1, 28, 28, 1])

    # First conv layer will compute 32 features for each 5x5 patch
    with tf.compat.v1.variable_scope('conv_layer1'):
        h_conv1 = layers.conv2d(feature, 32, kernel_size=[5, 5],
                                activation=tf.nn.relu, padding="SAME")
        h_pool1 = tf.nn.max_pool(
            h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Second conv layer will compute 64 features for each 5x5 patch.
    with tf.compat.v1.variable_scope('conv_layer2'):
        h_conv2 = layers.conv2d(h_pool1, 64, kernel_size=[5, 5],
                                activation=tf.nn.relu, padding="SAME")
        h_pool2 = tf.nn.max_pool(
            h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # reshape tensor into a batch of vectors
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

    # Densely connected layer with 1024 neurons.
    h_fc1 = layers.dense(h_pool2_flat, 1024, activation=tf.nn.relu)#layers.dropout(#,
        #rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Compute logits (1 per class) and compute loss.
    logits = layers.dense(h_fc1, 10, activation=None)
    return logits
