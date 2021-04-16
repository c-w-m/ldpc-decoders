import tensorflow as tf

'''
Number of parameters in following model: 4,607,178
'''

def create_plh(with_data=True):
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    flag_training = tf.placeholder(tf.bool, name='flag_training')
    # feed_train, feed_test
    feed_dicts = {keep_prob:0.7, flag_training:True}, {keep_prob:1.0, flag_training:False}
    kwargs = {'keep_prob':keep_prob, 'flag_training':flag_training}
    if with_data:
        x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='input_x')
        y = tf.placeholder(tf.float32, shape=[None], name='output_y')
        return (x, y), kwargs, feed_dicts
    else:
        return kwargs, feed_dicts


def create_conv10(x, keep_prob, flag_training):
    def initializer(shape): return (lambda: tf.truncated_normal(shape=shape, mean=0, stddev=0.08))
    conv1_filter = tf.Variable(initializer([3, 3, 3, 64]))
    conv2_filter = tf.Variable(initializer([3, 3, 64, 128]))
    conv3_filter = tf.Variable(initializer([5, 5, 128, 256]))
    conv4_filter = tf.Variable(initializer([5, 5, 256, 512]))

    # 1, 2
    conv1 = tf.nn.conv2d(x, conv1_filter, strides=[1,1,1,1], padding='SAME')
    conv1 = tf.nn.relu(conv1)
    conv1_pool = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    conv1_bn = tf.layers.batch_normalization(conv1_pool, training=flag_training)

    # 3, 4
    conv2 = tf.nn.conv2d(conv1_bn, conv2_filter, strides=[1,1,1,1], padding='SAME')
    conv2 = tf.nn.relu(conv2)
    conv2_pool = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    conv2_bn = tf.layers.batch_normalization(conv2_pool, training=flag_training)

    # 5, 6
    conv3 = tf.nn.conv2d(conv2_bn, conv3_filter, strides=[1,1,1,1], padding='SAME')
    conv3 = tf.nn.relu(conv3)
    conv3_pool = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    conv3_bn = tf.layers.batch_normalization(conv3_pool, training=flag_training)

    # 7, 8
    conv4 = tf.nn.conv2d(conv3_bn, conv4_filter, strides=[1,1,1,1], padding='SAME')
    conv4 = tf.nn.relu(conv4)
    conv4_pool = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    conv4_bn = tf.layers.batch_normalization(conv4_pool, training=flag_training)

    # 9
    flat = tf.contrib.layers.flatten(conv4_bn)

    # 10
    full1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=128, activation_fn=tf.nn.relu)
    full1 = tf.nn.dropout(full1, keep_prob)
    full1 = tf.layers.batch_normalization(full1, training=flag_training)

    # 11
    full2 = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=256, activation_fn=tf.nn.relu)
    full2 = tf.nn.dropout(full2, keep_prob)
    full2 = tf.layers.batch_normalization(full2, training=flag_training)

    # 12
    full3 = tf.contrib.layers.fully_connected(inputs=full2, num_outputs=512, activation_fn=tf.nn.relu)
    full3 = tf.nn.dropout(full3, keep_prob)
    full3 = tf.layers.batch_normalization(full3, training=flag_training)

    # # 13
    # full4 = tf.contrib.layers.fully_connected(inputs=full3, num_outputs=1024, activation_fn=tf.nn.relu)
    # full4 = tf.nn.dropout(full4, keep_prob)
    # full4 = tf.layers.batch_normalization(full4, training=flag_training)

    return tf.contrib.layers.fully_connected(inputs=full3, num_outputs=10, activation_fn=None)
