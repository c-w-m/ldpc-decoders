import tensorflow as tf

'''
Number of parameters in following model: 62,376,970
'''


def create_plh(with_data=True):
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    feed_dicts = {keep_prob:0.5}, {keep_prob:1.0}
    kwargs = {'keep_prob': keep_prob}
    if with_data:
        image = tf.placeholder(tf.float32, shape=[None, 227, 227, 3])
        label = tf.placeholder(tf.float32, shape=[None])
        return (image, label), kwargs, feed_dicts
    else:
        return kwargs, feed_dicts


def create_model(images, keep_prob):
    # https://github.com/gholomia/AlexNet-Tensorflow/blob/master/src/alexnet.py
    # https://www.learnopencv.com/wp-content/uploads/2018/05/AlexNet-1.png
    # consists of 62376968 parameters
    # counted using: https://stackoverflow.com/questions/38160940/how-to-count-total-number-of-trainable-parameters-in-a-tensorflow-model

    layers = tf.layers
    vv = lambda shape: tf.Variable(lambda: tf.truncated_normal(shape=shape, mean=0, stddev=0.08))

    # Convolution Layer 1 | Response Normalization | Max Pooling | ReLU
    c_layer_1 = tf.nn.conv2d(images, vv([11,11,3,96]), strides=[1, 4, 4, 1], padding="VALID", name="c_layer_1")
    c_layer_1 = tf.nn.relu(c_layer_1)
    # c_layer_1 = tf.nn.lrn(c_layer_1, depth_radius=N_DEPTH_RADIUS, bias=K_BIAS, alpha=ALPHA, beta=BETA)
    c_layer_1 = tf.nn.max_pool(c_layer_1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

    # Convolution Layer 2 | Response Normalization | Max Pooling | ReLU
    c_layer_2 = tf.nn.conv2d(c_layer_1, vv([5,5,96,256]), strides=[1, 1, 1, 1], padding="SAME", name="c_layer_2")
    c_layer_2 = tf.nn.relu(c_layer_2)
    # c_layer_2 = tf.nn.lrn(c_layer_2, depth_radius=5, bias=K_BIAS, alpha=ALPHA, beta=BETA)
    c_layer_2 = tf.nn.max_pool(c_layer_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

    # Convolution Layer 3 | ReLU
    c_layer_3 = tf.nn.conv2d(c_layer_2, vv([3,3,256,384]), strides=[1, 1, 1, 1], padding="SAME", name="c_layer_3")
    c_layer_3 = tf.nn.relu(c_layer_3)

    # Convolution Layer 4 | ReLU
    c_layer_4 = tf.nn.conv2d(c_layer_3, vv([3,3,384,384]), strides=[1, 1, 1, 1], padding="SAME", name="c_layer_4")
    c_layer_4 = tf.nn.relu(c_layer_4)

    # Convolution Layer 5 | ReLU | Max Pooling
    c_layer_5 = tf.nn.conv2d(c_layer_4, vv([3,3,384,256]), strides=[1, 1, 1, 1], padding="SAME", name="c_layer_5")
    c_layer_5 = tf.nn.relu(c_layer_5)
    c_layer_5 = tf.nn.max_pool(c_layer_5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

    # Flatten the multi-dimensional outputs to feed fully connected layers
    # feature_map = tf.reshape(c_layer_5, [-1, 13*13*256], name="myreshape")
    feature_map = tf.contrib.layers.flatten(c_layer_5)

    fc = tf.contrib.layers.fully_connected
    # Fully Connected Layer 1 | Dropout
    fc_layer_1 = fc(inputs=feature_map, num_outputs=4096, activation_fn=tf.nn.relu)
    fc_layer_1 = tf.nn.dropout(fc_layer_1, keep_prob)

    # Fully Connected Layer 2 | Dropout
    fc_layer_2 = fc(inputs=fc_layer_1, num_outputs=4096, activation_fn=tf.nn.relu)
    fc_layer_2 = tf.nn.dropout(fc_layer_2, keep_prob=keep_prob)

    # Fully Connected Layer 3 | Softmax
    fc_layer_3 = fc(inputs=fc_layer_2, num_outputs=1000, activation_fn=None)
    # cnn_output = tf.nn.softmax(fc_layer_3)
    return fc_layer_3
