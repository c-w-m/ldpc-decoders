'''
code taken from: https://github.com/arberzela/ImageNet32-tensorflow/blob/master/WRN_main.py
TF implementation of results in: https://arxiv.org/pdf/1707.08819.pdf

Number of parameters in following model: imagenet32 - 1,595,320
'''

import tensorflow as tf


_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-4


def create_plh(with_data=True):
    is_training = tf.placeholder(tf.bool, name='is_training')
    feed_dicts = {is_training:True}, {is_training:False}
    kwargs = {'is_training': is_training}
    return kwargs, feed_dicts


def batch_norm_relu(inputs, is_training, data_format):
  """Performs a batch normalization followed by a ReLU."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  inputs = tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=is_training, fused=True)
  inputs = tf.nn.relu(inputs)
  return inputs


def fixed_padding(inputs, kernel_size, data_format):
  """Pads the input along the spatial dimensions independently of input size.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
  return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding='SAME', use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(scale=2.0, distribution='normal'),
      data_format=data_format)


def building_block(inputs, filters, is_training, projection_shortcut, strides,
                   dropoutrate, data_format):
  """Standard building block for residual networks with BN before convolutions.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    is_training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    The output tensor of the block.
  """
  shortcut = inputs
  inputs = batch_norm_relu(inputs, is_training, data_format)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)

  inputs = batch_norm_relu(inputs, is_training, data_format)
  inputs = tf.layers.dropout(inputs=inputs, rate=dropoutrate, training=is_training)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=1,
      data_format=data_format)

  return inputs + shortcut


def bottleneck_block(inputs, filters, is_training, projection_shortcut,
                     strides, data_format):
  """Bottleneck block variant for residual networks with BN before convolutions.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first two convolutions. Note that the
      third and final convolution will use 4 times as many filters.
    is_training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    The output tensor of the block.
  """
  shortcut = inputs
  inputs = batch_norm_relu(inputs, is_training, data_format)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=1, strides=1,
      data_format=data_format)

  inputs = batch_norm_relu(inputs, is_training, data_format)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)

  inputs = batch_norm_relu(inputs, is_training, data_format)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
      data_format=data_format)

  return inputs + shortcut


def block_group(inputs, filters, block_fn, blocks, strides, dropoutrate, is_training, name,
                data_format):
  """Creates one layer of blocks for the ResNet model.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first convolution of the layer.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    is_training: Either True or False, whether we are currently training the
      model. Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    The output tensor of the block layer.
  """
  # Bottleneck blocks end with 4x the number of filters as they start with
  filters_out = 4 * filters if block_fn is bottleneck_block else filters

  def projection_shortcut(inputs):
    return tf.layers.conv2d(
      inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
      padding='SAME', use_bias=False,
      kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
      data_format=data_format)

  # Only the first block per block_layer uses projection_shortcut and strides
  inputs = block_fn(inputs, filters, is_training, projection_shortcut, strides,
                    dropoutrate, data_format)

  for _ in range(1, blocks):
    inputs = block_fn(inputs, filters, is_training, None, 1,
                      dropoutrate, data_format)

  return tf.identity(inputs, name)


# ##################### Build the neural network model #######################


def create_model(inputs, is_training, depth=28, k=2, num_classes=1000, dropoutrate=0):
    """Constructs the ResNet model given the inputs."""

    img_size = inputs.shape[1]
    num_blocks = (depth - 4) // 6
    if depth % 6 != 4: raise ValueError('depth must be 6n + 4:', depth)

    # https://stackoverflow.com/questions/41651628/negative-dimension-size-caused-by-subtracting-3-from-1-for-conv2d
    data_format = 'channels_last' #('channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=16, kernel_size=3, strides=1,
        data_format=data_format)
    inputs = tf.identity(inputs, 'initial_conv')

    num_filters = int(16*k)
    inputs = block_group(
        inputs=inputs, filters=num_filters, block_fn=building_block, blocks=num_blocks,
        strides=1, dropoutrate=dropoutrate, is_training=is_training, name='block_layer1',
        data_format=data_format)

    if img_size >= 16:
        num_filters = int(32*k)
        inputs = block_group(
            inputs=inputs, filters=num_filters, block_fn=building_block, blocks=num_blocks,
            strides=2, dropoutrate=dropoutrate, is_training=is_training, name='block_layer2',
            data_format=data_format)

    if img_size >= 32:
        num_filters = int(64*k)
        inputs = block_group(
            inputs=inputs, filters=num_filters, block_fn=building_block, blocks=num_blocks,
            strides=2, dropoutrate=dropoutrate, is_training=is_training, name='block_layer3',
            data_format=data_format)

    if img_size >= 64:
        num_filters = int(128*k)
        inputs = block_group(
            inputs=inputs, filters=num_filters, block_fn=building_block, blocks=num_blocks,
            strides=2, dropoutrate=dropoutrate, is_training=is_training, name='block_layer4',
            data_format=data_format)

    inputs = batch_norm_relu(inputs, is_training, data_format)
    inputs = tf.layers.average_pooling2d(
        inputs=inputs, pool_size=8, strides=1, padding='VALID',
        data_format=data_format)
    inputs = tf.identity(inputs, 'final_avg_pool')
    inputs = tf.reshape(inputs, [-1, num_filters])
    inputs = tf.layers.dense(inputs=inputs, units=num_classes)
    inputs = tf.identity(inputs, 'final_dense')

    logits = inputs
    return logits #, [v for v in tf.trainable_variables()]
