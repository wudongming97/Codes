from functools import partial

import tensorflow as tf

# global varible
is_training = tf.placeholder(tf.bool, name='is_training')
global_step = tf.train.get_or_create_global_step()

## ops alias
relu = tf.nn.relu
lrelu = tf.nn.leaky_relu
sigmoid = tf.nn.sigmoid

## layers alias
dense = tf.layers.dense
flatten = tf.layers.flatten
conv2d = tf.layers.conv2d
dconv2d = tf.layers.conv2d_transpose
bn = partial(tf.layers.batch_normalization, training=is_training)


def residual(name, x):
    '''
    original residual
    '''
    ci = x.get_shape().as_list()[3]
    with tf.variable_scope(name):
        net = conv2d(x, ci, [3, 3], [1, 1], 'SAME')
        net = relu(bn(net, training=is_training))
        net = conv2d(net, ci, [3, 3], [1, 1], 'SAME')
        net = bn(net, training=is_training)
    return relu(net + x)


def residual_pre(name, x):
    '''
    pre-activation residual
    '''
    ci = x.get_shape().as_list()[3]
    with tf.variable_scope(name):
        net = relu(bn(x, training=is_training))
        net = conv2d(net, ci, [3, 3], [1, 1], 'SAME')
        net = relu(bn(net, training=is_training))
        net = conv2d(net, ci, [3, 3], [1, 1], 'SAME')
    return x + net
