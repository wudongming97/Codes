import tensorflow as tf


def bn(x, is_training):
    return tf.layers.batch_normalization(x, training=is_training)

