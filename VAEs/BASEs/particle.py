import tensorflow as tf


class encoder:
    def __init__(self):
        self.name = 'mnist/encoder'

    def __call__(self, x):
        with tf.variable_scope(self.name):
            conv1 = tf.layers.conv2d(x, 128, [4, 4], [2, 2], 'SAME', activation=tf.nn.leaky_relu)
            conv2 = tf.layers.conv2d(conv1, 32, [4, 4], [2, 2], 'SAME', activation=tf.nn.leaky_relu)
            fc0 = tf.layers.flatten(conv2)
            latent_variable = tf.layers.dense(fc0, 64, activation=tf.nn.leaky_relu)

            return latent_variable

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class decoder:
    def __init__(self):
        self.name = 'mnist/decoder'

    def __call__(self, z, reuse=True):
        with tf.variable_scope(self.name, reuse=reuse):
            fc0 = tf.layers.dense(z, 7 * 7 * 32, activation=tf.nn.leaky_relu)
            fc0 = tf.reshape(fc0, [-1, 7, 7, 32])
            dconv1 = tf.layers.conv2d_transpose(fc0, 128, [4, 4], [2, 2], 'SAME', activation=tf.nn.leaky_relu)
            dconv2 = tf.layers.conv2d_transpose(dconv1, 32, [4, 4], [2, 2], 'SAME', activation=tf.nn.leaky_relu)
            logits = tf.layers.conv2d_transpose(dconv2, 1, [4, 4], [1, 1], 'SAME')
            out = tf.nn.sigmoid(logits)

            return out, logits

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

