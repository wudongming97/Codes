import tensorflow as tf


class D(object):
    def __init__(self):
        self.name = 'mnist/d_net'

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name, reuse=reuse):
            conv1 = tf.layers.conv2d(x, 128, [4, 4], [2, 2], activation=tf.nn.leaky_relu)
            conv2 = tf.layers.conv2d(conv1, 64, [4, 4], [2, 2], activation=tf.nn.leaky_relu)
            fc1 = tf.layers.dense(tf.layers.flatten(conv2), 512, activation=tf.nn.leaky_relu)
            fc2 = tf.layers.dense(fc1, 1)

            return fc2

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class G(object):
    def __init__(self):
        self.name = 'mnist/g_net'

    def __call__(self, z):
        with tf.variable_scope(self.name):
            bs = tf.shape(z)[0]
            fc1 = tf.layers.dense(z, 7 * 7 * 128, activation=tf.nn.relu)
            conv1 = tf.reshape(fc1, [bs, 7, 7, 128])
            conv1 = tf.layers.conv2d_transpose(conv1, 128, [4, 4], [2, 2], 'SAME', activation=tf.nn.relu)
            conv2 = tf.layers.conv2d_transpose(conv1, 32, [4, 4], [2, 2], 'SAME', activation=tf.nn.relu)
            fake = tf.layers.conv2d_transpose(conv2, 1, [4, 4], [1, 1], 'SAME', activation=tf.nn.sigmoid)
            return fake

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]
