# https://github.com/jiamings/wgan/blob/master/mnist/dcgan.py
#
import tensorflow as tf


class Discriminator(object):
    def __init__(self):
        self.x_dim = 784
        self.name = 'mnist/dcgan/d_net'

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            conv1 = tf.layers.conv2d(x, 64, [4, 4], [2, 2], activation=tf.nn.leaky_relu)
            conv2 = tf.layers.conv2d(conv1, 128, [4, 4], [2, 2], activation=tf.nn.leaky_relu)
            fc1 = tf.layers.dense(tf.layers.flatten(conv2), 1024, activation=tf.nn.leaky_relu)
            fc2 = tf.layers.dense(fc1, 1)

            return fc2

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class Generator(object):
    def __init__(self):
        self.z_dim = 100
        self.imshape = [28, 28, 1]
        self.name = 'mnist/dcgan/g_net'

    def __call__(self, z):
        with tf.variable_scope(self.name):
            bs = tf.shape(z)[0]
            fc1 = tf.layers.dense(z, 7*7*128, activation=tf.nn.relu)
            conv1 = tf.reshape(fc1, [bs, 7, 7, 128])
            conv1 = tf.layers.conv2d_transpose(conv1, 64, [4, 4], [2, 2],
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                                               activation=tf.nn.relu)
            conv2 = tf.layers.conv2d_transpose(conv1, 1, [4, 4], [2, 2],
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                                               activation=tf.nn.relu)

            fc2 = tf.layers.dense(tf.layers.flatten(conv2), 784, activation=tf.nn.sigmoid)
            fake = tf.reshape(fc2, [bs] + self.imshape)
            return fake

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

