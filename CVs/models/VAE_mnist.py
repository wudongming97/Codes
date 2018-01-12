import tensorflow as tf

import models.VAE_base


class VAE_(models.VAE_base.VAE):
    def __init__(self, flags, X):
        super().__init__(flags, X)

    def _encoder(self, X, ru=False):
        with tf.variable_scope('mlp_encoder', reuse=ru):
            fc0 = tf.layers.flatten(X)
            fc1 = tf.layers.dense(fc0, 784, activation=tf.nn.elu, name='L1')
            fc2 = tf.layers.dense(fc1, 512, activation=tf.nn.elu, name='L2')
            fc3 = tf.layers.dense(fc2, 512, activation=tf.nn.elu, name='L3')

            mu = tf.layers.dense(fc3, self.flags.z_size, name='fc_mu')
            logvar = tf.layers.dense(fc3, self.flags.z_size, name='fc_logvar')

        return mu, logvar

    def _decoder(self, z, ru=False):
        with tf.variable_scope('mlp_decoder', reuse=ru):
            dc0 = tf.layers.dense(z, 32, activation=tf.nn.elu, name='D0')
            dc1 = tf.layers.dense(dc0, 512, activation=tf.nn.elu, name='D1')
            dc2 = tf.layers.dense(dc1, 512, activation=tf.nn.elu, name='D2')
            dc3 = tf.layers.dense(dc2, 784, name='D4')
            dc4 = tf.reshape(dc3, [-1, 28, 28, 1])
        return dc4
