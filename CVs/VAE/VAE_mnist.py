import tensorflow as tf

import VAE_base


class VAE_(VAE_base.VAE):
    def __init__(self, flags, X):
        self.initializer = tf.contrib.layers.xavier_initializer()
        super().__init__(flags, X)

    def _encoder(self, X, ru=False):
        with tf.variable_scope('mlp_encoder', reuse=ru):
            fc0 = tf.layers.flatten(X)
            fc1 = tf.layers.dense(fc0, 784, activation=tf.nn.elu, kernel_initializer=self.initializer, name='L1')
            fc2 = tf.layers.dense(fc1, 256, activation=tf.nn.elu, kernel_initializer=self.initializer, name='L2')
            # fc3 = tf.layers.dense(fc2, 512, activation=tf.nn.elu, kernel_initializer=self.initializer, name='L3')

            mu = tf.layers.dense(fc2, self.flags.z_size, kernel_initializer=self.initializer, name='fc_mu')
            logvar = tf.layers.dense(fc2, self.flags.z_size, kernel_initializer=self.initializer, name='fc_logvar')

        return mu, logvar

    def _decoder(self, z, ru=False):
        with tf.variable_scope('mlp_decoder', reuse=ru):
            dc1 = tf.layers.dense(z, 256, activation=tf.nn.elu, kernel_initializer=self.initializer, name='D1')
            # dc2 = tf.layers.dense(dc1, 512, activation=tf.nn.elu, kernel_initializer=self.initializer, name='D2')
            dc3 = tf.layers.dense(dc1, 784, activation=tf.nn.sigmoid, kernel_initializer=self.initializer, name='D4')
            dc4 = tf.reshape(dc3, [-1, 28, 28, 1])
        return dc4
