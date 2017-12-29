import tensorflow as tf

import models.VAE


class VAE_(models.VAE.VAE):
    def __init__(self, flags, X):
        super().__init__(flags, X)

    def _encoder(self, X, ru=False):
        with tf.variable_scope('cnn_encoder', reuse=ru):
            c1 = self._conv_with_bn(X, 64, 3, 2, 'SAME', name='L1')
            c2 = tf.layers.conv2d(c1, 32, 3, 2, 'SAME', name='L2')
            c4 = tf.layers.conv2d(c2, 16, 3, 2, 'SAME', name='L3')
            fc0 = tf.reshape(c4, [-1, 4 * 4 * 16])

            mu = tf.layers.dense(fc0, self.flags.z_size, name='fc_mu')
            logvar = tf.layers.dense(fc0, self.flags.z_size, name='fc_logvar')

        return mu, logvar

    def _decoder(self, z, ru=False):
        with tf.variable_scope('cnn_decoder', reuse=ru):
            df1 = tf.layers.dense(z, 4 * 4 * 16, name='df1')
            d0 = tf.reshape(df1, [-1, 4, 4, 16])
            d1 = tf.layers.conv2d_transpose(d0, 64, 3, 2, 'SAME', name='L1')
            d2 = tf.layers.conv2d_transpose(d1, 32, 3, 2, 'SAME', name='L2')
            d3 = self._dconv_with_bn(d2, 3, 3, 2, 'SAME', 'L3')
        return d3
