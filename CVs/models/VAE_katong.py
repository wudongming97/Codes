import tensorflow as tf

import models.VAE_base


class VAE_(models.VAE_base.VAE):
    def __init__(self, flags, X):
        super().__init__(flags, X)

    def _encoder(self, X, ru=False):
        with tf.variable_scope('cnn_encoder', reuse=ru):
            c1 = self._conv_with_bn(X, 128, 3, 2, 'SAME', name='L1')
            c2 = tf.layers.conv2d(c1, 64, 3, 2, 'SAME', name='L2')
            c3 = tf.layers.conv2d(c2, 32, 3, 2, 'SAME', name='L3')
            c4 = tf.layers.conv2d(c3, 8, 3, 2, 'SAME', name='L4')
            fc0 = tf.reshape(c4, [-1, 6 * 6 * 8])

            mu = tf.layers.dense(fc0, self.flags.z_size, name='fc_mu')
            logvar = tf.layers.dense(fc0, self.flags.z_size, name='fc_logvar')

        return mu, logvar

    def _decoder(self, z, ru=False):
        with tf.variable_scope('cnn_decoder', reuse=ru):
            df1 = tf.layers.dense(z, 6 * 6 * 128, name='df1')
            d0 = tf.reshape(df1, [-1, 6, 6, 128])
            d1 = tf.layers.conv2d_transpose(d0, 128, 3, 2, 'SAME', name='L1')
            d2 = tf.layers.conv2d_transpose(d1, 64, 3, 2, 'SAME', name='L2')
            d3 = tf.layers.conv2d_transpose(d2, 64, 3, 2, 'SAME', name='L3')
            d4 = self._dconv_with_bn(d3, 3, 3, 2, 'SAME', 'L4')
        return d4
