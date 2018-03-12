import tensorflow as tf

import VAE_base


class VAE_(VAE_base.VAE):
    def __init__(self, flags, X):
        super().__init__(flags, X)

    def _encoder(self, X, ru=False):
        with tf.variable_scope('cnn_encoder', reuse=ru):
            c1 = tf.layers.conv2d(X, 128, 3, 2, padding='SAME', name='L1')
            c2 = tf.layers.conv2d(c1, 64, 3, 2, padding='SAME', name='L2',
                                  activation=tf.nn.relu,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
            c4 = tf.layers.conv2d(c2, 16, 3, 2, padding='SAME', name='L3',
                                  activation=tf.nn.relu,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
            fc0 = tf.reshape(c4, [-1, 4 * 4 * 16])

            mu = tf.layers.dense(fc0, self.flags.z_size, name='fc_mu', activation=tf.nn.relu)
            logvar = tf.layers.dense(fc0, self.flags.z_size, name='fc_logvar', activation=tf.nn.relu)

        return mu, logvar

    def _decoder(self, z, ru=False):
        with tf.variable_scope('cnn_decoder', reuse=ru):
            df1 = tf.layers.dense(z, 4 * 4 * 64, name='df1', activation=tf.nn.relu)
            d0 = tf.reshape(df1, [-1, 4, 4, 64])
            d1 = tf.layers.conv2d_transpose(d0, 128, 3, 2, padding='SAME', name='L1',
                                            activation=tf.nn.relu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer())
            d2 = tf.layers.conv2d_transpose(d1, 64, 3, 2, padding='SAME', name='L2',
                                            activation=tf.nn.relu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer())
            d3 = tf.layers.conv2d_transpose(d2, 3, 3, 2, padding='SAME', name='L3',
                                            activation=tf.nn.relu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer())
        return d3
