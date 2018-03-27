import numpy as np
import tensorflow as tf


class D(object):
    def __init__(self):
        self.name = 'mnist/d_net'

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
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
        self.z_dim = 64
        self.imshape = [28, 28, 1]
        self.name = 'mnist/g_net'

    def __call__(self, z):
        with tf.variable_scope(self.name):
            bs = tf.shape(z)[0]
            fc1 = tf.layers.dense(z, 7 * 7 * 128, activation=tf.nn.relu)
            conv1 = tf.reshape(fc1, [bs, 7, 7, 128])
            conv1 = tf.layers.conv2d_transpose(conv1, 128, [4, 4], [2, 2], 'SAME',
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                                               activation=tf.nn.relu)
            conv2 = tf.layers.conv2d_transpose(conv1, 32, [4, 4], [2, 2], 'SAME',
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                                               activation=tf.nn.relu)
            fake = tf.layers.conv2d_transpose(conv2, 1, [4, 4], [1, 1], 'SAME',
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
                                              activation=tf.nn.sigmoid)
            return fake

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class wgan(object):
    def __init__(self):
        self.G = G()
        self.D = D()
        self.imshape = self.G.imshape
        self.z_dim = self.G.z_dim

        self.real = tf.placeholder(tf.float32, [None] + self.imshape, name='x')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.fake = self.G(self.z)

        self.g_loss = tf.reduce_mean(self.D(self.fake, reuse=False))
        self.d_loss = tf.reduce_mean(self.D(self.real)) - tf.reduce_mean(self.D(self.fake)) + self._dx()
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_adam = tf.train.AdamOptimizer(1e-4, beta1=0.5, beta2=0.9).minimize(self.d_loss, var_list=self.D.vars)
            self.g_adam = tf.train.AdamOptimizer(1e-4, beta1=0.5, beta2=0.9).minimize(self.g_loss, var_list=self.G.vars)

    def _dx(self):
        eps = tf.random_uniform([], 0.0, 1.0)
        x_hat = eps * self.real + (1 - eps) * self.fake
        d_hat = self.D(x_hat)
        dx = tf.layers.flatten(tf.gradients(d_hat, x_hat)[0])
        dx = tf.sqrt(tf.reduce_sum(tf.square(dx), axis=1))
        dx = tf.reduce_mean(tf.square(dx - 1.0) * 10)
        return dx

    def _z_sample(self, bz):
        return np.random.normal(0, 1, [bz, self.z_dim])

    def gen(self, sess, bz):
        return sess.run(self.fake, feed_dict={self.z: self._z_sample(bz)})

    def fit(self, sess, bx, step):
        bz = bx.shape[0]
        for t in range(step):
            for _ in range(3):
                sess.run(self.d_adam, feed_dict={self.real: bx, self.z: self._z_sample(bz)})
            sess.run(self.g_adam, feed_dict={self.real: bx, self.z: self._z_sample(bz)})
