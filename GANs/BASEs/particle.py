import tensorflow as tf

from ops import bn


class G(object):
    def __init__(self, phase):
        self.name = 'mnist/g_net'
        self.phase = phase

    def __call__(self, z):
        with tf.variable_scope(self.name):
            fc1 = tf.nn.relu(bn(tf.layers.dense(z, 1024), self.phase))
            fc2 = tf.nn.relu(bn(tf.layers.dense(fc1, 7 * 7 * 128), self.phase))
            fc2 = tf.reshape(fc2, [-1, 7, 7, 128])
            cv1 = tf.nn.relu(bn(tf.layers.conv2d_transpose(fc2, 64, [4, 4], [2, 2], 'SAME'), self.phase))
            fake = tf.layers.conv2d_transpose(cv1, 1, [4, 4], [2, 2], 'SAME', activation=tf.nn.sigmoid)
            return fake

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class D(object):
    def __init__(self, phase):
        self.name = 'mnist/d_net'
        self.phase = phase

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name, reuse=reuse):
            cv1 = tf.nn.leaky_relu(bn(tf.layers.conv2d(x, 64, [4, 4], [2, 2]), self.phase))
            cv2 = tf.nn.leaky_relu(bn(tf.layers.conv2d(cv1, 128, [4, 4], [2, 2]), self.phase))
            fc1 = tf.nn.leaky_relu(bn(tf.layers.dense(tf.layers.flatten(cv2), 1024), self.phase))
            fc2 = tf.layers.dense(fc1, 1)

            return fc2

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]
