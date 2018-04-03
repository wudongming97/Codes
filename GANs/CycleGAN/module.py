# 网络结构参照 https://github.com/xhujoy/CycleGAN-tensorflow/blob/master/module.py
import random

from ops import *


class generator_resnet:
    """
    Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
    The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128, R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
    """

    def __init__(self, name):
        self.name = name
        self.ngf = 64

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name, reuse=reuse):
            c3 = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
            c3 = relu(instance_norm(conv2d(c3, self.ngf, 7, 1, name='g_e1_c'), 'g_e1_bn'))
            c3 = relu(instance_norm(conv2d(c3, self.ngf * 2, 3, 2, 'SAME', name='g_e2_c'), 'g_e2_bn'))
            r9 = relu(instance_norm(conv2d(c3, self.ngf * 4, 3, 2, 'SAME', name='g_e3_c'), 'g_e3_bn'))

            # define G network with 9 resnet blocks
            for i in range(9):
                r9 = residual_in(r9, name='g_r{}'.format(i + 1))

            d1 = dconv2d(r9, self.ngf * 2, 3, 2, 'SAME', name='g_d1_dc')
            d1 = relu(instance_norm(d1, 'g_d1_bn'))
            d2 = dconv2d(d1, self.ngf, 3, 2, 'SAME', name='g_d2_dc')
            d2 = relu(instance_norm(d2, 'g_d2_bn'))
            d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
            pred = tf.nn.tanh(conv2d(d2, 3, 7, 1, name='g_pred_c'))

        return pred

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class discriminator:
    def __init__(self, name):
        self.name = name

    def __call__(self, input_, reuse=True):
        with tf.variable_scope(self.name, reuse=reuse):
            f, ndf = 4, 64
            h0 = lrelu(conv2d(input_, ndf, f, 2, 'SAME'))  # 128 x 128 x 64
            h1 = lrelu(conv2d(h0, ndf * 2, f, 2, 'SAME'))  # 64 x 64 x 128
            h2 = lrelu(conv2d(h1, ndf * 4, f, 2, 'SAME'))
            h3 = lrelu(conv2d(h2, ndf * 8, f, 2, 'SAME'))
            h4 = conv2d(h3, 1, 4, 1)
            return h4  # (32 x 32 x 1)

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class discriminator_patch:
    def __init__(self, name):
        self.name = name

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name, reuse=reuse):
            f, ndf = 4, 64

            patch_ = tf.random_crop(input, [1, 70, 70, 3])
            c1 = lrelu(instance_norm(conv2d(patch_, ndf, f, 2, 'SAME'), 'i1'))
            c2 = lrelu(instance_norm(conv2d(c1, ndf * 2, f, 2, 'SAME'), 'i2'))
            c3 = lrelu(instance_norm(conv2d(c2, ndf * 4, f, 2, 'SAME'), 'i3'))
            c4 = lrelu(instance_norm(conv2d(c3, ndf * 8, f, 2, 'SAME'), 'i4'))
            c5 = conv2d(c4, 1, f, 1, 'SAME')

        return c5


class image_pool:
    def __init__(self, capacity=50):
        self.capacity = capacity
        self.level = 0
        self.fake_pool = []

    def __call__(self, image):
        if self.level < self.capacity:
            self.fake_pool.append(image)
            self.level += 1
            return image
        else:
            p = random.random()
            if p > 0.5:
                rix = random.randint(0, self.capacity - 1)
                temp = self.fake_pool[rix]
                self.fake_pool[rix] = image
                return temp
            else:
                return image


def dis_loss(d_real, d_fake):
    return tf.reduce_mean(tf.square(d_real - tf.ones_like(d_real)) + tf.square(d_fake))


def gen_loss(d_fake):
    return tf.reduce_mean(tf.square(d_fake - tf.ones_like(d_fake)))


def cyc_loss(x, cyc_x, y, cyc_y):
    return tf.reduce_mean(tf.abs(cyc_x - x) + tf.abs(cyc_y - y))
