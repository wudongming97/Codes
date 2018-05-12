import os

import numpy as np
import tensorflow as tf
from data import imcombind_, imsave_
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./datas/mnist')


def G(z):
    with tf.variable_scope('net_G'):
        fc1 = tf.layers.dense(z, 128, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 784, activation=tf.nn.sigmoid)
    return fc2


def D(im, reuse=True):
    with tf.variable_scope('net_D', reuse=reuse):
        fc1 = tf.layers.dense(im, 128, activation=tf.nn.leaky_relu)
        logit = tf.layers.dense(fc1, 1)
        prob = tf.sigmoid(logit)
    return logit, prob


z_dim = 16
lr = 1e-3
batch_size = 32
n_iters = 10000
to_display = 100

log_path = './results/'
os.makedirs(log_path, exist_ok=True)

z = tf.placeholder(tf.float32, [None, z_dim])
x_real = tf.placeholder(tf.float32, [None, 784], name='x')
x_fake = G(z)

real_logit, real_score = D(x_real, False)
fake_logit, fake_score = D(x_fake)

d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logit, labels=tf.ones_like(
    real_logit)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit, labels=tf.zeros_like(fake_logit)))
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit, labels=tf.ones_like(fake_logit)))

g_vars = [var for var in tf.global_variables() if 'net_G' in var.name]
d_vars = [var for var in tf.global_variables() if 'net_D' in var.name]

d_opt = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(d_loss, var_list=d_vars)
g_opt = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(g_loss, var_list=g_vars)

summery = tf.summary.merge([
    tf.summary.scalar('g_loss', g_loss),
    tf.summary.scalar('d_loss', d_loss),
    tf.summary.scalar('real_score', tf.reduce_mean(real_score)),
    tf.summary.scalar('fake_score', tf.reduce_mean(fake_score)),
    tf.summary.image('x_fake', tf.reshape(x_fake, [-1, 28, 28, 1]), max_outputs=4)
])

# train
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    _writer = tf.summary.FileWriter(log_path, sess.graph)

    for it in range(n_iters):
        x, _ = mnist.train.next_batch(batch_size, shuffle=True)
        noise = np.random.normal(size=[batch_size, z_dim])

        sess.run(d_opt, {x_real: x, z: noise})
        sess.run(g_opt, {x_real: x, z: noise})

        if it % to_display == 0:
            res = sess.run([g_loss, d_loss, tf.reduce_mean(real_score), tf.reduce_mean(fake_score), x_fake, summery],
                           {x_real: x, z: noise})
            _writer.add_summary(res[5], it)
            im = imcombind_(np.reshape(res[4], [batch_size, 28, 28, 1]))
            imsave_(log_path + '%d.png' % it, im)
            print('%4d/%4d g_loss: %.3f d_loss %.3f real_score: %.3f fake_score: %.3f ' % (
                it, n_iters, res[0], res[1], res[2], res[3]))
