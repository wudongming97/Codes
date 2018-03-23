import os
import tensorflow as tf


class wgan(object):
    def __init__(self, g_net, d_net, data_, flags):
        self.scale = 10
        self.G = g_net
        self.D = d_net
        self.data = data_
        self.flags = flags
        self.imshape = self.G.imshape
        self.z_dim = self.G.z_dim

        self.real = tf.placeholder(tf.float32, [None] + self.imshape, name='x')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.fake = self.G(self.z)

        self.g_loss = tf.reduce_mean(self.D(self.fake, reuse=False))
        self.d_loss = tf.reduce_mean(self.D(self.real)) - tf.reduce_mean(self.D(self.fake)) + self._dx()
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_adam = tf.train.AdamOptimizer(1e-4, beta1=0.5, beta2=0.9).minimize(self.d_loss, var_list=self.D.vars)
            self.g_adam = tf.train.AdamOptimizer(1e-4, beta1=0.5, beta2=0.9).minimize(self.g_loss, var_list=self.G.vars,
                                                                                      global_step=tf.train.get_or_create_global_step())
        with tf.name_scope('train_summary'):
            self.train_summary = tf.summary.merge([
                tf.summary.scalar('g_loss', self.g_loss),
                tf.summary.scalar('d_loss', self.d_loss),
                tf.summary.image('X', self.real, 16),
                tf.summary.image('fake', self.fake, 16)
            ])

    def _dx(self):
        eps = tf.random_uniform([], 0.0, 1.0)
        x_hat = eps * self.real + (1 - eps) * self.fake
        d_hat = self.D(x_hat)
        dx = tf.layers.flatten(tf.gradients(d_hat, x_hat)[0])
        dx = tf.sqrt(tf.reduce_sum(tf.square(dx), axis=1))
        dx = tf.reduce_mean(tf.square(dx-1.0) * self.flags.scale)
        return dx

    def gen(self, sess, bz, it):
        images = sess.run(self.fake, feed_dict={self.z: self.data.z_sample_(bz, self.z_dim)})
        sv_ = self.flags.logs_path + '{}.png'.format(it)
        self.data.imsave_(sv_, self.data.imcombind_(images))

    def train(self, sess, writer, saver):
        bz = self.flags.batch_sz
        for bx in self.data.next_batch_(bz, 'fashion'):
            for _ in range(self.flags.d_iters):
                sess.run(self.d_adam, feed_dict={self.real: bx, self.z: self.data.z_sample_(bz, self.z_dim)})
            sess.run(self.g_adam, feed_dict={self.real: bx, self.z: self.data.z_sample_(bz, self.z_dim)})
            step_ = sess.run(tf.train.get_global_step())

            if step_ >= self.flags.steps:
                print('Train is ok ...')
                break

            if step_ % 20 == 0:
                d_loss, g_loss, summary = sess.run([self.d_loss, self.g_loss, self.train_summary],
                                                   feed_dict={self.real: bx, self.z: self.data.z_sample_(bz, self.z_dim)})
                self.gen(sess, 64, step_)
                writer.add_summary(summary, step_)
                saver.save(sess, self.flags.ckpt_path, step_, write_meta_graph=False)
                print('Step [%d/%d] d_loss [%.4f] g_loss [%.4f]' % (step_, self.flags.steps, d_loss, g_loss))















