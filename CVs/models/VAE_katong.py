# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class VAE(object):
    def __init__(self, flags, X):
        self.flags = flags
        self.X = X
        self.phase = tf.placeholder(dtype=tf.bool, name='phase')
        self.normal_z = tf.placeholder(dtype=tf.float32, shape=[None, self.flags.z_size], name='normal_z')
        self._build_graph()

    def _build_graph(self):
        tf.train.create_global_step()
        self.optim_op, self.train_summary_op, self.loss, self.rec_loss, self.kld_loss = self._train_graph()
        self.infer_samples, self.infer_summary_op = self._infer_graph()
        self.recon_samoles, self.recon_summary_op = self._recon_graph()

    def _train_graph(self):
        mu, logvar = self._cnn_encoder(self.X)
        z = self._sample_z(mu, logvar)
        X_ = self._cnn_decoder(z)
        kld_loss = self._kld_loss(mu, logvar)
        rec_loss = self._rec_loss(self.X, X_)

        loss, optim_op = self._optim_op(rec_loss, kld_loss)

        with tf.name_scope('train_summary'):
            train_summary_op = tf.summary.merge([
                tf.summary.histogram('mu', mu),
                tf.summary.histogram('logvar', logvar),
                tf.summary.histogram('z', z),
                tf.summary.scalar('loss', loss),
                tf.summary.scalar('rec_loss', rec_loss),
                tf.summary.scalar('kld_loss', kld_loss)
            ])
        return optim_op, train_summary_op, loss, rec_loss, kld_loss

    def _infer_graph(self):
        samples = self._cnn_decoder(self.normal_z, True)
        with tf.name_scope('infer_summary'):
            infer_summary_op = tf.summary.merge([
                tf.summary.image('infer_images', samples, 8)
            ])
        return samples, infer_summary_op

    def _recon_graph(self):
        mu, logvar = self._cnn_encoder(self.X, True)
        z = self._sample_z(mu, logvar)
        X_ = self._cnn_decoder(z, True)
        with tf.name_scope('recon_summary'):
            recon_summary_op = tf.summary.merge([
                tf.summary.histogram('mu', mu),
                tf.summary.histogram('logvar', logvar),
                tf.summary.histogram('z', z),
                tf.summary.image('input_X_', self.X, 8),
                tf.summary.image('recon_X_', X_, 8)
            ])
        return X_, recon_summary_op

    def _conv_with_bn(self, input, filters, kernel_size, strides, padding, name):
        with tf.variable_scope(name):
            conv = tf.layers.conv2d(input, filters, kernel_size, strides, padding)
            bn = tf.nn.relu(tf.layers.batch_normalization(conv, training=self.phase))
        return bn

    def _cnn_encoder(self, X, ru=False):
        with tf.variable_scope('cnn_encoder', reuse=ru):
            c1 = self._conv_with_bn(X, 128, 3, 2, 'SAME', name='L1')
            c2 = tf.layers.conv2d(c1, 64, 3, 2, 'SAME', name='L2')
            c3 = tf.layers.conv2d(c2, 32, 3, 2, 'SAME', name='L3')
            c4 = tf.layers.conv2d(c3, 8, 3, 2, 'SAME', name='L4')
            fc0 = tf.reshape(c4, [-1, 6 * 6 * 8])

            mu = tf.layers.dense(fc0, self.flags.z_size, name='fc_mu')
            logvar = tf.layers.dense(fc0, self.flags.z_size, name='fc_logvar')

        return mu, logvar

    def _dconv_with_bn(self, input, filters, kernel_size, strides, padding, name):
        with tf.variable_scope(name):
            conv_t = tf.layers.conv2d_transpose(input, filters, kernel_size, strides, padding)
            bn = tf.nn.relu(tf.layers.batch_normalization(conv_t, training=self.phase))
            return bn

    def _cnn_decoder(self, z, ru=False):
        with tf.variable_scope('cnn_decoder', reuse=ru):
            df1 = tf.layers.dense(z, 6 * 6 * 128, name='df1')
            d0 = tf.reshape(df1, [-1, 6, 6, 128])
            d1 = tf.layers.conv2d_transpose(d0, 128, 3, 2, 'SAME', name='L1')
            d2 = tf.layers.conv2d_transpose(d1, 64, 3, 2, 'SAME', name='L2')
            d3 = tf.layers.conv2d_transpose(d2, 64, 3, 2, 'SAME', name='L3')
            d4 = self._dconv_with_bn(d3, 3, 3, 2, 'SAME', 'L4')
        return d4

    def _sample_z(self, mu, logvar):
        with tf.name_scope('sample_z'):
            bz = mu.get_shape().as_list()[0]
            eps = tf.truncated_normal((bz, self.flags.z_size), stddev=1.0)
            z = mu + tf.exp(0.5 * logvar) * eps
        return z

    @staticmethod
    def _kld_loss(mu, log_var):
        with tf.name_scope('kld_loss'):
            kld_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(log_var - tf.square(mu) - tf.exp(log_var) + 1, axis=1))
        return kld_loss

    # @staticmethod
    def _rec_loss(self, X, X_):
        with tf.name_scope('rec_loss'):
            rec_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=X, labels=X_)
            return tf.reduce_mean(rec_loss)

    def _optim_op(self, rec_loss, kld_loss):
        with tf.name_scope('optim_op'):
            # for batch_normal to work correct
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                loss = rec_loss + self.flags.beta * kld_loss
                optim_op = tf.train.AdamOptimizer(learning_rate=self.flags.lr).minimize(loss, tf.train.get_global_step())
        return loss, optim_op

    def fit(self, sess, writer, saver):
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        for _ in range(self.flags.steps):
            _, _summary, loss, rec_loss, kld_loss = sess.run(
                [self.optim_op, self.train_summary_op, self.loss, self.rec_loss, self.kld_loss], {self.phase: True})
            step_ = sess.run(tf.train.get_global_step())
            writer.add_summary(_summary, step_)

            if step_ % 5 == 0:
                print("TRAIN: | step %d/%d | train_loss: %.3f | rec_loss %.3f | kld_loss %.6f|" % (
                    step_, self.flags.steps, loss, rec_loss, kld_loss))
            if step_ % 20 == 0:
                self.infer_from_normal(sess, writer)
                self.infer_from_encoder(sess, writer)
                saver.save(sess, self.flags.ckpt_path, global_step=step_, write_meta_graph=False)

        coord.request_stop()
        coord.join(threads)

    def infer_from_normal(self, sess, writer):
        z = np.random.normal(0, 1, [self.flags.batch_size, self.flags.z_size])
        samples, _summary = sess.run([self.infer_samples, self.infer_summary_op], {self.normal_z: z, self.phase: False})
        writer.add_summary(_summary)
        return samples

    def infer_from_encoder(self, sess, writer):
        samples, _summary = sess.run([self.recon_samoles, self.recon_summary_op], {self.phase: False})
        writer.add_summary(_summary)
        return samples
