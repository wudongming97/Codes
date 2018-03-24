# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class VAE_cifar10(object):
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
        self.recon_samples, self.recon_summary_op = self._recon_graph()

    def _train_graph(self):
        mu, logvar = self._encoder(self.X)
        z = self._sample_z(mu, logvar)
        X_ = self._decoder(z)
        kld_loss = self._kld_loss(mu, logvar)
        rec_loss = self._rec_loss(X_, self.X)

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
        logits = self._decoder(self.normal_z, True)
        samples = tf.sigmoid(logits)
        with tf.name_scope('infer_summary'):
            infer_summary_op = tf.summary.merge([
                tf.summary.image('infer_images', samples, 8)
            ])
        return samples, infer_summary_op

    def _recon_graph(self):
        mu, logvar = self._encoder(self.X, True)
        z = self._sample_z(mu, logvar)
        samples = tf.sigmoid(self._decoder(z, True))
        with tf.name_scope('recon_summary'):
            recon_summary_op = tf.summary.merge([
                tf.summary.histogram('mu', mu),
                tf.summary.histogram('logvar', logvar),
                tf.summary.histogram('z', z),
                tf.summary.image('input_X_', self.X, 8),
                tf.summary.image('recon_X_', samples, 8)
            ])
        return samples, recon_summary_op

    def _encoder(self, X, ru=False):
        with tf.variable_scope('encoder', reuse=ru):
            c0 = tf.layers.conv2d(X, 128, [3, 3], [2, 2], 'SAME', activation=tf.nn.relu, reuse=ru, name='C_0')
            c0 = tf.layers.batch_normalization(c0, training=self.phase, reuse=ru, name='NC0')
            c1 = tf.layers.conv2d(c0, 128, [3, 3], [2, 2], 'SAME', activation=tf.nn.relu, reuse=ru, name='C_1')
            c1 = tf.layers.batch_normalization(c1, training=self.phase, reuse=ru, name='NC1')
            c2 = tf.layers.conv2d(c1, 16, [3, 3], [2, 2], 'SAME', activation=tf.nn.relu, reuse=ru, name='C_2')

            fc0 = tf.layers.flatten(c2)
            mu = tf.layers.dense(fc0, self.flags.z_size, name='fc_mu')
            logvar = tf.layers.dense(fc0, self.flags.z_size, name='fc_logvar')

        return mu, logvar

    def _decoder(self, z, ru=False):
        with tf.variable_scope('decoder', reuse=ru):
            f1 = tf.layers.dense(z, 8*8*16, activation=tf.nn.relu, name='f1')
            f1 = tf.reshape(f1, [-1, 4, 4, 64])

            dc1 = tf.layers.conv2d_transpose(f1, 128, [3, 3], [2, 2], 'SAME', activation=tf.nn.relu, reuse=ru, name='DC1')
            dc1 = tf.layers.batch_normalization(dc1, training=self.phase, reuse=ru, name='NDC1')
            dc2 = tf.layers.conv2d_transpose(dc1, 128, [3, 3], [2, 2], 'SAME', activation=tf.nn.relu, reuse=ru, name='DC2')
            dc2 = tf.layers.batch_normalization(dc2, training=self.phase, reuse=ru, name='NDC2')
            dc3 = tf.layers.conv2d_transpose(dc2, 3, [3, 3], [2, 2], 'SAME', reuse=ru, name='DC3')
        return dc3

    def _sample_z(self, mu, logvar):
        with tf.name_scope('sample_z'):
            eps = tf.random_normal(tf.shape(mu))
            z = mu + tf.exp(0.5 * logvar) * eps
        return z

    @staticmethod
    def _kld_loss(mu, logvar):
        with tf.name_scope('kld_loss'):
            kld_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(logvar - tf.square(mu) - tf.exp(logvar) + 1, axis=1))
        return kld_loss

    @staticmethod
    def _rec_loss(logits, labels):
        with tf.name_scope('rec_loss'):
            flatten_logits = tf.layers.flatten(logits)
            flatten_labels = tf.layers.flatten(labels)
            rec_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=flatten_logits, labels=flatten_labels), 1))
            return rec_loss

    def _optim_op(self, rec_loss, kld_loss):
        with tf.name_scope('optim_op'):
            loss = rec_loss + self.flags.beta * kld_loss
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optim_op = tf.train.AdamOptimizer(learning_rate=self.flags.lr).minimize(loss, tf.train.get_global_step())
        return loss, optim_op

    def fit(self, sess, writer, saver):
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        for _ in range(self.flags.steps):
            _, _summary, loss, rec_loss, kld_loss = sess.run(
                [self.optim_op, self.train_summary_op, self.loss, self.rec_loss, self.kld_loss],
                {self.phase: True})
            step_ = sess.run(tf.train.get_global_step())
            writer.add_summary(_summary, step_)

            if step_ % 1 == 0:
                print("TRAIN: | step %d/%d | train_loss: %.3f | rec_loss %.3f | kld_loss %.6f| " % (
                    step_, self.flags.steps, loss, rec_loss, kld_loss))
            if step_ % 1000 == 0:
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
        samples, _summary = sess.run([self.recon_samples, self.recon_summary_op], {self.phase: False})
        writer.add_summary(_summary)
        return samples

    def train_is_ok(self, sess):
        return sess.run(tf.train.get_global_step()) >= self.flags.steps