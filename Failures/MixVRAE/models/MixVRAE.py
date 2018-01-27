# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

import numpy as np
import tensorflow as tf
import math

import utils.Utils as U

TI = namedtuple('train_inputs', ['X', 'X_lengths', 'Y_t', 'Y_mask'])
TL = namedtuple('train_losses', ['loss', 'rec_loss', 'kld_loss'])
TO = namedtuple('train_ops', ['optim_op', 'summery_op'])


class MixVRAE(object):
    def __init__(self, flags):
        self.flags = flags
        self.initializer = tf.contrib.layers.xavier_initializer()
        self.normal_z = tf.placeholder(dtype=tf.float32, shape=[self.flags.batch_size, self.flags.z_size],
                                       name='normal_z')
        self.embedding = self._embedding_init()
        self.train_i = self._train_input()
        self.build_graph()

    def build_graph(self):
        tf.train.create_global_step()
        (self.train_op, self.train_summery_op), self.losses = self.train_graph(self.train_i.X, self.train_i.X_lengths,
                                                                               self.train_i.Y_t, self.train_i.Y_mask)
        t1_ = len(tf.global_variables())
        self.preds_z = self.infer_graph()
        t2_ = len(tf.global_variables())
        assert t1_ == t2_

    def _embedding_init(self):
        with tf.name_scope('embedding'):
            embedding = tf.get_variable(name='embedding',
                                        shape=[self.flags.vocab_size, self.flags.embed_size],
                                        dtype=tf.float32,
                                        initializer=tf.random_normal_initializer(stddev=0.1))
        return embedding

    def _train_input(self):
        with tf.name_scope('train_inputs'):
            X = tf.placeholder(tf.int32, shape=[self.flags.batch_size, self.flags.max_seq_len], name='X')
            X_lengths = tf.placeholder(tf.int32, shape=[self.flags.batch_size], name='X_lengths')
            Y_t = tf.placeholder(tf.int32, shape=[self.flags.batch_size, self.flags.max_seq_len], name='Y_t')
            Y_mask = tf.placeholder(tf.int32, shape=[self.flags.batch_size, self.flags.max_seq_len], name='Y_mask')
        return TI(X, X_lengths, Y_t, Y_mask)

    def _rnn_cell(self, name):
        with tf.variable_scope(name):
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.flags.hidden_size, reuse=tf.get_variable_scope().reuse)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=0.9, output_keep_prob=0.9)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.flags.n_layers)
        return cell

    def train_graph(self, X, X_lengths, X_t, X_mask):
        with tf.variable_scope('MixVRAE', reuse=False):
            cell_q = self._rnn_cell('cell_p')
            X_embed = tf.nn.embedding_lookup(self.embedding, X)
            H_i = tf.layers.dense(X_embed, self.flags.hidden_size, kernel_initializer=self.initializer, name='H_i')
            Hq_, Sq_ = tf.nn.dynamic_rnn(cell_q, H_i, X_lengths, dtype=tf.float32, scope='fuck_tf_q')
            mu = tf.layers.dense(Sq_[-1].h, self.flags.z_size, kernel_initializer=self.initializer, name='to_mu')
            logvar = tf.layers.dense(Sq_[-1].h, self.flags.z_size, kernel_initializer=self.initializer,
                                     name='to_logvar')

            cell_p = self._rnn_cell('cell_q')
            z = self._sample_z(mu, logvar)
            H, _ = tf.nn.dynamic_rnn(cell_p, Hq_, X_lengths, self._z_to_state(z), tf.float32, scope='fuck_tf_p')
            logits = tf.layers.dense(H, self.flags.vocab_size, kernel_initializer=self.initializer, name='H_o')

            kld_loss = self._kld_loss(mu, logvar)
            rec_loss = self._rec_loss(logits, X_t, X_mask)
            loss = 0.1 * self._kld_coef() * kld_loss + rec_loss

            train_op = self._train_op(loss)
            train_summary_op = tf.summary.merge([
                tf.summary.histogram('mu', mu),
                tf.summary.histogram('logvar', logvar),
                tf.summary.histogram('z', z),
                tf.summary.scalar('kld_loss', kld_loss),
                tf.summary.scalar('rec_loss', rec_loss),
                tf.summary.scalar('loss', loss)
            ])
        return TO(train_op, train_summary_op), TL(loss, rec_loss, kld_loss)

    def infer_graph(self):
        with tf.variable_scope('MixVRAE', reuse=True):
            cell_p = self._rnn_cell('cell_p')
            p_state = self._z_to_state(self.normal_z)
            cell_q = self._rnn_cell('cell_q')
            q_state = cell_q.zero_state(self.flags.batch_size, dtype=tf.float32)
            preds = []
            # 1表示G_TOKEN的idx，要跟DataLoader里的一致
            next_input = tf.constant(1, shape=[self.flags.batch_size, 1], dtype=tf.int32)
            for i in range(self.flags.max_seq_len):
                next_input = tf.squeeze(tf.nn.embedding_lookup(self.embedding, next_input))
                H_i = tf.layers.dense(next_input, self.flags.hidden_size, kernel_initializer=self.initializer,
                                      name='H_i')
                with tf.variable_scope('fuck_tf_q'):
                    step_h, q_state = cell_q(H_i, state=q_state)
                with tf.variable_scope('fuck_tf_p'):
                    step_pred, p_state = cell_p(step_h, state=p_state)
                logits = tf.layers.dense(step_pred, self.flags.vocab_size, kernel_initializer=self.initializer,
                                         name='H_o')
                next_input = tf.stop_gradient(tf.argmax(logits, 1))
                preds.append(next_input)
            return preds

    def _z_to_state(self, z):
        hidden_states = []
        with tf.name_scope('z_to_state'):
            for i in range(self.flags.n_layers):
                hidden_states.append(tf.nn.rnn_cell.LSTMStateTuple(
                    tf.layers.dense(z, self.flags.hidden_size, kernel_initializer=self.initializer),
                    tf.layers.dense(z, self.flags.hidden_size, kernel_initializer=self.initializer)
                ))
        return tuple(hidden_states)

    def _sample_z(self, mu, logvar):
        with tf.name_scope('sample_z'):
            eps = tf.random_normal((self.flags.batch_size, self.flags.z_size), stddev=1.0)
            z = mu + tf.exp(0.5 * logvar) * eps
        return z

    def _kld_coef(self):
        # if self.flags.kl_anealing:
        #     coef = tf.clip_by_value(tf.sigmoid(-15 + 20 * tf.train.get_global_step() / self.flags.steps), 0, 1)
        #     tf.summary.scalar('coef', coef)
        #     return tf.cast(coef, tf.float32)
        # else:
        return 1

    @staticmethod
    def _kld_loss(mu, log_var):
        with tf.name_scope('kld_loss'):
            kld_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(log_var - tf.square(mu) - tf.exp(log_var) + 1, axis=1))
        return kld_loss

    @staticmethod
    def _rec_loss(logits, targets, masks):
        with tf.name_scope('rec_loss'):
            rec_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
            rec_loss = tf.reduce_mean(tf.reduce_sum(rec_loss * tf.cast(masks, tf.float32), axis=1))
        return rec_loss

    def _train_op(self, loss):
        with tf.name_scope('train_op'):
            t_vars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, t_vars), 5)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.flags.lr)
            train_op = optimizer.apply_gradients(zip(grads, t_vars), global_step=tf.train.get_global_step())
        return train_op

    def train_is_ok(self, sess):
        return sess.run(tf.train.get_global_step()) >= self.flags.steps

    def fit(self, sess, data_loader, t_writer, _saver):
        for data in data_loader.next_batch(self.flags.batch_size, train=True, shuffle=True):
            X_i, X_lengths, Y_t, Y_mask = data_loader.unpack_for_mixvare(data, self.flags.max_seq_len)

            _, t_summery_, loss_, rec_loss_, kld_loss_ = sess.run(
                [self.train_op, self.train_summery_op] + list(self.losses),
                {self.train_i.X: X_i,
                 self.train_i.X_lengths: X_lengths,
                 self.train_i.Y_t: Y_t,
                 self.train_i.Y_mask: Y_mask
                 })
            step_ = sess.run(tf.train.get_global_step())
            t_writer.add_summary(t_summery_, step_)

            if step_ % 20 == 0:
                epoch_ = U.step_to_epoch(step_, data_loader.train_size, self.flags.batch_size)
                print("TRAIN: | Epoch %d | step %d/%d | train_loss: %.4f | rec_loss %.4f | kld_loss %.4f|" % (
                    epoch_, step_, self.flags.steps, loss_, rec_loss_, kld_loss_))

            # 每5个epoch存储下模型
            if step_ % U.epoch_to_step(1, data_loader.train_size, self.flags.batch_size) == 0:
                # self.valid(sess, data_loader, v_writer)
                _saver.save(sess, self.flags.ckpt_path, global_step=step_, write_meta_graph=False)
                print('model saved ...')

            if step_ >= self.flags.steps:
                _saver.save(sess, self.flags.ckpt_path, global_step=step_, write_meta_graph=False)
                print('train is end ...')
                break

    def infer_by_z(self, sess, data_loader, z):
        preds_z = sess.run(self.preds_z, {self.normal_z: z})
        return data_loader.to_seqs(np.array(preds_z).transpose())

    def infer_by_normal(self, sess, data_loader):
        z = np.random.normal(0, 1, [self.flags.batch_size, self.flags.z_size])
        return self.infer_by_z(sess, data_loader, z)

    # def infer_by_encoder(self, sess, data_loader, sentences):
    #     tensor = data_loader.to_tensor([s.split() for s in sentences])
    #     X, X_lengths, Y_t, Y_masks = data_loader.unpack_for_mixvare(tensor, self.flags.max_seq_len)
    #     preds_e = sess.run(self.preds_e, {self.train_i.X: X,
    #                                       self.train_i.X_lengths: X_lengths,
    #                                       self.train_i.Y_i: Y_i,
    #                                       self.train_i.Y_lengths: Y_lengths,
    #                                       self.train_i.Y_t: Y_t,
    #                                       self.train_i.Y_mask: Y_masks})
    #
    #     return data_loader.to_seqs(np.array(preds_e).transpose())
