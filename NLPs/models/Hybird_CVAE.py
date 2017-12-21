from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

import tensorflow as tf

import utils.Utils as U

TI = namedtuple('train_inputs', ['X', 'Y_i', 'Y_lengths', 'Y_t', 'Y_mask'])
TL = namedtuple('train_losses', ['loss', 'rec_loss', 'kld_loss', 'aux_loss'])
TO = namedtuple('train_ops', ['optim_op', 'train_summery_op'])

class Hybird_CVAE(object):
    def __init__(self, flags):
        self.flags = flags
        self.phase = tf.placeholder(dtype=tf.bool, name='phase')
        self.normal_z = tf.placeholder(dtype=tf.float32, shape=[1, self.flags.z_size], name='normal_z')
        self.embedding = self._embedding_init()
        self.rnn_cell = self._decoder_rnn_cell_init()
        self.train_i = self._train_input()
        self.build_graph()

    def build_graph(self):
        tf.train.create_global_step()
        self.losses, self.train_ops = self._train_graph()
        self._build_sample_from_normal_graph()

    def _train_graph(self):
        mu, log_var = self._cnn_encoder(self.train_i.X, False)
        kld_loss = self._kld_loss(mu, log_var)

        z = self._sample_z(mu, log_var)
        vocab_logits = self._cnn_decoder(z, reuse=False)
        aux_loss = self._aux_loss(vocab_logits, self.train_i.X)

        rnn_logits = self._rnn_train(vocab_logits, self.train_i.Y_i, self.train_i.Y_lengths)
        rec_loss = self._rec_loss(rnn_logits, self.train_i.Y_t, self.train_i.Y_mask)
        loss = self._train_loss(kld_loss, rec_loss, aux_loss)

        optim_op = self._train_op(loss)
        train_summery_op = tf.summary.merge_all()

        return TL(loss, rec_loss, kld_loss, aux_loss), TO(optim_op, train_summery_op)

    def _build_sample_from_normal_graph(self):
        vocab_logits = self._cnn_decoder(self.normal_z, reuse=True)
        #self._rnn_infer_layer(vocab_logits)
        # aux_loss, logits = self._cnn_decoder_subgraph('forward_z_subgraph', self.sample_input, reuse=True)
        # todo
        # next_symbol = tf.stop_gradient(tf.argmax(logits, 1))
        # next_input = tf.nn.embedding_lookup(self.embedding, next_symbol)
        None

    def _cnn_encoder(self, input, reuse_=False):
        with tf.name_scope('cnn_encoder'):
            e1 = tf.nn.embedding_lookup(self.embedding, input)
            with tf.name_scope('cnn'):
                c1 = tf.layers.conv1d(e1, 128, 3, strides=2, padding='SAME', name='c1', reuse=reuse_)
                bn1 = tf.nn.relu(tf.layers.batch_normalization(c1, training=self.phase, name='b1', reuse=reuse_))
                c2 = tf.layers.conv1d(bn1, 256, 3, strides=2, padding='SAME', name='c2', reuse=reuse_)
                bn2 = tf.nn.relu(tf.layers.batch_normalization(c2, training=self.phase, name='b2', reuse=reuse_))
            with tf.name_scope('liner'):
                cnn_output = tf.reshape(bn2, shape=[-1, 256 * int(self.flags.seq_len / 4)])
                mu = tf.layers.dense(cnn_output, units=self.flags.z_size, name='dense_mu', reuse=reuse_)
                log_var = tf.layers.dense(cnn_output, units=self.flags.z_size, name='dense_log_var', reuse=reuse_)
        return mu, log_var

    def _cnn_decoder(self, z, reuse=False):
        with tf.name_scope('cnn_decoder'):
            dconv_in = tf.layers.dense(z, 256 * int(self.flags.seq_len / 4), name='dense_dconv_input')
            dconv_out = self._decoder_dconv_layer(dconv_in, name='decoder_dconv_layer')
            vocab_logits = tf.layers.dense(dconv_out, self.flags.vocab_size, name='dc2vocab')

        return vocab_logits

    def _rnn_train(self, vocab_logits, inputs, lengths):
        with tf.name_scope('rnn_train_layer'):
            embed_word_inputs = tf.nn.embedding_lookup(self.embedding, inputs)
            rnn_cated_inputs = tf.concat([embed_word_inputs, vocab_logits], axis=-1)
            rnn_hidden_input = tf.layers.dense(rnn_cated_inputs, self.flags.rnn_hidden_size, name='rnn_input')
            rnn_hidden_output, _ = tf.nn.dynamic_rnn(self.rnn_cell, rnn_hidden_input, lengths, dtype=tf.float32)
            rnn_logits = tf.layers.dense(rnn_hidden_output, self.flags.vocab_size, name='rnn_output')
        return rnn_logits

    def _rnn_infer_layer(self, vocab_logits, go_input):
        with tf.name_scope('rnn_infer_layer'):
            U.print_shape(vocab_logits)
            next_input = tf.nn.embedding_lookup(self.embedding, go_input)
            for i in range(self.flags.seq_len):
                rnn_input = tf.concat([])

    def _train_op(self, loss):
        with tf.name_scope('train_op'):
            # for batch_normal to work correct
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                t_vars = tf.trainable_variables()
                grads, _ = tf.clip_by_global_norm(tf.gradients(loss, t_vars), 5)
                optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
                train_op = optimizer.apply_gradients(zip(grads, t_vars),
                                                     global_step=tf.train.get_global_step())
        return train_op

    def _embedding_init(self):
        with tf.name_scope('embedding'):
            with tf.variable_scope('embedding', reuse=False):
                embedding = tf.get_variable(name='embedding',
                                            shape=[self.flags.vocab_size, self.flags.embed_size],
                                            dtype=tf.float32,
                                            initializer=tf.random_normal_initializer(stddev=0.1))
        return embedding

    def _train_input(self):
        with tf.name_scope('train_inputs'):
            X = tf.placeholder(tf.int32, shape=[self.flags.batch_size, self.flags.seq_len], name='X')
            Y_i = tf.placeholder(tf.int32, shape=[self.flags.batch_size, self.flags.seq_len], name='Y_i')
            Y_lengths = tf.placeholder(tf.int32, shape=[self.flags.batch_size], name='Y_lengths')
            Y_t = tf.placeholder(tf.int32, shape=[self.flags.batch_size, self.flags.seq_len], name='Y_t')
            Y_mask = tf.placeholder(tf.int32, shape=[self.flags.batch_size, self.flags.seq_len], name='Y_mask')
        return TI(X, Y_i, Y_lengths, Y_t, Y_mask)

    def _sample_z(self, mu, log_var):
        with tf.name_scope('sample_z'):
            eps = tf.truncated_normal((self.flags.batch_size, self.flags.z_size), stddev=1.0)
            z = mu + tf.exp(0.5 * log_var) * eps

            tf.summary.histogram('z', z)
        return z

    def _decoder_dconv_layer(self, dc_input, name):
        with tf.name_scope(name):
            dc_input_ = tf.reshape(dc_input, [-1, 1, int(self.flags.seq_len / 4), 256],
                                   name='in_shape')
            dct1_out = self._decoder_conv2d_transpose_layer(input=dc_input_,
                                                            filter_shape=[1, 3, 128, 256],
                                                            out_shape=[self.flags.batch_size, 1,
                                                                       int(self.flags.seq_len / 2), 128],
                                                            stride=[1, 1, 2, 1],
                                                            padding='SAME',
                                                            name='dct1')
            dct2_out = self._decoder_conv2d_transpose_layer(input=dct1_out,
                                                            filter_shape=[1, 3, 200, 128],
                                                            out_shape=[self.flags.batch_size, 1,
                                                                       self.flags.seq_len, 200],
                                                            stride=[1, 1, 2, 1],
                                                            padding='SAME',
                                                            name='dct2')

            decoder_cnn_output = tf.reshape(dct2_out, [-1, self.flags.seq_len, 200])
        return decoder_cnn_output

    def _decoder_conv2d_transpose_layer(self, input, filter_shape, out_shape, stride, padding, name):
        with tf.name_scope(name):
            with tf.variable_scope(name):
                filter = tf.get_variable(name='filter',
                                         shape=filter_shape,
                                         dtype=tf.float32,
                                         initializer=tf.random_normal_initializer(stddev=0.1))
                conv1d_transpose = tf.nn.conv2d_transpose(input, filter, out_shape, stride, padding, name='conv2d_t')
                res = tf.nn.relu(tf.layers.batch_normalization(conv1d_transpose, training=self.phase), name='relu')
        return res

    def _train_loss(self, kld_loss, rec_loss, aux_loss):
        with tf.name_scope('loss'):
            train_loss = rec_loss + self.flags.alpha * aux_loss + self._kld_coef() * kld_loss
        return train_loss

    def _kld_coef(self):
        with tf.name_scope('kld_coef'):
            coef = tf.clip_by_value((tf.train.get_global_step() - self.flags.kld_anneal_start) / (
                    self.flags.kld_anneal_end - self.flags.kld_anneal_start), 0, 1)
            return tf.cast(coef, tf.float32)

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

    @staticmethod
    def _aux_loss(logits, targets):
        with tf.name_scope('aux_loss'):
            aux_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
            aux_loss = tf.reduce_mean(tf.reduce_sum(aux_loss, axis=1))
        return aux_loss


    def _decoder_rnn_cell_init(self):
        with tf.name_scope('decoder_rnn_cell'):
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.flags.rnn_hidden_size, reuse=False)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=0.8, output_keep_prob=0.8)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 2)
        return cell

    def train_is_ok(self, sess):
        return sess.run(tf.train.get_global_step()) >= self.flags.global_steps

    def fit(self, sess, data_loader, _writer, _saver):
        for data in data_loader.next_batch(self.flags.batch_size, train=True):
            X, Y_i, Y_lengths, Y_t, Y_masks = data_loader.unpack_for_hybird_cvae(data, self.flags.seq_len)

            _, summery_, loss_, rec_loss_, kld_loss_, aux_loss_ = sess.run(list(self.train_ops) + list(self.losses),
                                                                           {self.train_i.X: X,
                                                                            self.train_i.Y_i: Y_i,
                                                                            self.train_i.Y_lengths: Y_lengths,
                                                                            self.train_i.Y_t: Y_t,
                                                                            self.train_i.Y_mask: Y_masks,
                                                                            self.phase: True
                                                                            })
            step_ = sess.run(tf.train.get_global_step())
            _writer.add_summary(summery_, step_)  # tf.train.get_global_step())

            if step_ % 20 == 0:
                epoch_ = U.step_to_epoch(step_, data_loader.num_line, self.flags.batch_size)
                print("Epoch %d | step %d/%d | train_loss: %.3f | rec_loss %.3f | kld_loss %3f | aux_loss %3f |" % (
                    epoch_, step_, self.flags.global_steps, loss_, rec_loss_, kld_loss_, aux_loss_))

            if step_ % 1000 == 0:
                _saver.save(sess, self.flags.ckpt_path, global_step=step_, write_meta_graph=False)
                print('model saved ...')

            if step_ >= self.flags.global_steps:
                _saver.save(sess, self.flags.ckpt_path, global_step=step_, write_meta_graph=False)
                print('train is end ...')
                break

    def infer(self):
        None

    def valid(self):
        None
