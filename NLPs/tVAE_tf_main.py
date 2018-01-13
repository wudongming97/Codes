# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import utils.DataLoader as D
import utils.Utils as U
from models.tVAE_tf import tVAE_tf

data_loader = D.DataLoader(D.Vocab('europarl_tvae_tf', D.Level.WORD))

flags = tf.app.flags

flags.DEFINE_string('model_name', 'tVAE_tf', '')
flags.DEFINE_string('ckpt_path', './results/tVAE_tf/ckpt/', '')
flags.DEFINE_string('logs_path', './results/tVAE_tf/logs/', '')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('steps', U.epoch_to_step(10, data_loader.train_size, batch_size=32), '')
flags.DEFINE_integer('lr', 0.001, 'learning rate')
flags.DEFINE_integer('z_size', 32, '')
flags.DEFINE_integer('max_seq_len', 15, '')
flags.DEFINE_integer('n_layers', 1, '')
flags.DEFINE_integer('embed_size', 512, '')
flags.DEFINE_integer('vocab_size', data_loader.vocab_size, '')
flags.DEFINE_integer('hidden_size', 512, '')
flags.DEFINE_bool('kl_anealing', False, '是否使用kl_anealing技巧')
flags.DEFINE_float('beta', 1.0, 'kl_loss coef')
flags.DEFINE_float('gamma', 5, '')
flags.DEFINE_float('word_drop', '0.2', 'word dropout probability')

FLAGS = flags.FLAGS

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    # gpu memory
    sess_conf = tf.ConfigProto(
        # log_device_placement=True,
        gpu_options=tf.GPUOptions(
            allow_growth=True,
            per_process_gpu_memory_fraction=0.6,
        )
    )

    graph = tf.Graph()
    with graph.as_default():
        model = tVAE_tf(FLAGS)
        saver = tf.train.Saver(  # max_to_keep=5,
            keep_checkpoint_every_n_hours=1,
            pad_step_number=True)

    with tf.Session(graph=graph, config=sess_conf) as sess:
        train_writer = tf.summary.FileWriter(FLAGS.logs_path + 'train/', sess.graph)
        valid_writer = tf.summary.FileWriter(FLAGS.logs_path + 'valid/')
        sess.run(tf.global_variables_initializer())

        tf.train.export_meta_graph(FLAGS.ckpt_path + FLAGS.model_name + '.meta')
        ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        if model.train_is_ok(sess):
            # 查看随机生成的效果
            sentences = model.infer_by_normal(sess, data_loader)
            for ix in range(len(sentences)):
                t_s_ = sentences[ix].split(D.E_TOKEN)[0]
                print("{:3d}. {}".format(ix, t_s_))

            # 查看重构的情况
            data = data_loader.one_batch(FLAGS.batch_size, True)
            input_s_ = sorted(data_loader.to_seqs(data), key=len, reverse=True)
            out_s_ = model.infer_by_encoder(sess, data_loader, input_s_)
            pair_s_ = zip(input_s_, out_s_)
            for ix, _p in enumerate(pair_s_):
                print('In {:3d}: {}'.format(ix, _p[0]))
                t_s_ = _p[1].split(D.E_TOKEN)[0]
                print('Out{:3d}: {}'.format(ix, t_s_))
        else:
            print('\nbegin fit ...')
            model.fit(sess, data_loader, train_writer, saver)
            model.valid(sess, data_loader, valid_writer)


if __name__ == "__main__":
    tf.app.run()
