# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from utils import Utils as U, DataLoader as D
from Hybird_tVAE import Hybird_tVAE

flags = tf.app.flags

# data_loader
# data_loader = D.DataLoader(D.Vocab('europarl_hybird_tvae', D.Level.CHAR))
data_loader = D.DataLoader(D.Vocab('train128', D.Level.CHAR))

# hybird_tvae config
flags.DEFINE_string('model_name', 'Hybird_tVAE', '')
flags.DEFINE_string('ckpt_path', './results/Hybird_tVAE/ckpt_aux02_f2_1/', '')
flags.DEFINE_string('logs_path', './results/Hybird_tVAE/logs_aux02_f2_1/', '')

flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('steps', U.epoch_to_step(5, data_loader.train_size, batch_size=64), '')
flags.DEFINE_integer('lr', 0.001, 'learning rate')
flags.DEFINE_integer('z_size', 32, '')
flags.DEFINE_integer('seq_len', 129, '')
flags.DEFINE_integer('rnn_num', 2, '')
flags.DEFINE_integer('embed_size', 80, '')
flags.DEFINE_integer('vocab_size', data_loader.vocab_size, '')
flags.DEFINE_integer('rnn_hidden_size', 512, '')
flags.DEFINE_float('alpha', 0.2, '')  # 额外的重构误差的权重
flags.DEFINE_float('beta', 1.0, '')
flags.DEFINE_float('mu_foring', 8.0, '') # 不用mu_foring
flags.DEFINE_float('gamma', 2.0, '')
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
        model = Hybird_tVAE(FLAGS)
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
            # # 1)用标准正太分布来生成样本
            sentences = model.infer_by_normal(sess, data_loader)
            for ix in range(len(sentences)):
                t_s_ = sentences[ix].split(D.E_TOKEN)[0]
                print("{:3d}. {}".format(ix, t_s_))
            # # 2)infer by encoder, 直接从训练集中取数据
            data = data_loader.one_batch(FLAGS.batch_size)
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
    print('end ...')
