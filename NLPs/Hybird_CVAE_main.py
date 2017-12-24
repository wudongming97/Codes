# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import utils.DataLoader as D
import utils.Utils as U
from models.Hybird_CVAE import Hybird_CVAE

flags = tf.app.flags

# data_loader
data_loader_c = D.DataLoader(D.Vocab('hybird_cvae', D.Level.CHAR))

# hybird_cvae config
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('steps', U.epoch_to_step(18, data_loader_c.num_line, batch_size=32), '')
flags.DEFINE_integer('lr', 0.001, 'learning rate')
flags.DEFINE_integer('z_size', 32, '')
flags.DEFINE_integer('seq_len', 60, '')
flags.DEFINE_float('alpha', 0.2, '')
flags.DEFINE_float('beta', 0.02, '')
flags.DEFINE_string('model_name', 'Hybird_CVAE', '')
flags.DEFINE_string('ckpt_path', './results/Hybird_CVAE/ckpt/', '')
flags.DEFINE_string('logs_path', './results/Hybird_CVAE/log/', '')

# encoder
flags.DEFINE_integer('rnn_num', 2, '')
flags.DEFINE_integer('embed_size', 80, '')
flags.DEFINE_integer('vocab_size', data_loader_c.vocab_size, '')

#  decoder
flags.DEFINE_integer('rnn_hidden_size', 512, '')
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
        model = Hybird_CVAE(FLAGS)
        saver = tf.train.Saver(  # max_to_keep=5,
            keep_checkpoint_every_n_hours=1,
            pad_step_number=True)

    with tf.Session(graph=graph, config=sess_conf) as sess:
        summery_writer = tf.summary.FileWriter(FLAGS.logs_path, sess.graph)
        sess.run(tf.global_variables_initializer())

        tf.train.export_meta_graph(FLAGS.ckpt_path + FLAGS.model_name + '.meta')
        ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        if model.train_is_ok(sess):
            # some interest test

            # 1)用标准正太分布来生成样本
            print('\n=============begin infer_by_normal============\n')
            sentences = model.infer_by_normal(sess, data_loader_c)
            for ix in range(len(sentences)):
                t_s_ = sentences[ix].split(D.E_TOKEN)[0]
                print("{:3d}. {}".format(ix, t_s_))

            # 2)infer by encoder, 直接从训练集中取数据
            print('\n============begin infer_by_encoder============\n')
            for i in data_loader_c.next_batch(FLAGS.batch_size, True):
                input_s_ = data_loader_c.to_seqs(i)
                out_s_ = model.infer_by_encoder(sess, data_loader_c, input_s_)
                pair_s_ = zip(input_s_, out_s_)
                for ix, _p in enumerate(pair_s_):
                    print('In {:3d}: {}'.format(ix, _p[0]))
                    t_s_ = _p[1].split(D.E_TOKEN)[0]
                    print('Out{:3d}: {}'.format(ix, t_s_))
                break

            # 3)z空间的线性渐变，查看输出的连续变化
            print('\n=======begin linear infer between z1 and z2====\n')
            linear = np.linspace(-0.0001, 0.0001, num=FLAGS.batch_size)
            z = np.tile(linear, [FLAGS.z_size, 1]).transpose()
            sentences = model.infer_by_z(sess, data_loader_c, z)
            pair_s_ = zip(linear, sentences)
            for ix, _p in enumerate(pair_s_):
                t_s_ = _p[1].split(D.E_TOKEN)[0]
                print('{:3d}, Z={:.3f}, Out: {}'.format(ix, _p[0], t_s_))

        else:
            print('\nbegin fit ...')
            model.fit(sess, data_loader_c, summery_writer, saver)

            print('\nbegin valid ...')
            model.valid(sess, data_loader_c)


if __name__ == "__main__":
    tf.app.run()
    print('end ...')
