# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import experiments.Hybird_tVAE_test as T
import utils.DataLoader as D
import utils.Utils as U
from models.Hybird_tVAE import Hybird_tVAE

flags = tf.app.flags

# data_loader
data_loader = D.DataLoader(D.Vocab('europarl_hybird_tvae', D.Level.CHAR))
ptb_loader = D.DataLoader(D.Vocab('ptb_hybird_tvae', D.Level.CHAR))

# hybird_tvae config
flags.DEFINE_string('model_name', 'Hybird_tVAE', '')
flags.DEFINE_string('ckpt_path', './results/Hybird_tVAE/ckpt/', '')
flags.DEFINE_string('logs_path', './results/Hybird_tVAE/log/', '')

flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('steps', U.epoch_to_step(20, data_loader.train_size, batch_size=32), '')
flags.DEFINE_integer('lr', 0.001, 'learning rate')
flags.DEFINE_integer('z_size', 32, '')
flags.DEFINE_integer('seq_len', 60, '')
flags.DEFINE_integer('rnn_num', 2, '')
flags.DEFINE_integer('embed_size', 80, '')
flags.DEFINE_integer('vocab_size', data_loader.vocab_size, '')
flags.DEFINE_integer('rnn_hidden_size', 512, '')
flags.DEFINE_float('alpha', 0.2, '')
flags.DEFINE_float('beta', 1, '')
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
        summery_writer = tf.summary.FileWriter(FLAGS.logs_path, sess.graph)
        sess.run(tf.global_variables_initializer())

        tf.train.export_meta_graph(FLAGS.ckpt_path + FLAGS.model_name + '.meta')
        ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        if model.train_is_ok(sess):
            # # 1)用标准正太分布来生成样本
            # T.infer_by_normal_test(model, sess, data_loader)
            # # 2)infer by encoder, 直接从训练集中取数据
            T.infer_by_encoder_test(model, sess, data_loader, FLAGS.batch_size)
            # # 3)infer by encoder，从另外一个不同的数据集取数据
            # T.infer_by_encoder_test(model, sess, ptb_loader, FLAGS.batch_size)
            # # 4)z空间的线性渐变，查看输出的连续变化
            # T.infer_by_linear_z_test(model, sess, data_loader, FLAGS.batch_size, FLAGS.z_size)
            T.infer_by_same_test(model, sess, data_loader, 'the vote will take place tomorrow .', FLAGS.batch_size)
            # T.infer_by_same_test(model, sess, data_loader, 'i would like to make four point .', FLAGS.batch_size)


        else:
            print('\nbegin fit ...')
            model.fit(sess, data_loader, summery_writer, saver)
            model.valid(sess, europarl_valid_loader)


if __name__ == "__main__":
    tf.app.run()
    print('end ...')
