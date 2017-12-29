# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

import utils.Utils as U
from models.VAE_katong import VAE

flags = tf.app.flags
flags.DEFINE_string('model_name', 'VAE_katong', '')
flags.DEFINE_integer('steps', 20000, '')
flags.DEFINE_integer('z_size', 32, '')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_float('lr', 0.0005, '')
flags.DEFINE_float('beta', 1.0, '')

flags.DEFINE_string('images_path', 'E:\\data\\katong\\', '')
flags.DEFINE_string('file_name', './data/katong.tfrecords', '')

flags.DEFINE_string('ckpt_path', './results/VAE_katong/ckpt/', '')
flags.DEFINE_string('logs_path', './results/VAE_katong/logs/', '')
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

    # 写tfrecord文件
    if not os.path.exists(FLAGS.file_name):
        U.fixed_image_writer(FLAGS.images_path, FLAGS.file_name)

    graph = tf.Graph()
    with graph.as_default():

        with tf.name_scope('fixed_image_reader'):
            images = U.fixed_image_reader([FLAGS.file_name], [96, 96, 3])
            batch_images = tf.train.shuffle_batch([images],
                                                  batch_size=FLAGS.batch_size,
                                                  capacity=2000,
                                                  min_after_dequeue=100)
        model = VAE(FLAGS, batch_images)

        saver = tf.train.Saver(  # max_to_keep=5,
            keep_checkpoint_every_n_hours=1,
            pad_step_number=True)

    with tf.Session(graph=graph, config=sess_conf) as sess:
        train_writer = tf.summary.FileWriter(FLAGS.logs_path, sess.graph)
        sess.run(tf.global_variables_initializer())

        tf.train.export_meta_graph(FLAGS.ckpt_path + FLAGS.model_name + '.meta')
        ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        print('\nbegin fit ...')
        model.fit(sess, train_writer, saver)


if __name__ == "__main__":
    tf.app.run()
