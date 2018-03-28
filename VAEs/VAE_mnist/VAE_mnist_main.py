# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import Utils as U
from VAE_mnist import VAE_mnist

flags = tf.app.flags
flags.DEFINE_integer('steps', 200000, '')
flags.DEFINE_integer('z_size', 20, '')
flags.DEFINE_integer('batch_size', 60, '')
flags.DEFINE_float('lr', 0.0005, '')
flags.DEFINE_float('beta', 1.0, '')

# ================== VAE_mnist =======================
flags.DEFINE_string('model_name', 'VAE_mnist', '')
flags.DEFINE_string('file_name', './mnist.tfrecords', '')
flags.DEFINE_string('ckpt_path', './results/VAE_mnist/ckpt/', '')
flags.DEFINE_string('logs_path', './results/VAE_mnist/logs/', '')
# ================= VAE_mnist ========================
FLAGS = flags.FLAGS
SHAPE = [28, 28, 1]


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    sess_conf = tf.ConfigProto(
        # log_device_placement=True,
        gpu_options=tf.GPUOptions(
            allow_growth=True,
            per_process_gpu_memory_fraction=0.6,
        )
    )

    graph = tf.Graph()
    with graph.as_default():
        with tf.name_scope('fixed_image_reader'):
            images = U.fixed_image_reader([FLAGS.file_name], SHAPE)
            batch_images = tf.train.shuffle_batch([images],
                                                  batch_size=FLAGS.batch_size,
                                                  capacity=2000,
                                                  min_after_dequeue=100)
        model = VAE_mnist(FLAGS, batch_images)

        saver = tf.train.Saver(
            keep_checkpoint_every_n_hours=1,
            pad_step_number=True)

    with tf.Session(graph=graph, config=sess_conf) as sess:
        _writer = tf.summary.FileWriter(FLAGS.logs_path, sess.graph)
        sess.run(tf.global_variables_initializer())

        tf.train.export_meta_graph(FLAGS.ckpt_path + FLAGS.model_name + '.meta')
        ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        print('\nbegin fit ...')
        if not model.train_is_ok(sess):
            model.fit(sess, _writer, saver)


if __name__ == "__main__":
    tf.app.run()
