# -*- coding: utf-8 -*-
import tensorflow as tf
# ----------- mnist ---------
from mnist.data import data_
from mnist.dcgan import D, G
# ----------- mnist ---------

type_ = 'wgan'
flags = tf.app.flags
flags.DEFINE_string('ckpt_path', './results/{}/ckpt/'.format(type_), '')
flags.DEFINE_string('logs_path', './results/{}/logs/'.format(type_), '')

flags.DEFINE_integer('steps', 100000, '')
flags.DEFINE_integer('batch_sz', 32, '')
flags.DEFINE_float('lr', 0.001, '')
flags.DEFINE_float('scale', 10.0, '')
flags.DEFINE_integer('d_iters', 3, '')
FLAGS = flags.FLAGS


def model_(t):
    import wgan_gp as wgan_gp
    import wgan as wgan
    if t == 'wgan-gp':
        return wgan_gp.wgan(G(), D(), data_, FLAGS),
    elif t == 'wgan':
        return wgan.wgan(G(), D(), data_, FLAGS)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    sess_conf = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            allow_growth=True,
            per_process_gpu_memory_fraction=0.6,
        )
    )

    graph = tf.Graph()
    with graph.as_default():
        model = model_(type_)
        saver = tf.train.Saver(  # max_to_keep=5,
            keep_checkpoint_every_n_hours=1,
            pad_step_number=True)

    with tf.Session(graph=graph, config=sess_conf) as sess:
        writer = tf.summary.FileWriter(FLAGS.logs_path, sess.graph)
        sess.run(tf.global_variables_initializer())

        tf.train.export_meta_graph(FLAGS.ckpt_path + 'wgan.meta')
        ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        if sess.run(tf.train.get_global_step() < FLAGS.steps):
            model.train(sess, writer, saver)
        else:
            model.gen(sess, 36, FLAGS.steps + 1)


if __name__ == "__main__":
    tf.app.run()
