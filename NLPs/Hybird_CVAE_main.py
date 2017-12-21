import tensorflow as tf

import utils.DataLoader as D
import utils.Utils as U
from models.Hybird_CVAE import Hybird_CVAE

flags = tf.app.flags

# data_loader
data_loader_c = D.DataLoader(D.Vocab('hybird_cvae', D.Level.CHAR))

# hybird_cvae config
flags.DEFINE_integer('global_steps', U.epoch_to_step(10, data_loader_c.num_line, batch_size=32), '')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('lr', 0.001, 'learning rate')
flags.DEFINE_integer('z_size', 32, '')
flags.DEFINE_integer('seq_len', 60, '')
flags.DEFINE_float('alpha', 0.2, '')
flags.DEFINE_string('model_name', 'Hybird_CVAE', '')
flags.DEFINE_string('ckpt_path', './results/Hybird_CVAE/ckpt/', '')
flags.DEFINE_string('logs_path', './results/Hybird_CVAE/log/', '')

# encoder
flags.DEFINE_integer('embed_size', 80, '')
flags.DEFINE_integer('vocab_size', data_loader_c.vocab_size, '')
flags.DEFINE_integer('kld_anneal_start', 1000 * 8, '')
flags.DEFINE_integer('kld_anneal_end', 1000 * 13, '')

#  decoder
flags.DEFINE_integer('rnn_hidden_size', 512, '')

FLAGS = flags.FLAGS


def main(_):
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
        saver = tf.train.Saver(#max_to_keep=5,
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
            print('begin infer ...')
            model.infer()
        else:
            print('begin fit ...')
            model.fit(sess, data_loader_c, summery_writer, saver)


if __name__ == "__main__":
    tf.app.run()
    print('end ...')
