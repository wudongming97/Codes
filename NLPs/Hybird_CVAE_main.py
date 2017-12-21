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
flags.DEFINE_integer('aux_loss_alpha', 10, '')
flags.DEFINE_string('model_name', 'Hybird_CVAE', '')
flags.DEFINE_string('ckpt_path', './results/Hybird_CVAE/ckpt/', '')
flags.DEFINE_string('log_path', './results/Hybird_CVAE/log/', '')

# encoder
flags.DEFINE_integer('embed_size', 80, '')
flags.DEFINE_integer('vocab_size', data_loader_c.vocab_size, '')

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
        saver = tf.train.Saver(max_to_keep=5,
                               keep_checkpoint_every_n_hours=2,
                               pad_step_number=True)

    with tf.Session(graph=graph, config=sess_conf) as sess:
        summery_writer = tf.summary.FileWriter(FLAGS.log_path, sess.graph)
        sess.run(tf.global_variables_initializer())
        saver.save(sess, FLAGS.ckpt_path, tf.train.get_global_step())

        for data in data_loader_c.next_batch(FLAGS.batch_size, train=True):
            X, Y_i, Y_lengths, Y_t, Y_masks = data_loader_c.unpack_for_hybird_cvae(data, FLAGS.seq_len)

            _, loss_, summery_ = sess.run([model.train_op, model.train_loss, model.summary_op],
                                          {model.train_input[0]: X,
                                           model.train_input[1]: Y_i,
                                           model.train_input[2]: Y_lengths,
                                           model.train_input[3]: Y_t,
                                           model.train_input[4]: Y_masks,
                                           model.phase: True
                                           })

            step_ = sess.run(tf.train.get_global_step())
            summery_writer.add_summary(summery_, step_)  # tf.train.get_global_step())

            if step_ % 10 == 0:
                epoch_ = U.step_to_epoch(step_, data_loader_c.num_line, FLAGS.batch_size)
                print("Epoch %d | step %d/%d | train_loss: %.3f "
                      % (epoch_, step_, FLAGS.global_steps, loss_))
            if step_ >= FLAGS.global_steps:
                saver.save(sess, FLAGS.ckpt_path, tf.train.get_global_step(), write_meta_graph=False)
                print('Training is end ...')
                break


if __name__ == "__main__":
    tf.app.run()
    print('end ...')
