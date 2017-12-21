import tensorflow as tf

from models.Hybird_CVAE import Hybird_CVAE
from utils.DataLoader import DataLoader, Vocab, Level

flags = tf.app.flags

# hybird_cvae config
flags.DEFINE_integer('n_epochs', 10, '')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('lr', 0.001, 'learning rate')
flags.DEFINE_integer('z_size', 32, '')
flags.DEFINE_integer('seq_len', 60, '')
flags.DEFINE_integer('aux_loss_alpha', 10, '')
flags.DEFINE_string('model_name', 'Hybird_CVAE', '')
flags.DEFINE_string('ckpt_path', './results/Hybird_CVAE/ckpt/', '')

# encoder
flags.DEFINE_integer('embed_size', 80, '')
flags.DEFINE_integer('vocab_size', 20000, '')

#  decoder
flags.DEFINE_integer('rnn_hidden_size', 512, '')

FLAGS = flags.FLAGS


def main(_):
    data_loader_c = DataLoader(Vocab('hybird_cvae', Level.CHAR))
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
        summery_writer = tf.summary.FileWriter('./results/Hybird_CVAE/log/', sess.graph)
        sess.run(tf.global_variables_initializer())
        saver.save(sess, FLAGS.ckpt_path, tf.train.get_global_step())

        for data in data_loader_c.next_batch(FLAGS.batch_size, train=True):
            X, Y_i, Y_lengths, Y_t, Y_masks = data_loader_c.unpack_for_hybird_cvae(data, FLAGS.seq_len)

            _, loss_, summery_ = sess.run([model.train_op, model.train_loss, model.summary_op],
                                          {model.train_input[0]: X,
                                           model.train_input[1]: Y_i,
                                           model.train_input[2]: Y_lengths,
                                           model.train_input[3]: Y_t,
                                           model.phase: True
                                           })

            global_step_ = sess.run(tf.train.get_global_step())
            summery_writer.add_summary(summery_, global_step_)  # tf.train.get_global_step())

            if global_step_ % 10 == 0:
                epoch_ = int(global_step_ * FLAGS.batch_size / data_loader_c.num_line)
                print("Epoch %d | global_step %d | train_loss: %.3f "
                      % (epoch_, global_step_, loss_))
            saver.save(sess, FLAGS.ckpt_path, tf.train.get_global_step(), write_meta_graph=False)


if __name__ == "__main__":
    tf.app.run()
    print('end ...')
