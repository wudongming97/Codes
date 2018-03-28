import tensorflow as tf

from aae import aae
from config import FLAGS
from data import imcombind_, imsave_

SESS_CONF = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        allow_growth=True,
        per_process_gpu_memory_fraction=0.6))


def main(_):
    graph = tf.Graph()
    with graph.as_default():
        _model = aae()
        _saver = tf.train.Saver(pad_step_number=True)

    with tf.Session(graph=graph, config=SESS_CONF) as sess:
        _writer = tf.summary.FileWriter(FLAGS.log_path, sess.graph)
        tf.global_variables_initializer().run()

        ckpt = tf.train.get_checkpoint_state(FLAGS.log_path)
        if ckpt and ckpt.model_checkpoint_path:
            _saver.restore(sess, ckpt.model_checkpoint_path)

        _step = tf.train.get_global_step().eval()
        for _step in range(_step, FLAGS.steps // 20):
            a_loss, g_loss, d_loss, fit_summary = _model.fit(sess, 20)
            _writer.add_summary(fit_summary, _step)
            _saver.save(sess, FLAGS.log_path)
            print("Train [%d\%d] g_loss [%3f] d_loss [%3f] a_loss [%3f]" % (_step, FLAGS.steps, g_loss, d_loss, a_loss))

            images, gen_summary = _model.gen(sess, 100)
            _writer.add_summary(gen_summary)
            imsave_(FLAGS.log_path + 'train{}.png'.format(_step), imcombind_(images))


if __name__ == "__main__":
    tf.app.run()
