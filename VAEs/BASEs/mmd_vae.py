import numpy as np
import tensorflow as tf

from dataset import next_batch_, imcombind_, imsave_, plot_q_z
from particle import encoder, decoder
from sampler import gaussian

flags = tf.app.flags
flags.DEFINE_integer('steps', 20000, '')
flags.DEFINE_integer('bz', 64, '')
flags.DEFINE_integer('z_dim', 16, '')
flags.DEFINE_string('log_path', './logs/mmd_vae/', '')
FLAGS = flags.FLAGS


def compute_kernel(x, y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))


def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)


class mmd_vae:
    def __init__(self):
        self.en = encoder()
        self.de = decoder()

        self.z = tf.placeholder(tf.float32, [None, FLAGS.z_dim], name='z')
        self.x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x')
        en_ = self.en(self.x)
        self.mu = tf.layers.dense(en_, FLAGS.z_dim)
        self.log_var = tf.layers.dense(en_, FLAGS.z_dim)
        eps = tf.random_normal(tf.shape(self.mu))
        self.z_latent = self.mu + tf.exp(0.5 * self.log_var) * eps
        self.rec_x, _ = self.de(self.z_latent, False)
        self.gen_x, _ = self.de(self.z)

        flat_x = tf.layers.flatten(self.x)
        flat_rx = tf.layers.flatten(self.rec_x)
        self.loss_nll = tf.reduce_mean(tf.square(flat_x - flat_rx))
        self.loss_mmd = compute_mmd(self.z, self.z_latent)
        self.loss = self.loss_nll + self.loss_mmd
        self.optim = tf.train.AdamOptimizer(1e-3).minimize(self.loss, tf.train.get_or_create_global_step())

        self.fit_summary = tf.summary.merge([
            tf.summary.scalar('loss', self.loss),
            tf.summary.scalar('loss_nll', self.loss_nll),
            tf.summary.scalar('loss_mmd', self.loss_mmd),
            tf.summary.image('x', self.x, 8),
            tf.summary.image('rec_x', self.rec_x, 8),
            tf.summary.histogram('z', self.z_latent),
            tf.summary.histogram('mu', self.mu),
            tf.summary.histogram('logvar', self.log_var)
        ])
        self.gen_summary = tf.summary.merge([
            tf.summary.image('gen_x', self.gen_x, 8)
        ])

    def fit(self, sess, local_):
        for _ in range(local_):
            x, _ = next_batch_(FLAGS.bz)
            sess.run(self.optim, {self.x: x, self.z: gaussian(FLAGS.bz, FLAGS.z_dim)})
        x, _ = next_batch_(FLAGS.bz * 5)
        return sess.run([self.loss, self.loss_nll, self.loss_mmd, self.fit_summary], {
            self.x: x, self.z: gaussian(FLAGS.bz, FLAGS.z_dim)
        })

    def gen(self, sess, bz):
        return sess.run([self.gen_x, self.gen_summary], {self.z: gaussian(bz, FLAGS.z_dim)})

    def latent_z(self, sess, bz):
        x, y = next_batch_(bz)
        return sess.run(self.z_latent, {self.x: x}), y


def main(_):
    _model = mmd_vae()
    _gpu = tf.GPUOptions(allow_growth=True)
    _saver = tf.train.Saver(pad_step_number=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=_gpu)) as sess:
        _writer = tf.summary.FileWriter(FLAGS.log_path, sess.graph)
        tf.global_variables_initializer().run()

        ckpt = tf.train.get_checkpoint_state(FLAGS.log_path)
        if ckpt and ckpt.model_checkpoint_path:
            _saver.restore(sess, FLAGS.log_path)

        _step = tf.train.get_global_step().eval()
        while True:
            if _step >= FLAGS.steps:
                break
            loss, loss_nll, loss_mmd, fit_summary = _model.fit(sess, 100)

            _step = _step + 100
            _writer.add_summary(fit_summary, _step)
            _saver.save(sess, FLAGS.log_path)
            print("Train [%d\%d] loss [%3f] loss_nll [%3f] loss_mmd [%3f]" % (
                _step, FLAGS.steps, loss, loss_nll, loss_mmd))

            images, gen_summary = _model.gen(sess, 100)
            _writer.add_summary(gen_summary)
            imsave_(FLAGS.log_path + 'train{}.png'.format(_step), imcombind_(images))

            if _step % 1000 == 0:
                latent_z, y = _model.latent_z(sess, 2000)
                plot_q_z(latent_z, y, FLAGS.log_path + 'mmd_vae_z_{}.png'.format(_step))


if __name__ == "__main__":
    tf.app.run()
