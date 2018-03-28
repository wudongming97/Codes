import numpy as np
import tensorflow as tf

from config import FLAGS
from dataset import next_batch_, imcombind_, imsave_, embedding_viz_
from particle import encoder, decoder

LogPath = FLAGS.log_path + 'VAE/'


class vae:
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
        self.rec_x, logits = self.de(self.z_latent, False)
        self.gen_x, _ = self.de(self.z)

        flat_x = tf.layers.flatten(self.x)
        flat_logits = tf.layers.flatten(logits)
        self.rec_loss = tf.reduce_mean(tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=flat_logits, labels=flat_x), 1))
        self.kld_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(
            self.log_var - tf.square(self.mu) - tf.exp(self.log_var) + 1, 1))
        self.loss = self.rec_loss + self.kld_loss
        self.optim = tf.train.AdamOptimizer(1e-3).minimize(self.loss, tf.train.get_or_create_global_step())

        self.fit_summary = tf.summary.merge([
            tf.summary.scalar('loss', self.loss),
            tf.summary.scalar('rec_loss', self.rec_loss),
            tf.summary.scalar('kld_loss', self.kld_loss),
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
            sess.run(self.optim, {self.x: x})
        x, _ = next_batch_(FLAGS.bz * 5)
        return sess.run([self.loss, self.rec_loss, self.kld_loss, self.fit_summary], {
            self.x: x
        })

    def gen(self, sess, bz):
        z = np.random.normal(0, 1, [bz, FLAGS.z_dim])
        return sess.run([self.gen_x, self.gen_summary], {self.z: z})

    def latent_z(self, sess, bz):
        x, y = next_batch_(bz)
        return sess.run(self.z_latent, {self.x: x}), y


def main(_):
    _model = vae()
    _gpu = tf.GPUOptions(allow_growth=True)
    _saver = tf.train.Saver(pad_step_number=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=_gpu)) as sess:
        _writer = tf.summary.FileWriter(LogPath, sess.graph)
        tf.global_variables_initializer().run()

        ckpt = tf.train.get_checkpoint_state(LogPath)
        if ckpt and ckpt.model_checkpoint_path:
            _saver.restore(sess, LogPath)

        _step = tf.train.get_global_step().eval()
        while True:
            if _step >= FLAGS.steps:
                break
            loss, rec_loss, kld_loss, fit_summary = _model.fit(sess, 100)

            _step = _step + 100
            _writer.add_summary(fit_summary, _step)
            _saver.save(sess, LogPath)
            print("Train [%d\%d] loss [%3f] rec_loss [%3f] kld_loss [%3f]" % (
            _step, FLAGS.steps, loss, rec_loss, kld_loss))

            images, gen_summary = _model.gen(sess, 100)
            _writer.add_summary(gen_summary)
            imsave_(LogPath + 'train{}.png'.format(_step), imcombind_(images))

            if _step % 500 == 0:
                latent_z, y = _model.latent_z(sess, 1000)
                embedding_viz_(latent_z, y, _step, LogPath)


if __name__ == "__main__":
    tf.app.run()
