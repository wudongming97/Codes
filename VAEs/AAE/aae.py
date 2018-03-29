import tensorflow as tf

from dataset import next_batch_, imcombind_, imsave_, embedding_viz_
from particle import encoder, decoder, discriminator
from sampler import gaussian_mixture, gaussian, swiss_roll

flags = tf.app.flags
flags.DEFINE_integer('steps', 20000, '')
flags.DEFINE_integer('bz', 64, '')
flags.DEFINE_integer('z_dim', 16, '')
flags.DEFINE_float('alpha', 2, 'wgan 惩罚正则项的权重')

flags.DEFINE_string('z_dist', 'mg', '')
flags.DEFINE_string('datasets', 'mnist', '')
flags.DEFINE_string('log_path', './results_aae/', '')
FLAGS = flags.FLAGS


def z_real_(bz):
    if FLAGS.z_dist == 'g':
        return gaussian(bz, FLAGS.z_dim, 0, 2)
    elif FLAGS.z_dist == 'mg':
        return gaussian_mixture(bz, FLAGS.z_dim, 10)
    else:
        return swiss_roll(bz, FLAGS.z_dim, 10)


class aae:
    def __init__(self):
        self.en = encoder(FLAGS.z_dim)
        self.de = decoder()
        self.di = discriminator()

        self.x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x')
        self.real_z = tf.placeholder(tf.float32, [None, FLAGS.z_dim])
        self.fake_z = self.en(self.x)
        self.rec_x, _ = self.de(self.fake_z, False)
        self.gen_x, _ = self.de(self.real_z)

        self.g_loss = tf.reduce_mean(self.di(self.fake_z, False))
        self.d_loss = tf.reduce_mean(self.di(self.real_z)) - tf.reduce_mean(self.di(self.fake_z)) + self._dd()
        self.a_loss = tf.reduce_mean(tf.square(self.rec_x - self.x))

        self.g_optim = tf.train.AdamOptimizer(1e-3).minimize(self.g_loss, var_list=self.en.vars)
        self.d_optim = tf.train.AdamOptimizer(1e-3).minimize(self.d_loss, var_list=self.di.vars)
        self.a_optim = tf.train.AdamOptimizer(1e-3).minimize(self.a_loss, tf.train.get_or_create_global_step())

        self.fit_summary = tf.summary.merge([
            tf.summary.scalar('g_loss', self.g_loss),
            tf.summary.scalar('d_loss', self.d_loss),
            tf.summary.scalar('a_loss', self.a_loss),
            tf.summary.histogram('real_z', self.real_z),
            tf.summary.histogram('fake_z', self.fake_z),
            tf.summary.image('x', self.x, 8),
            tf.summary.image('rec_x', self.rec_x, 8)
        ])
        self.gen_summary = tf.summary.merge([
            tf.summary.image('gen_x', self.gen_x, 8)
        ])

    def fit(self, sess, local_):
        for _ in range(local_):
            x_real, _ = next_batch_(FLAGS.bz)
            sess.run(self.a_optim, {self.x: x_real})
            for _ in range(3):
                sess.run(self.d_optim, {self.x: x_real, self.real_z: z_real_(FLAGS.bz)})
            sess.run(self.g_optim, {self.x: x_real})

        x_real, _ = next_batch_(FLAGS.bz * 5)
        return sess.run([self.a_loss, self.g_loss, self.d_loss, self.fit_summary], {
            self.x: x_real, self.real_z: z_real_(FLAGS.bz * 5)})

    def gen(self, sess, bz):
        return sess.run([self.gen_x, self.gen_summary], {self.real_z: z_real_(bz)})

    def latent_z(self, sess, bz):
        x, y = next_batch_(bz)
        return sess.run(self.fake_z, {self.x: x}), y

    def _dd(self):
        eps = tf.random_uniform([], 0.0, 1.0)
        z_hat = eps * self.real_z + (1 - eps) * self.fake_z
        d_hat = self.di(z_hat)
        dx = tf.layers.flatten(tf.gradients(d_hat, z_hat)[0])
        dx = tf.sqrt(tf.reduce_sum(tf.square(dx), axis=1))
        dx = tf.reduce_mean(tf.square(dx - 1.0) * FLAGS.alpha)
        return dx


def main(_):
    _model = aae()
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
            a_loss, g_loss, d_loss, fit_summary = _model.fit(sess, 100)

            _step = _step + 100
            _writer.add_summary(fit_summary, _step)
            _saver.save(sess, FLAGS.log_path)
            print("Train [%d\%d] g_loss [%3f] d_loss [%3f] a_loss [%3f]" % (_step, FLAGS.steps, g_loss, d_loss, a_loss))

            images, gen_summary = _model.gen(sess, 100)
            _writer.add_summary(gen_summary)
            imsave_(FLAGS.log_path + 'train{}.png'.format(_step), imcombind_(images))

            if _step % 500 == 0:
                latent_z, y = _model.latent_z(sess, 2000)
                embedding_viz_(latent_z, y, _step, FLAGS.log_path)


if __name__ == "__main__":
    tf.app.run()
