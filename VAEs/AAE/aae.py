import tensorflow as tf

from config import FLAGS
from data import next_batch_, z_real_
from particle import encoder, decoder, discriminator


class aae:
    def __init__(self):
        self.en = encoder()
        self.de = decoder()
        self.di = discriminator()

        self.x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x')
        self.real_z = tf.placeholder(tf.float32, [None, FLAGS.z_dim])
        self.fake_z = self.en(self.x)
        self.rec_x = self.de(self.fake_z, False)
        self.gen_x = self.de(self.real_z)

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
                sess.run(self.d_optim, {self.x: x_real, self.real_z: z_real_(FLAGS.bz, FLAGS.z_dim)})
            sess.run(self.g_optim, {self.x: x_real})

        x_real, _ = next_batch_(FLAGS.bz * 5)
        return sess.run([self.a_loss, self.g_loss, self.d_loss, self.fit_summary], {
                    self.x: x_real, self.real_z: z_real_(FLAGS.bz * 5, FLAGS.z_dim)})

    def gen(self, sess, bz):
        return sess.run([self.gen_x, self.gen_summary], {self.real_z: z_real_(bz, FLAGS.z_dim)})

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
