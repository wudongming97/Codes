from data import next_batch_, imcombind_, imsave_
from ops import *
from sampler import gaussian

flags = tf.app.flags
flags.DEFINE_string('log_path', './logs/wgan_gp/', '')

flags.DEFINE_integer('steps', 20000, '')
flags.DEFINE_integer('bz', 100, '')
flags.DEFINE_integer('z_dim', 64, '')
flags.DEFINE_float('lr', 0.001, '')
flags.DEFINE_float('scale', 2.0, '')
FLAGS = flags.FLAGS


class G(object):
    def __init__(self):
        self.name = 'mnist/g_net'

    def __call__(self, z):
        with tf.variable_scope(self.name):
            fc2 = relu(dense(z, 7 * 7 * 128))
            fc2 = tf.reshape(fc2, [-1, 7, 7, 128])
            cv1 = relu(conv2d_t(fc2, 32, [4, 4], [2, 2], 'SAME'))
            fake = sigmoid(conv2d_t(cv1, 1, [4, 4], [2, 2], 'SAME'))
            return fake

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class D(object):
    def __init__(self):
        self.name = 'mnist/d_net'

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name, reuse=reuse):
            cv1 = lrelu(conv2d(x, 64, [4, 4], [2, 2]))
            fc1 = lrelu(dense(flatten(cv1), 784))
            fc2 = dense(fc1, 1)

            return fc2

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class wgan_gp(object):
    def __init__(self):
        self.G = G()
        self.D = D()

        self.real = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x')
        self.z = tf.placeholder(tf.float32, [None, FLAGS.z_dim], name='z')
        self.fake = self.G(self.z)

        self.g_loss = tf.reduce_mean(self.D(self.fake, reuse=False))
        self.d_loss = tf.reduce_mean(self.D(self.real)) - tf.reduce_mean(self.D(self.fake)) + self._dx()
        self.d_adam = tf.train.AdamOptimizer(FLAGS.lr).minimize(self.d_loss, var_list=self.D.vars)
        self.g_adam = tf.train.AdamOptimizer(FLAGS.lr).minimize(self.g_loss, var_list=self.G.vars,
                                                                global_step=tf.train.get_or_create_global_step())

        self.fit_summary = tf.summary.merge([
            tf.summary.scalar('g_loss', self.g_loss),
            tf.summary.scalar('d_loss', self.d_loss),
            tf.summary.image('X', self.real, 16),
            tf.summary.image('fake', self.fake, 16)
        ])

    def _dx(self):
        eps = tf.random_uniform([], 0.0, 1.0)
        x_hat = eps * self.real + (1 - eps) * self.fake
        d_hat = self.D(x_hat)
        dx = tf.layers.flatten(tf.gradients(d_hat, x_hat)[0])
        dx = tf.sqrt(tf.reduce_sum(tf.square(dx), axis=1))
        dx = tf.reduce_mean(tf.square(dx - 1.0) * FLAGS.scale)
        return dx

    def gen(self, sess, bz):
        return sess.run(self.fake, feed_dict={self.z: gaussian(bz, FLAGS.z_dim)})

    def fit(self, sess, local_):
        for _ in range(local_):
            x_real, _ = next_batch_(FLAGS.bz)
            for _ in range(3):
                sess.run(self.d_adam,
                         feed_dict={self.real: x_real, self.z: gaussian(FLAGS.bz, FLAGS.z_dim)})
            sess.run(self.g_adam,
                     feed_dict={self.real: x_real, self.z: gaussian(FLAGS.bz, FLAGS.z_dim)})
        x_real, _ = next_batch_(FLAGS.bz)
        return sess.run([self.d_loss, self.g_loss, self.fit_summary], feed_dict={
            self.real: x_real, self.z: gaussian(FLAGS.bz, FLAGS.z_dim)})


def main(_):
    _model = wgan_gp()
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
            d_loss, g_loss, fit_summary = _model.fit(sess, 100)

            _step = _step + 100
            _writer.add_summary(fit_summary, _step)
            print("Train [%d\%d] g_loss [%3f] d_loss [%3f]" % (_step, FLAGS.steps, g_loss, d_loss))

            images = _model.gen(sess, 100)
            imsave_(FLAGS.log_path + 'train_{}.png'.format(_step), imcombind_(images))
            _saver.save(sess, FLAGS.log_path) if _step % 5000 == 0 else None


if __name__ == "__main__":
    tf.app.run()
