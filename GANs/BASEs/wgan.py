import tensorflow as tf

from dataset import next_batch_, imcombind_, imsave_
from sampler import gaussian

flags = tf.app.flags
flags.DEFINE_string('log_path', './logs/wgan/', '')

flags.DEFINE_integer('steps', 20000, '')
flags.DEFINE_integer('bz', 100, '')
flags.DEFINE_integer('z_dim', 64, '')
flags.DEFINE_float('lr', 0.001, '')
flags.DEFINE_float('clip_v', 0.02, '')
FLAGS = flags.FLAGS


class G(object):
    def __init__(self, phase):
        self.name = 'mnist/g_net'
        self.phase = phase

    def __call__(self, z):
        with tf.variable_scope(self.name):
            fc1 = tf.nn.relu(bn(tf.layers.dense(z, 1024), self.phase))
            fc2 = tf.nn.relu(bn(tf.layers.dense(fc1, 7 * 7 * 128), self.phase))
            fc2 = tf.reshape(fc2, [-1, 7, 7, 128])
            cv1 = tf.nn.relu(bn(tf.layers.conv2d_transpose(fc2, 64, [4, 4], [2, 2], 'SAME'), self.phase))
            fake = tf.layers.conv2d_transpose(cv1, 1, [4, 4], [2, 2], 'SAME', activation=tf.nn.sigmoid)
            return fake

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class D(object):
    def __init__(self, phase):
        self.name = 'mnist/d_net'
        self.phase = phase

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name, reuse=reuse):
            cv1 = tf.nn.leaky_relu(bn(tf.layers.conv2d(x, 64, [4, 4], [2, 2]), self.phase))
            cv2 = tf.nn.leaky_relu(bn(tf.layers.conv2d(cv1, 128, [4, 4], [2, 2]), self.phase))
            fc1 = tf.nn.leaky_relu(bn(tf.layers.dense(tf.layers.flatten(cv2), 1024), self.phase))
            fc2 = tf.layers.dense(fc1, 1)

            return fc2

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class wgan(object):
    def __init__(self):
        self.phase = tf.placeholder(tf.bool, name='phase')
        self.G = G(self.phase)
        self.D = D(self.phase)

        self.real = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x')
        self.z = tf.placeholder(tf.float32, [None, FLAGS.z_dim], name='z')
        self.fake = self.G(self.z)

        self.g_loss = tf.reduce_mean(self.D(self.fake, reuse=False))
        self.d_loss = tf.reduce_mean(self.D(self.real)) - tf.reduce_mean(self.D(self.fake))
        self.d_clip = [v.assign(tf.clip_by_value(v, -FLAGS.clip_v, FLAGS.clip_v)) for v in self.D.vars]
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.d_optim = tf.train.RMSPropOptimizer(FLAGS.lr).minimize(self.d_loss, var_list=self.D.vars)
            self.g_optim = tf.train.RMSPropOptimizer(FLAGS.lr).minimize(self.g_loss, var_list=self.G.vars,
                                                                        global_step=tf.train.get_or_create_global_step())
        self.fit_summary = tf.summary.merge([
            tf.summary.scalar('g_loss', self.g_loss),
            tf.summary.scalar('d_loss', self.d_loss),
            tf.summary.image('X', self.real, 16),
            tf.summary.image('fake', self.fake, 16)
        ])

    def gen(self, sess, bz):
        return sess.run(self.fake, feed_dict={self.z: gaussian(bz, FLAGS.z_dim), self.phase: False})

    def fit(self, sess, local_):
        for _ in range(local_):

            x_real, _ = next_batch_(FLAGS.bz)
            for _ in range(3):
                sess.run(self.d_optim,
                         feed_dict={self.real: x_real, self.z: gaussian(FLAGS.bz, FLAGS.z_dim), self.phase: True})
                sess.run(self.d_clip)
            sess.run(self.g_optim,
                     feed_dict={self.real: x_real, self.z: gaussian(FLAGS.bz, FLAGS.z_dim), self.phase: True})

        x_real, _ = next_batch_(FLAGS.bz)
        return sess.run([self.d_loss, self.g_loss, self.fit_summary], feed_dict={
            self.real: x_real, self.z: gaussian(FLAGS.bz, FLAGS.z_dim), self.phase: False})


def main(_):
    _model = wgan()
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
