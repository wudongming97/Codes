import tensorflow as tf
from utils import imsave_, imcombind_, L1Loss, L2Loss, inverse_transform
from module import generator_resnet, discriminator, image_pool
from data import Reader
import numpy as np

flags = tf.app.flags
flags.DEFINE_string('ds_x', 'horse', 'x数据集')
flags.DEFINE_string('ds_y', 'zebra', 'y数据集')
flags.DEFINE_string('log_path', './logs/cycgan/', '')

flags.DEFINE_integer('steps', 20000, '')
flags.DEFINE_integer('bz', 100, '')
flags.DEFINE_float('lr', 0.001, '')
flags.DEFINE_float('lamda', 10, '')
FLAGS = flags.FLAGS


class cycle_gan:
    def __init__(self):
        self.fake_pool_x = image_pool(50)
        self.fake_pool_y = image_pool(50)
        self.reader_x = Reader(FLAGS.ds_x)
        self.reader_y = Reader(FLAGS.ds_y)

        self.G = generator_resnet('G')
        self.F = generator_resnet('F')
        self.Dx = discriminator('Dx')
        self.Dy = discriminator('Dy')

        self.real_x = self.reader_x.feed()
        self.real_y = self.reader_y.feed()
        self.pool_x = tf.placeholder(tf.float32, [None, 256, 256, 3], name='Dx_pool')
        self.pool_y = tf.placeholder(tf.float32, [None, 256, 256, 3], name='Dy_pool')

        self.fake_y = self.G(self.real_x, False)
        self.cyc_x = self.F(self.fake_y, False)
        self.fake_x = self.F(self.real_y)
        self.cyc_y = self.G(self.fake_x)

        self.Dy_fake = self.Dy(self.fake_y, False)
        self.Dy_real = self.Dy(self.real_y)
        self.Dx_fake = self.Dx(self.fake_x, False)
        self.Dx_real = self.Dx(self.real_x)

        # loss def
        self.cyc_loss = L1Loss(self.real_x, self.cyc_x) + L1Loss(self.real_y, self.cyc_y)

        self.gg_loss = L2Loss(self.Dy_fake, tf.ones_like(self.Dy_fake))
        self.gf_loss = L2Loss(self.Dx_fake, tf.ones_like(self.Dx_fake))
        self.g_loss = self.gg_loss + self.gf_loss + FLAGS.lamda * self.cyc_loss

        self.dx_loss = L2Loss(self.Dx_real, tf.ones_like(self.Dx_real)) + \
                       L2Loss(self.pool_x, tf.zeros_like(self.pool_x))
        self.dy_loss = L2Loss(self.Dy_real, tf.ones_like(self.Dy_real)) + \
                       L2Loss(self.pool_y, tf.zeros_like(self.pool_y))
        self.d_loss = self.dx_loss + self.dy_loss

        # optim
        optimizer = tf.train.AdamOptimizer(FLAGS.lr)
        self.d_optim = optimizer.minimize(self.d_loss, var_list=self.Dx.vars + self.Dy.vars)
        self.g_optim = optimizer.minimize(self.g_loss, var_list=self.G.vars + self.F.vars)

        self.inc_step = tf.assign_add(tf.train.get_or_create_global_step(), 1)
        self.fit_summ = tf.summary.merge([
            tf.summary.scalar('gg_loss', self.gg_loss),
            tf.summary.scalar('gf_loss', self.gf_loss),
            tf.summary.scalar('g_loss', self.g_loss),
            tf.summary.scalar('dx_loss', self.dx_loss),
            tf.summary.scalar('dy_loss', self.dy_loss),
            tf.summary.scalar('d_loss', self.d_loss)
        ])

    def fit(self, sess, local_):
        pool_x, pool_y = None, None
        for _ in range(local_):
            fake_x, fake_y, _ = sess.run([self.fake_x, self.fake_y, self.g_optim])
            pool_x, pool_y = self.fake_pool_x(fake_x), self.fake_pool_y(fake_y)
            sess.run([self.d_optim, self.inc_step], {self.pool_x: pool_x, self.pool_y: pool_y})
        return sess.run(
            [self.gg_loss, self.gf_loss, self.g_loss, self.dx_loss, self.dy_loss, self.d_loss, self.fit_summ],
            {self.pool_x: pool_x, self.pool_y: pool_y})

    def gen(self, sess, bz):
        res = []
        for _ in range(bz):
            images = sess.run(
                [self.real_x, self.fake_x, self.cyc_x, self.real_y, self.fake_y, self.cyc_y])
            res.extend(images)
        return np.concatenate(res, axis=0)


def main(_):
    _model = cycle_gan()
    _gpu = tf.GPUOptions(allow_growth=True)
    _saver = tf.train.Saver(pad_step_number=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=_gpu)) as sess:
        _writer = tf.summary.FileWriter(FLAGS.log_path, sess.graph)
        tf.global_variables_initializer().run()

        ckpt = tf.train.get_checkpoint_state(FLAGS.log_path)
        if ckpt and ckpt.model_checkpoint_path:
            _saver.restore(sess, FLAGS.log_path)
        _step = tf.train.get_global_step().eval()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        while True:
            if _step >= FLAGS.steps:
                break
            res = _model.fit(sess, 2)

            _step = _step + 2
            _writer.add_summary(res[6], _step)
            print("Train [%d\%d] gg_loss [%3f] gf_loss [%3f] g_loss [%3f] dx_loss [%3f] dy_loss [%3f] d_loss [%3f]" % (
                _step, FLAGS.steps, res[0], res[1], res[2], res[3], res[4], res[5]))

            images = inverse_transform(_model.gen(sess, 6))
            imsave_(FLAGS.log_path + 'train_{}.png'.format(_step), imcombind_(images, 6))
            _saver.save(sess, FLAGS.log_path) if _step % 5000 == 0 else None

        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    tf.app.run()
