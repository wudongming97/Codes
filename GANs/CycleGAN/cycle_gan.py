import numpy as np

from data import Reader
from module import *
from ops import global_step
from utils import imsave_, imcombind_, inverse_transform

flags = tf.app.flags
flags.DEFINE_string('ds_x', 'horse', 'x数据集')
flags.DEFINE_string('ds_y', 'zebra', 'y数据集')
flags.DEFINE_string('log_path', './logs/cycgan/', '')

flags.DEFINE_integer('steps', 200000, '')
flags.DEFINE_float('init_lr', 0.0002, '')
flags.DEFINE_integer('start_decay', 100000, '')
flags.DEFINE_integer('decay_step', 100000, '')
flags.DEFINE_float('lamda', 10, '')
FLAGS = flags.FLAGS


class cycle_gan:
    def __init__(self):
        self.G = generator_resnet('G')
        self.F = generator_resnet('F')
        self.Dx = discriminator('Dx')
        self.Dy = discriminator('Dy')

        # train graph
        self.fake_pool_x = image_pool(50)
        self.fake_pool_y = image_pool(50)
        self.reader_x = Reader(FLAGS.ds_x)
        self.reader_y = Reader(FLAGS.ds_y)

        self.real_x = self.reader_x.feed()
        self.fake_y = self.G(self.real_x, False)
        self.cyc_x = self.F(self.fake_y, False)

        self.real_y = self.reader_y.feed()
        self.fake_x = self.F(self.real_y)
        self.cyc_y = self.G(self.fake_x)

        self.pool_x = tf.placeholder(tf.float32, [None, 256, 256, 3], name='Dx_pool')
        self.pool_y = tf.placeholder(tf.float32, [None, 256, 256, 3], name='Dy_pool')

        self.Dy_fake = self.Dy(self.fake_y, False)
        self.Dy_real = self.Dy(self.real_y)
        self.Dx_fake = self.Dx(self.fake_x, False)
        self.Dx_real = self.Dx(self.real_x)

        # loss def
        self.Cyc_loss = cyc_loss(self.real_x, self.cyc_x, self.real_y, self.cyc_y)

        self.G_loss = gen_loss(self.Dy_fake) + FLAGS.lamda * self.Cyc_loss
        self.F_loss = gen_loss(self.Dx_fake) + FLAGS.lamda * self.Cyc_loss

        self.Dx_loss = dis_loss(self.Dx_real, self.Dx(self.pool_x))
        self.Dy_loss = dis_loss(self.Dy_real, self.Dy(self.pool_y))

        # optim
        self.lr = tf.where(
            tf.greater(global_step, FLAGS.start_decay),
            tf.train.polynomial_decay(FLAGS.init_lr, global_step - FLAGS.start_decay, FLAGS.decay_step, 0),
            FLAGS.init_lr
        )
        optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5)
        self.G_optim = optimizer.minimize(self.G_loss, var_list=self.G.vars)
        self.F_optim = optimizer.minimize(self.F_loss, var_list=self.F.vars)
        self.D_optim_x = optimizer.minimize(self.Dx_loss, var_list=self.Dx.vars)
        self.D_optim_y = optimizer.minimize(self.Dy_loss, var_list=self.Dy.vars)

        self.inc_step = tf.assign_add(global_step, 1)
        self.fit_summ = tf.summary.merge([
            tf.summary.scalar('lr', self.lr),
            tf.summary.scalar('G_loss', self.G_loss),
            tf.summary.scalar('F_loss', self.F_loss),
            tf.summary.scalar('Dx_loss', self.Dx_loss),
            tf.summary.scalar('Dy_loss', self.Dy_loss),
            tf.summary.histogram('Dx_fake', self.Dx_fake),
            tf.summary.histogram('Dx_real', self.Dx_real),
            tf.summary.histogram('Dy_fake', self.Dy_fake),
            tf.summary.histogram('Dy_real', self.Dx_real)
        ])

        # test graph
        # forward
        self.reader_x_t = Reader(FLAGS.ds_x + '_t')
        self.test_x = self.reader_x_t.feed(6)
        self.test_fake_y = self.G(self.test_x)
        self.test_cyc_x = self.F(self.test_fake_y)

        # backward
        self.reader_y_t = Reader(FLAGS.ds_y + '_t')
        self.test_y = self.reader_y_t.feed(6)
        self.test_fake_x = self.F(self.test_y)
        self.test_cyc_y = self.G(self.test_fake_x)

    def fit(self, sess, local_):
        pool_x, pool_y = None, None
        for _ in range(local_):
            fake_y, _ = sess.run([self.fake_y, self.G_optim])
            fake_x, _ = sess.run([self.fake_x, self.F_optim])
            pool_x, pool_y = self.fake_pool_x(fake_x), self.fake_pool_y(fake_y)

            sess.run(self.D_optim_x, {self.pool_x: pool_x})
            sess.run(self.D_optim_y, {self.pool_y: pool_y})
            sess.run(self.inc_step)

        return sess.run(
            [self.G_loss, self.F_loss, self.Dx_loss, self.Dy_loss, self.fit_summ],
            {self.pool_x: pool_x, self.pool_y: pool_y})

    def gen(self, sess):
        images = sess.run(
            [self.real_x, self.fake_y, self.cyc_x, self.real_y, self.fake_x, self.cyc_y])
        return np.concatenate(images, axis=0)

    def test(self, sess):
        images = sess.run(
            [self.test_x, self.test_fake_y, self.test_cyc_x, self.test_y, self.test_fake_x, self.test_cyc_y])
        return np.concatenate(images, axis=0)


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
            res = _model.fit(sess, 100)

            _step = _step + 100
            _writer.add_summary(res[4], _step)
            print("Train [%d\%d] G_loss [%3f] F_loss [%3f]  Dx_loss [%3f] Dy_loss [%3f] " % (
                _step, FLAGS.steps, res[0], res[1], res[2], res[3]))

            images = inverse_transform(_model.gen(sess))
            imsave_(FLAGS.log_path + 'train_{}.png'.format(_step), imcombind_(images, 3))
            images = inverse_transform(_model.test(sess))
            imsave_(FLAGS.log_path + 'test_{}.png'.format(_step), imcombind_(images, 6))
            _saver.save(sess, FLAGS.log_path) if _step % 1000 == 0 else None

        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    tf.app.run()
