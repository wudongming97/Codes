import math

import numpy as np
import tensorflow as tf

from mnist.cnn import cl
from mnist.data import imsave_, imcomb_, implot_, data_
from mnist.wgan_gp import wgan

Gi = 10  # 生成器每轮迭代次数
Di = 4  # Solver每轮迭代次数
Bz = 100  # 每批次数据的大小
mem = 2  # 记忆力，过去记忆与新到数据的比例
Data = data_()
Tn = math.floor(Data.num / Bz)  # 总的数据批次
LogPath = './results/'


class scholar(object):
    def __init__(self):
        tf.train.create_global_step()
        self.add1 = tf.assign_add(tf.train.get_global_step(), tf.constant(1, dtype=tf.int64))
        self.generator = wgan()
        self.solver = cl()

    @staticmethod
    def current(ss_):
        return ss_.run(tf.train.get_global_step())

    def replay(self, ss_, bz):
        return self.generator.gen(ss_, bz)

    def teach(self, ss_, old):
        return self.solver.pred(ss_, old)

    def fit(self, ss_, nd):
        ox = self.replay(ss_, mem * Bz)
        oy = self.teach(ss_, ox)
        md = self._mix((ox, oy), nd)
        self.generator.fit(ss_, md[0], Gi)
        self.solver.fit(ss_, md, Di)
        ss_.run(self.add1)

    def valid(self, ss_, da, tn):
        images = self.generator.gen(ss_, 16)
        img = imcomb_(images)
        imsave_(LogPath + "{}.png".format(tn), img)
        implot_(img)
        loss, acc = self.solver.valid(ss_, da)
        print('Valid [%3d\%3d] loss [%4f] acc [%4d]' % (tn, Tn, loss, acc))

    def _mix(self, old, new):
        import random
        images = np.concatenate([old[0], new[0]], 0)
        labels = np.concatenate([old[1], new[1]], 0)
        pairs = list(zip(images, labels))
        random.shuffle(pairs)
        mix = list(zip(*pairs))
        return np.array(mix[0]), np.array(mix[1])


if __name__ == '__main__':
    scholar = scholar()
    saver = tf.train.Saver(pad_step_number=True)
    with tf.Session() as ss:
        ss.run(tf.global_variables_initializer())
        tf.train.export_meta_graph(LogPath + 'clDGR.meta')
        ckpt = tf.train.get_checkpoint_state(LogPath)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(ss, ckpt.model_checkpoint_path)
        while True:
            ti = scholar.current(ss)
            if ti > Tn:
                break
            data = Data.next_bt(ti, Bz)
            scholar.fit(ss, data)
            scholar.valid(ss, data, ti)
            saver.save(ss, LogPath, ti, write_meta_graph=False)
