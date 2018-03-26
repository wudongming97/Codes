import math

import numpy as np
import tensorflow as tf

from mnist.cnn import cl
from mnist.data import imsave_, imcomb_, implot_, dataset
from mnist.wgan_gp import wgan

LocalStep = 8
Bz = 100  # 每批次数据的大小
Mem = 4  # 记忆力，过去记忆与新到数据的比例
Mnist = dataset()
Tn = int(math.floor(Mnist.num / Bz))  # 总的数据批次
LogPath = './results/'


class scholar(object):
    def __init__(self):
        self.generator = wgan()
        self.solver = cl()

    def replay(self, ss_, bz):
        return self.generator.gen(ss_, bz)

    def teach(self, ss_, old):
        return self.solver.pred(ss_, old)

    def pre_train(self, ss, nd_):
        scholar.generator.fit(ss, nd_[0], 400)  # 先对generator进行预训练
        scholar.solver.fit(ss, nd_)

    def fit(self, ss_, nd_, ti_):
        ox = self.replay(ss_, Mem * Bz)
        oy = self.teach(ss_, ox)
        md = self._mix((ox, oy), nd_)
        imsave_(LogPath + "mix_{}_{}.png".format(ti_, _local), imcomb_(md[0]))
        self.generator.fit(ss_, md[0], 36)
        loss, acc = self.solver.fit(ss_, md)
        print('Train [%d] loss [%4f] acc [%4f]' % (ti_, loss, acc))

    def valid(self, ss_, da_, ti_):
        image = imcomb_(self.generator.gen(ss_, da_[0].shape[0]))
        imsave_(LogPath + "{}.png".format(ti_), image)
        loss, acc = self.solver.valid(ss_, da_)
        print(' --Valid [%d\%d] loss [%4f] acc [%4f]' % (ti_, Tn, loss, acc))

    def _mix(self, old, new):
        import random
        images = np.concatenate([old[0], new[0]], 0)
        labels = np.concatenate([old[1], new[1]], 0)
        pairs = list(zip(images, labels))
        random.shuffle(pairs)
        mix = list(zip(*pairs))
        return np.array(mix[0]), np.array(mix[1])


if __name__ == '__main__':
    _conf = tf.ConfigProto(gpu_options=tf.GPUOptions(
        allow_growth=True, per_process_gpu_memory_fraction=0.6))
    scholar = scholar()
    saver = tf.train.Saver(pad_step_number=True)
    with tf.Session(config=_conf) as ss:
        ss.run(tf.global_variables_initializer())
        tf.train.export_meta_graph(LogPath + 'clDGR.meta')
        ckpt = tf.train.get_checkpoint_state(LogPath)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(ss, ckpt.model_checkpoint_path)
        scholar.pre_train(ss, Mnist.next_bt(Bz))
        for ti in range(Tn):
            scholar.fit(ss, Mnist.next_bt(Bz), ti)
            scholar.valid(ss, Mnist.pre_bt(), ti)
            saver.save(ss, LogPath, write_meta_graph=False) if ti % 200 == 0 else None
