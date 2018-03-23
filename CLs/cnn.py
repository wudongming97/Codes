import numpy as np
import tensorflow as tf
from mnist import next_batch_

Bz = 36
Lr = 1e-4
Iter = 60000
SaverPath = './results/cnn/ckpt/'


class cl(object):
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.y = tf.placeholder(tf.float32, [None, 10])
        self.logits = self.net_(self.x, False)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits))
        self.optim = tf.train.AdamOptimizer(Lr).minimize(self.loss, tf.train.get_or_create_global_step())

    def net_(self, x, reuse=True):
        with tf.variable_scope('net_', reuse):
            conv1 = tf.layers.conv2d(x, 128, [4, 4], [2, 2], activation=tf.nn.relu)
            conv2 = tf.layers.conv2d(conv1, 64, [4, 4], [2, 2], activation=tf.nn.relu)
            fc1 = tf.layers.dense(tf.layers.flatten(conv2), 128, activation=tf.nn.relu)
            logits = tf.layers.dense(fc1, 10)
        return logits

    def fit(self, sess, saver):
        for it, data in enumerate(next_batch_(Bz)):
            sess.run(self.optim, {self.x: data[0], self.y: data[1]})
            _step = sess.run(tf.train.get_global_step())
            if it % 20 == 0:
                loss, logits = sess.run([self.loss, self.logits], {self.x: data[0], self.y: data[1]})
                saver.save(sess, SaverPath, _step, write_meta_graph=False)
                pred = np.argmax(logits, 1) == np.argmax(data[1], 1)
                acc = np.mean(pred)
                print('Train: [%d/%d] loss [%.4f] acc [%.4f]' % (_step, Iter, loss, acc))
            if Iter < _step:
                break

    def valid(self, sess):
        for it, data in enumerate(next_batch_(500, False)):
            loss, logits = sess.run([self.loss, self.logits], {self.x: data[0], self.y: data[1]})
            pred = np.argmax(logits, 1) == np.argmax(data[1], 1)
            acc = np.mean(pred)
            print('Valid: [%d/%d] loss [%.4f] acc [%.4f]' % (it, 20, loss, acc))
            if it > 20:
                break


if __name__ == '__main__':
    model = cl()
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1, pad_step_number=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.train.export_meta_graph(SaverPath + 'cnn.meta')
        ckpt = tf.train.get_checkpoint_state(SaverPath)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        model.fit(sess, saver)
        model.valid(sess)
