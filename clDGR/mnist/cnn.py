import numpy as np
import tensorflow as tf


class cl(object):
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.y = tf.placeholder(tf.float32, [None, 10])
        self.logits = self.net_(self.x, False)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits))
        self.optim = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

    def net_(self, x, reuse=True):
        with tf.variable_scope('net_', reuse):
            conv1 = tf.layers.conv2d(x, 128, [4, 4], [2, 2], activation=tf.nn.relu)
            conv2 = tf.layers.conv2d(conv1, 64, [4, 4], [2, 2], activation=tf.nn.relu)
            fc1 = tf.layers.dense(tf.layers.flatten(conv2), 128, activation=tf.nn.relu)
            logits = tf.layers.dense(fc1, 10)
        return logits

    def fit(self, sess, data, its):
        for t in range(its):
            _, loss, logits = sess.run([self.optim, self.loss, self.logits],
                                       {self.x: data[0], self.y: data[1]})


    def pred(self, sess, old):
        return sess.run(self.logits, {self.x: old})

    def valid(self, sess, data):
        loss, logits = sess.run([self.loss, self.logits], {self.x: data[0], self.y: data[1]})
        acc = np.mean(np.argmax(logits, 1) == np.argmax(data[1], 1))
        return loss, acc
