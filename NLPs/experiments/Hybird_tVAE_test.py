# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import utils.DataLoader as D


# 查看随机生成的情况
def infer_by_normal_test(model, sess, data_loader):
    sentences = model.infer_by_normal(sess, data_loader)
    for ix in range(len(sentences)):
        t_s_ = sentences[ix].split(D.E_TOKEN)[0]
        print("{:3d}. {}".format(ix, t_s_))


# 此测试是为了查看训练数据或者测试数据重构的情况
def infer_by_encoder_test(model, sess, data_loader, batch_size):
    for i in data_loader.next_batch(batch_size):
        input_s_ = sorted(data_loader.to_seqs(i), key=len, reverse=True)
        out_s_ = model.infer_by_encoder(sess, data_loader, input_s_)
        pair_s_ = zip(input_s_, out_s_)
        for ix, _p in enumerate(pair_s_):
            print('In {:3d}: {}'.format(ix, _p[0]))
            t_s_ = _p[1].split(D.E_TOKEN)[0]
            print('Out{:3d}: {}'.format(ix, t_s_))
        break


# 此测试是为了查看VAE（去kl_loss项和不去kl_loss项）的连续性
def infer_by_same_test(model, sess, data_loader, sentence, batch_size):
    sentences = [sentence] * batch_size
    out_s_ = model.infer_by_encoder(sess, data_loader, sentences)
    for ix, s_ in enumerate(out_s_):
        print('Out{:3d}: {}'.format(ix, s_.split(D.E_TOKEN)[0]))


# 次测试是为了查看VAE的z空间的连续性
def infer_by_linear_z_test(model, sess, data_loader, batch_size, z_size):
    linear = np.linspace(-0.0001, 0.0001, num=batch_size)
    z = np.tile(linear, [z_size, 1]).transpose()
    sentences = model.infer_by_z(sess, data_loader, z)
    pair_s_ = zip(linear, sentences)
    for ix, _p in enumerate(pair_s_):
        t_s_ = _p[1].split(D.E_TOKEN)[0]
        print('{:3d}, Z={:.3f}, Out: {}'.format(ix, _p[0], t_s_))
