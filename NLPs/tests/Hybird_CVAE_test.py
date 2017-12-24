# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import utils.DataLoader as D


def infer_by_normal_test(model, sess, data_loader):
    print('\n=============begin infer_by_normal============\n')
    sentences = model.infer_by_normal(sess, data_loader)
    for ix in range(len(sentences)):
        t_s_ = sentences[ix].split(D.E_TOKEN)[0]
        print("{:3d}. {}".format(ix, t_s_))

def infer_by_encoder_test(model, sess, data_loader, batch_size):
    print('\n============begin infer_by_encoder============\n')
    for i in data_loader.next_batch(batch_size, True):
        input_s_ = data_loader.to_seqs(i)
        out_s_ = model.infer_by_encoder(sess, data_loader, input_s_)
        pair_s_ = zip(input_s_, out_s_)
        for ix, _p in enumerate(pair_s_):
            print('In {:3d}: {}'.format(ix, _p[0]))
            t_s_ = _p[1].split(D.E_TOKEN)[0]
            print('Out{:3d}: {}'.format(ix, t_s_))
        break

def infer_by_linear_z_test(model, sess, data_loader, batch_size, z_size):
    print('\n=======begin linear infer between z1 and z2====\n')
    linear = np.linspace(-0.0001, 0.0001, num=batch_size)
    z = np.tile(linear, [z_size, 1]).transpose()
    sentences = model.infer_by_z(sess, data_loader, z)
    pair_s_ = zip(linear, sentences)
    for ix, _p in enumerate(pair_s_):
        t_s_ = _p[1].split(D.E_TOKEN)[0]
        print('{:3d}, Z={:.3f}, Out: {}'.format(ix, _p[0], t_s_))