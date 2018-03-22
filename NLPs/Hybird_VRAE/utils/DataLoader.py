# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import enum
import math
import os
import pickle
import random
from functools import reduce

import unidecode

P_TOKEN = 'P'
G_TOKEN = 'G'
E_TOKEN = 'E'
U_TOKEN = 'U'


class Level(enum.Enum):
    CHAR = 'char'
    WORD = 'word'


class Vocab:
    def __init__(self, vocab_name, level, path_='./data'):
        self.vocab_name = vocab_name
        self.level = level
        self.r_file = os.path.join(path_, self.vocab_name + '.txt')
        self.p_file = os.path.join(path_, self.vocab_name + '_' + str(self.level) + '.pkl')

        is_processed = os.path.exists(self.p_file)
        if is_processed:
            self._restore()
        else:
            self._process()
            self._persist()

    def _process(self):
        seqs = unidecode.unidecode(open(self.r_file, encoding='UTF-8').read()).split('\n')
        if self.level == Level.CHAR:
            self.vocab = self._build_vocab(seqs)
        elif self.level == Level.WORD:
            seqs = [s_.split() for s_ in seqs]
            self.vocab = self._build_vocab(seqs)
        else:
            raise ValueError('level: {} not support.'.format(self.level))

        self.vocab_size = len(self.vocab)
        self.idx = dict(zip(self.vocab, range(self.vocab_size)))
        self.idx_t = self._to_tensor(seqs)

    def _build_vocab(self, seqs, lf=2, sz=None):
        flatten_seq = reduce(lambda x1, x2: x1 + x2, seqs)
        counts_ = collections.Counter(flatten_seq).most_common(sz)
        counts_ = [c for c in counts_ if c[1] > lf]
        vocab_ = [P_TOKEN, G_TOKEN, E_TOKEN, U_TOKEN] + [c[0] for c in counts_]
        return vocab_

    def _to_tensor(self, seqs):
        get_ = lambda x: self.idx.get(x, self.idx.get(U_TOKEN))
        tensor = [list(map(get_, s_)) for s_ in seqs]
        return tensor

    def _to_seqs(self, tensor):
        seqs_ = [[self.vocab[ix] for ix in s_] for s_ in tensor]
        if self.level == Level.CHAR:
            seqs = [''.join(s_) for s_ in seqs_]
        else:
            seqs = [' '.join(s_) for s_ in seqs_]
        return seqs

    def _persist(self):
        with open(self.p_file, 'wb') as f:
            pickle.dump(self.vocab, f)
            pickle.dump(self.idx, f)
            pickle.dump(self.idx_t, f)

    def _restore(self):
        with open(self.p_file, 'rb') as f:
            self.vocab = pickle.load(f)
            self.idx = pickle.load(f)
            self.idx_t = pickle.load(f)
        self.vocab_size = len(self.vocab)


class DataLoader:
    def __init__(self, vocab):
        self.vocab = vocab
        self.vocab_size = self.vocab.vocab_size
        self.max_seq_len = max(map(len, self.vocab.idx_t))
        self.sample_size = len(self.vocab.idx_t)
        self.train_t = self.vocab.idx_t[:math.floor(self.sample_size * 0.50)]
        self.valid_t = self.vocab.idx_t[math.floor(self.sample_size * 0.50):]
        self.train_size = len(self.train_t)
        self.valid_size = len(self.valid_t)
        self.to_seqs = lambda x: self.vocab._to_seqs(x)
        self.to_tensor = lambda x: self.vocab._to_tensor(x)
        self._show()

    def _show(self):
        print('=================vocab info===================')
        print(
            'name: {}, Level: {}, vocab_size: {}, train_size: {}, valid_size: {}'.format(self.vocab.vocab_name, self.vocab.level,
                                                                                         self.vocab_size, self.train_size,
                                                                                         self.valid_size))

    def next_batch(self, batch_size, train=True, shuffle=True):
        tensor_ = self.train_t if train else self.valid_t
        num_ = self.train_size if train else self.valid_size
        while True:
            if shuffle:
                index_ = random.sample(range(num_), batch_size)
            batch_tensor_ = [tensor_[ix] for ix in index_]
            yield batch_tensor_

    # 方便在测试的时候使用
    def one_batch(self, batch_size, train=True):
        tensor_ = self.train_t if train else self.valid_t
        num_ = self.train_size if train else self.valid_size
        return [tensor_[ix] for ix in random.sample(range(num_), batch_size)]

    def unpack_for_tvae(self, batch_tensor):
        sorted_tensor = sorted(batch_tensor, key=len, reverse=True)
        X, X_lengths = self._pad(sorted_tensor)
        Y_i, Y_lengths = self._pad([[self.vocab.idx[G_TOKEN]] + line for line in sorted_tensor])
        Y_t, _ = self._pad([line + [self.vocab.idx[E_TOKEN]] for line in sorted_tensor])
        return X, X_lengths, Y_i, self._mask(Y_lengths), Y_t

    def unpack_for_hybird_tvae(self, batch_tensor, seq_len):
        sorted_tensor = sorted(batch_tensor, key=len, reverse=True)
        _input, _lengths = self._pad([[self.vocab.idx[G_TOKEN]] + line for line in sorted_tensor], seq_len)
        Y_t, _ = self._pad([line + [self.vocab.idx[E_TOKEN]] for line in sorted_tensor], seq_len)
        return _input, _input, _lengths, Y_t, self._mask(_lengths, seq_len)

    def unpack_for_tvae_tf(self, batch_tensor, seq_len):
        sorted_tensor = sorted(batch_tensor, key=len, reverse=True)
        X, X_lengths = self._pad(sorted_tensor, seq_len)
        Y_i, Y_lengths = self._pad([[self.vocab.idx[G_TOKEN]] + line for line in sorted_tensor], seq_len)
        Y_t, _ = self._pad([line + [self.vocab.idx[E_TOKEN]] for line in sorted_tensor], seq_len)
        return X, X_lengths, Y_i, Y_lengths, Y_t, self._mask(Y_lengths, seq_len)

    def g_input(self, batch_size):
        g_input_ = [[self.vocab.idx[G_TOKEN]] for _ in range(batch_size)]
        return g_input_

    def _pad(self, inputs, pad_max_len=None):
        seq_lengths = [len(s) for s in inputs]
        seq_max_len_ = max(seq_lengths)
        if (pad_max_len is not None) and (pad_max_len < seq_max_len_):
            raise ValueError('seq_max_len({}) > pad_max_len({})'.format(seq_max_len_, pad_max_len))
        pad_max_len_ = seq_max_len_ if pad_max_len is None else pad_max_len
        padded = [s + [self.vocab.idx[P_TOKEN]] * (pad_max_len_ - len(s)) for s in inputs]
        return padded, seq_lengths

    def _mask(self, seq_lengths, pad_max_len=None):
        seq_max_len_ = max(seq_lengths)
        if (pad_max_len is not None) and (pad_max_len < seq_max_len_):
            raise ValueError('_mask param error!')
        pad_max_len_ = seq_max_len_ if pad_max_len is None else pad_max_len
        mask = [[1] * l_ + [0] * (pad_max_len_ - l_) for l_ in seq_lengths]
        return mask
