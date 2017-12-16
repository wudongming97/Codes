import os
import math
import random
import re
import collections
import pickle

import unidecode

import numpy as np
from functools import reduce

class CorpusLoader:
    def __init__(self ,params):
        self.params = params
        self.raw_data_file = './data.txt'
        self.preprocessed_data_path = './preprocessed_data/'
        if not os.path.exists(self.preprocessed_data_path):
            os.makedirs(self.preprocessed_data_path)
        self.data_words_file = self.preprocessed_data_path + 'words.pkl'
        self.data_idxs_file = self.preprocessed_data_path + 'idxs.pkl'
        self.idx2word_file = self.preprocessed_data_path + 'vocab.pkl'


        self.pad_token = '<pad>'
        self.go_token = '<go>'
        self.end_token = '<end>'
        self.unk_token = '<unk>'

        words_exist = os.path.exists(self.data_words_file)
        idx_exist = os.path.exists(self.idx2word_file)
        tensors_exist = os.path.exists(self.data_idxs_file)

        if words_exist and idx_exist and tensors_exist:
            self.load_preprocessed()
            print('preprocessed data loaded ...')
        else:
            self.preprocess(lf=self.params.get('lf', 0))
            print('data have preprocessed ...')

    def global_txt_process(self, file):
        print('global txt processing ...')
        # tolow
        data = open(self.raw_data_file, encoding='UTF-8').read().lower()

        # 把unicode转换成ascii
        data = unidecode.unidecode(data)

        return data

    def global_seqs_process(self, data_words):
        # 只保留指定长度的seq
        low_level, high_level = self.params['keep_seq_lens']
        data_words = [words for words in data_words if len(words) >= low_level and len(words) < high_level]
        # 根据seq_len进行排序，decent
        if self.params['global_seqs_sort']:
            data_words = sorted(data_words, key=len, reverse=True)

        return data_words


    def preprocess(self, lf=0):
        print('begin preprocessing ...')
        # 一些全局的文本处理
        data = self.global_txt_process(self.raw_data_file)

        # data_words = [sentence=[word, ...], ...], 同时删除空行或者只有一个字母的行
        self.data_words = [line.split() for line in data.split('\n') if len(line) >= 1]

        # 全局的seqs&token处理
        self.data_words = self.global_seqs_process(self.data_words)
        with open(self.data_words_file, 'wb') as f:
            pickle.dump(self.data_words, f)

        self.num_line = len(self.data_words)

        self.word_vocab_size, self.idx_to_word, self.word_to_idx = self.build_word_vocab(self.data_words, lf)
        with open(self.idx2word_file, 'wb') as f:
            pickle.dump(self.idx_to_word, f)

        unk_get = lambda x: self.word_to_idx.get(x, self.word_to_idx.get(self.unk_token))
        self.data_idxs = [list(map(unk_get, line)) for line in self.data_words]

        with open(self.data_idxs_file, 'wb') as f:
            pickle.dump(self.data_idxs, f)

        self.show_corpus_info()

    def load_preprocessed(self):
        self.data_words = pickle.load(open(self.data_words_file, 'rb'))
        self.data_idxs = pickle.load(open(self.data_idxs_file, 'rb'))
        self.idx_to_word = pickle.load(open(self.idx2word_file, 'rb'))
        self.word_to_idx = {x: i for i, x in enumerate(self.idx_to_word)}
        self.word_vocab_size = len(self.idx_to_word)
        self.word_idx_file = dict(zip(self.idx_to_word, range(self.word_vocab_size)))
        self.num_line = len(self.data_words)

        self.show_corpus_info()

    def show_corpus_info(self):
        print('vocab_size: {}'.format(self.word_vocab_size))
        print('total data num_lines: {}, train_data/test_data: {}'.format(self.num_line, self.params['train_fraction']))
        print('\n')

    def build_word_vocab(self, data_words, lf):
        flatten_words = [w for ws in data_words for w in ws ]
        word_counts = collections.Counter(flatten_words).most_common()
        # 删除低频的词
        word_counts = [w for w in word_counts if w[1] > lf]

        # idx2word
        idx_to_word = [self.pad_token, self.go_token, self.end_token, self.unk_token] + [w[0] for w in word_counts]
        vocab_size = len(idx_to_word)
        word_to_idx = {x: i for i, x in enumerate(idx_to_word)}

        return vocab_size, idx_to_word, word_to_idx

    def sort_and_pad(self, inputs):
        sorted_seqs = sorted(inputs, key=len, reverse=True)
        seqs_len = [len(s) for s in sorted_seqs]

        #从idx还原为txt
        words_list = [[self.idx_to_word[idx] for idx in idx_list] for idx_list in sorted_seqs]
        sentences = [reduce(lambda x1,x2: x1 + ' ' + x2, words) for words in words_list]

        padded_seqs = []
        masks = []
        max_sentence_len = max(seqs_len)
        pad_idx = self.word_to_idx.get(self.pad_token)
        for s in sorted_seqs:
            padded_seqs.append(s + [pad_idx]*(max_sentence_len-len(s)))
            masks.append([1]*len(s) + [0]*(max_sentence_len-len(s)))
        return sentences, padded_seqs, seqs_len, masks

    def target_data_idxs(self, target='train'):
        if target=='train':
            data = self.data_idxs[:math.floor(self.num_line * self.params['train_fraction'])]
            return data
        else:
            data = self.data_idxs[math.ceil(self.num_line * self.params['train_fraction']):]
            return data

    # 数据默认用
    def next_batch(self, batch_size, target):
        data = self.target_data_idxs(target)
        data_len = len(data)
        for i in range(data_len // batch_size):

            indexes = range(data_len)[i*batch_size:(i+1)*batch_size]
            if self.params['shuffle']:
                indexes = random.sample(range(data_len), batch_size)

            encoder_word_input = [data[index] for index in indexes]

            decoder_word_input = [[self.word_to_idx[self.go_token]] + line for line in encoder_word_input]
            decoder_word_output = [line + [self.word_to_idx[self.end_token]] for line in encoder_word_input]
            sentences, encoder_word_input, input_seq_len, _ = self.sort_and_pad(encoder_word_input)
            _, decoder_word_input, _, decoder_mask = self.sort_and_pad(decoder_word_input)
            _, decoder_word_output, _, _ = self.sort_and_pad(decoder_word_output)

            yield sentences, encoder_word_input, input_seq_len, decoder_word_input, decoder_word_output, decoder_mask

    def go_input(self, batch_size):
        go_word_input = [[self.word_to_idx[self.go_token]] for _ in range(batch_size)]
        return go_word_input

    def sample_word_from_distribution(self, distribution):
        ix = np.random.choice(range(self.word_vocab_size), p=distribution.ravel())
        return ix, self.idx_to_word[ix]

    def top_k(self, array, k):
        ixs = array.argsort()[-k:][::-1]
        words = [self.idx_to_word[ix] for ix in ixs]

        return ixs.tolist(), words


if __name__ == '__main__':
    corpus_loader_params = {
        'lf': 1,  # 低频词
        'keep_seq_lens': [5, 20],
    }
    loader = CorpusLoader(corpus_loader_params)
    encoder_word_input, input_seq_len, decoder_word_input, decoder_word_output, decoder_mask = loader.next_batch(
        batch_size=2)
