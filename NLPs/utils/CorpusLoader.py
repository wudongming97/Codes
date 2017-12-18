import collections
import math
import os
import pickle
import random
from functools import reduce

import unidecode


class CorpusLoader:
    def __init__(self, params):
        self.params = params

        self.char_lf = self.params.get('char_lf')
        self.word_lf = self.params.get('word_lf')
        self.keep_words_lens = self.params.get('keep_words_lens')
        self.keep_chars_lens = self.params.get('keep_chars_lens')
        self.shuffle = self.params.get('shuffle')
        self.global_seqs_sort = self.params.get('global_seqs_sort')
        self.train_fraction = self.params.get('train_fraction')

        self.data_path_prefix = self.params.get('data_path_prefix', './')
        self.raw_data_file = self.data_path_prefix + './data.txt'
        self.preprocessed_data_path = self.data_path_prefix + './preprocessed_data/'
        if not os.path.exists(self.preprocessed_data_path):
            os.makedirs(self.preprocessed_data_path)

        self.idxs_persistent_files = [self.preprocessed_data_path + 'char_idxs.pkl',
                                      self.preprocessed_data_path + 'word_idxs.pkl']
        self.vocab_persistent_files = [self.preprocessed_data_path + 'char_vocab.pkl',
                                       self.preprocessed_data_path + 'word_vocab.pkl']

        self.vocabs = []
        self.vocabs_to_idx = []
        self.data_idxs = []

        self.pad_token = 'P'
        self.go_token = 'G'
        self.end_token = 'E'
        self.unk_token = 'U'

        idxs_files_exist = reduce(
            lambda x1, x2: x1 and x2,
            [os.path.exists(p) for p in self.idxs_persistent_files]
        )

        vocab_files_exists = reduce(
            lambda x1, x2: x1 and x2,
            [os.path.exists(p) for p in self.vocab_persistent_files]
        )

        if idxs_files_exist and vocab_files_exists:
            self.load_preprocessed()
            print('preprocessed data loaded ...')
        else:
            self.preprocess()
            print('data have preprocessed ...')

        self.show_corpus_info()

    def _global_txt_process(self, file):
        print('global txt processing ...')
        # tolow
        data = open(self.raw_data_file, encoding='UTF-8').read().lower()

        # 把unicode转换成ascii
        data = unidecode.unidecode(data)

        return data

    def _global_seqs_process(self, seqs, target):
        # 只保留指定长度的seq
        left = self.keep_words_lens[0] if target == 0 else self.keep_chars_lens[0]
        right = self.keep_words_lens[1] if target == 0 else self.keep_chars_lens[1]
        seqs = [seq for seq in seqs if
                len(seq) >= left and len(seq) < right]
        # 根据seq_len进行排序，decent
        if self.global_seqs_sort:
            seqs = sorted(seqs, key=len, reverse=True)

        return seqs

    def preprocess(self):
        print('begin preprocessing ...')
        # 一些全局的文本处理
        data = self._global_txt_process(self.raw_data_file)

        sentences = self._global_seqs_process(data.split('\n'), 0)
        words_list = self._global_seqs_process([line.split() for line in data.split('\n') if len(line) >= 1], 1)

        self.vocabs = [self._build_vocab(sentences, self.char_lf), self._build_vocab(words_list, self.word_lf)]
        self.vocab_sizes = [len(t) for t in self.vocabs]
        self.vocabs_to_idx = [dict(zip(v, range(len(v)))) for v in self.vocabs]

        self.data_idxs = [self._build_idxs(sentences, 0), self._build_idxs(words_list, 1)]
        self.num_lines = [len(idx) for idx in self.data_idxs]

        # presist
        for target, p in enumerate(self.idxs_persistent_files):
            with open(p, 'wb') as f:
                pickle.dump(self.data_idxs[target], f)
        for target, p in enumerate(self.vocab_persistent_files):
            with open(p, 'wb') as f:
                pickle.dump(self.vocabs[target], f)

    def load_preprocessed(self):
        self.data_idxs = [pickle.load(open(f, 'rb')) for f in self.idxs_persistent_files]
        self.vocabs = [pickle.load(open(f, 'rb')) for f in self.vocab_persistent_files]

        self.num_lines = [len(t) for t in self.data_idxs]
        self.vocab_sizes = [len(t) for t in self.vocabs]
        self.vocabs_to_idx = [dict(zip(v, range(len(v)))) for v in self.vocabs]

    def show_corpus_info(self):
        print('char_vocab_size: {}, word_vocab_size: {}'.format(self.vocab_sizes[0], self.vocab_sizes[1]))
        print('char_num_lines: {}, word_num_lines: {}, train_data/test_data: {}'.format(self.num_lines[0],
                                                                                        self.num_lines[1],
                                                                                        self.train_fraction))

    def _build_vocab(self, seqs, lf):
        flatten_seq = [token for seq in seqs for token in seq]
        counts_ = collections.Counter(flatten_seq).most_common()
        counts_ = [c for c in counts_ if c[1] > lf]
        vocab_ = [self.pad_token, self.go_token, self.end_token, self.unk_token] + [c[0] for c in counts_]
        return vocab_

    def _build_idxs(self, seqs, target):
        unk_get = lambda x: self.vocabs_to_idx[target].get(x, self.vocabs_to_idx[target].get(self.unk_token))
        idxs = [list(map(unk_get, seq)) for seq in seqs]
        return idxs

    def _sort_and_pad(self, inputs, target):
        sorted_seqs = sorted(inputs, key=len, reverse=True)
        seqs_len = [len(s) for s in sorted_seqs]

        # 从idx还原为txt
        sentences = self.decode_idxs_list(sorted_seqs, target)

        padded_seqs = []
        masks = []
        max_seq_len = max(seqs_len)
        pad_idx = self.vocabs_to_idx[target].get(self.pad_token)
        for s in sorted_seqs:
            padded_seqs.append(s + [pad_idx] * (max_seq_len - len(s)))
            masks.append([1] * len(s) + [0] * (max_seq_len - len(s)))
        return sentences, padded_seqs, seqs_len, masks

    def _target_data_idxs(self, target, train):
        if train:
            return self.data_idxs[target][:math.floor(self.num_lines[target] * self.train_fraction)]
        else:
            return self.data_idxs[target][math.ceil(self.num_lines[target] * self.train_fraction):]

    # 数据默认用
    def next_batch(self, batch_size, target, train=True):  # target: 0 or 1
        data = self._target_data_idxs(target, train)
        data_len = len(data)
        for i in range(data_len // batch_size):

            indexes = range(data_len)[i * batch_size:(i + 1) * batch_size]
            if self.shuffle:
                indexes = random.sample(range(data_len), batch_size)

            encoder_input = [data[index] for index in indexes]

            decoder_input = [[self.vocabs_to_idx[target][self.go_token]] + line for line in encoder_input]
            decoder_output = [line + [self.vocabs_to_idx[target][self.end_token]] for line in decoder_input]
            sentences, encoder_input, input_seq_len, _ = self._sort_and_pad(decoder_output, target)
            _, decoder_input, _, decoder_mask = self._sort_and_pad(decoder_output, target)
            _, decoder_output, _, _ = self._sort_and_pad(decoder_output, target)

            yield sentences, encoder_input, input_seq_len, decoder_input, decoder_output, decoder_mask

    def go_input(self, target, batch_size, ):
        go_input = [[self.vocabs_to_idx[target][self.go_token]] for _ in range(batch_size)]
        return go_input

    def decode_idxs_list(self, idxs_list, target):
        seq_list = [[self.vocabs[target][id] for id in idxs] for idxs in idxs_list]
        if target == 0:
            sentences = [reduce(lambda x1, x2: x1 + x2, seq) for seq in seq_list]
        else:
            sentences = [reduce(lambda x1, x2: x1 + ' ' + x2, seq) for seq in seq_list]
        return sentences

    def vocab2idx(self, target):
        return self.vocabs_to_idx[target]

    def max_seq_len(self, target):
        max_lens = [self.keep_chars_lens[1], self.keep_words_lens[1]]
        return max_lens[target]
