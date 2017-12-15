import os
import re
import collections
import pickle

import unidecode

import numpy as np
from functools import reduce

class CorpusLoader:
    def __init__(self ,params):
        self.params = params
        self.raw_data_files = ['./train.txt', './test.txt']
        self.preprocessed_data_path = './preprocessed_data/'
        if not os.path.exists(self.preprocessed_data_path):
            os.makedirs(self.preprocessed_data_path)
        self.word_data_file = self.preprocessed_data_path + 'words.pkl'
        self.word_idx_file = self.preprocessed_data_path + 'vocab.pkl'
        self.word_tensor_files = [self.preprocessed_data_path + 'train.npy', self.preprocessed_data_path + 'valid.npy']

        self.pad_token = '<pad>'
        self.go_token = '<go>'
        self.end_token = '<end>'
        self.unk_token = '<unk>'

        words_exists = os.path.exists(self.word_data_file)
        idx_exists = os.path.exists(self.word_idx_file)
        tensors_exists = reduce(lambda x1,x2: x1 and x2, [os.path.exists(f) for f in self.word_tensor_files])

        if idx_exists and tensors_exists and words_exists:
            self.load_preprocessed()
            print('preprocessed data loaded ...')
        else:
            self.preprocess(lf=self.params.get('lf', 0))
            print('data have preprocessed ...')

    def global_txt_process(self, data):
        # tolow
        data = [target.lower() for target in data]

        # 把unicode转换成ascii
        data = [unidecode.unidecode(target) for target in data]

        return data
        #

    def global_seqs_process(self, data_words):
        # 只保留指定长度的seq
        low_level, high_level = self.params['keep_seq_lens']
        data_words = [[words for words in target if len(words) >= low_level and len(words) < high_level] for target
                           in data_words]
        # 根据seq_len进行排序，decent
        if self.params['global_seqs_sort']:
            data_words = [sorted(target, key=len, reverse=True) for target in data_words]

        return data_words


    def preprocess(self, lf=0):
        print('begin preprocessing ...')
        data = [open(file, encoding='UTF-8').read() for file in self.raw_data_files]

        # 一些全局的文本处理
        data = self.global_txt_process(data)

        # merged_data_words = [sentence=[word, ...], ...], 同时删除空行或者只有一个字母的行
        self.data_words = [[line.split() for line in fi.split('\n') if len(line) >= 1] for fi in data]

        # 全局的seqs处理
        self.data_words = self.global_seqs_process(self.data_words)

        with open(self.word_data_file, 'wb') as f:
            pickle.dump(self.data_words, f)

        merged_data_words = (data[0] + '\n' + data[1]).split()
        self.word_vocab_size, self.idx_to_word, self.word_to_idx = self.build_word_vocab(merged_data_words, lf)
        self.num_lines = [len(target) for target in self.data_words]
        with open(self.word_idx_file, 'wb') as f:
            pickle.dump(self.idx_to_word, f)

        unk_get = lambda x: self.word_to_idx.get(x, self.word_to_idx.get(self.unk_token))
        self.word_tensors = np.array(
            [[list(map(unk_get, line)) for line in target] for target in self.data_words]
        )
        print(self.word_tensors.shape)
        for i, path in enumerate(self.word_tensor_files):
            np.save(path, self.word_tensors[i])

        self.show_corpus_info()

    def load_preprocessed(self):
        self.data_words = pickle.load(open(self.word_data_file, 'rb'))
        self.idx_to_word = pickle.load(open(self.word_idx_file, 'rb'))
        self.word_to_idx = {x: i for i, x in enumerate(self.idx_to_word)}
        self.word_vocab_size = len(self.idx_to_word)
        self.word_idx_file = dict(zip(self.idx_to_word, range(self.word_vocab_size)))
        self.num_lines = [len(target) for target in self.data_words]
        self.word_tensors = np.array([np.load(target) for target in self.word_tensor_files])

        self.show_corpus_info()

    def show_corpus_info(self):
        print('\n')
        print('--------- Corpus Info ---------')
        print('train num_lines : {} , test num_lines: {}'.format(self.num_lines[0], self.num_lines[1]))
        print('vocab_size: {}'.format(self.word_vocab_size))
        print('\n')
        print('\n')

    def build_word_vocab(self, merged_data_words, lf):
        word_counts = collections.Counter(merged_data_words).most_common()
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

    def next_batch(self, batch_size, target_str='train'):
        target = 0 if target_str == 'train' else 1
        for i in range(self.num_lines[target] // batch_size):

            indexes = range(self.num_lines[target])[i*batch_size:(i+1)*batch_size]
            if self.params['shuffle']:
                indexes = np.array(np.random.randint(self.num_lines[target], size=batch_size))

            encoder_word_input = [self.word_tensors[target][index] for index in indexes]
            decoder_word_input = [[self.word_to_idx[self.go_token]] + line for line in encoder_word_input]
            decoder_word_output = [line + [self.word_to_idx[self.end_token]] for line in encoder_word_input]
            sentences, encoder_word_input, input_seq_len, _ = self.sort_and_pad(encoder_word_input)
            _, decoder_word_input, _, decoder_mask = self.sort_and_pad(decoder_word_input)
            _, decoder_word_output, _, _ = self.sort_and_pad(decoder_word_output)

            yield sentences, np.array(encoder_word_input), input_seq_len, np.array(decoder_word_input), np.array(
                decoder_word_output), decoder_mask

    def go_input(self, batch_size):
        go_word_input = [[self.word_to_idx[self.go_token]] for _ in range(batch_size)]
        return np.array(go_word_input)

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
