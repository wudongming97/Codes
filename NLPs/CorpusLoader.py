import numpy as np
from functools import reduce
from Corpus import UNK_token, PAD_token, ParallelCorpus, SOS_token, EOS_token


def pad_sentence_batch(bt):
    padded_seqs = []
    seq_lens = []
    masks = []
    max_sentence_len = max([len(s) for s in bt])
    for s in bt:
        padded_seqs.append(s + [PAD_token]*(max_sentence_len-len(s)))
        seq_lens.append(len(s))
        masks.append([1]*len(s) + [0]*(max_sentence_len-len(s)))
    return padded_seqs, seq_lens, masks


def sentences2indices(sentences, word2idx):
    indices = [[word2idx.get(word, UNK_token) for word in s] for s in sentences]
    return indices


def indices2sentences(indices, idx2word):
    unk_word = idx2word[UNK_token]
    sentences = [[idx2word.get(idx, unk_word) for idx in idxs] for idxs in indices]
    return sentences


def to_inputs(sentences, word2idx):
    indices = sentences2indices(sentences, word2idx)
    padded_input, lens, masks = pad_sentence_batch(indices)
    return np.array(padded_input), lens, masks


def to_outputs(indices, idx2word):
    words_lists = indices2sentences(indices, idx2word)
    sentences = [reduce(lambda x, y: x + ' ' + y, words) for words in words_lists]
    return  sentences


class CorpusLoader:
    def __init__(self, sentences, word2idx, idx2word):
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.s_len = len(sentences)
        self.sentences = sentences
        self.vocab_sz = len(word2idx)

    def sentences_2_inputs(self, sentences, is_sort=True):
        split_sentences = [s.split() for s in sentences]
        if is_sort:
            split_sentences.sort(key=len, reverse=True)
        sorted_sentences = [reduce(lambda s1,s2:s1+ ' ' + s2, s, '') for s in split_sentences]
        return to_inputs(split_sentences, self.word2idx), sorted_sentences

    def to_outputs(self, indices):
        return to_outputs(indices, self.idx2word)

    def next_batch(self, batch_sz, target=False):
        for i in range(0, self.s_len - self.s_len % batch_sz, batch_sz):
            split_sentences_bt = [[word for word in s.split()] for s in self.sentences[i: i + batch_sz]]
            sorted_sentences_bt = sorted(split_sentences_bt, key=len, reverse=True)
            padded, lens, masks = to_inputs(sorted_sentences_bt, self.word2idx)
            # 如果作为目标， 则在每句的末尾分别添加'<EOS>'
            if target:
                taget_sorted_sentences_bt = [word_list + [self.idx2word[EOS_token]] for word_list in sorted_sentences_bt]
                t_padded, t_lens, t_masks = to_inputs(taget_sorted_sentences_bt, self.word2idx)
                yield (np.array(padded), lens, masks), (np.array(t_padded), t_lens, t_masks)
            else:
                yield (np.array(padded), lens, masks)


class ParallelCorpusLoader:
    def __init__(self, sentences_pair, X_word2idx, X_idx2word, Y_word2idx, Y_idx2word):
        self.sentences_pair = sentences_pair
        self.s_len = len(sentences_pair)
        self.X_word2idx = X_word2idx
        self.X_idx2word = X_idx2word
        self.Y_word2idx = Y_word2idx
        self.Y_idx2word = Y_idx2word
        self.X_vocab_sz = len(X_word2idx)
        self.Y_vocab_sz = len(Y_word2idx)

    def sentences_2_inputs(self, sentences_list):
        split_sentences = [s.split() for s in sentences_list]
        return to_inputs(split_sentences, self.X_word2idx)

    def to_outputs(self, indices):
        return to_outputs(indices, self.Y_idx2word)

    def next_batch(self, batch_sz):
        for i in range(0, self.s_len - self.s_len % batch_sz, batch_sz):
            split_sentences_pair_bt = [list(map(lambda x:x.split(), sp)) for sp in self.sentences_pair[i: i + batch_sz]]
            split_sentences_pair_bt.sort(key=lambda x: len(x[0]), reverse=True)
            X, Y = list(zip(*split_sentences_pair_bt))
            X_padded_bt, X_bt_lens, X_bt_masks = to_inputs(X, self.X_word2idx)
            Y_padded_bt, Y_bt_lens, Y_bt_masks = to_inputs(Y, self.Y_word2idx)
            yield ((np.array(X_padded_bt), X_bt_lens, X_bt_masks), (np.array(Y_padded_bt), Y_bt_lens, Y_bt_masks))


if __name__ == '__main__':
    file = '../datasets/en_vi_nlp/tst2012.en'
    file1 = '../datasets/en_vi_nlp/tst2012.vi'
    '''
    corpus = Corpus(file).process().trim(3)
    corpus_loader = CorpusLoader(corpus.sentences, corpus.word2idx, corpus.idx2word)
    for it, (padded_bt, bt_lens, bt_masks) in enumerate(corpus_loader.next_batch(3)):
        print(it)
        print(bt_lens)
    print('end')
    '''
    corpus2 = ParallelCorpus(file, file1).process().trim(2, 2)
    corpus_loader2 = ParallelCorpusLoader(corpus2.pair_sentences, corpus2.X_word2idx, corpus2.X_idx2word, corpus2.Y_word2idx, corpus2.Y_idx2word)
    for it, (X, Y) in enumerate(corpus_loader2.next_batch(2)):
        X_padded_bt, X_bt_lens, X_bt_masks = X
        Y_padded_bt, Y_bt_lens, Y_bt_masks = Y
        print(it)
        print(X_bt_lens)
        print(Y_bt_lens)
    print('end')


