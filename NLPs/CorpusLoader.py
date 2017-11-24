import numpy as np
from Corpus import UNK_token, PAD_token, Corpus, ParallelCorpus


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


class CorpusLoader:
    def __init__(self, sentences, word2idx, idx2word):
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.s_len = len(sentences)
        self.sentences = sentences
        self.vocab_sz = len(word2idx)

    def sentences2indices(self, sentences):
        indices = [[self.word2idx.get(word, UNK_token) for word in s] for s in sentences]
        return indices

    def indices2sentences(self, indices):
        unk_word = self.idx2word[UNK_token]
        sentences = [[self.idx2word.get(idx, unk_word) for idx in idxs] for idxs in indices]
        return sentences

    def next_batch(self, batch_sz):
        for i in range(0, self.s_len - self.s_len % batch_sz, batch_sz):
            split_sentences_bt = [[word for word in s.split()] for s in self.sentences[i: i + batch_sz]]
            sorted_sentences_bt = sorted(split_sentences_bt, key=len, reverse=True)
            indices_bt = self.sentences2indices(sorted_sentences_bt)
            # pad
            padded_bt, bt_lens, bt_masks = pad_sentence_batch(indices_bt)
            yield (np.array(padded_bt), bt_lens, bt_masks)


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

    def next_batch(self, batch_sz):
        for i in range(0, self.s_len - self.s_len % batch_sz, batch_sz):
            split_sentences_pair_bt = [list(map(lambda x:x.split(), sp)) for sp in self.sentences_pair[i: i + batch_sz]]
            split_sentences_pair_bt.sort(key=lambda x: len(x[0]), reverse=True)
            X_indices, Y_indices = self.spairs2indices(split_sentences_pair_bt)
            # pad
            X_padded_bt, X_bt_lens, X_bt_masks = pad_sentence_batch(X_indices)
            Y_padded_bt, Y_bt_lens, Y_bt_masks = pad_sentence_batch(Y_indices)
            yield ((np.array(X_padded_bt), X_bt_lens, X_bt_masks), (np.array(Y_padded_bt), Y_bt_lens, Y_bt_masks))

    def spairs2indices(self, s_pairs):
        X, Y = list(zip(*s_pairs))
        X_indices = [[self.X_word2idx.get(word, UNK_token) for word in s] for s in X]
        Y_indices = [[self.Y_word2idx.get(word, UNK_token) for word in s] for s in Y]
        return X_indices, Y_indices



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


