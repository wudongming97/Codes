import numpy as np
from Corpus import UNK_token, PAD_token, Corpus

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
            padded_bt, bt_lens, bt_masks = self.pad_sentence_batch(indices_bt)
            yield (np.array(padded_bt), bt_lens, bt_masks)

    @staticmethod
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


if __name__ == '__main__':
    file = 'nlp_dataset/Corpus.test.txt'
    corpus = Corpus(file).process().trim(3)
    corpus_loader = CorpusLoader(corpus.sentences, corpus.word2idx, corpus.idx2word)
    for it, (padded_bt, bt_lens, bt_masks) in enumerate(corpus_loader.next_batch(3)):
        print(it)
        print(bt_lens)
    print('end')


