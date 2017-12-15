import pickle

PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3


class Corpus:
    def __init__(self, file_path):
        self.file = file_path
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.word2count = {}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.n_words = 4
        self.sentences = open(self.file, encoding='UTF-8').read().strip().lower().split('\n')

    def filter_sentences(self, f):
        sentences = [s for s in self.sentences if not f(s)]
        self.sentences = sentences
        return self

    def process(self):
        print('process raw data...\n')
        for s in self.sentences:
            self.__index_words(s)
        print('sentences num : {}, vocab size: {}\n'.format(len(self.sentences), self.n_words))
        return self

    def save(self, path='./'):
        print('saving corpus ...')
        sentences = [s.split() for s in self.sentences]
        with open(path + 'data.pkl', 'wb') as f:
            pickle.dump(sentences, f)
        with open(path + 'idx2word.pkl', 'wb') as f:
            pickle.dump(self.idx2word, f)
        with open(path + 'word2idx.pkl', 'wb') as f:
            pickle.dump(self.word2idx, f)

    def trim(self, count):
        keep_words = []
        for k, v in self.word2count.items():
            if v >= count:
                keep_words.append(k)
        print('keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.word2idx), len(keep_words) / len(self.word2idx)
        ))
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.word2count = {}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.n_words = 4
        for word in keep_words:
            self.__index_word(word)
        return self

    def __index_words(self, sentence):
        for word in sentence.split(' '):
            self.__index_word(word)

    def __index_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.word2count[word] = 1
            self.idx2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class ParallelCorpus:
    def __init__(self, src_file, target_file):
        self.pair_sentences = self.pair_sentences(src_file, target_file)
        self.X_word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.X_word2count = {}
        self.X_idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.X_n_words = 4
        self.Y_word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.Y_word2count = {}
        self.Y_idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.Y_n_words = 4

    def pair_sentences(self, src_file, target_file):
        with open(src_file, encoding='UTF-8') as f1:
            lines1 = f1.read().strip().lower().split('\n')
        with open(target_file, encoding='UTF-8') as f2:
            lines2 = f2.read().strip().lower().split('\n')
        pair_sentences = list(zip(lines1, lines2))
        return pair_sentences

    def process(self):
        print('process raw data...\n')
        for s in self.pair_sentences:
            self.__index_sentence_pair(s)
        print('source corpus info: sentences num : {}, vocab size: {}\n'.format(len(self.pair_sentences), self.X_n_words))
        print('target corpus info: sentences num : {}, vocab size: {}\n'.format(len(self.pair_sentences), self.Y_n_words))
        return self

    def filter_sentence_pairs(self, src_f=None, target_f=None):
        if src_f is not None:
            sentences_pairs = [(s, t) for (s, t) in self.pair_sentences if not src_f(s)]
            self.pair_sentences = sentences_pairs
        if target_f is not None:
            sentences_pairs = [(s, t) for (s, t) in self.pair_sentences if not target_f(t)]
            self.pair_sentences = sentences_pairs
        return self

    def trim(self, src_count, target_count):
        self.trimX(src_count)
        self.trimY(target_count)
        return self

    def trimX(self, src_count):
        # trim x
        keep_words = []
        for k, v in self.X_word2count.items():
            if v >= src_count:
                keep_words.append(k)
        print('source corpus: keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.X_word2idx), len(keep_words) / len(self.X_word2idx)
        ))
        self.X_word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.X_word2count = {}
        self.X_idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.X_n_words = 4
        for word in keep_words:
            self.__X_index_word(word)

    def trimY(self, target_count):
        # trim y
        keep_words = []
        for k, v in self.Y_word2count.items():
            if v >= target_count:
                keep_words.append(k)
        print('target corpus: keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.Y_word2idx), len(keep_words) / len(self.Y_word2idx)
        ))
        self.Y_word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.Y_word2count = {}
        self.Y_idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.Y_n_words = 4
        for word in keep_words:
            self.__Y_index_word(word)

    def __index_sentence_pair(self, sentence_pair):
        for word in sentence_pair[0].split(' '):
            self.__X_index_word(word)
        for word in sentence_pair[1].split(' '):
            self.__Y_index_word(word)

    def __X_index_word(self, word):
        if word not in self.X_word2idx:
            self.X_word2idx[word] = self.X_n_words
            self.X_word2count[word] = 1
            self.X_idx2word[self.X_n_words] = word
            self.X_n_words += 1
        else:
            self.X_word2count[word] += 1

    def __Y_index_word(self, word):
        if word not in self.Y_word2idx:
            self.Y_word2idx[word] = self.Y_n_words
            self.Y_word2count[word] = 1
            self.Y_idx2word[self.Y_n_words] = word
            self.Y_n_words += 1
        else:
            self.Y_word2count[word] += 1

if __name__ == '__main__':
    file = '../datasets/en_vi_nlp/train.en'
    corpus = Corpus(file).process().trim(1)
    corpus.save()

    print('end')








