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
        self.sentences = open(self.file, encoding='UTF-8').read().strip().split('\n')

    def filter_sentences(self, f):
        sentences = [s for s in self.sentences if f(s)]
        self.sentences = sentences
        return self

    def process(self):
        print('process raw data...\n')
        for s in self.sentences:
            self.__index_words(s)
        print('sentences num : {}, vocab size: {}\n'.format(len(self.sentences), self.n_words))
        return self

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


if __name__ == '__main__':
    file = 'nlp_dataset/Corpus.test.txt'
    corpus = Corpus(file).process().trim(3)

    print('end')








