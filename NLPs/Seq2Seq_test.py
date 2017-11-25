import os

TRAIN = 0
EVALUTION = 1

mode = TRAIN

model_args = {
    'name': 'seq2seq_test',
    'encoder_embedding_dim': 256,
    'decoder_embedding_dim': 256,
    'hidden_size': 128,
    'n_layers': 3
    }

hyper_params = {
    'epoch': 10,
    'lr': 0.0005,
    'batch_sz': 60,
    'max_grad_norm': 5
}

train_data_source = '../datasets/en_vi_nlp/train.en'
train_data_target = '../datasets/en_vi_nlp/train.vi'
test_data_source = '../datasets/en_vi_nlp/tst2012.en'
test_data_target = '../datasets/en_vi_nlp/tst2012.vi'

from Corpus import ParallelCorpus, Corpus
from CorpusLoader import ParallelCorpusLoader
from Seq2Seq import Seq2Seq


is_blank_line = lambda s: len(s) == 0
corpus = ParallelCorpus(train_data_source, train_data_target).filter_sentence_pairs(is_blank_line, is_blank_line).process()
corpusLoader = ParallelCorpusLoader(corpus.pair_sentences, corpus.X_word2idx, corpus.X_idx2word,
                                    corpus.Y_word2idx, corpus.Y_idx2word)
seq2seq = Seq2Seq(corpusLoader, model_args, hyper_params)


def test(batch_sz):
    test_lines = Corpus(test_data_source).filter_sentences(is_blank_line).sentences
    test_lines_len = len(test_lines)
    for i in range(0, test_lines_len - test_lines_len % batch_sz, batch_sz):
        bas = test_lines[i: i + batch_sz]
        bas.sort(key=lambda x:len(x.split()), reverse=True)
        X_input, X_lens, _= corpusLoader.sentences_2_inputs(bas)
        ret = seq2seq.predict(X_input, X_lens)
        sentences_list = corpusLoader.to_outputs(ret.tolist())
        with open('result.txt', 'a', encoding='UTF-8') as f:
            write_ret = [s+'\n' for s in sentences_list]
            f.writelines(write_ret)


if __name__ == '__main__':
    if mode == TRAIN:
        seq2seq.fit()
        seq2seq.save()
        test(60)
    else:
        print('begin evaluate...')
        seq2seq.load()
        test(60)
        print('end evaluate...')



