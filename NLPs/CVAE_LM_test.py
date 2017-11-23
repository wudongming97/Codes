import datetime, os
from Corpus import Corpus
from CorpusLoader import CorpusLoader
from CVAE_LM import CVAE_LM
from functools import reduce

TRAIN = 0
EVALUTION = 1
mode = EVALUTION

model_args = {
        'name': 'CVAE_LM_test',
        'emb_dim': 312,
        'hid_sz': 156,
        'n_layers': 2,
        'z_dim': 32
    }

hyper_params = {
        'epoch': 10,
        'lr': 0.001,
        'batch_sz': 60,
        'max_grad_norm': 5
    }

sentence_filter = lambda s:len(s.split()) > 5 and len(s.split()) < 15
corpus = Corpus('nlp_dataset/Corpus.train.txt').filter_sentences(sentence_filter).process()
corpus_loader = CorpusLoader(corpus.sentences, corpus.word2idx, corpus.idx2word)
cvae_lm = CVAE_LM(corpus_loader, model_args, hyper_params)

if __name__ == '__main__':
    if mode == TRAIN:
        cvae_lm.fit()
        cvae_lm.save()
    elif mode == EVALUTION:
        print('begin evaluate...')
        cvae_lm.load()

        # generate
        indices = cvae_lm.generate(100)
        word_lists = corpus_loader.indices2sentences(indices.numpy().tolist())
        sentences = [reduce(lambda x, y: x + ' ' + y, words) for words in word_lists]
        # save sentences to file
        save_dir = './res_samples/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        res_file = save_dir + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") + '_generated_sentences.txt'
        with open(res_file, 'a') as f:
            for s in sentences:
                f.write(s + '\n')

        print('end evaluate and result is in the file : {}'.format(res_file))
    else:
        print("Done nothing...")


