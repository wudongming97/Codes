import datetime, os
import numpy as np
import torch
from Corpus import Corpus
from CorpusLoader import CorpusLoader
from CVAE_LM import CVAE_LM
from functools import reduce

TRAIN = 0
EVALUTION = 1
mode = TRAIN

model_args = {
        'name': 'CVAE_LM_test',
        'encoder_bid': True,
        'decoder_bid': False,
        'emb_dim': 512,
        'hid_sz': 256,
        'n_layers': 2,
        'z_dim': 16
    }

hyper_params = {
        'epoch': 10,
        'lr': 0.0002,
        'batch_sz': 1,
        'max_grad_norm': 5
    }

train_data_path = '../datasets/en_vi_nlp/train.en'
test_data_path = '../datasets/en_vi_nlp/tst2012.en'

remove_sentences_by_lenght = lambda s:len(s.split()) < 4 or len(s.split()) > 15
remove_blank_sentences = lambda s: len(s) == 0
corpus = Corpus(train_data_path).filter_sentences(remove_sentences_by_lenght).filter_sentences(remove_blank_sentences).process()
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
        g_size = [100, model_args['z_dim']]
        mu = torch.zeros(g_size)
        log_var = torch.ones(g_size)
        sentences = cvae_lm.generate_by_z(mu, log_var)

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


