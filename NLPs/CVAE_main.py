# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.CVAE import CVAE, USE_GPU
from utils.DataLoader import DataLoader, Vocab, Level


# ========================================================================
#  word_level_params
# ========================================================================
class word_level_params:
    encoder_params = {
        'rnn_cell_str': 'gru',
        'emb_size': 512,
        'hidden_size': 512,
        'n_layers': 1,
        'bidirectional': False,
    }

    decoder_params = {
        'rnn_cell_str': 'gru',
        'emb_size': 512,
        'hidden_size': 512,
        'n_layers': 1,
        'bidirectional': False,
        'input_dropout_p': 0.8,
    }

    params = {
        'n_epochs': 30,
        'lr': 0.001,
        'batch_size': 64,
        'z_size': 16,
        'max_grad_norm': 5,
        'top_k': 1,
        'word_dropout_p': 0.2,
        'kl_lss_anneal': True,
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        'model_name': 'trained_word_CVAE.model',
        'only_rec_loss': True,  # for debug
    }


if __name__ == '__main__':

    ptb_train_loader = DataLoader(Vocab('europarl_train_cvae', Level.WORD))

    level = word_level_params()
    level.params['vocab_size'] = ptb_train_loader.vocab.vocab_size

    model = CVAE(level.encoder_params, level.decoder_params, level.params)
    if USE_GPU:
        model = model.cuda()

    if model.have_saved_model:
        model.load()
    else:
        # train
        model.fit(ptb_train_loader)
        model.save()

    # 随机生成1000个句子
    for i in range(1000):
        print('{}, {}'.format(i, model.sample_from_normal(ptb_train_loader)))
