# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random

import numpy as np
import torch

import utils.Utils as U
from utils.DataLoader import U_TOKEN, E_TOKEN
from utils.Nll import nll

# constant
USE_GPU = torch.cuda.is_available()
TORCH_VERSION = torch.__version__


# helper function
def _rnn_cell_helper(str_rnn_cell):
    if str_rnn_cell.lower() == 'lstm':
        rnn_cell = torch.nn.LSTM
    elif str_rnn_cell.lower() == 'gru':
        rnn_cell = torch.nn.GRU
    else:
        raise ValueError("Unsupported RNN Cell: {0}".format(str_rnn_cell))

    return rnn_cell


# model
class Encoder(torch.nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params

        self.emb_size = self.params.get('emb_size')
        self.vocab_size = self.params.get('vocab_size')
        self.n_layers = self.params.get('n_layers')
        self.hidden_size = self.params.get('hidden_size')
        self.bidirectional = self.params.get('bidirectional')
        self.num_directions = 2 if self.params.get('bidirectional') else 1
        self.rnn_cell_str = self.params.get('rnn_cell_str')
        self.rnn_cell = _rnn_cell_helper(self.rnn_cell_str)

        # model
        self.embedding = torch.nn.Embedding(self.vocab_size, self.emb_size)
        self.rnn = self.rnn_cell(self.emb_size, self.hidden_size, self.n_layers,
                                 bidirectional=self.bidirectional, batch_first=True)

    def forward(self, inputs, inputs_len):
        embedded = self.embedding(inputs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, inputs_len, batch_first=True)
        _, hidden = self.rnn(packed)
        if self.rnn_cell_str == 'lstm':
            return hidden[0]
        else:
            return hidden


class Decoder(torch.nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params

        self.emb_size = self.params.get('emb_size')
        self.vocab_size = self.params.get('vocab_size')
        self.n_layers = self.params.get('n_layers')
        self.hidden_size = self.params.get('hidden_size')
        self.bidirectional = self.params.get('bidirectional')
        self.num_directions = 2 if self.params.get('bidirectional') else 1
        self.rnn_cell_str = self.params.get('rnn_cell_str')
        self.rnn_cell = _rnn_cell_helper(self.rnn_cell_str)

        self.input_dropout_p = self.params.get('input_dropout_p')

        # model
        self.embedding = torch.nn.Embedding(self.vocab_size, self.emb_size)
        self.dropout_layer = torch.nn.AlphaDropout(self.input_dropout_p)
        self.rnn = self.rnn_cell(self.emb_size, self.hidden_size, self.n_layers,
                                 bidirectional=self.bidirectional, batch_first=True)

    def forward(self, inputs, h0):
        embedded = self.embedding(inputs)
        embedded_dropout = self.dropout_layer(embedded)
        output, hidden = self.rnn(embedded_dropout, h0)
        return output, hidden


class CVAE(torch.nn.Module):
    def __init__(self, encoder_params, decoder_params, params):
        super(CVAE, self).__init__()

        self.params = params
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params

        self.encoder_params['vocab_size'] = self.params.get('vocab_size')
        self.decoder_params['vocab_size'] = self.params.get('vocab_size')

        self.batch_size = self.params.get('batch_size')
        self.vocab_size = self.params.get('vocab_size')
        self.z_size = self.params.get('z_size')
        self.model_name = self.params.get('model_name')
        self.lr = self.params.get('lr')
        self.beta = self.params.get('beta')
        self.kl_lss_anneal = self.params.get('kl_lss_anneal')
        self.n_epochs = self.params.get('n_epochs')
        self.word_dropout_p = self.params.get('word_dropout_p')
        self.top_k = self.params.get('top_k')
        self.max_grad_norm = self.params.get('max_grad_norm')

        # model
        self.encoder = Encoder(self.encoder_params)
        self.decoder = Decoder(self.decoder_params)

        self.fc_mu = torch.nn.Linear(
            self.encoder.num_directions * self.encoder.hidden_size,
            self.z_size)
        self.fc_logvar = torch.nn.Linear(
            self.encoder.num_directions * self.encoder.hidden_size,
            self.z_size)
        self.fc_h = torch.nn.Linear(
            self.z_size,
            self.decoder.n_layers * self.decoder.num_directions * self.decoder.hidden_size
        )
        # only for lstm
        if self.decoder.rnn_cell_str == 'lstm':
            self.fc_c = torch.nn.Linear(
                self.z_size,
                self.decoder.n_layers * self.decoder.num_directions * self.decoder.hidden_size
            )

        self.fc_out = torch.nn.Linear(
            self.decoder.num_directions * self.decoder.hidden_size, self.vocab_size)

        # train
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.have_saved_model = os.path.exists(self.model_name)

    def _sample_z(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.autograd.Variable(torch.randn(mu.size()))
        if USE_GPU:
            eps = eps.cuda()
        ret = mu + std * eps
        return ret

    @staticmethod
    def _kld_loss(mu, log_var):
        kld_loss = (-0.5 * torch.sum(log_var - torch.pow(mu, 2) - torch.exp(log_var) + 1, 1)).mean().squeeze()
        return kld_loss

    def _forward_d(self, Y_i, decoder_hidden):
        decoder_output, hidden = self.decoder(Y_i, decoder_hidden)
        decoder_output = decoder_output.contiguous().view(-1, self.decoder.num_directions * self.decoder.hidden_size)
        output = self.fc_out(decoder_output)

        return output, hidden

    def _stats_from_z(self, z):
        batch_size = z.size()[0]
        if self.decoder.rnn_cell_str == 'lstm':
            decoder_hidden = self.fc_h(z), self.fc_c(z)
            decoder_hidden = (h.view(self.decoder.n_layers * self.decoder.num_directions, batch_size, -1) for h in
                              decoder_hidden)
            return tuple(decoder_hidden)
        elif self.decoder.rnn_cell_str == 'gru':
            decoder_hidden = self.fc_h(z).view(self.decoder.n_layers * self.decoder.num_directions, batch_size, -1)
            return decoder_hidden
        else:
            raise ValueError("Unsupported RNN Cell")

    def forward(self, X, X_lengths, Y_i):
        context = self.encoder(X, X_lengths)

        context = context[-self.encoder.num_directions:, :, :].view(self.batch_size, -1)
        mu, log_var = self.fc_mu(context), self.fc_logvar(context)
        kld_loss = self._kld_loss(mu, log_var)

        z = self._sample_z(mu, log_var)

        decoder_hidden = self._stats_from_z(z)
        output, _ = self._forward_d(Y_i, decoder_hidden)

        return output, kld_loss

    @staticmethod
    def _rec_loss(d_output, d_target, d_mask):
        # recon loss 。 note：这里的log_softmax是必须的， 应为nll的输入需要log_prob
        losses = nll(torch.nn.functional.log_softmax(d_output), d_target.view(-1, 1))
        target_mask = torch.autograd.Variable(torch.FloatTensor(d_mask)).view(-1)
        if USE_GPU:
            target_mask = target_mask.cuda()
        loss = torch.mul(losses, target_mask).mean().squeeze()
        return loss

    def _kld_coef(self, cur_epoch, cur_iter):
        scalar_sigmoid = lambda x: 1 / (1 + math.exp(-x))
        if self.kl_lss_anneal:
            return scalar_sigmoid(-10 + 20 * cur_epoch / self.n_epochs)
        else:
            return 1

    def _word_dropout_helper(self, loader, p, x):
        if np.random.binomial(1, p) == 1:
            return loader.vocab.idx[U_TOKEN]
        else:
            return x

    def fit(self, loader, display_step=15):
        print('begin fit ...\n')
        for it, data in enumerate(loader.next_batch(self.batch_size)):
            e = U.step_to_epoch(it, loader.num_line, self.batch_size)

            X, X_lengths, Y_i, Y_masks, Y_t = loader.unpack_for_cvae(data)
            sentences = loader.to_seqs(X)

            if self.word_dropout_p >= 0:
                drop_word_f = lambda x: self._word_dropout_helper(loader, self.word_dropout_p, x)
                Y_i = [list(map(drop_word_f, s)) for s in Y_i]

            X = torch.autograd.Variable(torch.Tensor(X).long())
            Y_i = torch.autograd.Variable(torch.Tensor(Y_i).long())
            Y_t = torch.autograd.Variable(torch.Tensor(Y_t).long())

            if USE_GPU:
                X, Y_i, Y_t = X.cuda(), Y_i.cuda(), Y_t.cuda()

            kld_coef = self._kld_coef(e, it)
            kl_lss, rec_lss = self.train_bt(X, X_lengths, Y_i, Y_t, Y_masks, kld_coef)

            if it % display_step == 0:
                global_step_ = U.epoch_to_step(self.n_epochs, loader.num_line, self.batch_size)
                str_ = "Epoch %d/%d | step %d/%d | train_loss: %.3f | rec_loss: %.3f | kl_loss: %.6f | kld_coef: %.6f |"
                print(
                    str_.format(e + 1, self.n_epochs, it, global_step_, self.beta * kld_coef * kl_lss + rec_lss, rec_lss, kl_lss,
                                kld_coef))

            if it % (display_step * 20) == 0:
                # 查看重构情况
                print('\n------------ reconstruction --------------')
                for i, s in enumerate(sentences):
                    if i > 4:
                        break
                    print('-----')
                    print("Input: {}\nOutput: {} ".format(s, self.sample_from_encoder(loader, s)))
                # 查看随机生成情况
                print('\n------------ sample_from_normal ----------')
                for i in range(self.batch_size):
                    if i > 4:
                        break
                    print('{}, {}'.format(i, self.sample_from_normal(loader)))
                print('\n')

            # 训练结束
            if e >= self.n_epochs:
                break

    def sample_from_normal(self, loader):
        z = torch.autograd.Variable(torch.randn(1, self.z_size))
        if USE_GPU: z = z.cuda()
        return self._sample_from_z(loader, z)

    def sample_from_encoder(self, loader, sentence):
        words = sentence.split()
        e_input = loader.to_tensor([words])
        e_input_len = list(map(len, e_input))
        e_input = torch.autograd.Variable(torch.from_numpy(np.array(e_input)))
        if USE_GPU:
            e_input = e_input.cuda()

        context = self.encoder(e_input, e_input_len).view(1, -1)
        mu, log_var = self.fc_mu(context), self.fc_logvar(context)
        z = self._sample_z(mu, log_var)
        return self._sample_from_z(loader, z)

    def _sample_from_z(self, loader, z):
        # 一句一句的采样，所以batch_size都填1
        Y_i = np.array(loader.g_input(batch_size=1))
        Y_i = torch.autograd.Variable(torch.from_numpy(Y_i).long())

        result_idx = []
        decoder_hidden = self._stats_from_z(z)
        for i in range(loader.max_seq_len):
            if USE_GPU: Y_i = Y_i.cuda()
            d_output, decoder_hidden = self._forward_d(Y_i, decoder_hidden)

            d_output_np = d_output.data.squeeze().cpu().numpy()
            ixs = d_output_np.argsort()[-self.top_k:][::-1].tolist()  # top_k
            ix = ixs[random.randint(0, len(ixs) - 1)]

            if loader.vocab.vocab[ix] == E_TOKEN:
                break

            result_idx.append(ix)
            Y_i = torch.autograd.Variable(torch.from_numpy(np.array([[ix]])).long())

        return loader.to_seqs([result_idx])[0]

    def save(self):
        torch.save(self.state_dict(), self.model_name)
        print('model saved ...')

    def load(self):
        self.load_state_dict(torch.load(self.model_name))
        print('model loaded ...')

    def train_bt(self, X, X_lengths, Y_i, Y_t, Y_mask, kld_coef):
        self.optimizer.zero_grad()
        d_output, kld_lss = self(X, X_lengths, Y_i)
        rec_lss = self._rec_loss(d_output, Y_t, Y_mask)

        # update
        lss = self.beta * kld_coef * kld_lss + rec_lss
        lss.backward()
        torch.nn.utils.clip_grad_norm(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return kld_lss.data[0], rec_lss.data[0]
