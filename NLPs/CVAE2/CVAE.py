import os
import math
import random
import torch
import numpy as np
from Utils import nll
from CorpusLoader import CorpusLoader

# 解决输出报UnicodeEncodeError
import sys, codecs
if sys.stdout.encoding != 'UTF-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
if sys.stderr.encoding != 'UTF-8':
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

USE_GPU = torch.cuda.is_available()
if USE_GPU:
    torch.cuda.set_device(1)

class Encoder(torch.nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.num_directions = 2 if self.params['bidirectional'] else 1
        
        # model
        self.embedding = torch.nn.Embedding(self.params['vocab_size'], self.params['emb_size'])
        self.rnn = torch.nn.LSTM(self.params['emb_size'], self.params['hidden_size'], self.params['n_layers'],
                                 bidirectional=self.params['bidirectional'], batch_first=True)
        
    def init_hidden(self, batch_size):
        h0 = torch.autograd.Variable(
            torch.zeros(self.params['n_layers'] * self.num_directions, batch_size, self.params['hidden_size']))
        c0 = torch.autograd.Variable(
            torch.zeros(self.params['n_layers'] * self.num_directions, batch_size, self.params['hidden_size']))
        if USE_GPU:
            h0, c0 = h0.cuda(), c0.cuda()
        return h0, c0
    
    def forward(self, inputs, inputs_len, hidden):
        embedded = self.embedding(inputs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, inputs_len, batch_first=True)
        _, (hidden, _) = self.rnn(packed, hidden)
        return hidden


class Decoder(torch.nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.num_directions = 2 if self.params['bidirectional'] else 1

        # model
        self.embedding = torch.nn.Embedding(self.params['vocab_size'], self.params['emb_size'])
        self.drop_out_layer = torch.nn.AlphaDropout(self.params['drop_out'])
        self.rnn = torch.nn.LSTM(self.params['emb_size'], self.params['hidden_size'], self.params['n_layers'],
                                 bidirectional=self.params['bidirectional'], batch_first=True)

    def forward(self, inputs, h0, c0):
        embedded = self.embedding(inputs)
        embedded_dropout = self.drop_out_layer(embedded)
        output, (hidden, _) = self.rnn(embedded_dropout, (h0, c0))
        return output, hidden

    def init_hidden(self, batch_size):
        h0 = torch.autograd.Variable(
            torch.zeros(self.params['n_layers'] * self.num_directions, batch_size, self.params['hidden_size']))
        c0 = torch.autograd.Variable(
            torch.zeros(self.params['n_layers'] * self.num_directions, batch_size, self.params['hidden_size']))
        if USE_GPU:
            h0, c0 = h0.cuda(), c0.cuda()
        return h0, c0


class CVAE(torch.nn.Module):
    def __init__(self, encoder_params, decoder_params, params):
        super(CVAE, self).__init__()

        self.params = params
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params
        self.encoder_params['vocab_size'] = self.params['vocab_size']
        self.decoder_params['vocab_size'] = self.params['vocab_size']
        self.batch_size = self.params['batch_size']

        # model
        self.encoder = Encoder(self.encoder_params)
        self.decoder = Decoder(self.decoder_params)

        self.fc_mu = torch.nn.Linear(
            self.encoder_params['n_layers'] * self.encoder.num_directions * self.encoder_params['hidden_size'],
            self.params['z_size'])
        self.fc_logvar = torch.nn.Linear(
            self.encoder_params['n_layers'] * self.encoder.num_directions * self.encoder_params['hidden_size'],
            self.params['z_size'])
        self.fc_h = torch.nn.Linear(
            self.params['z_size'],
            self.decoder_params['n_layers'] * self.decoder.num_directions * self.decoder_params['hidden_size']
        )
        self.fc_out = torch.nn.Linear(
            self.decoder.num_directions * self.decoder_params['hidden_size'], self.params['vocab_size'])

        # train
        self.optimizer = torch.optim.Adam(self.parameters(),  lr=self.params['lr'])
        self.have_saved_model = os.path.exists(self.params['model_name'])

    def sample_z(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.autograd.Variable(torch.randn(mu.size()))
        if USE_GPU:
            eps = eps.cuda()
        ret = mu + std * eps
        return ret

    @staticmethod
    def kld_loss(mu, log_var):
        kld_loss = (-0.5 * torch.sum(log_var - torch.pow(mu, 2) - torch.exp(log_var) + 1, 1)).mean().squeeze()
        return kld_loss

    def forward_d(self, d_inputs, decoder_hidden):
        batch_size = d_inputs.size()[0]

        _, c0 = self.decoder.init_hidden(batch_size)
        decoder_output, hidden = self.decoder(d_inputs, decoder_hidden, c0)
        decoder_output = decoder_output.contiguous().view(-1, self.decoder.num_directions * self.decoder_params[
            'hidden_size'])
        output = self.fc_out(decoder_output)

        return output, hidden

    def forward(self, e_inputs, e_inputs_len, d_inputs):
        encoder_hidden = self.encoder.init_hidden(self.batch_size)
        context = self.encoder(e_inputs, e_inputs_len, encoder_hidden)

        context = context.view(self.batch_size, -1)
        mu, logvar = self.fc_mu(context), self.fc_logvar(context)
        kld_loss = self.kld_loss(mu, logvar)

        z = self.sample_z(mu, logvar)

        decoder_hidden = self.fc_h(z)
        decoder_hidden = decoder_hidden.view(self.decoder_params['n_layers'] * self.decoder.num_directions,
                                             self.batch_size, -1)
        output, _ = self.forward_d(d_inputs, decoder_hidden)

        return output, kld_loss

    @staticmethod
    def rec_loss(d_output, d_target, d_mask):
        # recon loss
        losses = nll(torch.nn.functional.log_softmax(d_output), d_target.view(-1, 1))
        target_mask = torch.autograd.Variable(torch.FloatTensor(d_mask)).view(-1)
        if USE_GPU:
            target_mask = target_mask.cuda()
        loss = torch.mul(losses, target_mask).mean().squeeze()
        return loss

    def kld_coef(self, cur_epoch, cur_iter):
        if self.params['kl_lss_anneal']:
            return math.exp(cur_epoch - self.params['n_epochs'])
        else:
            return 1

    def fit(self, corpus_loader, display_step=15):
        print('begin fit ...')
        n_epochs = self.params['n_epochs']
        for e in range(n_epochs):
            for it, inputs in enumerate(corpus_loader.next_batch(self.batch_size, target_str='train')):
                sentences, encoder_word_input, input_seq_len, decoder_word_input, decoder_word_output, decoder_mask = inputs

                encoder_word_input = torch.autograd.Variable(torch.from_numpy(encoder_word_input))
                decoder_word_input = torch.autograd.Variable(torch.from_numpy(decoder_word_input))
                decoder_word_output = torch.autograd.Variable(torch.from_numpy(decoder_word_output))

                if USE_GPU:
                    encoder_word_input, decoder_word_input = encoder_word_input.cuda(), decoder_word_input.cuda()
                    decoder_word_output = decoder_word_output.cuda()

                kl_lss, rec_lss = self.train_bt(encoder_word_input, input_seq_len, decoder_word_input,
                                                 decoder_word_output, decoder_mask, self.kld_coef(e, it))

                if it % display_step == 0:
                    print(
                        "Epoch %d/%d | Batch %d/%d | train_loss: %.3f | kl_loss: %.3f | rec_loss: %.3f | kld_coef: %.3f |" %
                        (e+1, n_epochs, it, corpus_loader.num_lines[0] // self.batch_size, kl_lss + rec_lss, kl_lss,
                         rec_lss, self.kld_coef(e+1, it+1)))

                if it % (display_step * 20) == 0:
                    # 查看重构情况
                    print('\n------------ reconstruction --------------')
                    for i, s in enumerate(sentences):
                        if i > 4:
                            break
                        print('-----')
                        print("Input: {}\nOutput: {} ".format(s, self.sample_from_encoder(corpus_loader, s)))
                    print('\n------------ sample_from_normal ----------')
                    for i in range(self.params['batch_size']):
                        if  i > 4:
                            break
                        print('{}, {}'.format(i, self.sample_from_normal(corpus_loader)))
                    print('\n')


    def sample_from_normal(self, corpus_loader):
        z_np = np.random.normal(0, 1, self.params['z_size'])
        return self.sample_from_z(corpus_loader, z_np)

    def sample_from_encoder(self, corpus_loader, sentence):
        words = sentence.split()
        e_input = [corpus_loader.word_to_idx.get(w, corpus_loader.word_to_idx[corpus_loader.unk_token]) for w in words]
        e_input_len = [len(e_input)]
        e_input = torch.autograd.Variable(torch.from_numpy(np.atleast_2d(e_input)))
        if USE_GPU:
            e_input = e_input.cuda()

        encoder_hidden = self.encoder.init_hidden(1)
        context = self.encoder(e_input, e_input_len, encoder_hidden).view(1, -1)
        mu, logvar = self.fc_mu(context), self.fc_logvar(context)
        z = self.sample_z(mu, logvar)
        return self.sample_from_z(corpus_loader, z.data.cpu().numpy())

    def sample_from_z(self, corpus_loader, z_np):
        # 一句一句的采样，所以batch_size都填1
        z = torch.autograd.Variable(torch.from_numpy(z_np)).float().view(-1, self.params['z_size'])
        decoder_word_input_np = corpus_loader.go_input(1)
        decoder_word_input = torch.autograd.Variable(torch.from_numpy(decoder_word_input_np).long())
        if USE_GPU:
            z, decoder_word_input = z.cuda(), decoder_word_input.cuda()
        result = ''

        decoder_hidden = self.fc_h(z)
        decoder_hidden = decoder_hidden.view(self.decoder_params['n_layers'] * self.decoder.num_directions, 1, -1)

        for i in range(corpus_loader.params['keep_seq_lens'][1]):
            d_output, decoder_hidden = self.forward_d(decoder_word_input, decoder_hidden)

            # prediction = torch.nn.functional.softmax(d_output)
            # ix, word = corpus_loader.sample_word_from_distribution(prediction .data.cpu().numpy())

            d_output_np = d_output.data.squeeze().cpu().numpy()
            ixs, words = corpus_loader.top_k(d_output_np, self.params['top_k'])
            choice = random.randint(0, len(ixs)-1)
            ix, word = ixs[choice], words[choice]

            if word == corpus_loader.end_token:
                break

            result += ' ' + word

            decoder_word_input_np = np.array([[ix]])
            decoder_word_input = torch.autograd.Variable(torch.from_numpy(decoder_word_input_np).long())
            if USE_GPU:
                decoder_word_input = decoder_word_input.cuda()

        return result

    def save(self):
        torch.save(self.state_dict(), self.params['model_name'])
        print('model saved ...')

    def load(self):
        self.load_state_dict(torch.load(self.params['model_name']))
        print('model loaded ...')




    def train_bt(self, encoder_word_input, input_seq_len, decoder_word_input,decoder_word_output, decoder_mask, kld_coef):
        self.optimizer.zero_grad()

        d_output, kld_lss = self(encoder_word_input, input_seq_len, decoder_word_input)
        rec_lss = self.rec_loss(d_output, decoder_word_output, decoder_mask)

        # update
        # lss = kld_coef * kld_lss + rec_lss
        lss = rec_lss
        lss.backward()
        torch.nn.utils.clip_grad_norm(self.parameters(), self.params['max_grad_norm'])
        self.optimizer.step()

        return kld_lss.data[0], rec_lss.data[0]


if __name__ == '__main__':
    encoder_params = {
        'emb_size': 512,
        'hidden_size': 512,
        'n_layers': 2,
        'bidirectional': False,
    }

    decoder_params = {
        'emb_size': 512,
        'hidden_size': 512,
        'n_layers': 2,
        'bidirectional': False,
        'drop_out': 0.8,
    }

    params = {
        'n_epochs': 5,
        'lr': 0.0005,
        'batch_size': 64,
        'z_size': 16,
        'max_grad_norm': 5,
        'kl_lss_anneal': True,
        'top_k': 2,
        # 'use_gpu': True,
        'model_name': 'trained_CVAE.model',
    }

    corpus_loader_params = {
        'lf': 5, #低频词
        'keep_seq_lens': [5, 20],
    }
    corpus_loader = CorpusLoader(corpus_loader_params)
    params['vocab_size'] = corpus_loader.word_vocab_size

    model = CVAE(encoder_params, decoder_params, params)
    if USE_GPU:
        model = model.cuda()

    if model.have_saved_model:
        model.load()
    else:
        # train
        model.fit(corpus_loader)
        model.save()

    # evaluate
    result = model.sample_from_encoder(corpus_loader, 'hao are you .')
    print(result)










