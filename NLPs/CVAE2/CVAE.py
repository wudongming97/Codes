import os
import random
import torch
from Utils import nll
import CorpusLoader

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
        self.fc_out = torch.nn.Linear(self.num_directions * self.params['hidden_size'], self.params['vocab_size'])

    def forward(self, inputs, hidden):
        embedded = self.embedding(inputs)
        embedded_dropout = self.drop_out_layer(embedded)
        output, _ = self.rnn(embedded_dropout, hidden)
        output = self.fc_out(output.contiguous().view(-1, self.params['hidden_size'] * self.num_directions))
        return output


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

    def sample_z(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.autograd.Variable(torch.randn(mu.size()))
        if USE_GPU:
            eps = eps.cuda()
        ret = mu + std * eps
        return ret

    @staticmethod
    def kld_loss(mu, log_var):
        # kl_loss
        kld_element = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
        kld_loss = torch.sum(kld_element, 1).mul_(-0.5).mean().squeeze()
        return kld_loss 

    def forward(self, e_inputs, e_inputs_len, d_inputs):
        encoder_hidden = self.encoder.init_hidden(self.batch_size)
        context = self.encoder(e_inputs, e_inputs_len, encoder_hidden)

        context = context.view(self.batch_size, -1)
        mu, logvar = self.fc_mu(context), self.fc_logvar(context)
        z = self.sample_z(mu, logvar)

        kld_loss = self.kld_loss(mu, logvar)

        decoder_hidden = self.fc_h(z)
        decoder_hidden = decoder_hidden.view(self.batch_size,
                                             self.decoder_params['n_layers'] * self.decoder.num_directions, -1)

        decoder_output = self.decoder(d_inputs, decoder_hidden)
        decoder_output = decoder_output.view(self.batch_size, -1)
        output = self.fc_out(decoder_output)

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

    def fit(self, corpus_loader, display_step=10):
        print('begin fit ...')
        n_epochs = self.params['n_epochs']
        for e in range(n_epochs):
            for it, (
            encoder_word_input, input_seq_len, decoder_word_input, decoder_word_output, decoder_mask) in enumerate(
                corpus_loader.next_batch(self.batch_size ,target_str='train')):
                encoder_word_input = torch.autograd.Variable(torch.from_numpy(encoder_word_input))
                decoder_word_input = torch.autograd.Variable(torch.from_numpy(decoder_word_input))
                decoder_word_output = torch.autograd.Variable(torch.from_numpy(decoder_word_output))
                if USE_GPU:
                    encoder_word_input, decoder_word_input, decoder_word_output = encoder_word_input.cuda(), decoder_word_input.cuda(), decoder_word_output.cuda()

                kl_lss, rec_lss = self.train(encoder_word_input, input_seq_len, decoder_word_input,
                                                 decoder_word_output, decoder_mask)

                if it % display_step == 0:
                    print("Epoch %d/%d | Batch %d/%d | train_loss: %.3f | kl_loss: %.3f | rec_loss: %.3f |" %
                          (e, n_epochs, it, corpus_loader.num_lines[0] // self.batch_size, kl_lss+rec_lss, kl_lss, rec_lss))


    def train(self, encoder_word_input, input_seq_len, decoder_word_input,decoder_word_output, decoder_mask):
        self.optimizer.zero_grad()

        d_output, kl_lss = self(encoder_word_input, input_seq_len, decoder_word_input)
        rec_lss = self.rec_loss(d_output, decoder_word_output, decoder_mask)

        # update
        lss = kl_lss + rec_lss
        lss.backward()
        torch.nn.utils.clip_grad_norm(self.parameters(), self.params['max_grad_norm'])
        self.optimizer.step()

        return kl_lss.data[0], rec_lss.data[0]


if __name__ == '__main__':
    encoder_params = {
        'emb_size': 256,
        'hidden_size': 256,
        'n_layers': 2,
        'bidirectional': True,
    }

    decoder_params = {
        'emb_size': 256,
        'hidden_size': 256,
        'n_layers': 2,
        'bidirectional': True,
        'drop_out': 0.8,
    }

    params = {
        'n_epochs': 10,
        'lr': 0.0005,
        'batch_size': 100,
        'vocab_size': 256,
        'z_size': 16,
    }

    corpus_loader = CorpusLoader()
    params['vocab_size'] = corpus_loader.word_vocab_size

    model = CVAE(encoder_params, decoder_params, params)

    # train
    model.fit(corpus_loader)








