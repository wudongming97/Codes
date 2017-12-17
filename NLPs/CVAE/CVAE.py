import os
import math
import random
import torch
import numpy as np
from Utils import nll, USE_GPU

# helper function
def _rnn_cell_helper(str_rnn_cell):
    if str_rnn_cell.lower() == 'lstm':
        rnn_cell = torch.nn.LSTM
    elif str_rnn_cell.lower() == 'gru':
        rnn_cell = torch.nn.GRU
    else:
        raise ValueError("Unsupported RNN Cell: {0}".format(str_rnn_cell))

    return rnn_cell

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
        self.only_rec_loss = self.params.get('only_rec_loss')
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
        self.optimizer = torch.optim.Adam(self.parameters(),  lr=self.lr)
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

    def _forward_d(self, d_inputs, decoder_hidden):
        decoder_output, hidden = self.decoder(d_inputs, decoder_hidden)
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

    def forward(self, e_inputs, e_inputs_len, d_inputs):
        context = self.encoder(e_inputs, e_inputs_len)

        context = context[-self.encoder.num_directions:,:,:].view(self.batch_size, -1)
        mu, log_var = self.fc_mu(context), self.fc_logvar(context)
        kld_loss = self._kld_loss(mu, log_var)

        z = self._sample_z(mu, log_var)

        decoder_hidden = self._stats_from_z(z)
        output, _ = self._forward_d(d_inputs, decoder_hidden)

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
        if self.only_rec_loss:
            return 0
        elif self.kl_lss_anneal:
            # return math.exp(cur_epoch - self.params['n_epochs'])
            # return math.tanh(cur_epoch * 8 / self.params['n_epochs'] )
            return scalar_sigmoid(-15 + 20 * cur_epoch / self.n_epochs)
        else:
            return 1

    def _word_dropout_helper(self, corpus_loader, p, x):
        if np.random.binomial(1, p) == 1:
            return corpus_loader.word_to_idx[corpus_loader.unk_token]
        else:
            return x

    def fit(self, corpus_loader, display_step=15):
        print('begin fit ...\n')
        for e in range(self.n_epochs):
            for it, inputs in enumerate(corpus_loader.next_batch(self.batch_size, target='train')):
                sentences, encoder_word_input, input_seq_len, decoder_word_input, decoder_word_output, decoder_mask = inputs

                if self.word_dropout_p >= 0:
                    drop_word_f = lambda x: self._word_dropout_helper(corpus_loader, self.word_dropout_p, x)
                    decoder_word_input = [list(map(drop_word_f, s)) for s in decoder_word_input]

                encoder_word_input = torch.autograd.Variable(torch.Tensor(encoder_word_input).long())
                decoder_word_input = torch.autograd.Variable(torch.Tensor(decoder_word_input).long())
                decoder_word_output = torch.autograd.Variable(torch.Tensor(decoder_word_output).long())

                if USE_GPU:
                    encoder_word_input, decoder_word_input = encoder_word_input.cuda(), decoder_word_input.cuda()
                    decoder_word_output = decoder_word_output.cuda()

                kld_coef = self._kld_coef(e, it)
                kl_lss, rec_lss = self.train_bt(encoder_word_input, input_seq_len, decoder_word_input,
                                                 decoder_word_output, decoder_mask, kld_coef)

                if it % display_step == 0:
                    print(
                        "Epoch %d/%d | Batch %d/%d | train_loss: %.3f | rec_loss: %.3f | kl_loss: %.6f | kld_coef: %.6f | kld_coef*kl_loss: %.6f |" % (e+1, self.n_epochs, it, corpus_loader.num_line // self.batch_size, kl_lss*kld_coef + rec_lss, rec_lss, kl_lss, kld_coef, kld_coef*kl_lss))

                if it % (display_step * 20) == 0:
                    # 查看重构情况
                    print('\n------------ reconstruction --------------')
                    for i, s in enumerate(sentences):
                        if i > 4:
                            break
                        print('-----')
                        print("Input: {}\nOutput: {} ".format(s, self.sample_from_encoder(corpus_loader, s)))
                    # 查看随机生成情况
                    print('\n------------ sample_from_normal ----------')
                    for i in range(self.batch_size):
                        if  i > 4:
                            break
                        print('{}, {}'.format(i, self.sample_from_normal(corpus_loader)))
                    print('\n')


    def sample_from_normal(self, corpus_loader):
        z = torch.autograd.Variable(torch.randn(1, self.z_size))
        if USE_GPU: z = z.cuda()
        return self._sample_from_z(corpus_loader, z)

    def sample_from_encoder(self, corpus_loader, sentence):
        words = sentence.split()
        e_input = [corpus_loader.word_to_idx.get(w, corpus_loader.word_to_idx[corpus_loader.unk_token]) for w in words]
        e_input_len = [len(e_input)]
        e_input = torch.autograd.Variable(torch.from_numpy(np.atleast_2d(e_input)))
        if USE_GPU:
            e_input = e_input.cuda()

        context = self.encoder(e_input, e_input_len).view(1, -1)
        mu, log_var = self.fc_mu(context), self.fc_logvar(context)
        z = self._sample_z(mu, log_var)
        return self._sample_from_z(corpus_loader, z)

    def _sample_from_z(self, corpus_loader, z):
        # 一句一句的采样，所以batch_size都填1
        decoder_word_input_np = np.array(corpus_loader.go_input(1))
        decoder_word_input = torch.autograd.Variable(torch.from_numpy(decoder_word_input_np).long())
        if USE_GPU: decoder_word_input = decoder_word_input.cuda()

        result = ''
        decoder_hidden = self._stats_from_z(z)
        for i in range(corpus_loader.keep_seq_lens[1]):
            d_output, decoder_hidden = self._forward_d(decoder_word_input, decoder_hidden)

            # prediction = torch.nn.functional.softmax(d_output)
            # ix, word = corpus_loader.sample_word_from_distribution(prediction .data.cpu().numpy())

            d_output_np = d_output.data.squeeze().cpu().numpy()
            ixs, words = corpus_loader.top_k(d_output_np, self.top_k)
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
        torch.save(self.state_dict(), self.model_name)
        print('model saved ...')

    def load(self):
        self.load_state_dict(torch.load(self.model_name))
        print('model loaded ...')

    def train_bt(self, encoder_word_input, input_seq_len, decoder_word_input,decoder_word_output, decoder_mask, kld_coef):
        self.optimizer.zero_grad()

        d_output, kld_lss = self(encoder_word_input, input_seq_len, decoder_word_input)
        rec_lss = self._rec_loss(d_output, decoder_word_output, decoder_mask)

        # update
        lss = kld_coef * kld_lss + rec_lss
        # lss = rec_lss
        lss.backward()
        torch.nn.utils.clip_grad_norm(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return kld_lss.data[0], rec_lss.data[0]
