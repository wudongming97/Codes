import os
import datetime
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from Corpus import SOS_token, EOS_token
from Utils import nll


class Encoder(nn.Module):
    def __init__(self, input_sz, emb_dim, hid_sz, n_layers, z_dim):
        super(Encoder, self).__init__()
        self.input_sz = input_sz
        self.hid_sz = hid_sz
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.z_dim = z_dim
        self.build_model()

    def build_model(self):
        self.embedding = nn.Embedding(self.input_sz, self.emb_dim)
        self.rnn = nn.LSTM(self.emb_dim, self.hid_sz, self.n_layers, batch_first=True)
        self.fc_mu = nn.Linear(self.hid_sz//2, self.z_dim)
        self.fc_var = nn.Linear(self.hid_sz - self.hid_sz//2, self.z_dim)

    def init_hidden(self, batch_sz):
        result = (Variable(torch.zeros(self.n_layers, batch_sz, self.hid_sz)),
                  Variable(torch.zeros(self.n_layers, batch_sz, self.hid_sz)))
        return result

    def forward(self, inputs, inputs_len, hidden=None):
        batch_sz = inputs.size(0)
        if hidden is None:
            hidden = self.init_hidden(batch_sz)

        embedded = self.embedding(inputs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, inputs_len, batch_first=True)
        _, (hidden, _) = self.rnn(packed, hidden)
        # output, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        last_hidden = hidden[self.n_layers-1:, :, :].view(batch_sz, -1)
        mu = self.fc_mu(last_hidden[:, 0: self.hid_sz//2])
        log_var = self.fc_var(last_hidden[:, self.hid_sz//2: self.hid_sz])
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, vocab_sz, emb_dim, hid_sz, n_layers, z_dim):
        super(Decoder, self).__init__()
        self.vocab_sz = vocab_sz
        self.emb_dim = emb_dim
        self.hid_sz = hid_sz
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.build_model()

    def build_model(self):
        self.embedding = nn.Embedding(self.vocab_sz, self.emb_dim)
        self.rnn = nn.LSTM(self.emb_dim+self.z_dim, self.hid_sz, self.n_layers, batch_first=True)
        self.fc_out = nn.Linear(self.hid_sz, self.vocab_sz)

    def sample_z(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        z = Variable(torch.randn(mu.size()))
        z = z * mu + std
        return z

    def init_hidden(self, batch_sz):
        result = (Variable(torch.zeros(self.n_layers, batch_sz, self.hid_sz)),
                  Variable(torch.zeros(self.n_layers, batch_sz, self.hid_sz)))
        return result

    def forward(self, inputs, mu, log_var, hidden=None):
        batch_sz = inputs.size(0)
        if hidden is None:
            hidden = self.init_hidden(batch_sz)
        embedded = self.embedding(inputs)
        z = self.sample_z(mu, log_var)
        rnn_input = torch.cat([z.view(batch_sz, -1, self.z_dim).expand(batch_sz, embedded.size(1), self.z_dim), embedded], 2)
        output, hidden = self.rnn(rnn_input, hidden)
        output = self.fc_out(output.contiguous().view(-1, self.hid_sz))
        return output, hidden


class CVAE_LM:
    def __init__(self, corpus_loader, model_args, hyper_params):
        self.corpus_loader = corpus_loader
        self.model_args = model_args
        self.hyper_params = hyper_params
        self.save_dir = './saved_models/'
        self.build_model()

    def build_model(self):
        self.encoder = Encoder(self.corpus_loader.vocab_sz, self.model_args['emb_dim'], self.model_args['hid_sz'], self.model_args['n_layers'], self.model_args['z_dim'])
        self.decoder = Decoder(self.corpus_loader.vocab_sz, self.model_args['emb_dim'], self.model_args['hid_sz'], self.model_args['n_layers'], self.model_args['z_dim'])
        self.encoder_optimizer = optim.Adam(self.encoder.parameters())
        self.decoder_optimizer = optim.Adam(self.decoder.parameters())
        # self.criterion = torch.nn.CrossEntropyLoss()

    def train(self, inputs, inputs_len, inputs_mask):
        batch_sz = inputs.size(0)

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        mu, log_var = self.encoder(inputs, inputs_len)
        decoder_inputs = self.process_decoder_input(inputs)
        decoder_out, decoder_hidden = self.decoder(decoder_inputs, mu, log_var)

        # kl_loss
        kld_element = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
        kld_loss = torch.sum(kld_element).mul_(-0.5)

        # recon loss
        losses = nll(nn.functional.log_softmax(decoder_out), inputs.view(-1, 1))
        inputs_mask = torch.autograd.Variable(torch.FloatTensor(inputs_mask)).view(-1)
        rec_loss = torch.mul(losses, inputs_mask).sum() / batch_sz

        # loss = kl_loss + recon_loss
        loss = kld_loss + rec_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm(self.encoder.parameters(), self.hyper_params['max_grad_norm'])
        torch.nn.utils.clip_grad_norm(self.decoder.parameters(), self.hyper_params['max_grad_norm'])
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.data[0], kld_loss.data[0], rec_loss.data[0]

    def fit(self, display_step=10):
        print('begin fit...')
        n_epoch = self.hyper_params['epoch']
        batch_sz = self.hyper_params['batch_sz']
        for epoch in range(1, n_epoch+1):
            for it, (padded_bt, bt_lens, bt_masks) in enumerate(
                    self.corpus_loader.next_batch(batch_sz)):
                batch_inputs = Variable(torch.from_numpy(padded_bt))
                lss, kl_lss, rec_lss = self.train(batch_inputs, bt_lens, bt_masks)

                if it % display_step == 0:
                    print("Epoch %d/%d | Batch %d/%d | train_loss: %.3f | kl_loss: %.3f | rec_loss: %.3f |" %
                          (epoch, n_epoch, it, self.corpus_loader.s_len // batch_sz, lss, kl_lss, rec_lss))
            # generated samples
            indices = self.generate(2)
            word_lists = self.corpus_loader.indices2sentences(indices.numpy().tolist())
            sentences = [reduce(lambda x, y: x + ' ' + y, words) for words in word_lists]
            print(sentences)

    def generate(self, batch_sz=5, maxLen=10):
        z_dim = self.model_args['z_dim']
        res_indices = torch.LongTensor([[SOS_token]]).expand(batch_sz, 1)
        go_inputs = Variable(torch.LongTensor([[SOS_token]])).expand((batch_sz, 1))
        mu = Variable(torch.zeros(batch_sz, z_dim))
        log_var = Variable(torch.ones(batch_sz, z_dim))
        for i in range(maxLen):
            decoder_output, decoder_hidden = self.decoder(go_inputs, mu, log_var)
            decoder_output = torch.nn.functional.log_softmax(decoder_output)
            _, topi = decoder_output.data.topk(1)
            res_indices = torch.cat((res_indices, topi), 1)
            go_inputs = Variable(topi)
        return res_indices

    def save(self):
        print('save model...')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        model_name = self.model_args['name']
        torch.save(self.encoder.state_dict(), os.path.join(self.save_dir, model_name+'_E.pkl'))
        torch.save(self.decoder.state_dict(), os.path.join(self.save_dir, model_name+'_D.pkl'))
        save_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        torch.save(self.encoder.state_dict(), os.path.join(self.save_dir, model_name + '_E.pkl' + save_time))
        torch.save(self.decoder.state_dict(), os.path.join(self.save_dir, model_name + '_D.pkl' + save_time))

    def load(self, encoder_file=None, decoder_file=None):
        print('load model...')
        model_name = self.model_args['name']
        if encoder_file is None:
            self.encoder.load_state_dict(torch.load(os.path.join(self.save_dir, model_name + '_E.pkl')))
        else:
            self.encoder.load_state_dict(torch.load(encoder_file))
        if decoder_file is None:
            self.decoder.load_state_dict(torch.load(os.path.join(self.save_dir, model_name + '_D.pkl')))
        else:
            self.decoder.load_state_dict(torch.load(decoder_file))

    @staticmethod
    def process_decoder_input(target):
        target = target[:, :-1]
        go = torch.autograd.Variable((torch.zeros(target.size(0), 1) + SOS_token).long())
        decoder_input = torch.cat((go, target), 1)
        return decoder_input


# ------------------------------------------------------------
if __name__ == '__main__':
    # test model
    from Corpus import Corpus
    from CorpusLoader import CorpusLoader
    from functools import reduce

    small_batch_sz = 3
    small_hid_sz = 9
    small_emb_sz = 10
    small_z_dim = 8
    small_n_layers = 2

    corpus = Corpus('nlp_dataset/Corpus.test.txt').process()
    corpus_loader = CorpusLoader(corpus.sentences, corpus.word2idx, corpus.idx2word)

    '''
    # test encoder and decoder
    encoder = Encoder(corpus.n_words, small_emb_sz, small_hid_sz, small_n_layers, small_z_dim)
    decoder = Decoder(corpus.n_words, small_emb_sz, small_hid_sz, small_n_layers, small_z_dim)
    for it, (padded_bt, bt_lens, bt_masks) in enumerate(corpus_loader.next_batch(small_batch_sz)):
        print(bt_lens)
        if it == 20:
            print(it)
        mu, log_var = encoder(Variable(torch.from_numpy(padded_bt)), bt_lens)
        print("iter: {}".format(it))
        print(mu.size())
        print(log_var.size())

        de_out, de_hid = decoder(Variable(torch.from_numpy(padded_bt)), mu, log_var)
        print(de_out.size())
    '''

    # test cvae_lm
    model_args = {
        'name': 'CVAE_LM',
        'emb_dim': small_emb_sz,
        'hid_sz': small_hid_sz,
        'n_layers': small_n_layers,
        'z_dim': small_z_dim,
        'max_grad_norm': 5
    }

    hyper_params = {
        'epoch': 6,
        'lr': 0.001,
        'batch_sz': 10
    }
    cvae_lm = CVAE_LM(corpus_loader, model_args, hyper_params)
    cvae_lm.fit()
    cvae_lm.save()
    cvae_lm.load()
    indices = cvae_lm.generate(2)
    word_lists = corpus_loader.indices2sentences(indices.numpy().tolist())
    print(indices)
    print(word_lists)
    sentences = [reduce(lambda x, y :x + ' ' + y, words) for words in word_lists]
    print(sentences)



