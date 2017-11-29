import os
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from Corpus import SOS_token, EOS_token
from Utils import nll, print_sentences

USE_GPU = torch.cuda.is_available()
if USE_GPU:
    torch.cuda.set_device(1)

class Encoder(nn.Module):
    def __init__(self, input_sz, emb_dim, hid_sz, n_layers, z_dim, bidirectional=False):
        super(Encoder, self).__init__()
        self.input_sz = input_sz
        self.hid_sz = hid_sz
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.z_dim = z_dim
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.build_model()

    def build_model(self):
        self.embedding = nn.Embedding(self.input_sz, self.emb_dim)
        self.rnn = nn.LSTM(self.emb_dim, self.hid_sz, self.n_layers, batch_first=True, bidirectional=self.bidirectional)
        self.fc_mu = nn.Linear(self.num_directions * self.hid_sz, self.z_dim)
        self.fc_var = nn.Linear(self.num_directions * self.hid_sz, self.z_dim)

    def init_hidden(self, batch_sz):
        h0 = Variable(torch.zeros(self.n_layers * self.num_directions, batch_sz, self.hid_sz))
        c0 = Variable(torch.zeros(self.n_layers * self.num_directions, batch_sz, self.hid_sz))
        if USE_GPU:
            h0, c0 = h0.cuda(), c0.cuda()
        return h0, c0

    def forward(self, inputs, inputs_len, hidden=None):
        batch_sz = inputs.size(0)
        if hidden is None:
            hidden = self.init_hidden(batch_sz)

        embedded = self.embedding(inputs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, inputs_len, batch_first=True)
        _, (hidden, _) = self.rnn(packed, hidden)
        # output, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        last_hidden = hidden[(self.n_layers-1) * self.num_directions:, :, :].view(batch_sz, -1)
        mu = self.fc_mu(last_hidden)
        log_var = self.fc_var(last_hidden)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, vocab_sz, emb_dim, hid_sz, n_layers, z_dim, bidirectional=False):
        super(Decoder, self).__init__()
        self.vocab_sz = vocab_sz
        self.emb_dim = emb_dim
        self.hid_sz = hid_sz
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.build_model()

    def build_model(self):
        self.embedding = nn.Embedding(self.vocab_sz, self.emb_dim)
        self.rnn = nn.LSTM(self.emb_dim+self.z_dim, self.hid_sz, self.n_layers, batch_first=True, bidirectional=self.bidirectional)
        self.fc_out = nn.Linear(self.num_directions * self.hid_sz, self.vocab_sz)

    def sample_z(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = Variable(torch.randn(mu.size()))
        if USE_GPU:
            eps = eps.cuda()
        ret = mu + std * eps
        return ret

    def init_hidden(self, batch_sz):
        h0 = Variable(torch.zeros(self.n_layers * self.num_directions, batch_sz, self.hid_sz))
        c0 = Variable(torch.zeros(self.n_layers * self.num_directions, batch_sz, self.hid_sz))
        if USE_GPU:
            h0, c0 = h0.cuda(), c0.cuda()
        return h0, c0

    def forward(self, inputs, mu, log_var, hidden=None):
        batch_sz = inputs.size(0)
        if hidden is None:
            hidden = self.init_hidden(batch_sz)
        embedded = self.embedding(inputs)
        z = self.sample_z(mu, log_var)
        rnn_input = torch.cat([z.view(batch_sz, -1, self.z_dim).expand(batch_sz, embedded.size(1), self.z_dim), embedded], 2)
        output, hidden = self.rnn(rnn_input, hidden)
        output = self.fc_out(output.contiguous().view(-1, self.hid_sz * self.num_directions))
        return output, hidden

class CVAE_LM:
    def __init__(self, corpus_loader, model_args, hyper_params):
        self.corpus_loader = corpus_loader
        self.model_args = model_args
        self.hyper_params = hyper_params
        self.save_dir = './saved_models/'
        self.alpha = 0      # kl-loss annealing trick
        self.build_model()

    def build_model(self):
        en_bid = self.model_args.get('encoder_bid', None)
        de_bid = self.model_args.get('decoder_bid', None)
        self.encoder = Encoder(self.corpus_loader.vocab_sz, self.model_args['emb_dim'], self.model_args['hid_sz'], self.model_args['n_layers'], self.model_args['z_dim'], en_bid)
        self.decoder = Decoder(self.corpus_loader.vocab_sz, self.model_args['emb_dim'], self.model_args['hid_sz'], self.model_args['n_layers'], self.model_args['z_dim'], de_bid)
        if USE_GPU:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.hyper_params['lr'])
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=self.hyper_params['lr'])
        # self.criterion = torch.nn.CrossEntropyLoss()

    @staticmethod
    def kld_loss(mu, log_var):
        # kl_loss
        kld_element = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
        kld_loss = torch.sum(kld_element).mul_(-0.5)
        return kld_loss

    @staticmethod
    def rec_loss(decoder_out, target_inputs, target_mask):
        # recon loss
        batch_sz = target_inputs.size(0)
        losses = nll(nn.functional.log_softmax(decoder_out), target_inputs.view(-1, 1))
        target_mask = torch.autograd.Variable(torch.FloatTensor(target_mask)).view(-1)
        if USE_GPU:
            target_mask = target_mask.cuda()
        loss = torch.mul(losses, target_mask).sum()
        return loss

    def kl_loss_annealing_policy(self, n_epoch, cur_epoch):
        #line policy
        if n_epoch <= 1:
            self.alpha = 1
        else:
            self.alpha = torch.linspace(0, 1, n_epoch)[cur_epoch]

    def train(self, source_inputs, source_len, target_inputs, target_mask):
        batch_sz = source_inputs.size(0)
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        mu, log_var = self.encoder(source_inputs, source_len)
        decoder_inputs = self.process_decoder_input(target_inputs)
        decoder_out, decoder_hidden = self.decoder(decoder_inputs, mu, log_var)

        # kl_loss
        kld_loss = self.kld_loss(mu, log_var)

        # recon loss
        rec_loss = self.rec_loss(decoder_out, target_inputs, target_mask)

        # loss = (kl_loss + recon_loss) / batch_sz
        loss = (rec_loss + kld_loss * self.alpha) / batch_sz

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
            for it, ((source_padded, source_len, _), (target_inputs, _, target_mask)) in enumerate(
                    self.corpus_loader.next_batch(batch_sz, target=True)):
                source_inputs = Variable(torch.from_numpy(source_padded))
                target_inputs = Variable(torch.from_numpy(target_inputs))
                if USE_GPU:
                    source_inputs, target_inputs = source_inputs.cuda(), target_inputs.cuda()
                lss, kl_lss, rec_lss = self.train(source_inputs, source_len, target_inputs, target_mask)

                if it % display_step == 0:
                    print("Epoch %d/%d | Batch %d/%d | train_loss: %.3f | kl_loss: %.3f | rec_loss: %.3f |" %
                          (epoch, n_epoch, it, self.corpus_loader.s_len // batch_sz, lss, kl_lss, rec_lss))

            # kl-loss policy
            self.kl_loss_annealing_policy(n_epoch, epoch-1)
            # test
            self.epoch_test(2)

    def epoch_test(self, batch_sz=5):
        # random test
        print('======== test with z ~ N(0, 1)=======')
        g_size = [batch_sz, model_args['z_dim']]
        mu = torch.zeros(g_size)
        log_var = torch.ones(g_size)
        z_sentences = self.generate_by_z(mu, log_var)
        print_sentences(z_sentences)

        # test use train data
        print('======== test use train data ========')
        maxlen = batch_sz if corpus_loader.s_len >= batch_sz else corpus_loader.s_len
        sampled_sentences = random.sample(corpus_loader.sentences, maxlen)
        g_sentences, sorted_source_sentences = self.generate_by_encoder(sampled_sentences)
        print('train sentences ==>')
        print_sentences(sorted_source_sentences)
        print('generated sentences ==>')
        print_sentences(g_sentences)

    def generate_by_encoder(self, sentence_list, maxLen=10):
        (X_input, X_lens, _), sorted_sentences = self.corpus_loader.sentences_2_inputs(sentence_list)
        X_input = Variable(torch.from_numpy(X_input))
        if USE_GPU:
            X_input = X_input.cuda()
        mu, log_var = self.encoder(X_input, X_lens)

        return self.generate_by_z(mu.data, log_var.data, maxLen), sorted_sentences

    def generate_by_z(self, mu, log_var, maxLen=10):
        batch_sz = mu.size(0)
        z_dim = self.model_args['z_dim']
        res_indices = torch.LongTensor([[SOS_token]]).expand(batch_sz, 1)
        go_inputs = Variable(torch.LongTensor([[SOS_token]])).expand((batch_sz, 1))
        mu = Variable(mu)
        log_var = Variable(log_var)

        if USE_GPU:
            go_inputs, mu, log_var = go_inputs.cuda(), mu.cuda(), log_var.cuda()
            res_indices = torch.cuda.LongTensor([[SOS_token]]).expand(batch_sz, 1)

        for i in range(maxLen):
            decoder_output, decoder_hidden = self.decoder(go_inputs, mu, log_var)
            decoder_output = torch.nn.functional.log_softmax(decoder_output)
            _, topi = decoder_output.data.topk(1)
            res_indices = torch.cat((res_indices, topi), 1)
            go_inputs = Variable(topi)
        sentences = self.corpus_loader.to_outputs(res_indices.cpu().numpy().tolist())
        return sentences

    def save(self):
        print('save model...')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        model_name = self.model_args['name']
        torch.save(self.encoder.state_dict(), os.path.join(self.save_dir, model_name+'_E.pkl'))
        torch.save(self.decoder.state_dict(), os.path.join(self.save_dir, model_name+'_D.pkl'))

    def load(self):
        print('load model...')
        model_name = self.model_args['name']
        self.encoder.load_state_dict(torch.load(os.path.join(self.save_dir, model_name + '_E.pkl')))
        self.decoder.load_state_dict(torch.load(os.path.join(self.save_dir, model_name + '_D.pkl')))

    @staticmethod
    def process_decoder_input(target):
        target = target[:, :-1]
        go = torch.autograd.Variable((torch.zeros(target.size(0), 1) + SOS_token).long())
        if USE_GPU:
            go = go.cuda()
        decoder_input = torch.cat((go, target), 1)
        return decoder_input


# ------------------------------------------------------------
if __name__ == '__main__':
    # test model
    from Corpus import Corpus
    from CorpusLoader import CorpusLoader
    from functools import reduce

    debug_data_path = '../datasets/en_vi_nlp/debug.en'

    corpus = Corpus(debug_data_path).process()
    corpus_loader = CorpusLoader(corpus.sentences, corpus.word2idx, corpus.idx2word)

    # test cvae_lm
    model_args = {
        'name': 'CVAE_LM_debug',
        'encoder_bid': True,
        'decoder_bid': False,
        'emb_dim': 64,
        'hid_sz': 64,
        'n_layers': 1,
        'z_dim': 8
    }

    hyper_params = {
        'epoch': 28,
        'lr': 0.001,
        'batch_sz': 2,
        'max_grad_norm': 5
    }
    cvae_lm = CVAE_LM(corpus_loader, model_args, hyper_params)
    cvae_lm.fit()
    cvae_lm.save()
    cvae_lm.load()
    g_size = [1, model_args['z_dim']]
    mu = torch.zeros(g_size)
    log_var = torch.ones(g_size)
    sentences = cvae_lm.generate_by_z(mu, log_var)
    print('hahaha')
    print(corpus_loader.sentences)
    sentences, sorted_sentences = cvae_lm.generate_by_encoder([corpus_loader.sentences[0]], 10)
    print(sorted_sentences)
    print(sentences)
    sentences, sorted_sentences = cvae_lm.generate_by_encoder([corpus_loader.sentences[1]], 10)
    print(sorted_sentences)
    print(sentences)


