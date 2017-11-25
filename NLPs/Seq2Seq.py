# 参考： https://github.com/zhedongzheng/finch/blob/master/nlp-models
import os
import torch
from Utils import nll
from Corpus import SOS_token, EOS_token

USE_GPU = torch.cuda.is_available()
if USE_GPU:
    torch.cuda.set_device(1)


class Encoder(torch.nn.Module):
    def __init__(self, input_size, encoder_embedding_dim, hidden_size, n_layers):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoder_embedding_dim = encoder_embedding_dim
        self.n_layers = n_layers
        self.build_model()
    # end constructor

    def build_model(self):
        self.embedding = torch.nn.Embedding(self.input_size, self.encoder_embedding_dim)
        self.lstm = torch.nn.LSTM(self.encoder_embedding_dim, self.hidden_size,
                                  batch_first=True, num_layers=self.n_layers)

    def forward(self, inputs, X_lens, hidden=None):
        batch_sz = inputs.size(0)
        if hidden is None:
            hidden = self.init_hidden(batch_sz)
        embedded = self.embedding(inputs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, X_lens, batch_first=True)
        rnn_out, hidden = self.lstm(packed, hidden)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        return output, hidden

    def init_hidden(self, batch_sz):
        h0 = torch.autograd.Variable(torch.zeros(self.n_layers, batch_sz, self.hidden_size))
        c0 = torch.autograd.Variable(torch.zeros(self.n_layers, batch_sz, self.hidden_size))
        if USE_GPU:
            h0, c0 = h0.cuda(), c0.cuda()
        return h0, c0

class Decoder(torch.nn.Module):
    def __init__(self, output_size, decoder_embedding_dim, hidden_size, n_layers):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.decoder_embedding_dim = decoder_embedding_dim
        self.n_layers = n_layers
        self.build_model()

    def build_model(self):
        self.embedding = torch.nn.Embedding(self.output_size, self.decoder_embedding_dim)
        self.lstm = torch.nn.LSTM(self.decoder_embedding_dim, self.hidden_size,
                                  batch_first=True, num_layers=self.n_layers)
        self.out = torch.nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputs, hidden):
        embedded = self.embedding(inputs)
        output, hidden = self.lstm(embedded, hidden)
        output = self.out(output.contiguous().view(-1, self.hidden_size))
        return output, hidden

class Seq2Seq:
    def __init__(self, corpus_loader, model_args, hyper_params):
        self.corpus_loader = corpus_loader
        self.model_args = model_args
        self.hyper_params = hyper_params
        self.save_dir = './saved_models/'
        self.build_model()

    def build_model(self):
        self.encoder = Encoder(self.corpus_loader.X_vocab_sz, self.model_args['encoder_embedding_dim'], self.model_args['hidden_size'], self.model_args['n_layers'])
        self.decoder = Decoder(self.corpus_loader.Y_vocab_sz, self.model_args['decoder_embedding_dim'], self.model_args['hidden_size'], self.model_args['n_layers'])
        if USE_GPU:
            self.encoder, self.decoder = self.encoder.cuda(), self.decoder.cuda()
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters())
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters())
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self, X_inputs, X_lens, Y_inputs, Y_masks):
        batch_size = X_inputs.size(0)
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        encoder_output, encoder_hidden = self.encoder(X_inputs, X_lens)

        decoder_hidden = encoder_hidden

        decoder_input = self.process_decoder_input(Y_inputs)
        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

        losses = nll(torch.nn.functional.log_softmax(decoder_output), Y_inputs.view(-1, 1))
        Y_masks = torch.autograd.Variable(torch.FloatTensor(Y_masks)).view(-1)
        if USE_GPU:
            Y_masks = Y_masks.cuda()
        loss = torch.mul(losses, Y_masks).sum() / batch_size
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.encoder.parameters(), self.hyper_params['max_grad_norm'])
        torch.nn.utils.clip_grad_norm(self.decoder.parameters(), self.hyper_params['max_grad_norm'])
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.data[0]

    def fit(self, display_step=10, corpus_loader=None):
        print('begin fit...')
        n_epoch = self.hyper_params['epoch']
        batch_sz = self.hyper_params['batch_sz']
        for epoch in range(1, n_epoch+1):
            for local_step, ((X_input, X_lens, _), (Y_input, _, Y_masks)) in enumerate(self.corpus_loader.next_batch(batch_sz)):
                X_input = torch.autograd.Variable(torch.from_numpy(X_input))
                Y_input = torch.autograd.Variable(torch.from_numpy(Y_input))
                if USE_GPU:
                    X_input, Y_input = X_input.cuda(), Y_input.cuda()

                loss = self.train(X_input, X_lens, Y_input, Y_masks)

                if local_step % display_step == 0:
                    print("Epoch %d/%d | Batch %d/%d | train_loss: %.3f |" %
                          (epoch, n_epoch, local_step, self.corpus_loader.s_len // batch_sz, loss))

    def predict(self, X_input, X_lens, maxLen=None):
        batch_size = X_input.shape[0]
        if maxLen is None:
            maxLen = 2 * X_input.shape[1]

        encoder_inputs = torch.autograd.Variable(torch.from_numpy(X_input))
        if USE_GPU:
            encoder_inputs = encoder_inputs.cuda()

        encoder_output, encoder_hidden = self.encoder(encoder_inputs, X_lens)
        decoder_hidden = encoder_hidden

        output_indices = torch.LongTensor([[SOS_token]]).expand(batch_size, 1)
        if USE_GPU:
            output_indices = torch.cuda.LongTensor([[SOS_token]]).expand(batch_size, 1)

        decoder_inputs = torch.autograd.Variable(torch.LongTensor([[SOS_token]])).expand((batch_size, 1))
        if USE_GPU:
            decoder_inputs = decoder_inputs.cuda()

        for i in range(maxLen):
            decoder_output, decoder_hidden = self.decoder(decoder_inputs, decoder_hidden)
            decoder_output = torch.nn.functional.log_softmax(decoder_output)
            _, topi = decoder_output.data.topk(1)
            output_indices = torch.cat((output_indices, topi), 1)
            decoder_inputs = torch.autograd.Variable(topi)
        return output_indices.cpu().numpy()

    def process_decoder_input(self, target):
        target = target[:, :-1]
        go = torch.autograd.Variable((torch.zeros(target.size(0), 1) + SOS_token).long())
        if USE_GPU:
            go = go.cuda()
        decoder_input = torch.cat((go, target), 1)
        return decoder_input

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


if __name__ == '__main__':
    model_args = {
        'name': 'seq2seq',
        'encoder_embedding_dim': 10,
        'decoder_embedding_dim': 10,
        'hidden_size': 12,
        'n_layers': 2,
    }

    hyper_params = {
        'epoch': 1,
        'lr': 0.001,
        'batch_sz': 10,
        'max_grad_norm': 5
    }

    test_data_source = '../datasets/en_vi_nlp/tst2012.en'
    test_data_target = '../datasets/en_vi_nlp/tst2012.vi'

    from Corpus import ParallelCorpus
    from CorpusLoader import ParallelCorpusLoader
    from torch.autograd import Variable

    corpus = ParallelCorpus(test_data_source, test_data_target).process()
    corpusLoader = ParallelCorpusLoader(corpus.pair_sentences, corpus.X_word2idx, corpus.X_idx2word, corpus.Y_word2idx, corpus.Y_idx2word)
    seq2seq = Seq2Seq(corpusLoader, model_args, hyper_params)
    seq2seq.fit()
    source_sentences = ['how are you!', 'where are you?', 'are you ok?', 'I like you.']
    one_sentences = ['hello, are you ok!']
    X_input, X_lens, _ = corpusLoader.sentences_2_inputs(source_sentences)
    ret = seq2seq.predict(X_input, X_lens)
    sentence_list = corpusLoader.to_outputs(ret.tolist())
    print(sentence_list)
    X_input, X_lens, _ = corpusLoader.sentences_2_inputs(one_sentences)
    ret = seq2seq.predict(X_input, X_lens)
    sentence_list = corpusLoader.to_outputs(ret.tolist())
    print(sentence_list)









