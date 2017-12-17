from CorpusLoader import CorpusLoader
from CVAE import CVAE
from Utils import USE_GPU, TORCH_VERSION

# 解决输出报UnicodeEncodeError
# import sys, codecs
# if s# ys.stdout.encoding != 'UTF-8':
#     sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
# if sys.stderr.encoding != 'UTF-8':
#     sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

print('============ Env Info ===============')
print('pytorch version : {}\n'.format(TORCH_VERSION))
print('use gpu: {}'.format(str(USE_GPU)))

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
    'top_k': 5,
    'word_dropout_p': 0.6,
    'kl_lss_anneal': True,
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    'model_name': 'trained_CVAE.model',
    'only_rec_loss': False,  # for debug
}

corpus_loader_params = {
    'lf': 15, #低频词
    'keep_seq_lens': [5, 20],
    'shuffle': False,
    'global_seqs_sort': True,
    'train_fraction': 0.8,
}

if __name__ == '__main__':

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

    # 随机生成1000个句子
    for i in range(1000):
        print('{}, {}'.format(i, model.sample_from_normal(corpus_loader)))