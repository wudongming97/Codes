from models.CVAE import CVAE, USE_GPU, TORCH_VERSION
from utils.CorpusLoader import CorpusLoader

# 解决输出报UnicodeEncodeError
# import sys, codecs
# if s# ys.stdout.encoding != 'UTF-8':
#     sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
# if sys.stderr.encoding != 'UTF-8':
#     sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


print('============ Env Info ===============')
print('pytorch version : {}\n'.format(TORCH_VERSION))
print('use gpu: {}'.format(str(USE_GPU)))

#  word_level_params
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
        'target': 1, # or 0
        'n_epochs': 30,
        'lr': 0.001,
        'batch_size': 64,
        'z_size': 16,
        'max_grad_norm': 5,
        'top_k': 2,
        'word_dropout_p': 0.6,
        'kl_lss_anneal': True,
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        'model_name': 'trained_word_CVAE.model',
        'only_rec_loss': False,  # for debug
    }

# ========================================================================
# ========================================================================
#  char_level_params
class char_level_params:
    encoder_params = {
        'rnn_cell_str': 'gru',
        'emb_size': 32,
        'hidden_size': 32,
        'n_layers': 1,
        'bidirectional': False,
    }

    decoder_params = {
        'rnn_cell_str': 'gru',
        'emb_size': 32,
        'hidden_size': 32,
        'n_layers': 1,
        'bidirectional': False,
        'input_dropout_p': 0.8,
    }

    params = {
        'target': 0, #0 for char 1 for word
        'n_epochs': 30,
        'lr': 0.001,
        'batch_size': 4,
        'z_size': 16,
        'max_grad_norm': 5,
        'top_k': 2,
        'word_dropout_p': 0.6,
        'kl_lss_anneal': True,
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        'model_name': 'trained_char_CVAE.model',
        'only_rec_loss': True,  # for debug
    }

corpus_loader_params = {
    'data_path_prefix': './data/CVAE/',
    'char_lf': 15,  # 低频词
    'word_lf': 15,
    'keep_words_lens': [5, 20],
    'keep_chars_lens': [5, 100],
    'shuffle': False,
    'global_seqs_sort': True,
    'train_fraction': 0.8,
}

if __name__ == '__main__':

    # level = char_level_params()
    level = word_level_params()

    corpus_loader = CorpusLoader(corpus_loader_params)
    level.params['vocab_size'] = corpus_loader.vocab_sizes[level.params['target']]

    model = CVAE(level.encoder_params, level.decoder_params, level.params)
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
