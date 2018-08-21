from os import system
from tempfile import NamedTemporaryFile

import jieba
import spacy
import torch as T
import torchtext as tt

_use_cuda = T.cuda.is_available()
DEVICE = T.device('cuda' if _use_cuda else 'cpu')

spacy_en = spacy.load('en')
tokenize_en = lambda s: [tok.text for tok in spacy_en.tokenizer(s)]
tokenize_zh = lambda s: list(jieba.cut(s))

SRC = tt.data.Field(tokenize=tokenize_en, lower=True)
TRG = tt.data.Field(tokenize=tokenize_zh, init_token='<s>', eos_token='</s>', lower=True)

NEU_tri, NEU_val, NEU_tst = tt.datasets.TranslationDataset.splits(
    exts=('.en', '.cn'), fields=(SRC, TRG), path='../../Datasets/WMT17/neu2017',
    filter_pred=lambda ex: 10 <= len(ex.trg) <= 25
)

SRC.build_vocab(NEU_tri, max_size=30000, min_freq=5)  # max_size=30000 or min_freq=5
TRG.build_vocab(NEU_tri, max_size=30000, min_freq=5)

NEU_tri_iter, NEU_val_iter, NEU_tst_iter = tt.data.BucketIterator.splits(
    datasets=(NEU_tri, NEU_val, NEU_tst), batch_sizes=(64, 100, 100), repeat=False, device=DEVICE
)


def tensor_to_sentences(idxs):
    idxs = idxs.detach().cpu().numpy()
    batch_size = idxs.shape[1]
    seq_len = idxs.shape[0]
    outputs = []
    for batch in range(batch_size):
        one_batch = []
        for seq in range(seq_len):
            ss = TRG.vocab.itos[idxs[seq, batch]]
            if ss == TRG.init_token:
                continue
            if ss == TRG.eos_token:
                break
            one_batch.append(ss)
        one_batch = ' '.join(one_batch)
        outputs.append(one_batch)
    return outputs


def bleu(reference, output):
    with NamedTemporaryFile('w+t', delete=False) as rf, NamedTemporaryFile('w+t', delete=False) as of:
        rf.write('\n'.join(reference))
        of.write('\n'.join(output))
    bleu_f(rf.name, of.name)


def bleu_f(rf, of):
    system('./multi-bleu.perl {} < {}'.format(rf, of))


def test(model, data_iter, f_names=('test_1.txt', 'test_2.txt')):
    model.eval()
    with T.no_grad():
        trg, trn = [], []
        for _, batch in enumerate(data_iter):
            outputs = model(batch.src, batch.trg[:-1])
            outputs = outputs.max(2)[1]
            trg.extend(tensor_to_sentences(batch.trg[1:]))
            trn.extend(tensor_to_sentences(outputs))

    with open(f_names[0], 'w+t') as tf, open(f_names[1], 'w+t') as of:
        tf.writelines('\n'.join(trg))
        of.writelines('\n'.join(trn))
    bleu_f(tf.name, of.name)


def split_file_by_lines(f_name, *args):
    cur_line = 0
    i = 0
    with open(f_name) as f:
        lines = f.readlines()
        L = len(lines)
        assert sum(args) <= L
        for arg in args:
            with open(f_name + str(i), 'w') as wf:
                wf.writelines(lines[cur_line: cur_line + arg])
            cur_line += arg
            i += 1

        with open(f_name + str(i), 'w') as wf:
            wf.writelines(lines[cur_line: L])


if __name__ == '__main__':
    file = '../../Datasets/WMT17/neu2017/NEU_cn.txt'
    split_file_by_lines(file, 200000, 200000)
    file = '../../Datasets/WMT17/neu2017/NEU_en.txt'
    split_file_by_lines(file, 200000, 200000)

    # reference = 'Two person be in a small race car drive by a green hill .'
    # output = 'Two person be in race uniform in a street car .'
    # bleu([reference], [output])
