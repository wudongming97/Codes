import jieba
import spacy
import torchtext as tt

from utils import DEVICE

SOS_TOK = '<s>'
EOS_TOK = '</s>'
PAD_TOK = '<pad>'

spacy_en = spacy.load('en')
tokenize_en = lambda s: [tok.text for tok in spacy_en.tokenizer(s)]
tokenize_zh = lambda s: list(jieba.cut(s))

SRC = tt.data.Field(tokenize=tokenize_en, pad_token=PAD_TOK, lower=True)
TRG = tt.data.Field(tokenize=tokenize_zh, init_token=SOS_TOK, eos_token=EOS_TOK, pad_token=PAD_TOK, lower=True)

NEU_tri, NEU_val, NEU_tst = tt.datasets.TranslationDataset.splits(
    exts=('.en.txt', '.cn.txt'), fields=(SRC, TRG), path='../../Datasets/WMT17/neu2017',
    filter_pred=lambda ex: 10 <= len(ex.trg) <= 20
)

SRC.build_vocab(NEU_tri, max_size=20000)  # max_size=20000 or min_freq=5
TRG.build_vocab(NEU_tri, max_size=20000)

SOS_ID = TRG.vocab.stoi[SOS_TOK]
EOS_ID = TRG.vocab.stoi[EOS_TOK]
PAD_ID = TRG.vocab.stoi[PAD_TOK]

NEU_tri_iter, NEU_val_iter, NEU_tst_iter = tt.data.BucketIterator.splits(
    datasets=(NEU_tri, NEU_val, NEU_tst), batch_sizes=(100, 100, 1), repeat=False, device=DEVICE
)

print("[TRAIN]:%d (dataset:%d)\t[TEST]:%d (dataset:%d)\t[VAL]:%d (dataset:%d)"
      % (len(NEU_tri_iter), len(NEU_tri_iter.dataset),
         len(NEU_val_iter), len(NEU_val_iter.dataset),
         len(NEU_tst_iter), len(NEU_tst_iter.dataset)))


def to_str(idxs, itos, eos_id=None):
    """
    :param idxs:  1D tensor
    :param itos:
    :return:
    """
    tokens = []
    for i in idxs:
        if eos_id is not None and i.item() == eos_id:
            break
        tokens.append(itos[i.item()])
    return ' '.join(tokens)
