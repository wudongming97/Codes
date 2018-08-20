import jieba
import spacy
import torchtext as tt

from utils import DEVICE

spacy_en = spacy.load('en')

tokenize_en = lambda s: [tok.text for tok in spacy_en.tokenizer(s)]
tokenize_zh = lambda s: list(jieba.cut(s))

SRC = tt.data.Field(tokenize=tokenize_en, lower=True)
TRG = tt.data.Field(tokenize=tokenize_zh, init_token='<s>', eos_token='</s>', lower=True)

NEU_tri, NEU_val, NEU_tst = tt.datasets.TranslationDataset.splits(
    exts=('.en', '.cn'), fields=(SRC, TRG), path='../../Datasets/WMT17/neu2017',
    filter_pred=lambda ex: len(ex.src) <= 30 and len(ex.trg) <= 30
)

SRC.build_vocab(NEU_tri, NEU_val, NEU_tst, min_freq=5)
TRG.build_vocab(NEU_tri, NEU_val, NEU_tst, min_freq=5)

NEU_tri_iter, NEU_val_iter, NEU_tst_iter = tt.data.BucketIterator.splits(
    datasets=(NEU_tri, NEU_val, NEU_tst), batch_sizes=(32, 100, 100), repeat=False, device=DEVICE
)
