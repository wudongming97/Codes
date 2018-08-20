import spacy
import torchtext as tt

from utils import DEVICE

spacy_en = spacy.load('en')
spacy_de = spacy.load('de')

tokenize_en = lambda s: [tok.text for tok in spacy_de.tokenizer(s)]
tokenize_de = lambda s: [tok.text for tok in spacy_en.tokenizer(s)]

SRC = tt.data.Field(tokenize=tokenize_de, lower=True)
TRG = tt.data.Field(tokenize=tokenize_en, init_token='<s>', eos_token='</s>', lower=True)

NEU_tri, NEU_val, NEU_tst = tt.datasets.TranslationDataset.splits(
    exts=('.en', '.de'), fields=(SRC, TRG), path='../../Datasets/wmt14/',
    train='train.tok.clean.bpe.32000', validation='newstest2013.tok.bpe.32000',
    test='newstest2014.tok.bpe.32000', filter_pred=lambda ex: len(ex.src) <= 50 and len(ex.trg) <= 50
)

SRC.build_vocab(NEU_tri, NEU_val, NEU_tst, min_freq=5)
TRG.build_vocab(NEU_tri, NEU_val, NEU_tst, min_freq=5)

NEU_tri_iter, NEU_val_iter, NEU_tst_iter = tt.data.BucketIterator.splits(
    datasets=(NEU_tri, NEU_val, NEU_tst), batch_sizes=(32, 100, 100), repeat=False, device=DEVICE
)
