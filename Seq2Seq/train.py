from bleu import *
from corpus import *
from mnt import MNT

saved_model_name = None
# saved_model_name = 'best.pth'

mnt = MNT(
    save_dir='./',
    src=SRC,
    trg=TRG,
    tri_iter=NEU_tri_iter,
    val_iter=NEU_val_iter,
    e_n_layers=1,
    d_n_layers=1,
    n_epochs=10,
    use_attn=True,
)
if saved_model_name is None:
    mnt.fit()
else:
    mnt.load(saved_model_name)
    with open('./src.txt', 'w') as sf, open('./trg.txt', 'w') as tf, open('./trs.txt', 'w') as of:
        for sample in NEU_tst_iter:
            src = to_str(sample.src, SRC.vocab.itos)
            trg = to_str(sample.trg[1:-1], TRG.vocab.itos, EOS_ID)
            trs = mnt.translate(sample.src)
            sf.write(src), tf.write(trg), of.write(trs)
    # 计算bleu
    bleu_f('./trg.txt', './trs.txt')
