from mnt import MNT
from utils import *

mnt = MNT(
    save_dir='./vanilla/',
    src=SRC,
    trg=TRG,
    tri_iter=NEU_tri_iter,
    val_iter=NEU_val_iter,
    tst_iter=NEU_tst_iter,
    e_n_layers=1,
    d_n_layers=2,
    n_epochs=20,

    use_attn=True,
)
mnt.fit()
