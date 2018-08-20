from corpus import *
from mnt import MNT

mnt = MNT(
    save_dir='./vanilla/',
    src=SRC,
    trg=TRG,
    tri_iter=NEU_tri_iter,
    val_iter=NEU_val_iter,
    tst_iter=NEU_tst_iter,
    device_id=0,
    n_epochs=100,
)
mnt.fit()
