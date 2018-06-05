import os
import torch

from beta_vae_h import train, test, BetaVAE_H, DEVICE
from data import chairs_3d_iter

z_dim = 16
beta = 4
save_dir = './b4z16/'
os.makedirs(save_dir, exist_ok=True)

model = BetaVAE_H(z_dim, 3).to(DEVICE)
train(model,
      data_iter=chairs_3d_iter,
      lr=1e-4,
      n_epochs=5,
      beta=beta,
      save_dir=save_dir)

model.load_state_dict(torch.load(save_dir + 'beta_4_vae_4.pth'))

test(model,
     batch_size=8,
     width=3,
     save_dir=save_dir)
