# 一些VAE的实现

## vae
![vae1](https://github.com/yxue3357/MyResearchCodes/raw/master/VAEs/BASEs/results/train7000.png)
![vae1](https://github.com/yxue3357/MyResearchCodes/raw/master/VAEs/BASEs/results/vae_z_7000.png)

## mmd_vae
![mmd_vae1](https://github.com/yxue3357/MyResearchCodes/raw/master/VAEs/BASEs/results/mmd_train4000.png)
![mmd_vae1](https://github.com/yxue3357/MyResearchCodes/raw/master/VAEs/BASEs/results/mmd_vae_z_1000.png)

Notes:
> 1. loss_nll 用MSE或者用交叉熵效果差不多
> 2. loss_nll 换用l1 loss没效果， 生成几乎全黑图片, z空间也很混乱