## A simple **"Adversarial Autoencoders"**  Implementation.

### vae

<figure class="half">
    <img src="https://github.com/yxue3357/MyResearchCodes/raw/master/VAEs/AAE/results/vae_z16_train8000.png" width="50%" height="50%">
    <img src="https://github.com/yxue3357/MyResearchCodes/raw/master/VAEs/AAE/results/vae_z16_tsne_8000.png" width="50%" height="50%">
</figure>

![vae1](https://github.com/yxue3357/MyResearchCodes/raw/master/VAEs/AAE/results/vae_z16_train8000.png){:height="50%" width="50%"}
![vae2](https://github.com/yxue3357/MyResearchCodes/raw/master/VAEs/AAE/results/vae_z16_tsne_8000.png){:height="50%" width="50%"}

### aae
![aae1](https://github.com/yxue3357/MyResearchCodes/raw/master/VAEs/AAE/results/aae_train_6600.png){:height="50%" width="50%"}
![aae2](https://github.com/yxue3357/MyResearchCodes/raw/master/VAEs/AAE/results/aae_z_6000.png){:height="50%" width="50%"}

### label regularized aae
![vae_lr1](https://github.com/yxue3357/MyResearchCodes/raw/master/VAEs/AAE/results/aae_lr_train13500.png){:height="50%" width="50%"}
![vae_lr2](https://github.com/yxue3357/MyResearchCodes/raw/master/VAEs/AAE/results/aae_lr_z_13500.png){:height="50%" width="50%"}

Note:
> 1. 有label信息后， z的embedding效果很好，不过在swiss_roll不尽如人意。

### supervised aae
![supervised aae1](https://github.com/yxue3357/MyResearchCodes/raw/master/VAEs/AAE/results/supervised_aae_train_16900.png)
![supervised aae2](https://github.com/yxue3357/MyResearchCodes/raw/master/VAEs/AAE/results/supervised_aae_train_13700.png)
![supervised aae3](https://github.com/yxue3357/MyResearchCodes/raw/master/VAEs/AAE/results/supervised_aae_train_26100.png)

Note: 
> 1. 在加入了监督信息过后， z的embedding可视化后，效果很差。