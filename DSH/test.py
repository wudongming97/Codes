from network import get_network
from utils import *

save_dir = './Results/'

model = get_network().to(DEVICE)

pretrain = '20.pth'
model.load_state_dict(torch.load(save_dir + pretrain))

## test
