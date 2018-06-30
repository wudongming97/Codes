from model import get_model
from utils import *

save_dir = './Results/'

model = get_model().to(DEVICE)

pretrain = '20.pth'
model.load_state_dict(torch.load(save_dir + pretrain))

## test
