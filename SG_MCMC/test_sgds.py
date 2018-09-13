from model import *
from sgld import SGLD

sgld_model = MLP()
sgld_trainer = SGLD(sgld_model.parameters())
train(sgld_model, sgld_trainer, 'sgld')

sgd_model = MLP()
sgd_trainer = SGLD(sgld_model.parameters(), addnoise=False)
train(sgd_model, sgd_trainer, 'sgd')
