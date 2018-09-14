from model import *
from sgld import SGLD

sgld_model = MLP()
sgld_trainer = SGLD(sgld_model.parameters(), lr=0.01)
train(sgld_model, sgld_trainer, 'sgld')

sgd_model = MLP()
sgd_trainer = SGLD(sgd_model.parameters(), lr=0.01, addnoise=False)
train(sgd_model, sgd_trainer, 'sgd')
