import random
import torch.nn as nn
from  dataclasses import dataclass, field

@dataclass(repr=True)
class config:
    data_path : str
    design : str
    loss : str
    test : int
    device = "cuda:0"
    z_size = 100
    learning_rate = 0.0001
    betas = [0.0001,0.9]
    epoch = 50
    batch_size = 64
    seed:int=field(default_factory=lambda : random.randint(1,1000))
    critic_iter = 5
    reg_lambda = 10
    goodness_weight = 10
    distance:nn =field(default_factory=lambda : nn.HuberLoss())
    dev_metric = "L1"