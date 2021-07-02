import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

from tqdm import tqdm
from cifar_model import *
from utils import Normalize_net
from attack import *
from config import *

from src.train import Trainer
from src.eval import Evaluator
import json

def main():
    args = parse_args()
    configs = get_configs(args)

    model = WRN(depth=configs.model_depth, width=configs.model_width, num_classes=configs.num_class)
    model = Normalize_net(model) # apply the normalization before feeding the inputs into the classifier.
    
    if configs.mode == 'train':
        train = Trainer(configs, model)
        train.train_model()
    elif configs.mode == 'eval':
        test = Evaluator(configs, model)
        test.eval_model()
    else:
        raise ValueError('Specify the mode, `train` or `eval`')


if __name__ == '__main__':
    main()