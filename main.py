import torch
import os, argparse

import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from cifar_model import *
from utils import Normalize_net
from attack import Attacker
device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=8)

model = WRN(depth=34, width=1, num_classes=10).to(device)

checkpoint = torch.load('./cifar_model/state_dict/WRN_simple.pth')
model.load_state_dict(checkpoint['model'])
model = Normalize_net(model)
config = {
    'eps' : 8.0/255.0,
    'attack_steps': 20,
    'attack_lr': 2.0 / 255,
    'random_init': True,
}
attack = Attacker(model, config)

def adv_test():
    model.eval()
    adv_correct, correct = 0, 0 
    total = 0 
    tq = tqdm(enumerate(testloader), total=len(testloader), leave=True)
    for batch_idx, (inputs, targets) in tq:
        inputs, targets = inputs.to(device), targets.to(device)

        logit = model(inputs)
        _, pred = logit.max(1)
        correct += pred.eq(targets).sum().item()

        adv_inputs = attack.mi_fgsm(inputs, targets)
        logit = model(adv_inputs)
        _, pred = logit.max(1)
        adv_correct += pred.eq(targets).sum().item()

        total += targets.size(0)
        tq.set_description("Accuracy clean/adv: {:.4f}/{:.4f}".format(correct/total*100.0, adv_correct/total*100.0))

adv_test()