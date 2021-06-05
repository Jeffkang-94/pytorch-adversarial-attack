import torch
import os, argparse

import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from cifar_model import *
from utils import Normalize_net
from adv_attack import *
from config import parse_args

args = parse_args()
verbose=True

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

testset = torchvision.datasets.CIFAR10(root=args.datapath, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8)

if args.model == 'res':
    model = WRN(depth=34, width=1, num_classes=10).to(device)
    checkpoint = torch.load(os.path.join(args.checkpoint_dir, args.name))
elif args.model =='wideres':
    # not supported yet
    model = WRN(depth=34, width=10, num_classes=10).to(device)
    pass
model.load_state_dict(checkpoint['model'])
model = Normalize_net(model)
model.eval()

config = {
    'eps' : args.eps/255.0,
    'attack_steps': args.attack_steps,
    'attack_lr': args.attack_lr / 255.0,
    'random_init': args.random_init,
}

if verbose:
    print("Adversarial Attack : {}".format(args.attack))
    print(config)

if args.attack =='FGSM':
    attack = FGSM(model, config)
elif args.attack =='MIFGSM':
    attack = MIFGSM(model, config)
elif args.attack =='PGD':
    attack = PGD(model, config)

def adv_test():
    adv_correct, correct = 0, 0 
    total = 0 
    tq = tqdm(enumerate(testloader), total=len(testloader), leave=True)
    for batch_idx, (inputs, targets) in tq:
        inputs, targets = inputs.to(device), targets.to(device)

        logit = model(inputs)
        _, pred = logit.max(1)
        correct += pred.eq(targets).sum().item()

        adv_inputs = attack(inputs, targets)
        logit = model(adv_inputs)
        _, pred = logit.max(1)
        adv_correct += pred.eq(targets).sum().item()

        total += targets.size(0)
        tq.set_description("Accuracy clean/adv: {:.4f}/{:.4f}".format(correct/total*100.0, adv_correct/total*100.0))

if __name__ == '__main__':
    adv_test()