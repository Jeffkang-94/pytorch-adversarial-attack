import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision
import utils
import os
from tqdm import tqdm
from attack import *

class Evaluator:
    def __init__(self, configs, model):
        self.configs = configs
        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.save_path = os.path.join(configs.save_path, configs.model_name)
        transform_test = T.Compose([
            T.ToTensor()
        ])
        self.testloader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(root=configs.data_root, train=False, download=True, transform=transform_test)
            ,batch_size=configs.batch_size, shuffle=False, num_workers=8
        )
        if self.configs.spbn:
            self.model = models.convert_splitbn_model(self.model, momentum=0.5)
        else:
            self.model.to(self.device)
        self._load_network(os.path.join(self.save_path, "best.pth"))


        attack_config = {
            'attack': configs.attack,
            'eps' : configs.attack_eps/255.0,
            'attack_steps': configs.attack_steps,
            'attack_lr': configs.attack_lr / 255.0,
            'random_init': configs.random_init,
        }
        if configs.attack == 'FGSM':
            self.attacker = FGSM(self.model, attack_config)
        elif configs.attack =='PGD':
            self.attacker = PGD(self.model, attack_config)
        elif configs.attack =='MIFGSM':
            self.attacker = MIFGSM(self.model, attack_config)
        else:
            raise ValueError("[FGSM/MI-FGSM/PGD attack types are supported.")

    def _load_network(self, checkpoint_path):
        print("Loading model from {} ...".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model'])
        print("Loading Done..")

    def _accuracy(self, logits, target):
        _, pred = torch.max(logits, dim=1)
        correct = (pred == target).sum()
        total = target.size(0)
        acc = (float(correct) / total) * 100
        return acc

    def eval_model(self):
        self.model.eval()
        acc = 0 
        adv_acc = 0 
        tq = tqdm(enumerate(self.testloader), total=len(self.testloader), leave=True)
        for i, (x,y) in tq:
            x, y = x.to(self.device), y.to(self.device)
            x_adv = self.attacker(x,y)
            logits = self.model(x)
            adv_logits = self.model(x_adv)
            acc += self._accuracy(logits, y)
            adv_acc += self._accuracy(adv_logits, y)
            tq.set_description('Evaluation: clean/adv {:.4f}/{:.4f}'.format(
                    acc/(i+1), adv_acc/(i+1)))
        return acc/(i+1), adv_acc/(i+1)
