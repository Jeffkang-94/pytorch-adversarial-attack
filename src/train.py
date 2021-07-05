import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.optim as optim
import torchvision
import utils
import os
import time
from torch.utils.tensorboard import SummaryWriter 
from attack import *
from tqdm import tqdm

class Trainer:
    def __init__(self, configs, model):

        self.configs = configs
        self.model = model 
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.save_path = os.path.join(configs.save_path, configs.model_name)
        self.epoch = 0 
        self.best_acc = 0 
        assert os.path.exists(self.save_path), "The directory exists, modify `model_name` in train.json"


        # Creating data loaders
        transform_train = T.Compose([
            T.Pad(4, padding_mode='reflect'),
            T.RandomCrop(32),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ])

        transform_test = T.Compose([
            T.ToTensor()
        ])

        self.trainloader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(root=configs.data_root, train=True, download=True, transform=transform_train)
            ,batch_size=configs.batch_size, shuffle=True, num_workers=8
        )

        self.testloader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(root=configs.data_root, train=False, download=True, transform=transform_test)
            ,batch_size=configs.batch_size, shuffle=False, num_workers=8
        )

        if self.configs.spbn:
            self.model = models.convert_splitbn_model(self.model, momentum=0.5).to(self.device)
        else:
            self.model.to(self.device)
            
        if self.configs.resume:
            self._load_network(os.path.join(self.save_path, "latest.pth"))
        
        self.optimizer = optim.SGD(self.model.parameters(), configs.lr, momentum=0.9, weight_decay=configs.weight_decay)
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[60, 120, 160], gamma=0.1)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        
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

        print('Number of model parameters: {}'.format(
            sum([p.data.nelement() for p in self.model.parameters()])))
    
    def _load_network(self, checkpoint_path):
        print("Loading model from {} ...".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.epoch = checkpoint['epoch'] + 1
        self.best_acc = checkpoint['best_acc']
        print("Loading Done..")

    def _save_network(self, best=False):
        self.model.eval()
        checkpoint = dict()
        checkpoint['model'] = self.model.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()
        checkpoint['lr_scheduler'] = self.lr_scheduler.state_dict()
        checkpoint['epoch'] = self.epoch
        checkpoint['best_acc'] = self.best_acc
        if best:
            torch.save(checkpoint, os.path.join(self.save_path, 'best.pth'))
        else:
            torch.save(checkpoint, os.path.join(self.save_path, 'latest.pth'))

    def _accuracy(self, logits, target):
        _, pred = torch.max(logits, dim=1)
        correct = (pred == target).sum()
        total = target.size(0)
        acc = (float(correct) / total) * 100
        return acc

    def _logger(self, loss, acc):
        self.writer.add_scalar('training/loss', loss.item())
        self.writer.add_scalar('training/acc', acc)

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

    def train_model(self):
        log_dir = self.save_path + '/training_log'
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        best_acc = self.best_acc
        start_epoch = self.epoch
        for epoch in range(start_epoch, self.configs.epochs):
            self.model.train()
            acc = 0
            start_time = time.time()
            tq = tqdm(enumerate(self.trainloader), total=len(self.trainloader), leave=True)
            for i, (x,y) in tq:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                if self.configs.phase == 'clean':
                    logits = self.model(x)
                elif self.configs.phase == 'adv':
                    x_adv = self.attacker(x,y)
                    logits = self.model(x_adv)

                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()

                acc += self._accuracy(logits, y)
                self.epoch += 1
                tq.set_description('Epoch {}/{}, Loss: {:.4f}, Accuracy: {:.4f}'.format(
                    epoch, self.configs.epochs, loss.item(), acc/(i+1)))
                self._logger(loss,acc)
            self.lr_scheduler.step()
            end_time = time.time()

            if self.epoch %  self.configs.save_interval == 0:
                eval_acc, eval_adv_acc = self.eval_model()
                if eval_adv_acc > self.best_acc:
                    self.best_acc = eval_adv_acc
                self._save_network()









