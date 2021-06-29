import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

class Attacker(ABC):
    def __init__(self, model, config):
        """
        ## initialization ##
        :param model: Network to attack
        :param config : configuration to init the attack
        """
        self.config = config
        self.model = model
        self.clamp = (0,1)
    
    def _random_init(self, x):
        x = x + (torch.rand(x.size(), dtype=x.dtype, device=x.device) - 0.5) * 2 * self.config['eps']
        x = torch.clamp(x,*self.clamp)
        return x

    def __call__(self, x,y):
        x_adv = self.forward(x,y)
        return x_adv

class FGSM(Attacker):
    def __init__(self, model, config, target=None):
        super(FGSM, self).__init__(model, config)
        self.target = target

    def forward(self, x, y):
        x_adv = x.detach().clone()

        if self.config['random_init']:
            x_adv = self._random_init(x_adv)

        x_adv.requires_grad=True
        self.model.zero_grad()

        logit = self.model(x_adv)
        #y = torch.LongTensor(torch.randint(10, (64,1)).squeeze(0)).cuda()
        #y = [torch.randint(10,(1,1)).squeeze(0) for i in range(64)]
        #y = torch.LongTensor(y).cuda()
        if self.target is None:
            #
            cost = -F.cross_entropy(logit, y)
        else:
            cost = F.cross_entropy(logit, self.target)
        
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        cost.backward()

        x_adv.grad.sign_()
        x_adv = x_adv - self.config['eps']*x_adv.grad
        x_adv = torch.clamp(x_adv,*self.clamp)

        return x_adv

class PGD(Attacker):
    def __init__(self, model, config, target=None):
        super(PGD, self).__init__(model, config)
        self.target = target

    def forward(self, x, y):
        """
        :param x: Inputs to perturb
        :param y: Ground-truth label
        :param target : Target label 
        :return adversarial image
        """
        x_adv = x.detach().clone()
        if self.config['random_init'] :
            x_adv = self._random_init(x_adv)

        for step in range(self.config['attack_steps']):
            x_adv.requires_grad = True
            self.model.zero_grad()
            logits = self.model(x_adv) #f(T((x))
            if self.target is None:
                # Untargeted attacks - gradient ascent
                loss = F.cross_entropy(logits, y,  reduction="sum")
                loss.backward()
                grad = x_adv.grad.detach()
                grad = grad.sign()
                x_adv = x_adv + self.config['attack_lr'] * grad
            else:
                # Targeted attacks - gradient descent
                assert self.target.size() == y.size()
                loss = F.cross_entropy(logits, target)
                loss.backward()
                grad = x_adv.grad.detach()
                grad = grad.sign()
                x_adv = x_adv - self.config['attack_lr'] * grad

            # Projection
            x_adv = x + torch.clamp(x_adv - x, min=-self.config['eps'], max=self.config['eps'])
            x_adv = x_adv.detach()
            x_adv = torch.clamp(x_adv, *self.clamp)

        return x_adv

class MIFGSM(Attacker):
    def __init__(self, model, config, target=None):
        super(MIFGSM, self).__init__(model, config)
        self.target = target

    def forward(self, x,y):
        """
        :param x: Inputs to perturb
        :param y: Ground-truth label
        :param target : Target label 
        :return adversarial image
        """
        alpha = self.config['eps'] / self.config['attack_steps'] 
        decay = 1.0
        x_adv = x.detach().clone()
        momentum = torch.zeros_like(x_adv, device=x.device)
        if self.config['random_init'] :
            x_adv = x_adv + (torch.rand(x.size(), dtype=x.dtype, device=x.device) - 0.5) * 2 * self.config['eps']
            x_adv = torch.clamp(x_adv,*self.clamp)


        for step in range(self.config['attack_steps']):
            x.requires_grad=True
            logit = self.model(x)
            if self.target is None:
                cost = -F.cross_entropy(logit, y)
            else:
                cost = F.cross_entropy(logit, target)
            grad = torch.autograd.grad(cost, x, retain_graph=False, create_graph=False)[0]
            grad_norm = torch.norm(grad, p=1)
            grad /= grad_norm
            grad += momentum*decay
            momentum = grad
            x_adv = x - alpha*grad.sign()
            a = torch.clamp(x - self.config['eps'], min=0)
            b = (x_adv >= a).float()*x_adv + (a > x_adv).float()*a
            c = (b > x + self.config['eps']).float() * (x + self.config['eps']) + (
                x + self.config['eps'] >= b
            ).float() * b
            x = torch.clamp(c, max=1).detach()
        x_adv = torch.clamp(x, *self.clamp)
        return x_adv

