## Code for PGD attacker
import torch
import torch.nn.functional as F

class Attacker(object):
    def __init__(self, model, config):
        """
        ## initialization ##
        :param model: Network to attack
        :param config : configuration to init the attack
        """
        self.config = config
        self.model = model
        self.clamp = (0,1)
    
    def fgsm(self, x, y, target=None):
        """
        :param x: Inputs to perturb
        :param y: Ground-truth label
        :param target : Target label 
        :return adversarial image
        """
        
        x_adv = x.detach().clone()

        if self.config['random_init']:
            # Flag to use random initialization
            x_adv = x_adv + (torch.rand(x.size(), dtype=x.dtype, device=x.device) - 0.5) * 2 * self.config['eps']
            x_adv = torch.clamp(x_adv,0,1)

        x_adv.requires_grad=True
        self.model.zero_grad()

        logit = self.model(x_adv)
        if target is None:
            cost = -F.cross_entropy(logit, y)
        else:
            cost = F.cross_entropy(logit, target)
        
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        cost.backward()

        x_adv.grad.sign_()
        x_adv = x_adv - self.config['eps']*x_adv.grad
        x_adv = torch.clamp(x_adv,*self.clamp)

        return x_adv

    def pgd(self, x, y, target=None):
        """
        :param x: Inputs to perturb
        :param y: Ground-truth label
        :param target : Target label 
        :return adversarial image
        """
        x_adv = x.detach().clone()
        if self.config['random_init'] :
            x_adv = x_adv + (torch.rand(x.size(), dtype=x.dtype, device=x.device) - 0.5) * 2 * self.config['eps']
            x_adv = torch.clamp(x_adv,*self.clamp)

        for step in range(self.config['attack_steps']):
            x_adv.requires_grad = True
            self.model.zero_grad()
            logits = self.model(x_adv) #f(T((x))
            if target is None:
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
    
    def mi_fgsm(self, x,y, target=None):
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
            if target is None:
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

    
