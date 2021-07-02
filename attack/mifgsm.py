from attack import Attacker
import torch
import torch.nn.functional as F

class MIFGSM(Attacker):
    def __init__(self, model, config, target=None):
        super(MIFGSM, self).__init__(model, config)
        self.target = target

    def forward(self, x, y):
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
