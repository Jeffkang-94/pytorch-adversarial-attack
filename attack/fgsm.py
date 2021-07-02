from attack import Attacker
import torch
import torch.nn.functional as F

class FGSM(Attacker):
    def __init__(self, model, config, target=None):
        super(FGSM, self).__init__(model, config)
        self.target = target

    def forward(self, x, y):
        """
        :param x: Inputs to perturb
        :param y: Ground-truth label
        :param target : Target label 
        :return adversarial image
        """
        x_adv = x.detach().clone()

        if self.config['random_init']:
            x_adv = self._random_init(x_adv)

        x_adv.requires_grad=True
        self.model.zero_grad()

        logit = self.model(x_adv)

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