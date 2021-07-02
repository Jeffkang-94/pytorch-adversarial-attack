import torch
import torch.nn as nn

class MyBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(MyBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def get_mean_and_std(self):
        return torch.mean(self.running_mean), torch.mean(self.running_var)

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            
            mean = input.mean([0, 2, 3])
            # use biased var in train
            var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var

                
        else:
            mean = self.running_mean
            var = self.running_var
        input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))


        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input

class SplitBatchNorm2d(torch.nn.BatchNorm2d):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, num_splits=2):
        
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        assert num_splits > 1, 'Should have at least one aux BN layer (num_splits at least 2)'
        self.num_splits = num_splits
        self.aux_bn = nn.ModuleList([
            nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats) for _ in range(num_splits - 1)])
    def forward(self, input: torch.Tensor):
        if self.training:  # aux BN only relevant while training
            split_size = input.shape[0] // self.num_splits # 64 to 32 
            assert input.shape[0] == split_size * self.num_splits, "batch size must be evenly divisible by num_splits"
            split_input = input.split(split_size)
            x = [super().forward(split_input[0])]
            #mean_list, std_list= [] , []
            for i, a in enumerate(self.aux_bn):
                x.append(a(split_input[i + 1]))
            return torch.cat(x, dim=0)
        else:
            return super().forward(input)

def print_mean_std(module):
    mod = module
    print(mod)
    for name, child in module.named_children():
        print(child.running_mean)
    return mean, std

def convert_splitbn_model(module, momentum=0.01, num_splits=2):
    """
    Recursively traverse module and its children to replace all instances of
    ``torch.nn.modules.batchnorm._BatchNorm`` with `SplitBatchnorm2d`.
    Args:
        module (torch.nn.Module): input module
        num_splits: number of separate batchnorm layers to split input across
    Example::
        >>> # model is an instance of torch.nn.Module
        >>> model = timm.models.convert_splitbn_model(model, num_splits=2)
    """
    mod = module
    adv_momentum = momentum
    if isinstance(module, torch.nn.modules.instancenorm._InstanceNorm):
        return module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        mod = SplitBatchNorm2d(
            module.num_features, module.eps,adv_momentum, module.affine,
            module.track_running_stats, num_splits=num_splits)
        mod.running_mean = module.running_mean
        mod.running_var = module.running_var
        mod.num_batches_tracked = module.num_batches_tracked
        if module.affine:
            mod.weight.data = module.weight.data.clone().detach()
            mod.bias.data = module.bias.data.clone().detach()

        for aux in mod.aux_bn:
            aux.running_mean = module.running_mean.clone()
            aux.running_var = module.running_var.clone()
            aux.num_batches_tracked = module.num_batches_tracked.clone()
            if module.affine:
                aux.weight.data = module.weight.data.clone().detach()
                aux.bias.data = module.bias.data.clone().detach()
    for name, child in module.named_children():
        mod.add_module(name, convert_splitbn_model(child, num_splits=num_splits))
    del module
    return mod