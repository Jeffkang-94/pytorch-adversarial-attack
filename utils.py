import torch
import torch.nn as nn

class Normalize_net(nn.Module):
    def __init__(self, model):
        super(Normalize_net, self).__init__()
        self.model = model
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std  = (0.2023, 0.1994, 0.2010)
        
    def data_normalize(self, tensor, mean, std, inplace=False):
        if not torch.is_tensor(tensor):
            raise TypeError('tensor is not a torch image.')
        if not inplace:
            tensor = tensor.clone()
        dtype = tensor.dtype
        mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
        tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
        return tensor
    
    def forward(self, input):
        normalized_input = self.data_normalize(input, self.mean, self.std)
        out = self.model(normalized_input)
        return out