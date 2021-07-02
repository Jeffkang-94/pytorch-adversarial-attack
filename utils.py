import torch
import torch.nn as nn


def visualize_sample(batch_idx, image, adv_image):
    fig, axes = plt.subplots(2,5, figsize=(12,6))
    image = np.transpose(image.cpu().detach().numpy(), (0,2,3,1))
    adv_image = np.transpose(adv_image.cpu().detach().numpy(), (0,2,3,1))
    for k in range(5):
        axes[0, k].axis("off"), axes[1, k].axis("off")

        x_grid = torchvision.utils.make_grid(torch.from_numpy(image[k,:,:,:]), nrow=1, padding=0, normalize=True, pad_value=0)
        x_npgrid = x_grid.cpu().detach().numpy()
        axes[0, k].imshow(x_npgrid, interpolation='nearest')
        axes[0, k].set_title("Clean images")

        x_adv_grid = torchvision.utils.make_grid(torch.from_numpy(adv_image[k,:,:,:]), nrow=1, padding=0, normalize=True, pad_value=0)
        x_adv_npgrid = x_adv_grid.cpu().detach().numpy()
        axes[1, k].imshow(x_adv_npgrid, interpolation='nearest')
        axes[1, k].set_title("Adv images")
    plt.axis("off")
    if not os.path.isdir("results"):
        os.makedirs("results")
    plt.savefig("results/sample_{}.jpg".format(batch_idx))
    plt.close(fig)
    
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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ConfigMapper(object):
    def __init__(self, args):
        for key in args:
            self.__dict__[key] = args[key]

