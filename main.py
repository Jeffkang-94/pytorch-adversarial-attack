import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

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
    checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'ResNet', args.name))
elif args.model =='wideres':
    model = WRN(depth=34, width=10, num_classes=10).to(device)
    checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'WideResNet', args.name))
model.load_state_dict(checkpoint['model'])
model = Normalize_net(model) # apply the normalization before feeding the inputs into the classifier.
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
        if args.viz_result and batch_idx%10==0:
            visualize_sample(batch_idx, inputs, adv_inputs)
        logit = model(adv_inputs)
        _, pred = logit.max(1)
        adv_correct += pred.eq(targets).sum().item()

        total += targets.size(0)
        tq.set_description("Accuracy clean/adv: {:.4f}/{:.4f}".format(correct/total*100.0, adv_correct/total*100.0))

if __name__ == '__main__':
    adv_test()