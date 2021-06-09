'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
from tqdm import tqdm
from cifar_model import *


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch_size', default=512, type=int, help='size of batch')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--name', type=str, help='name of trial')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
num_epoch = 201
# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=16)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=16)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = WRN(depth=34, width=1, num_classes=10)
net = net.to(device)
cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/wrn.pth')
    net.load_state_dict(checkpoint['model'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer_lr = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[60,120,180], gamma=0.1)
checkpoint_path = os.path.join('checkpoint',args.name)
os.makedirs(checkpoint_path, exist_ok=True)
f = open(os.path.join(checkpoint_path, "log.txt"),"a")
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    print(optimizer)
    tq = tqdm(enumerate(trainloader), total=len(trainloader), leave=True)
    for batch_idx, (inputs, targets) in tq:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        tq.set_description('Training [%d/%d] Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (epoch, num_epoch, train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
        
    optimizer_lr.step()

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        tq = tqdm(enumerate(testloader), total=len(testloader), leave=True)
        for batch_idx, (inputs, targets) in tq:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            tq.set_description('Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    f.write('Epoch[%d] Test Acc : %.3f%% (%d/%d)\n' %(epoch, acc, correct, total))
    f.flush()
    if acc > best_acc:
        print('Saving..')
        state = {
            'model': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        
        torch.save(state, os.path.join(checkpoint_path,'checkpoint.pth'))
        best_acc = acc


for epoch in range(start_epoch, start_epoch+num_epoch):
    train(epoch)
    test(epoch)
f.close()
