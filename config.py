import argparse

def parse_args():
    desc ="Pytorch Adversarial Attack"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--model', type=str, required=True, help='[res|wideres]')
    parser.add_argument('--dataset', type=str, required=True, help='cifar10')
    parser.add_argument('--datapath', type=str, required=True, help='Denote the dataset path')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/')
    parser.add_argument('--name', type=str, default='ResNet_clean.pth')

    parser.add_argument('--attack', type=str, default='FGSM', help='The type of adversarial attacks [FGSM/MI-FGSM/PGD]')
    parser.add_argument('--attack_steps', type=int, default='7', help='The number of attack steps')
    parser.add_argument('--attack_lr', type=float, default='2', help='The number of attack learning rate')
    parser.add_argument('--eps', type=float, default=8, help='The magnitude of the adversarial attacks')
    parser.add_argument('--viz_result', action='store_true', help='The flag of generating visualization result')
    parser.add_argument('--random_init', action='store_true', help='The flag of random start')
    return parser.parse_args()