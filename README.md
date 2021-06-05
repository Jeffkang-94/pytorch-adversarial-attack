# Pytorch implementation of gradient-based adversarial attack

This repository covers pytorch implementation of FGSM, MI-FGSM, and PGD attack.
Attacks are implemented in `attack.py` file.
To explore adversarial attack, we deal with ![Madry](https://arxiv.org/pdf/1706.06083.pdf) model which had been trained with PGD adversarial examples.

## Preliminary

When we train the model with task-specific loss (e.g., classification), the model constructs a decision boundary and classifies given inputs based on that boundary. 
An adversarial attack aims to find noise distribution to cross the decision boundary within Lp ball.
In order to make sure that crafted adversarial images hold imperceptibility, the magnitude of perturbation will not be significant at human-level intuition.
However, those are capable of crossing the boundary, leading to misclassification.

<img src="https://github.com/Jeffkang-94/pytorch-adversarial-attack/blob/develop/asset/clean_result.jpg" width="200" height="400">
<img src="https://github.com/Jeffkang-94/pytorch-adversarial-attack/blob/develop/asset/diff.jpg" width="200" height="200">
<img src="https://github.com/Jeffkang-94/pytorch-adversarial-attack/blob/develop/asset/adv_result.jpg" width="200" height="200">

## Dependency

```
python 3.6
pytorch >= 1.4.0
tqdm
```

## Configuration
`adv_test.sh` file consists of several options to launch the adversarial attack with different configuration.
You can specify the attack configuration as below.

```python
config = {
    'eps' : args.eps/255.0, 
    'attack_steps': args.attack_steps,
    'attack_lr': args.attack_lr / 255.0, 
    'random_init': args.random_init, 
}
```

```bash
CUDA_VISIBLE_DEVICES=1 python main.py \
    --model 'res' \
    --name ${pretrained_file} \ # ResNet_clean.pth, ResNet_Madry_PGD7.pth
    --dataset 'cifar10' \
    --datapath 'data' \
    --attack ${attack_type}\ # FGSM/PGD/MI-FGSM supported
    --batch_size 64 \
    --attack_steps 40 \
    --attack_lr 2 \
    --random_init
```
  - `attack` is a attack type. *FGSM/PGD/MI-FGSM* are supported.
  - `attack_steps` is a number of attack steps. It is used for iterative attacks (e.g., MI-FGSM, PGD)
  - `attack_lr` is a learning rate of attack. 
  - `random_init` is a flag whether apply random initialization before launching an adversarial attack.



## Usage

```python
from adv_attack import PGD

config = {
    'eps' : 8.0/255.0, 
    'attack_steps': 7,
    'attack_lr': 2.0 / 255.0, 
    'random_init': True, 
}

attack = PGD(model, config)
adversarial_image = attack(image, label)
```
You can test out the adversarial attack using following command lines.

> mkdir data
> ln -s <datapath> data
> bash adv_test.sh


## Pre-trained model

We provide the pre-trained ResNet model which had been trained with CIFAR-10 dataset.
Note that `Madry` model had been trained with PGD-7 adversarial examples following introduced settings.

**ResNet** : ![link](https://drive.google.com/file/d/1zAiPdXLPYkikxVnjXR8zcgEGer8HR3Ca/view?usp=sharing)
**Madry**  : ![link](https://drive.google.com/file/d/1iAwkv18spCYaVEOi7IGDDHpgY9EB-MUi/view?usp=sharing)

## Experiment

|**Model** | **Clean** | **FGSM** | **MI-FGSM-20** | **PGD-7/40** |
:---: |:---: |:---: |:---: | :---: |
ResNet | 92.24 | 25.7 | 1.05  | 0.65/0.00
Madry-Simple  | 78.10 | 49.71 | 46.97 | 45.01/41.06
Wide-ResNetx10 |      |       |       |
Madry-Wide     |      |       |       | 

## Reference

![torchattack](https://github.com/Harry24k/adversarial-attacks-pytorch)