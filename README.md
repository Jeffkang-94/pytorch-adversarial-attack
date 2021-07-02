# Pytorch implementation of gradient-based adversarial attack

This repository covers pytorch implementation of `FGSM`, `MI-FGSM`, and `PGD` attack.
Attacks are implemented in `adv_attack.py` file.
To explore adversarial attack, we deal with [Madry](https://arxiv.org/pdf/1706.06083.pdf) model which had been trained with PGD adversarial examples.

## Preliminary

When we train the model with task-specific loss (e.g., classification), the model constructs a decision boundary and classifies given inputs based on that boundary. 
An adversarial attack aims to find noise distribution to cross the decision boundary within Lp ball.
In order to make sure that crafted adversarial images hold imperceptibility, the magnitude of perturbation will not be significant at human-level intuition.
However, those are capable of crossing the boundary, leading to misclassification.
<p align="center">
    <img src="https://github.com/Jeffkang-94/pytorch-adversarial-attack/blob/develop/asset/clean_result.jpg" alt width="250" height="250">
    <img src="https://github.com/Jeffkang-94/pytorch-adversarial-attack/blob/develop/asset/diff.jpg" width="250" height="250">
    <img src="https://github.com/Jeffkang-94/pytorch-adversarial-attack/blob/develop/asset/adv_result.jpg" width="250" height="250">
    <p align="center"> Original image / Difference / Adversarial image </p>
</p>

## Dependency

```
python 3.6
pytorch >= 1.4.0
tqdm
```

## :rocket: Usage

```python
from attack import PGD

attack_config = {
    'eps' : 8.0/255.0, 
    'attack_steps': 7,
    'attack_lr': 2.0 / 255.0, 
    'random_init': True, 
}

attack = PGD(model, attack_config)
adversarial_image = attack(image, label)
```

## :hammer: Adversarial Training

You can test out the adversarial training using following command lines.  
You have to specify the configuration path before launching the files.
```
> mkdir data  
> ln -s <datapath> data  
> python main.py --cfg_path config/train.json # training a victim model
> python main.py --cfg_path config/eval.json  # launching an adversarial attack to evaluate the pre-trained model
```

### Configuration
Under `config` file, `train.json` and `eval.json` files include the configurations to launch the training or evaluation.  
You can set the different options depending on your own environment.
This is the example of `train.json`. 


```
{
    "mode": "train",            // we are under train mode             
    "data_root": "./data",      // You can specify the own dataset root
    "model_name" : "resnet",    // name of the model
    "model_depth": 32,          // model depth
    "model_width": 1,           // model width
    "num_class":10,             // number of class, e.g., cifar-10 : 10
    "phase": "adv",             // [clean/adv] supported
  
/* Training Configuration */
    "lr": 0.1,
    "batch_size": 256,
    "weight_decay": 0.0005,
    "epochs": 200,
    "save_interval" : 5,
    "restore": false,
    "save_path": "results",
    "spbn": false,              // Split-batchnorm training, not supported
    "resume": false,
 
 /* Attack Configuration */
    "attack": "PGD",            // attack type
    "attack_steps": 7,          // attack steps
    "attack_eps": 8.0,          // magnitude of epsilon
    "attack_lr": 2.0,           // attack learning rate
    "random_init": true,        // flag for random start
  }
```


## 🚴 Pre-trained model

We provide the pre-trained `ResNet` model which had been trained with CIFAR-10 dataset.
Note that `Madry` model had been trained with *PGD-7 adversarial examples* following introduced settings.
For using a pre-trained model, you can use `download.sh` file. 
It will automatically download the whole pre-trained weight files and organize them to the designated path.

```
bash download.sh
```

Or you can directly access the link as below.

**ResNet** : [link](https://drive.google.com/file/d/1zAiPdXLPYkikxVnjXR8zcgEGer8HR3Ca/view?usp=sharing)  
**Madry**  : [link](https://drive.google.com/file/d/1iAwkv18spCYaVEOi7IGDDHpgY9EB-MUi/view?usp=sharing)  

**Wide-ResNet** : [link](https://drive.google.com/file/d/1uEQdXDIK4XPnm7ZkzLFtTrXb0WfAgi54/view?usp=sharing)  
**Wide-Madry**  : [link](https://drive.google.com/file/d/1xdocjfDQ88LzLjYfkIJVSL9ImzrzUXZB/view?usp=sharing)

## 📔 Experiment

|**Model** | **Clean** | **FGSM** | **MI-FGSM-20** | **PGD-7/40** |
:---: |:---: |:---: |:---: | :---: |
ResNet | 92.24 | 25.7 | 1.05  | 0.65/0.00
Madry-Simple  | 78.10 | 50.02 | 46.97 | 45.01/41.06
Wide-ResNetx10 | 94.70   | 31.15 | 0.43  | 0.14/0.00
Madry-Wide     | 86.71   | 52.27 | 47.27| 47.74/43.39

## :ghost: Examples
We visualize each sample of adversary.

### FGSM adversary
<img src="https://github.com/Jeffkang-94/pytorch-adversarial-attack/blob/develop/asset/FGSM.jpg" alt width="800" height="400">

### MI-FGSM 20 steps adversary

<img src="https://github.com/Jeffkang-94/pytorch-adversarial-attack/blob/develop/asset/MIFGSM.jpg" width="800" height="400">

### PGD 7 steps adversary

<img src="https://github.com/Jeffkang-94/pytorch-adversarial-attack/blob/develop/asset/PGD.jpg" width="800" height="400">
    

## Reference

- Harry24k [adversarial-attacks-pytorch](https://github.com/Harry24k/adversarial-attacks-pytorch)
