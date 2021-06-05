CUDA_VISIBLE_DEVICES=1 python main.py \
    --model 'res' \
    --name 'ResNet_clean.pth' \
    --dataset 'cifar10' \
    --datapath 'data' \
    --attack 'FGSM'\
    --batch_size 64 \
    --attack_steps 40 \
    --attack_lr 2 \
    --random_init