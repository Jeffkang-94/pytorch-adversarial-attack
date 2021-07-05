
for VAR in 1zAiPdXLPYkikxVnjXR8zcgEGer8HR3Ca 1iAwkv18spCYaVEOi7IGDDHpgY9EB-MUi 1uEQdXDIK4XPnm7ZkzLFtTrXb0WfAgi54 1xdocjfDQ88LzLjYfkIJVSL9ImzrzUXZB
do
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=$VAR" > /dev/null
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=$VAR" -o "$VAR.pth"
done

mkdir results
for VAR in ResNet WideResNet
do
    mkdir results/$VAR
    mkdir results/$VAR/clean
    mkdir results/$VAR/adv
done

mv 1zAiPdXLPYkikxVnjXR8zcgEGer8HR3Ca.pth best.pth
mv best.pth results/ResNet/clean
mv 1iAwkv18spCYaVEOi7IGDDHpgY9EB-MUi.pth best.pth
mv best.pth results/ResNet/adv

mv 1uEQdXDIK4XPnm7ZkzLFtTrXb0WfAgi54.pth best.pth
mv best.pth results/WideResNet/clean
mv 1xdocjfDQ88LzLjYfkIJVSL9ImzrzUXZB.pth best.pth
mv best.pth results/WideResNet/adv
rm cookie