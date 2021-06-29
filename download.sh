
for VAR in 1zAiPdXLPYkikxVnjXR8zcgEGer8HR3Ca 1iAwkv18spCYaVEOi7IGDDHpgY9EB 1uEQdXDIK4XPnm7ZkzLFtTrXb0WfAgi54 1xdocjfDQ88LzLjYfkIJVSL9ImzrzUXZB
do
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=$VAR" > /dev/null
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=$VAR" -o "$VAR.pth"
done

mkdir checkpoint
mkdir checkpoint/ResNet
mkdir checkpoint/WideResNet
mv 1zAiPdXLPYkikxVnjXR8zcgEGer8HR3Ca.pth clean.pth
mv 1iAwkv18spCYaVEOi7IGDDHpgY9EB.pth adv.pth
mv clean.pth adv.pth checkpoint/ResNet
mv 1uEQdXDIK4XPnm7ZkzLFtTrXb0WfAgi54.pth clean.pth
mv 1xdocjfDQ88LzLjYfkIJVSL9ImzrzUXZB.pth adv.pth
mv clean.pth adv.pth checkpoint/WideResNet
rm cookie