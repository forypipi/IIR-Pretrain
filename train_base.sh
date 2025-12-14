#!/bin/sh
cd /mnt/nasv3/fuourui/DeepLearning/MyModels/BAM

PARTITION=Segmentation
dataset=pascal
exp_name=split0

arch=PSPNet
net=resnet101

exp_dir=exp/${dataset}/${arch}/${exp_name}/${net}_v4
snapshot_dir=${exp_dir}/snapshot
result_dir=${exp_dir}/result
config=config/${dataset}/${exp_name}/${net}_base.yaml
mkdir -p ${exp_dir}
chmod 777 -R ${exp_dir}
mkdir -p ${snapshot_dir} ${result_dir}
chmod 777 -R ${snapshot_dir} ${result_dir}
now=$(date +"%Y%m%d_%H%M%S")
cp train_base.sh train_base.py ${config} ${exp_dir}

echo ${arch}
echo ${config}

python train_base.py --config=${config} --arch=${arch} 2>&1 | tee ${result_dir}/train-$now.logW