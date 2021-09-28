#!/bin/bash
FILES=./train_configs/multi_train/*

echo Select CUDA_VISIBLE_DEVICES
read cuda_device

for file in $FILES
do
    CUDA_VISIBLE_DEVICES=$cuda_device python train.py $file
done


# echo Start index?
# read idx

# i=0

# for file in $FILES
# do
#     (( i++ ))
#     [ $i -lt $idx ] && continue
#     # CUDA_VISIBLE_DEVICES=$cuda_device python train.py $file
#     echo $file
# done
# echo $i