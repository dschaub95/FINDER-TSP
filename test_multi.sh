#!/bin/bash
MODELS=./test_models/*
# MODELS=./saved_models/tsp_2d/nrange_20_20/*

echo Select CUDA_VISIBLE_DEVICES
read cuda_device

echo Select search strategy -> 0:greedy, 1:beam_search 2:augmentation
read search_strategy

echo Select beam width
read beamwidth

for model_path in $MODELS
do
    # my_array=($(echo $model | tr "/" "\n"))
    IFS='/' read -ra my_array <<< "$model_path"
    model_name=${my_array[-1]}
    echo $model_name
    echo $model_path
    CUDA_VISIBLE_DEVICES=$cuda_device python test_greedy.py $model_name
    CUDA_VISIBLE_DEVICES=$cuda_device python test_beamsearch.py $model_name $beamwidth
done
