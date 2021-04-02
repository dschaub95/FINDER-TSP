prompt="Training the model!"

CUDA_VISIBLE_DEVICES=gpu_id python train.py \
#    2>&1 | tee results/train_log.txt