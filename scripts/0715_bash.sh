#!/usr/bin/env bash

# screen -r 1807.pts-0.minga
# since 0714 17:06
/scratch_net/minga/liuka/recycle_gan/Recycle-GAN


python train.py --dataroot /srv/glusterfs/liuka/RecycleGAN/Viper_data/Viper/data/recycle-gan \
--name 0715 --model recycle_gan  --which_model_netG resnet_6blocks --which_model_netP unet_256 \
--dataset_mode unaligned_triplet  --no_dropout --gpu 0 --identity 0  --pool_size 0 --batchSize 4 --max_dataset_size 29968 \
--loadSize 256 --input_nc 3 --output_nc 3

