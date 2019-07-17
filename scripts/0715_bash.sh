#!/usr/bin/env bash

# screen -r 1807.pts-0.minga
# since 0714 17:06
/scratch_net/minga/liuka/recycle_gan/Recycle-GAN
cd /scratch_net/minga/liuka/recycle_gan/Recycle-GAN

# ongoing
# screen -r 1807.pts-0.minga
# continue at 0716 17:23, 48h
# not work
python train.py --dataroot /srv/glusterfs/liuka/RecycleGAN/Viper_data/Viper/data/recycle-gan \
--name 0715 --model recycle_gan  --which_model_netG resnet_6blocks --which_model_netP unet_256 \
--dataset_mode unaligned_triplet  --no_dropout --gpu 0 --identity 0  --pool_size 0 --batchSize 4 --max_dataset_size 29968 \
--loadSize 256 --input_nc 3 --output_nc 1 --display_freq 400 --continue_train --epoch_count 15




python train.py --dataroot /srv/glusterfs/liuka/RecycleGAN/Viper_data/Viper/data/recycle-gan \
--name 0717 --model recycle_GAN_v2  --which_model_netG resnet_6blocks --which_model_netP unet_256 \
--dataset_mode unaligned_triplet  --no_dropout --gpu 0 --identity 0  --pool_size 0 --batchSize 4 --max_dataset_size 29968 \
--loadSize 256 --input_nc 3 --output_nc 1 --display_freq 400



# todo ongoing
# screen -r 5269.pts-12.minga
# change to screen -r 24243.pts-0.minga  since 0715 12:44
# stop at 0716 10:46
# 24h continue at epoch 32
#since 0717 01:13
# continue at epoch 39
# stop since 0717 0827
python train.py --dataroot recycle-gan_day \
--name 0715_day --model recycle_gan  --which_model_netG resnet_6blocks --which_model_netP unet_256 \
--dataset_mode unaligned_triplet  --no_dropout --gpu 0 --identity 0  --pool_size 0 --batchSize 4 --max_dataset_size 29968 \
--loadSize 256 --input_nc 3 --output_nc 1 --display_freq 400 --continue_train --epoch_count 39

