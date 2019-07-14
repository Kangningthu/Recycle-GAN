#!/usr/bin/env bash
python train.py --dataroot /srv/glusterfs/liuka/RecycleGAN/Viper_data/Viper/data/recycle-gan \
--name 0715 --model recycle_gan  --which_model_netG resnet_6blocks --which_model_netP prediction \
--dataset_mode unaligned_triplet  --no_dropout --gpu 0 --identity 0  --pool_size 0 --batchSize 8 --max_dataset_size 29968 \
--loadSize 256 --input_nc 3 --output_nc 1

