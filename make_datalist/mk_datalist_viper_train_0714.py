import os
import pickle

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]

if __name__ == '__main__':


    dir_A = '/srv/beegfs02/scratch/video_trans_lkn/data/liuka/RecycleGAN/Viper/data/recycle-gan/trainA/'
    A_paths = sorted(make_dataset(dir_A))

    save_A_path = '/scratch_net/minga/liuka/recycle_gan/Recycle-GAN/pickle_list/total_viper_a_train_list.pickle'

    with open(save_A_path, 'wb') as fp:
        pickle.dump(A_paths, fp)


    # dir_B = '/srv/glusterfs/liuka/Seq_GTAV/combined_dataset/val/img_png/'
    # dir_B = '/srv/beegfs02/scratch/video_trans_lkn/data/liuka/RecycleGAN/Viper/data/recycle-gan/trainB_ID/'
    # B_paths = sorted(make_dataset(dir_B))
    #
    # save_B_path = '/scratch_net/minga/liuka/recycle_gan/Recycle-GAN/pickle_list/total_viper_b_train_ID_list.pickle'
    #
    # with open(save_B_path, 'wb') as fp:
    #     pickle.dump(B_paths, fp)

    dir_B = '/srv/beegfs02/scratch/video_trans_lkn/data/liuka/RecycleGAN/Viper/data/recycle-gan/trainB/'
    B_paths = sorted(make_dataset(dir_B))

    save_B_path = '/scratch_net/minga/liuka/recycle_gan/Recycle-GAN/pickle_list/total_viper_b_train_list.pickle'

    with open(save_B_path, 'wb') as fp:
        pickle.dump(B_paths, fp)
