import os
import pickle

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
Dict_train = {}
Dict_train['Day'] = [1, 2, 3, 4, 5, 44, 45, 46, 47, 48, 49, 50, 51, 65, 66, 67, 68, 69]
Dict_train['Night'] = [8, 9, 10, 11, 12, 13, 52, 53, 54, 55, 56, 57, 58, 70, 71, 72, 73, 74, 75, 76, 77]
Dict_train['Rain'] = [30, 31, 59, 60, 61, 62, 63, 64]
Dict_train['Snow'] = [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43]
Dict_train['Sunset'] = [6, 7, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

Train_Subset_Dict_key_list = ['Day', 'Night', 'Rain', 'Snow', 'Sunset']
Train_Subset_Dict = Dict_train


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir,sub_indx, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    subset_name = Train_Subset_Dict_key_list[sub_indx]
    subsets_list = Train_Subset_Dict[subset_name]

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                subset_num = int(fname.split('_')[0])
                if subset_num in subsets_list:
                    path = os.path.join(root, fname)
                    images.append(path)
    return images[:min(max_dataset_size, len(images))]

if __name__ == '__main__':
    # make day subset
    sub_indx = 0

    dir_A = '/srv/beegfs02/scratch/video_trans_lkn/data/liuka/RecycleGAN/Viper/data/recycle-gan/trainA/'
    A_paths = sorted(make_dataset(dir_A,sub_indx))

    save_A_path = '/scratch_net/minga/liuka/recycle_gan/Recycle-GAN/pickle_list/total_viper_a_day_train_list.pickle'

    with open(save_A_path, 'wb') as fp:
        pickle.dump(A_paths, fp)


    # dir_B = '/srv/glusterfs/liuka/Seq_GTAV/combined_dataset/val/img_png/'
    dir_B = '/srv/beegfs02/scratch/video_trans_lkn/data/liuka/RecycleGAN/Viper/data/recycle-gan/trainB_ID/'
    B_paths = sorted(make_dataset(dir_B,sub_indx))

    save_B_path = '/scratch_net/minga/liuka/recycle_gan/Recycle-GAN/pickle_list/total_viper_b_day_train_ID_list.pickle'

    with open(save_B_path, 'wb') as fp:
        pickle.dump(B_paths, fp)

