import os
import pickle

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

Dict = {}
Dict['Day'] = [1, 2, 3, 29,30, 31, 32, 41, 42, 43, 44]
Dict['Night'] = [5, 6 ,7, 33, 34, 35, 36, 45, 46, 47]
Dict['Rain'] = [18, 19, 20, 37, 38, 39, 40]
Dict['Snow'] = [21, 22, 23, 24, 25, 26, 27, 28]
Dict['Sunset'] = [4, 8, 9, 10,  11,  12, 13, 14, 15, 16, 17]
Dict_key_list = ['Day','Night','Rain','Snow','Sunset']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir,key_indx, max_dataset_size=float("inf")):
    images_real = []
    images_fake = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            subset_num = int(fname.split('_')[0])
            if subset_num in Dict[Dict_key_list[key_indx]]:
                if 'real_B.png' in fname:
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        images_real.append(fname)
                elif 'fake_B.png' in fname:
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        images_fake.append(fname)

    print(len(images_real))
    print(len(images_fake))

    return images_real[:min(max_dataset_size, len(images_real))], images_fake[:min(max_dataset_size, len(images_fake))]

def build_dict_for_subsets(dir):
    Dict_subsets_real = {}
    Dict_subsets_fake = {}
    for key_indx in range(len(Dict_key_list)):
        tempt_real, tempt_fake = make_dataset(dir, key_indx)
        Dict_subsets_real[Dict_key_list[key_indx]] = tempt_real
        Dict_subsets_fake[Dict_key_list[key_indx]] = tempt_fake

    return Dict_subsets_real,Dict_subsets_fake


if __name__ == '__main__':
    # dir_A = '/scratch_net/minga/liuka/cyclegan/pytorch-CycleGAN-and-pix2pix/recyclegan_result_0627/0626_recycle_seq_GTAV_ID/test_latest/images/'


    # dir_A = '/scratch_net/minga/liuka/cyclegan/pytorch-CycleGAN-and-pix2pix/recycle_results/recyclegan_result_0629/0627_total_gtav_cyclegan/test_latest/images/'
    #
    #
    # save_A_real_path = '/scratch_net/minga/liuka/cyclegan/pickle_list/test_mIoU_with_scenario/test_total_gtav_a_real_dict_0704.pickle'
    # save_A_fake_path = '/scratch_net/minga/liuka/cyclegan/pickle_list/test_mIoU_with_scenario/test_total_gtav_a_fake_dict_0704.pickle'
    #
    #
    # A_real_paths, A_fake_paths = build_dict_for_subsets(dir_A)
    # with open(save_A_real_path, 'wb') as fp:
    #     pickle.dump(A_real_paths, fp)
    #
    # with open(save_A_fake_path, 'wb') as fp:
    #     pickle.dump(A_fake_paths, fp)



    # dir_A = '/scratch_net/minga/liuka/cyclegan/pytorch-CycleGAN-and-pix2pix/recycle_results/recyclegan_result_0629/0626_recycle_seq_GTAV_ID/test_latest/images'
    #
    #
    # save_A_real_path = '/scratch_net/minga/liuka/cyclegan/pickle_list/test_mIoU_with_scenario/test_seq_gtav_a_real_dict_0704.pickle'
    # save_A_fake_path = '/scratch_net/minga/liuka/cyclegan/pickle_list/test_mIoU_with_scenario/test_seq_gtav_a_fake_dict_0704.pickle'
    #
    #
    # A_real_paths, A_fake_paths = build_dict_for_subsets(dir_A)
    # with open(save_A_real_path, 'wb') as fp:
    #     pickle.dump(A_real_paths, fp)
    #
    # with open(save_A_fake_path, 'wb') as fp:
    #     pickle.dump(A_fake_paths, fp)


    # dir_A = '/scratch_net/minga/liuka/cyclegan/pytorch-CycleGAN-and-pix2pix/recycle_results/recyclegan_result_0709/0630_recycle_freeze_seq_GTAV_ID/test_45/images/'
    #
    #
    # save_A_real_path = '/scratch_net/minga/liuka/cyclegan/pickle_list/test_mIoU_with_scenario/test_seq_4_gtav_a_real_dict_0709.pickle'
    # save_A_fake_path = '/scratch_net/minga/liuka/cyclegan/pickle_list/test_mIoU_with_scenario/test_seq_4_gtav_a_fake_dict_0709.pickle'
    #
    #
    # A_real_paths, A_fake_paths = build_dict_for_subsets(dir_A)
    # with open(save_A_real_path, 'wb') as fp:
    #     pickle.dump(A_real_paths, fp)
    #
    # with open(save_A_fake_path, 'wb') as fp:
    #     pickle.dump(A_fake_paths, fp)


    dir_A = '/srv/beegfs02/scratch/video_trans_lkn/data/liuka/ori_recycle_gan_results/0731_results_on_val/0725/test_16/images/'


    save_A_real_path = '/scratch_net/minga/liuka/cyclegan/pickle_list/test_mIoU_with_scenario/test_total_ori_recycle_30_gtav_b_real_dict_0731.pickle'
    save_A_fake_path = '/scratch_net/minga/liuka/cyclegan/pickle_list/test_mIoU_with_scenario/test_total_ori_recycle_30_gtav_b_fake_dict_0731.pickle'


    A_real_paths, A_fake_paths = build_dict_for_subsets(dir_A)
    with open(save_A_real_path, 'wb') as fp:
        pickle.dump(A_real_paths, fp)

    with open(save_A_fake_path, 'wb') as fp:
        pickle.dump(A_fake_paths, fp)
