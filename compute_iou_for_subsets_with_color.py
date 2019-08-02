import numpy as np
import argparse
import json
from PIL import Image
from os.path import join
import pickle
import os
import cv2


Dict = {}
Dict['Day'] = [1, 2, 3, 29,30, 31, 32, 41, 42, 43, 44]
Dict['Night'] = [5, 6 ,7, 33, 34, 35, 36, 45, 46, 47]
Dict['Rain'] = [18, 19, 20, 37, 38, 39, 40]
Dict['Snow'] = [21, 22, 23, 24, 25, 26, 27, 28]
Dict['Sunset'] = [4, 8, 9, 10,  11,  12, 13, 14, 15, 16, 17]
Dict_key_list = ['Day','Night','Rain','Snow','Sunset']

# some subsets are in bad condition, we do not want to take them into consideration
# Uncounted_list = range(35,48)
# Uncounted_list = []
# Uncounted_list = range(35,48)

cmap = np.array([(180, 130, 70), (153, 153, 153), (35, 182, 87), (70, 70, 70), (30, 170, 250),
                 (0, 0, 0), (153, 153, 190), (35, 142, 35), (0, 220, 220), (20, 20, 150), (70, 0, 0),
                 (0, 74, 111), (60, 20, 220), (152, 251, 152), (142, 0, 0), (81, 0, 81), (232, 35, 244),
                 (21, 0, 81), (128, 64, 128), (153, 153, 173), (100, 180, 180), (100, 80, 0), (100, 60, 0),
                 (153, 153, 168)],
dtype=np.uint8)


cmap_list = [(180, 130, 70), (153, 153, 153), (35, 182, 87), (70, 70, 70), (30, 170, 250),
                 (0, 0, 0), (153, 153, 190), (35, 142, 35), (0, 220, 220), (20, 20, 150), (70, 0, 0),
                 (0, 74, 111), (60, 20, 220), (152, 251, 152), (142, 0, 0), (81, 0, 81), (232, 35, 244),
                 (21, 0, 81), (128, 64, 128), (153, 153, 173), (100, 180, 180), (100, 80, 0), (100, 60, 0),
                 (153, 153, 168)]
def colormap2id(img_path):
    """

    :param img_path: the image path of the fake colored label
    :return: the label ID
    """
    img = cv2.imread(img_path)
    img_h = img.shape[0]
    img_w = img.shape[1]
    new_img = np.zeros((img_h, img_w))
    # sum_dist = 0

    for h_idx in range(img_h):
        for w_idx in range(img_w):
            tempt_val = img[h_idx, w_idx,:]
            cmap_dist = np.zeros(24)
            for c_idx, cmap_val in enumerate(cmap_list):
                # print('cmap_val')
                # print(cmap_val)
                # print('tempt_val')
                # print(tempt_val)
                cmap_dist[c_idx] = (tempt_val[0]-cmap_val[0])**2 + (tempt_val[1]-cmap_val[1])**2 \
                                   + (tempt_val[2]-cmap_val[2])**2
            # print(np.min(cmap_dist))
            new_img[h_idx,w_idx] = int(cmap_dist.argmin())
            # sum_dist += np.min(cmap_dist)

    # print(np.unique(new_img))

    # print(img_name)
    # print(sum_dist)
    return new_img.astype(np.int64)

# def colormap2id_exact(img_path):
#     """
#
#     :param img_path: the image path of the real colored label
#     :return: the label ID
#     """
#
#     img = cv2.imread(img_path)
#     img_h = img.shape[0]
#     img_w = img.shape[1]
#     new_img = np.zeros((img_h, img_w))
#     for idx, color in enumerate(cmap):
#         array = img == color
#         array_idx = array[:, :, 0] * array[:, :, 1] * array[:, :, 2]
#         new_img[array_idx] = idx
#
#     # print(np.unique(new_img))
#
#     # print(img_name)
#     return new_img.astype(np.int64)


def colormap2id_np(img_path):
    """

    :param img_path: the image path of the fake colored label
    :return: the label ID
    """
    img = cv2.imread(img_path)

    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_NEAREST)
    # print(img.shape)
    # print(img[0,0,:])
    img_h = img.shape[0]
    img_w = img.shape[1]
    new_img = np.zeros((img_h, img_w))
    # sum_dist = 0

    img_dis_array = np.zeros((img_h, img_w, 24))

    for c_idx, cmap_val in enumerate(cmap_list):
        img_dis_array[:,:,c_idx] = (img[:,:,0] - cmap_val[0]*np.ones([img_h,img_w]))**2 + (img[:,:,1] - cmap_val[1]*np.ones([img_h,img_w]))**2  + (img[:,:,2] - cmap_val[2]*np.ones([img_h,img_w]))**2

    new_img = np.argmin(img_dis_array,axis=2)

    return new_img.astype(np.int64)

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)


def result_stats(hist):
    acc_overall = np.diag(hist).sum() / hist.sum() * 100
    acc_percls = np.diag(hist) / (hist.sum(1) + 1e-8) * 100
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-8) * 100
    freq = hist.sum(1) / hist.sum()
    fwIU = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc_overall, acc_percls, iu, fwIU



def compute_mIoU(gt_dir, pred_dir, root, uncounted_list, seq_len):
    """
    Compute IoU given the predicted colorized images and
    """

    # if int(seq_len) == 4:
    #     roo_gt =
    # elif int(seq_len) == 4:
    #     roo_gt =
    # else:
    #     roo_gt =

    if uncounted_list == 'no':
        Uncounted_list = []
    elif  uncounted_list == 'v2':
        Uncounted_list = range(40, 48)
    else:
        Uncounted_list = range(35, 48)
    print('Uncounted_list')
    print(Uncounted_list)
    num_classes = 24
    # print('Num classes', num_classes)
    hist = np.zeros((num_classes, num_classes))
    with open(gt_dir, 'rb') as file:
        gt_imgs_dic = pickle.load(file)

    with open(pred_dir, 'rb') as file:
        pred_imgs_dic = pickle.load(file)

    gt_imgs_overall = []
    pred_imgs_overall = []
    for Dict_key in Dict_key_list:
        gt_imgs_overall += sorted(gt_imgs_dic[Dict_key])
        pred_imgs_overall += sorted(pred_imgs_dic[Dict_key])

    print('overal_all')
    gt_imgs = gt_imgs_overall
    pred_imgs = pred_imgs_overall
    # print(len(gt_imgs))
    # print(len(pred_imgs))
    # print(gt_imgs)
    for ind in range(len(gt_imgs)):
        # print(int(pred_imgs[ind].split('_')[0]))
        # if int(pred_imgs[ind].split('_')[0]) in Uncounted_list:
        #     print(int(pred_imgs[ind].split('_')[0]))

        if not int(pred_imgs[ind].split('_')[0]) in Uncounted_list:
            if os.path.exists(os.path.join(root, pred_imgs[ind])) and os.path.exists(os.path.join(root, gt_imgs[ind])):
                # pred = np.array(colormap2id(os.path.join(root, pred_imgs[ind])))
                # label = np.array(colormap2id(os.path.join(root, gt_imgs[ind])))
                # if len(label.flatten()) != len(pred.flatten()):
                #     print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()),
                #                                                                           len(pred.flatten()), gt_imgs[ind],
                #                                                                           pred_imgs[ind]))
                #     continue
                # hist += fast_hist(label.flatten(), pred.flatten(), num_classes)

                pred = np.array(colormap2id_np(os.path.join(root, pred_imgs[ind])))
                label = np.array(colormap2id_np(os.path.join(root, gt_imgs[ind])))
                if len(label.flatten()) != len(pred.flatten()):
                    print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()),
                                                                                          len(pred.flatten()),
                                                                                          gt_imgs[ind],
                                                                                          pred_imgs[ind]))
                    continue

                hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
                # try:
                #     pred = np.array(colormap2id_np(os.path.join(root, pred_imgs[ind])))
                #     label = np.array(colormap2id_np(os.path.join(root, gt_imgs[ind])))
                #     if len(label.flatten()) != len(pred.flatten()):
                #         print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()),
                #                                                                               len(pred.flatten()),
                #                                                                               gt_imgs[ind],
                #                                                                               pred_imgs[ind]))
                #         continue
                #
                #     hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
                # except:
                #     print(pred_imgs[ind])
                # if ind > 0 and ind % 10 == 0:
                #     print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100 * np.mean(per_class_iu(hist))))

    mIoUs = per_class_iu(hist)

    acc_overall, acc_percls, iu, fwIU = result_stats(hist)
    # for ind_class in range(num_classes):
    #     print('===>' + str(ind_class) + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    # print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
    print({'mIoU': ' {:0.2f}  fwIoU: {:0.2f} pixel acc: {:0.2f} per cls acc: {:0.2f}'.format(
        np.nanmean(iu), fwIU, acc_overall, np.nanmean(acc_percls))})


    for subset_indx in range(len(Dict_key_list)):
        hist = np.zeros((num_classes, num_classes))
        print(Dict_key_list[subset_indx])
        gt_imgs = sorted(gt_imgs_dic[Dict_key_list[subset_indx]])
        pred_imgs = sorted(pred_imgs_dic[Dict_key_list[subset_indx]])
        # print(len(gt_imgs))
        # print(len(pred_imgs))
        for ind in range(len(gt_imgs)):
            if not int(pred_imgs[ind].split('_')[0]) in Uncounted_list:
                if os.path.exists(os.path.join(root, pred_imgs[ind])) and os.path.exists(os.path.join(root, gt_imgs[ind])):

                    pred = np.array(colormap2id_np(os.path.join(root, pred_imgs[ind])))
                    label = np.array(colormap2id_np(os.path.join(root, gt_imgs[ind])))
                    if len(label.flatten()) != len(pred.flatten()):
                        print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()),
                                                                                              len(pred.flatten()),
                                                                                              gt_imgs[ind],
                                                                                              pred_imgs[ind]))
                        continue

                    hist += fast_hist(label.flatten(), pred.flatten(), num_classes)

                    # try:
                    #     pred = np.array(colormap2id_np(os.path.join(root, pred_imgs[ind])))
                    #     label = np.array(colormap2id_np(os.path.join(root, gt_imgs[ind])))
                    #     if len(label.flatten()) != len(pred.flatten()):
                    #         print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()),
                    #                                                                               len(pred.flatten()),
                    #                                                                               gt_imgs[ind],
                    #                                                                               pred_imgs[ind]))
                    #         continue
                    #     hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
                    # except:
                    #     print(pred_imgs[ind])
                    # if ind > 0 and ind % 10 == 0:
                    #     print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100*np.mean(per_class_iu(hist))))

        mIoUs = per_class_iu(hist)

        acc_overall, acc_percls, iu, fwIU = result_stats(hist)
        # for ind_class in range(num_classes):
        #     print('===>' + str(ind_class) + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
        # print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
        print({'mIoU':' {:0.2f}  fwIoU: {:0.2f} pixel acc: {:0.2f} per cls acc: {:0.2f}'.format(
                np.nanmean(iu), fwIU, acc_overall, np.nanmean(acc_percls))})



    return mIoUs, fwIU, acc_overall, np.nanmean(acc_percls)

def main(args):

    gt_dir_total = '/scratch_net/minga/liuka/cyclegan/pickle_list/test_mIoU_with_scenario/test_total_ori_recycle_30_gtav_b_real_dict_0731.pickle'
    pred_dir_total = '/scratch_net/minga/liuka/cyclegan/pickle_list/test_mIoU_with_scenario/test_total_ori_recycle_30_gtav_b_fake_dict_0731.pickle'

    # gt_dir_seq = '/scratch_net/minga/liuka/cyclegan/pickle_list/test_mIoU_with_scenario/test_seq_gtav_a_real_dict_0704.pickle'
    # pred_dir_seq = '/scratch_net/minga/liuka/cyclegan/pickle_list/test_mIoU_with_scenario/test_seq_gtav_a_fake_dict_0704.pickle'
    # print(args.root)
    #
    # print(args.mode )
    args.seq_len = 30
    gt_dir = gt_dir_total
    pred_dir = pred_dir_total

    # print(gt_dir)
    # print(pred_dir)


    compute_mIoU(gt_dir, pred_dir, args.root, args.uncounted_list, args.seq_len)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str,
                        default='/srv/beegfs02/scratch/video_trans_lkn/data/liuka/ori_recycle_gan_results/0731_results_on_val/0725/test_16/images/',
                        help='directory which stores GTAV val pred images')
    parser.add_argument('--mode', type=str,
                        default='total',
                        help='directory which stores GTAV val pred images')

    parser.add_argument('--seq_len', type=int,
                        default=30,
                        help='directory which stores GTAV val pred images')

    parser.add_argument('--uncounted_list', type=str,
                        default='no',
                        help='directory which stores GTAV val pred images')

    args = parser.parse_args()
    main(args)
