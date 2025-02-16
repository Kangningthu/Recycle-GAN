###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path
import pickle

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

def make_dataset_ID(dir, max_dataset_size=float("inf")):
    if 'day' in dir:
        if str(dir).endswith('trainA'):
            # with open('/scratch_net/minga/liuka/cyclegan/total_gtav_a_list.pickle', 'rb') as file:
            with open('/scratch_net/minga/liuka/recycle_gan/Recycle-GAN/pickle_list/total_viper_a_day_train_list.pickle',
                      'rb') as file:
                images = pickle.load(file)
        elif str(dir).endswith('trainB'):
            with open('/scratch_net/minga/liuka/recycle_gan/Recycle-GAN/pickle_list/total_viper_b_day_train_ID_list.pickle',
                      'rb') as file:
                images = pickle.load(file)

        elif str(dir).endswith('testB'):
            with open('/scratch_net/minga/liuka/cyclegan/pickle_list/total_gtav_a_test_list.pickle', 'rb') as file:
            # with open('/scratch_net/minga/liuka/cyclegan/pickle_list/total_gtav_a_test_ID_list.pickle', 'rb') as file:
                images = pickle.load(file)
        elif str(dir).endswith('testA'):
            with open('/scratch_net/minga/liuka/cyclegan/pickle_list/total_gtav_b_test_list.pickle', 'rb') as file:
                images = pickle.load(file)



    else:
        if str(dir).endswith('trainA'):
            # with open('/scratch_net/minga/liuka/cyclegan/total_gtav_a_list.pickle', 'rb') as file:
            with open('/scratch_net/minga/liuka/recycle_gan/Recycle-GAN/pickle_list/total_viper_a_train_list.pickle', 'rb') as file:
                images = pickle.load(file)
        elif str(dir).endswith('trainB'):
            with open('/scratch_net/minga/liuka/recycle_gan/Recycle-GAN/pickle_list/total_viper_b_train_ID_list.pickle', 'rb') as file:
                images = pickle.load(file)
        elif str(dir).endswith('testB'):
            with open('/scratch_net/minga/liuka/cyclegan/pickle_list/total_gtav_a_test_list.pickle', 'rb') as file:
                images = pickle.load(file)
        elif str(dir).endswith('testA'):
            with open('/scratch_net/minga/liuka/cyclegan/pickle_list/total_gtav_b_test_list.pickle', 'rb') as file:
                images = pickle.load(file)

        else:
            images = []
            assert os.path.isdir(dir), '%s is not a valid directory' % dir

            for root, _, fnames in sorted(os.walk(dir)):
                for fname in fnames:
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        images.append(path)

    return images[:min(max_dataset_size, len(images))]


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
