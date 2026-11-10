import os
import random
import re

import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


class BaseDataSets(Dataset):
    """ACDC 数据集（TransUNet 预处理格式）。
    train: 读 ACDC_training_slices/*.h5 切片；val/test: 读 ACDC_training_volumes/*.h5 体。
    划分写死(标准 70/10/20)：train=patient031-100, val=021-030, test=001-020。
    h5 的 key 为 'image'/'label'，3 个前景类(RV/Myo/LV) + 背景 = 4 类。"""

    def __init__(self, base_dir=None, split='train', list_dir=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        train_ids, val_ids, test_ids = self._get_ids()
        if self.split.find('train') != -1:
            self.all_slices = os.listdir(self._base_dir + "/ACDC_training_slices")
            self.sample_list = []
            for ids in train_ids:
                new_data_list = list(filter(lambda x: re.match('{}.*'.format(ids), x) is not None, self.all_slices))
                self.sample_list.extend(new_data_list)
        elif self.split.find('val') != -1:
            self.all_volumes = os.listdir(self._base_dir + "/ACDC_training_volumes")
            self.sample_list = []
            for ids in val_ids:
                new_data_list = list(filter(lambda x: re.match('{}.*'.format(ids), x) is not None, self.all_volumes))
                self.sample_list.extend(new_data_list)
        elif self.split.find('test') != -1:
            self.all_volumes = os.listdir(self._base_dir + "/ACDC_training_volumes")
            self.sample_list = []
            for ids in test_ids:
                new_data_list = list(filter(lambda x: re.match('{}.*'.format(ids), x) is not None, self.all_volumes))
                self.sample_list.extend(new_data_list)
        print("total {} samples".format(len(self.sample_list)))

    def _get_ids(self):
        all_cases_set = ["patient{:0>3}".format(i) for i in range(1, 101)]
        testing_set = ["patient{:0>3}".format(i) for i in range(1, 21)]
        validation_set = ["patient{:0>3}".format(i) for i in range(21, 31)]
        training_set = [i for i in all_cases_set if i not in testing_set + validation_set]
        return [training_set, validation_set, testing_set]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/ACDC_training_slices/{}".format(case), 'r')
            image = h5f['image'][:]
            label = h5f['label'][:]
            sample = {'image': image, 'label': label}
            sample = self.transform(sample)
        else:
            h5f = h5py.File(self._base_dir + "/ACDC_training_volumes/{}".format(case), 'r')
            image = h5f['image'][:]
            label = h5f['label'][:]
            sample = {'image': image, 'label': label}
        sample["idx"] = idx
        sample['case_name'] = case.replace('.h5', '')
        return sample


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        assert (image.shape[0] == self.output_size[0]) and (image.shape[1] == self.output_size[1])
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {'image': image, 'label': label}
        return sample
