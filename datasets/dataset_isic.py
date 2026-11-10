import os
import random

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset

# ConvNeXt-T 编码器是 ImageNet 预训练的，故 RGB 输入按 ImageNet 统计做逐通道归一化
# （与从零训练的 MALUNet/EGE-UNet 用数据集自身 mean/std 不同；用预训练就该对齐预训练分布）。
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 掩码候选命名：ISIC2018 同名(N.png)，ISIC2017 为 <stem>_segmentation.png；大小写都兜底。
_MASK_SUFFIXES = ('_segmentation.png', '.png', '_Segmentation.png', '_mask.png')


def random_rot_flip(image, label):
    """image: HxWx3 (uint8)，label: HxW。在空间平面 (H,W) 上做 0/90/180/270 旋转 + 翻转。"""
    k = np.random.randint(0, 4)
    image = np.rot90(image, k, axes=(0, 1))
    label = np.rot90(label, k, axes=(0, 1))
    axis = np.random.randint(0, 2)  # 只翻空间轴(0/1)，不动通道轴
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, axes=(0, 1), order=1, reshape=False)  # 双线性插值 RGB
    label = ndimage.rotate(label, angle, axes=(0, 1), order=0, reshape=False)  # 最近邻保持二值
    return image, label


def _to_tensor(image, label, output_size):
    """HxWx3 uint8 + HxW{0,1} -> (3xHxW float, HxW long)。已是目标尺寸时不缩放。"""
    h, w = image.shape[0], image.shape[1]
    if h != output_size[0] or w != output_size[1]:
        image = zoom(image, (output_size[0] / h, output_size[1] / w, 1), order=1)
        label = zoom(label, (output_size[0] / h, output_size[1] / w), order=0)
    image = image.astype(np.float32) / 255.0
    image = (image - _IMAGENET_MEAN) / _IMAGENET_STD          # HxWx3
    image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()  # 3xHxW
    label = torch.from_numpy(np.ascontiguousarray(label)).long()
    return image, label


class RandomGenerator(object):
    """训练增广：随机旋转/翻转后归一化。与 Synapse/ACDC 的同名类接口一致(吃 {'image','label'} 出 {'image','label'})。"""

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        image, label = _to_tensor(image, label, self.output_size)
        return {'image': image, 'label': label}


class ISIC_dataset(Dataset):
    """ISIC2017 / ISIC2018 皮肤镜病灶二分类分割（MALUNet/EGE-UNet 的 7:3 目录布局）。

    目录结构：base_dir/<split>/images/*.{jpg,png} 与 base_dir/<split>/masks/*.png。
    其中 split='train' 为 70%，split='val' 为 30%（即测试集，沿用该协议的命名）。

    要点：
      - 图像统一转 RGB，已是 256x256；按 img_size 兜底缩放(双线性)。
      - 掩码为带抗锯齿灰阶的 PNG（非纯 0/1），以阈值 127 二值化 -> {0,1}，num_classes=2。
      - 训练走 transform 增广；val/test 仅归一化(不增广)，并附 case_name 便于落盘可视化。
    """

    def __init__(self, base_dir, split='train', img_size=256, transform=None):
        self.split = split
        self.img_size = img_size
        self.transform = transform
        self.img_dir = os.path.join(base_dir, split, 'images')
        self.msk_dir = os.path.join(base_dir, split, 'masks')
        assert os.path.isdir(self.img_dir), "image dir not found: {}".format(self.img_dir)
        assert os.path.isdir(self.msk_dir), "mask dir not found: {}".format(self.msk_dir)

        msk_files = set(os.listdir(self.msk_dir))
        self.sample_list = []
        for n in sorted(os.listdir(self.img_dir)):
            stem = n.rsplit('.', 1)[0]
            cands = [n] + [stem + suf for suf in _MASK_SUFFIXES]
            mk = next((c for c in cands if c in msk_files), None)
            if mk is None:
                raise FileNotFoundError("no mask for image {} in {}".format(n, self.msk_dir))
            self.sample_list.append((n, mk))
        print("ISIC {} split: {} image/mask pairs (img_size={})".format(split, len(self.sample_list), img_size))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        img_name, msk_name = self.sample_list[idx]
        image = np.array(Image.open(os.path.join(self.img_dir, img_name)).convert('RGB'))   # HxWx3 uint8
        mask = np.array(Image.open(os.path.join(self.msk_dir, msk_name)).convert('L'))       # HxW
        label = (mask >= 127).astype(np.uint8)                                               # 二值化 {0,1}

        sample = {'image': image, 'label': label}
        if self.split == 'train' and self.transform is not None:
            sample = self.transform(sample)
        else:
            img, lab = _to_tensor(image, label, (self.img_size, self.img_size))
            sample = {'image': img, 'label': lab}
        sample['case_name'] = img_name.rsplit('.', 1)[0]
        return sample
