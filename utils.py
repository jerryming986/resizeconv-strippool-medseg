import os
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom, label as cc_label
import torch.nn as nn
import SimpleITK as sitk


def keep_largest_component(mask):
    """只保留二值 mask(2D/3D)的最大连通域。用于测试后处理，去掉离群假阳块以稳住 HD95。
    Synapse 的 8 个器官(含左右肾分别标号)都应是单连通体，故每类保留最大块是安全的。"""
    if mask.sum() == 0:
        return mask
    labeled, num = cc_label(mask)
    if num <= 1:
        return mask
    counts = np.bincount(labeled.ravel())
    counts[0] = 0  # 背景不算
    largest = counts.argmax()
    return (labeled == largest).astype(mask.dtype)


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def update_confusion(stats, pred, gt):
    """累加二分类混淆矩阵（前景=1）。stats 为 [TP,FP,TN,FN] 的可变 list，pred/gt 为 {0,1} 数组。
    用全像素累加(micro)汇总，得到与 MALUNet/EGE-UNet 一致的全局 DSC/IoU/Acc/Sen/Spe。"""
    p = pred.astype(bool)
    g = gt.astype(bool)
    stats[0] += int(np.logical_and(p, g).sum())     # TP
    stats[1] += int(np.logical_and(p, ~g).sum())    # FP
    stats[2] += int(np.logical_and(~p, ~g).sum())   # TN
    stats[3] += int(np.logical_and(~p, g).sum())    # FN
    return stats


def isic_metrics_from_confusion(stats):
    """由 [TP,FP,TN,FN] 计算皮肤病灶分割常用指标（前景类）。"""
    tp, fp, tn, fn = stats
    eps = 1e-7
    return {
        'DSC': 2 * tp / (2 * tp + fp + fn + eps),       # Dice / F1
        'IoU': tp / (tp + fp + fn + eps),               # Jaccard / mIoU(前景)
        'Acc': (tp + tn) / (tp + tn + fp + fn + eps),   # 像素准确率
        'Sen': tp / (tp + fn + eps),                    # 敏感度 / 召回
        'Spe': tn / (tn + fp + eps),                    # 特异度
    }


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    device = next(net.parameters()).device
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().to(device)
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().to(device)
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()

    # 测试后处理：每类只保留最大连通域，去掉远处假阳块以降低 HD95（不影响训练）。
    # 默认关闭；设环境变量 LCC=1 开启，便于做「带/不带后处理」对比。
    if os.environ.get('LCC', '0') == '1':
        pred_clean = np.zeros_like(prediction)
        for i in range(1, classes):
            comp = keep_largest_component(prediction == i)
            pred_clean[comp > 0] = i
        prediction = pred_clean

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list
