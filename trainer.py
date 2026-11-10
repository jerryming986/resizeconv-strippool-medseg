import copy
import logging
import math
import os
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from utils import DiceLoss


# EMA（权重指数滑动平均）配置：对训练后段的权重做滑动平均，平滑「只测最后一个 epoch」带来的抖动。
#   USE_EMA=0 关闭（退回原行为）；EMA_DECAY 越接近 1，平滑窗口越长。
#   0.9998 ≈ 平滑最近 ~5000 次更新（本配置下约后 50+ epoch）。
USE_EMA = os.environ.get('USE_EMA', '1') != '0'
EMA_DECAY = float(os.environ.get('EMA_DECAY', '0.9998'))

# BIM 微调旋钮（仅影响优化/调度，不改结构 -> 不影响严格加载）：
#   BIM_LR_MULT    BIM 分支参数(bim/norm_bim/gamma_bim)学习率相对 base_lr 的倍率（默认 1.0；想让 BIM 学快点设 2~3）。
#   WARMUP_EPOCHS  线性 warmup 的 epoch 数（默认 20；想缩短设 10~15）。
BIM_LR_MULT = float(os.environ.get('BIM_LR_MULT', '1.0'))
WARMUP_EPOCHS = int(os.environ.get('WARMUP_EPOCHS', '20'))


class ModelEMA:
    """模型权重的指数滑动平均（timm 风格），用于稳定最终模型、削弱单 epoch 抖动。

    维护一份影子权重 self.ema：每步 ema = d·ema + (1-d)·model。
    采用自适应 decay：d = min(EMA_DECAY, (1+step)/(10+step))，使训练早期 d 较小
    （快速跟踪、不被随机初始化污染），随 step 增大逐步升到目标 decay。

    影子模型是 model 的深拷贝，state_dict 的 key 与 model 完全一致，
    可被 test.py 直接加载（用 CKPT=ema.pth 指定）。
    """
    def __init__(self, model, decay=0.9998):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        self.num_updates = 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        d = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        msd = model.state_dict()
        for k, ema_v in self.ema.state_dict().items():
            model_v = msd[k].detach()
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(d).add_(model_v, alpha=1 - d)   # 浮点权重做滑动平均
            else:
                ema_v.copy_(model_v)                        # 整型 buffer 直接拷贝

    def save(self, path):
        torch.save(self.ema.state_dict(), path)


@torch.no_grad()
def log_bim_status(model, probe_image, writer, iter_num):
    """监控 BIM 模块的「激活程度」，每个 epoch 记一次（无 BIM 时直接返回，零开销）：
      (1) |gamma_bim| —— 学到的门控强度（初值 0 或 BIM_GAMMA0，越大说明 BIM 介入越深）；
      (2) act_ratio  —— BIM 残差分支相对主干的贡献比 ||γ·BIM(LN(x))|| / ||x||，
          用一次探针前向（eval 关 DropPath）实测，是「激活程度」最直接的度量。
    指标打到 tensorboard(bim/*) 和日志，便于判断「热启动 + 更高学习率」是否真把 BIM 用起来了。
    """
    blocks = [m for _, m in model.named_modules() if getattr(m, 'use_bim', False)]
    if not blocks:
        return
    gabs = [m.gamma_bim.detach().abs().mean().item() for m in blocks]
    for m in blocks:
        m.record_act = True
    was_training = model.training
    model.eval()
    model(probe_image)
    model.train(was_training)
    ratios = []
    for m in blocks:
        m.record_act = False
        if m.last_act_ratio is not None:
            ratios.append(m.last_act_ratio)
    g_mean, g_max = sum(gabs) / len(gabs), max(gabs)
    r_mean = sum(ratios) / len(ratios) if ratios else 0.0
    r_max = max(ratios) if ratios else 0.0
    writer.add_scalar('bim/gamma_abs_mean', g_mean, iter_num)
    writer.add_scalar('bim/gamma_abs_max', g_max, iter_num)
    writer.add_scalar('bim/act_ratio_mean', r_mean, iter_num)
    writer.add_scalar('bim/act_ratio_max', r_max, iter_num)
    logging.info('BIM monitor: |gamma_bim| mean=%.4e max=%.4e | act_ratio mean=%.4f max=%.4f (%d blocks)'
                 % (g_mean, g_max, r_mean, r_max, len(blocks)))


def trainer_synapse(args, model, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # 数据集切换：ACDC 用专属 .h5 加载器(BaseDataSets)，其余沿用 Synapse_dataset。
    if args.dataset == 'ACDC':
        from datasets.dataset_acdc import BaseDataSets, RandomGenerator
        db_train = BaseDataSets(base_dir=args.root_path, split="train",
                                transform=transforms.Compose(
                                    [RandomGenerator([args.img_size, args.img_size])]))
    else:
        from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
        db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                                   transform=transforms.Compose(
                                       [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    train_loader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True,
                              worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    # 取未包 DataParallel 的底层模型：EMA 从它建立、向它取参，保存的 state_dict 无 'module.' 前缀，
    # 与 test.py（加载未包装模型）的 key 一致。
    model_without_ddp = model.module if isinstance(model, nn.DataParallel) else model
    ema = ModelEMA(model_without_ddp, decay=EMA_DECAY) if USE_EMA else None
    model.train()
    # ACDC 标准：0.5·CE + 0.5·Dice，且 CE 忽略 label=4 的 padding 像素；Synapse：0.4·CE + 0.6·Dice。
    if args.dataset == 'ACDC':
        ce_loss = CrossEntropyLoss(ignore_index=4)
        ce_w, dice_w = 0.5, 0.5
    else:
        ce_loss = CrossEntropyLoss()
        ce_w, dice_w = 0.4, 0.6
    dice_loss = DiceLoss(num_classes)
    # E1（实验）：ConvNeXt 是 AdamW 原生设计，改用 AdamW 微调配方替代 Swin-Unet 沿用的 SGD。
    # 按 timm/ConvNeXt 惯例：1 维参数(LayerNorm 权重/偏置/LayerScale gamma)不施加 weight decay。
    # 注意：AdamW 的 base_lr 要比 SGD 的 0.01 小得多，命令行请传 --base_lr 3e-4（沿用 0.01 会发散）。
    # 再把 BIM 分支参数(bim/norm_bim/gamma_bim)单独成组，挂更高学习率倍率 lr_scale=BIM_LR_MULT，
    # 让这个从头训练的新模块学得比已收敛的预训练主干快一点（仅 USE_BIM 时存在这些参数）。
    def _is_bim(name):   # 按路径分量匹配，与嵌套深度无关：子模块 bim/norm_bim、参数 gamma_bim
        parts = name.split('.')
        return ('bim' in parts) or ('norm_bim' in parts) or ('gamma_bim' in parts)
    decay, no_decay, bim_decay, bim_no_decay = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_nd = (p.ndim <= 1 or name.endswith('.bias'))
        if _is_bim(name):
            (bim_no_decay if is_nd else bim_decay).append(p)
        else:
            (no_decay if is_nd else decay).append(p)
    param_groups = [
        {'params': decay,    'weight_decay': 0.05, 'lr_scale': 1.0},
        {'params': no_decay, 'weight_decay': 0.0,  'lr_scale': 1.0},
    ]
    if bim_decay or bim_no_decay:
        param_groups += [
            {'params': bim_decay,    'weight_decay': 0.05, 'lr_scale': BIM_LR_MULT},
            {'params': bim_no_decay, 'weight_decay': 0.0,  'lr_scale': BIM_LR_MULT},
        ]
    optimizer = optim.AdamW(param_groups, lr=base_lr)
    n_bim = sum(p.numel() for p in bim_decay + bim_no_decay)
    logging.info("optimizer: base groups + BIM group (lr_mult={}, {} params)".format(BIM_LR_MULT, n_bim)
                 if n_bim else "optimizer: base groups only (no BIM params)")
    # 关闭混合精度（AMP）：BIM 的协方差 + 四阶连乘在 fp16 下会溢出成 Inf/NaN，
    # 改为全程 fp32 训练以保证数值稳定（4090 fp32 仍然很快）。
    use_amp = False
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(train_loader)  # max_epoch = max_iterations // len(trainloader) + 1
    # E1：AdamW 配线性 warmup（默认 20 epoch，可由 WARMUP_EPOCHS 调短），之后 cosine 衰减到 0
    warmup_epochs = WARMUP_EPOCHS
    warmup_iters = max(1, warmup_epochs * len(train_loader))
    logging.info("{} iterations per epoch. {} max iterations ".format(len(train_loader), max_iterations))
    logging.info("EMA: {}".format("enabled, decay={}".format(EMA_DECAY) if ema is not None else "disabled"))
    iterator = tqdm(range(max_epoch), ncols=70)
    save_interval = 50  # 每隔多少 epoch 存一次 checkpoint
    probe_image = None   # 固定一小批图，供每 epoch 的 BIM 激活监控做探针前向
    for epoch_num in iterator:
        model.train()
        batch_dice_loss = 0
        batch_ce_loss = 0
        for i_batch, sampled_batch in tqdm(enumerate(train_loader), desc=f"Train: {epoch_num}", total=len(train_loader),
                                           leave=False):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            if probe_image is None:
                probe_image = image_batch[:4].detach().clone()   # 固定探针批，监控 BIM 激活程度
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(image_batch)
                loss_ce = ce_loss(outputs, label_batch[:].long())
                loss_dice = dice_loss(outputs, label_batch, softmax=True)
                loss = ce_w * loss_ce + dice_w * loss_dice
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model_without_ddp)   # 每步更新 EMA 影子权重
            # E1：线性 warmup -> cosine 衰减（替换原 poly schedule）
            if iter_num < warmup_iters:
                lr_ = base_lr * (iter_num + 1) / warmup_iters
            else:
                progress = (iter_num - warmup_iters) / max(1, max_iterations - warmup_iters)
                lr_ = 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_ * param_group.get('lr_scale', 1.0)   # BIM 组按倍率提速

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            # logging.info('Train: iteration : %d/%d, lr : %f, loss : %f, loss_ce: %f, loss_dice: %f' % (
            #     iter_num, epoch_num, lr_, loss.item(), loss_ce.item(), loss_dice.item()))
            batch_dice_loss += loss_dice.item()
            batch_ce_loss += loss_ce.item()
            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
        batch_ce_loss /= len(train_loader)
        batch_dice_loss /= len(train_loader)
        batch_loss = 0.4 * batch_ce_loss + 0.6 * batch_dice_loss
        logging.info('Train epoch: %d : loss : %f, loss_ce: %f, loss_dice: %f' % (
            epoch_num, batch_loss, batch_ce_loss, batch_dice_loss))
        if probe_image is not None:
            log_bim_status(model_without_ddp, probe_image, writer, iter_num)   # 监控 BIM 激活程度
        # Synapse 为 18 训练 / 12 测试的官方协议，无验证集：不做验证，按固定 epoch 训练，
        # 定期保存 checkpoint；训练结束后用 test.py 在 12 例测试集上评测 Dice/HD95。
        # 最终 epoch 存为 epoch_{max_epoch-1}.pth，与 test.py 的默认查找名一致。
        if (epoch_num + 1) % save_interval == 0 or epoch_num == max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model_without_ddp.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            # EMA 影子权重始终存为 ema.pth（每次覆盖，保留最新）；测试用 CKPT=ema.pth 加载。
            if ema is not None:
                ema_path = os.path.join(snapshot_path, 'ema.pth')
                ema.save(ema_path)
                logging.info("save EMA model to {}".format(ema_path))

    writer.close()
    return "Training Finished!"
