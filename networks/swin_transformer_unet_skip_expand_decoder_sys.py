import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# 兼容新旧 timm：新版 (>=0.9) 用 timm.layers，旧版用 timm.models.layers
try:
    from timm.layers import DropPath, trunc_normal_
except ImportError:
    from timm.models.layers import DropPath, trunc_normal_

# BIM 增强开关：默认 True，在每个 ConvNeXt block 主干（保留预训练 7×7 conv_dw）之上
# 叠加一个 BIM(AttModule) 残差增强分支（gamma_bim 初值 0 -> 起步严格等价预训练基线）。
# 设环境变量 USE_BIM=0 则退回纯 ConvNeXt 基线（消融对比），不含任何 BIM。
USE_BIM = os.environ.get('USE_BIM', '1') != '0'

# BIM 起步门控初值：默认 0.0（gamma_bim 全 0 -> 起步严格等价预训练基线，最安全）。
# 设 BIM_GAMMA0=0.01 等小正值可让 BIM 分支「热启动」——从 step 0 就有微弱贡献、更早学起来，
# 代价是一开始就轻微扰动已收敛的预训练主干（调大有发散/掉点风险，建议 0.01~0.05）。
# 注意：该值只决定初始化；checkpoint 里存的是训练后的 gamma_bim，故测试加载不受影响。
BIM_GAMMA0 = float(os.environ.get('BIM_GAMMA0', '0.0'))

# BIM 放置位置（仅 USE_BIM=1 时生效）：BIM_STAGES 指定哪些「编码器 stage 索引」启用 BIM，
# 逗号分隔，默认 "0,1,2,3"（全部 stage + 对应解码器镜像，即原行为，与已训 checkpoint 一致）。
# 例：BIM_STAGES=0,1 只在高分辨率轻量 stage 放 BIM（瘦身消融），低分辨率大通道 stage(384/768) 不放。
# 解码器 up_stage j 镜像编码器 stage (2-j)，其 BIM 开关随对应编码器 stage 联动。
# 注意：测试时 BIM_STAGES 必须与训练时一致（严格加载 state_dict）。
def _parse_bim_stages():
    raw = os.environ.get('BIM_STAGES', '0,1,2,3').replace(' ', '')
    return set(int(s) for s in raw.split(',') if s != '')

BIM_STAGES = _parse_bim_stages()

# BIM_DECODER=0 单独关闭解码器侧 BIM（做「仅编码器」消融）；默认 1=开（与已训 checkpoint 一致）。
BIM_DECODER = os.environ.get('BIM_DECODER', '1') != '0'

# 解码头切换（消融用）：
#   HEAD=convtr（默认）= 一发 4× ConvTranspose，与 75.96 的 SGD baseline 版一致；
#   HEAD=resize        = 两级 resize-conv + 从输入原图取全分辨率 skip，与服务器 83.91/84.37 版逐字一致。
# 两者参数量相近，resize 的 GFLOPs 更大（全分辨率 3×3 卷积）。测试时 HEAD 须与训练一致（严格加载）。
HEAD = os.environ.get('HEAD', 'convtr').lower()

# 注意力门控 skip（Attention U-Net, Oktay 2018）：USE_AG=1 在解码器每个 skip 上加一个
# 注意力门，用解码器特征门控 skip、抑制无关背景（减远端假阳性 -> 利好 HD95，突出小器官）。
# 参数很小（~0.2M）；per-channel 门控强度 γ 初值 0 -> 起步等价无门控 baseline。
# 默认关(0)，opt-in，与 BIM/HEAD 等开关独立组合。测试时 USE_AG 须与训练一致（严格加载）。
USE_AG = os.environ.get('USE_AG', '0') != '0'

# 条带池化上下文模块（Strip Pooling, Hou et al. CVPR2020 思路的轻量残差版）：USE_SP=1 在
# 768c@7×7 瓶颈处加一个分支——水平/垂直长条池化各自经深度 1D 卷积建模长程带状依赖、
# 广播回 HxW 后融合，补全 ConvNeXt 局部卷积缺乏的全局/细长结构上下文（利好大/细长器官 DSC）。
# per-channel 门控 γ 初值 0 -> 起步等价 baseline；约 +1.2M 参数。与 BIM/AG/HEAD 独立组合。
# 默认关(0)，opt-in。测试时 USE_SP 须与训练一致（严格加载）。
USE_SP = os.environ.get('USE_SP', '0') != '0'
# SP 注入形态：SP_MODE=add（默认，原行为）加法注入 `x + γ·ctx`（会凭空造远端假阳性，HD95 抽签）；
# SP_MODE=gate 乘法门 `x·(1+γ·tanh(ctx))`——作用在 x 上、背景 x≈0 处恒≈0，造不出远端假阳性
# （治 HD95 不稳），tanh 能升能降仍保留涨 DSC 的能力。两种 γ=0 都恒等。测试须与训练一致。
SP_MODE = os.environ.get('SP_MODE', 'add').lower()


class LayerNorm2d(nn.LayerNorm):
    """对 [B, C, H, W] 张量在通道维做 LayerNorm（channels-first）。
    自带实现以避免依赖特定 timm 版本；参数名 weight/bias 与 timm.LayerNorm2d 一致，
    可直接加载 ConvNeXt 预训练的 stem.1 / downsample.0 权重。"""
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class Cov_Block(nn.Module):
    """
    相似性感知加权模块（对应结构图 Similarity-aware Weighting 部分）
    核心：计算特征协方差矩阵，生成全局相似性权重，模仿大脑的特征匹配机制
    Args:
        dim: 输入通道数（双分支拼接后的总通道数）
    Input: 双分支拼接特征 [B, 2C, H, W]
    Output: 加权后的双分支拆分特征 [B, C, H, W] ×2
    """
    def __init__(self, dim):
        super().__init__()
        mid_dim = round(0.6 * dim)  # 隐藏层通道数，压缩 60% 降低计算量
        # MLP：将协方差矩阵映射为全局权重
        self.block = nn.Sequential(
            nn.Linear(dim, mid_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(mid_dim, 1),
        )

    def forward(self, x):
        shortcut = x.clone()  # 保留原始特征用于加权
        # 维度重排：将空间维度展平，适配协方差计算
        x = rearrange(x, 'B C H W -> B (H W) C')
        # L2 归一化，消除特征幅值差异对相似性计算的影响
        x = F.normalize(x, dim=-2)
        # 计算通道间协方差矩阵 [B, C, C]，衡量特征间的全局相似性
        cov = x.transpose(-2, -1) @ x
        # MLP 生成全局相似性权重 [B, 1, 1]
        cov = self.block(cov).unsqueeze_(-1)
        # 全局加权增强：原始特征 × (1 + 权重)，保留原始信息的同时增强相似特征
        out = shortcut * (1 + cov)
        # 拆分为两个分支，适配后续高阶交互
        return out.chunk(2, dim=1)


class AttModule(nn.Module):
    """
    Brain-Inspired Module (BIM) 核心实现
    模仿大脑视觉皮层的信息处理流程：多尺度编码 → 相似性匹配 → 高阶交互
    对应结构图 (c) 的完整流程：Encoding → Similarity-aware Weighting → High-Order Interaction
    Args:
        dim: 输入/输出通道数
        bias: 卷积是否带偏置
    Input: 输入特征 [B, C, H, W]
    Output: 增强特征 [B, C, H, W]（与输入尺寸完全一致，即插即用）
    """
    def __init__(self, dim, bias=True):
        super().__init__()
        self.dim = dim
        # 阶段1：双分支多尺度编码（对应结构图 Encoding 部分）
        # 1×1 卷积扩展通道为 2 倍，拆分为两个独立分支
        self.to_hidden = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        # 分支1：3×3 深度卷积，提取局部细节特征（对应大脑 V1 区的小感受野神经元）
        self.to_hidden_dw1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        # 分支2：9×9 深度卷积，提取全局结构特征（对应大脑 V2 区的大感受野神经元）
        self.to_hidden_dw2 = nn.Conv2d(dim, dim, kernel_size=9, stride=1, padding=4, groups=dim, bias=bias)

        # 阶段2：相似性感知加权（对应结构图 Similarity-aware Weighting 部分）
        self.cov1 = Cov_Block(dim * 2)

        # 阶段3：级联高阶交互（对应结构图 High-Order Interaction 部分）
        # 三级级联深度卷积 + 逐元素相乘，模拟神经元的级联激活与高阶信息整合
        self.dw1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.dw2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.dw3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        # 输出投影：1×1 卷积融合特征，恢复原始通道数
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def reset_bim_parameters(self):
        """方差保持初始化（关键）。
        BIM 的四阶级联乘法对初始化极度敏感：若用 std=0.02 的小初始化，
        各卷积输出 ~0.01，连乘四次后衰减到 ~1e-13，导致输出与梯度双双消失、
        空间混合层无法训练。改用 gain=1 的 kaiming 初始化（每个卷积近似保持
        输入方差），使连乘链中的激活与梯度都维持在 O(1)。
        注意：不动 Cov_Block 的 Linear（保持小初始化，让 cov≈0、(1+cov)≈1，
        通道加权在初始时接近恒等）。"""
        for conv in (self.to_hidden, self.to_hidden_dw1, self.to_hidden_dw2,
                     self.dw1, self.dw2, self.dw3, self.project_out):
            nn.init.kaiming_normal_(conv.weight, nonlinearity='linear')
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def forward(self, x):
        # 步骤1：双分支多尺度编码
        x_hidden = self.to_hidden(x)
        x1, x2 = x_hidden.chunk(2, dim=1)  # 拆分为两个分支
        x1 = self.to_hidden_dw1(x1)  # 3×3 局部特征
        x2 = self.to_hidden_dw2(x2)  # 9×9 全局特征

        # 步骤2：相似性感知全局加权
        x1, x2 = self.cov1(torch.cat([x1, x2], dim=1))

        # 步骤3：级联高阶交互（核心创新，模拟大脑神经元的级联激活）
        out = x1 * x2  # 一阶交互：局部 × 全局特征融合
        out = self.dw1(out) * x2  # 二阶交互：融合特征与全局特征交互
        out = self.dw2(out) * x1  # 三阶交互：融合特征与局部特征交互
        out = self.dw3(out) * x2  # 四阶交互：最终融合特征与全局特征交互

        # 步骤4：输出投影
        out = self.project_out(out)
        return out


class Mlp(nn.Module):
    """ConvNeXt block 内的倒置瓶颈 MLP（1×1 -> GELU -> 1×1，channels-last）。
    命名 fc1/fc2 与 timm ConvNeXt 一致，可直接加载预训练权重。"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvNeXtBIMBlock(nn.Module):
    r"""ConvNeXt 基本块（完整保留、加载预训练）+ 可选的 BIM 残差增强分支。

    主干 = 标准 ConvNeXt block（channels-first 输入 [B, C, H, W]）：
        shortcut = x
        x = conv_dw(x)                          # 7×7 深度卷积 ← 加载 ConvNeXt 预训练
        x = LayerNorm(x)                        # ← 加载 ConvNeXt 预训练
        x = MLP(x)  (fc1 -> GELU -> fc2)        # ← 加载 ConvNeXt 预训练
        x = gamma * x                           # LayerScale ← 加载 ConvNeXt 预训练
        x = shortcut + DropPath(x)

    BIM 增强（USE_BIM=True 时）在主干之上再叠加一个残差分支：
        x = x + DropPath(gamma_bim * BIM(LayerNorm(x)))
    其中 gamma_bim 初值为 0，故训练起步时 BIM 分支输出恒为 0，
    整个 block 严格等价于已验证可收敛的预训练 ConvNeXt 基线；gamma_bim 在
    训练中自行学习，BIM 作为可学习的增量增强逐步生效。

    主干子模块命名固定为 conv_dw / norm / mlp / gamma，与 timm ConvNeXt 的
    state_dict key 精确对齐（编码器直接命中、解码器镜像加载）。
    BIM 分支（norm_bim / bim / gamma_bim）为全新模块，从头训练。
    """
    def __init__(self, dim, drop_path=0., ls_init_value=1e-6, mlp_ratio=4, bias=True, use_bim=True):
        super().__init__()
        # 主干：标准 ConvNeXt block，全部加载预训练
        # 命名 conv_dw 对齐 timm，可直接加载 ConvNeXt 预训练的 7×7 深度卷积权重
        self.conv_dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(in_features=dim, hidden_features=int(mlp_ratio * dim), act_layer=nn.GELU)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value is not None else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # BIM 残差增强分支（gamma_bim 初值 0 -> 起步等价基线）；use_bim 由所在 stage 决定
        self.use_bim = use_bim
        if use_bim:
            self.norm_bim = LayerNorm2d(dim)
            self.bim = AttModule(dim, bias=bias)
            self.gamma_bim = nn.Parameter(torch.full((dim,), BIM_GAMMA0))
            # 激活监控用：训练侧把 record_act 置 True 跑一次探针前向后，从 last_act_ratio
            # 读「BIM 残差分支相对主干的贡献比 ||γ·BIM(LN(x))|| / ||x||」；平时 False，热路径零开销。
            self.record_act = False
            self.last_act_ratio = None

    def forward(self, x):
        # 主干：预训练 ConvNeXt block
        shortcut = x
        x = self.conv_dw(x)               # B, C, H, W
        x = x.permute(0, 2, 3, 1)         # B, H, W, C
        x = self.norm(x)
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2)         # B, C, H, W
        if self.gamma is not None:
            x = x * self.gamma.reshape(1, -1, 1, 1)
        x = shortcut + self.drop_path(x)
        # BIM 残差增强：x = x + γ_bim * BIM(LN(x))，γ_bim 初值 0（或 BIM_GAMMA0）
        if self.use_bim:
            bim_out = self.gamma_bim.reshape(1, -1, 1, 1) * self.bim(self.norm_bim(x))
            if self.record_act:   # 仅监控探针时计算贡献比，训练热路径不触发
                self.last_act_ratio = (bim_out.norm() / (x.norm() + 1e-6)).item()
            x = x + self.drop_path(bim_out)
        return x


class ConvNeXtStage(nn.Module):
    r"""编码器 stage：可选下采样 + 若干 ConvNeXtBIMBlock。

    downsample 命名（LayerNorm2d + Conv2d k2s2）对齐 timm 的
    stages.{s}.downsample.0/1；stage0 无下采样（紧跟 stem）。
    """
    def __init__(self, in_dim, out_dim, depth, drop_path_rates, downsample=True,
                 ls_init_value=1e-6, bias=True, use_bim=True):
        super().__init__()
        if downsample:
            self.downsample = nn.Sequential(
                LayerNorm2d(in_dim),
                nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2),
            )
        else:
            self.downsample = nn.Identity()
        self.blocks = nn.ModuleList([
            ConvNeXtBIMBlock(out_dim, drop_path=drop_path_rates[i],
                             ls_init_value=ls_init_value, bias=bias, use_bim=use_bim)
            for i in range(depth)
        ])

    def forward(self, x):
        x = self.downsample(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class AttentionGate(nn.Module):
    r"""Attention U-Net 风格的 skip 注意力门（Oktay et al., 2018）。

    用（上采样后的）解码器特征 g 作为门控信号，给编码器 skip 特征 x 计算逐位置
    注意力系数 α∈(0,1)，抑制 skip 中与当前解码语义无关的背景区域，从而减少远端
    假阳性（利好 HD95）、突出目标器官（利好小器官 DSC）。

    本处 g 与 x 分辨率/通道一致（解码器 upsample 之后），无需再对齐尺寸。
    输出用 per-channel 门控强度 γ 做残差混合：out = x · (1 − γ·(1−α))，
    γ 初值 0 ⇒ 起步 out=x，严格等价无门控 baseline；γ 训练中自学，门控渐进生效
    （和 BIM 的 gamma_bim 同一安全套路，避免新模块一上来扰动已收敛的预训练主干）。
    """
    def __init__(self, dim, inter_dim=None):
        super().__init__()
        inter_dim = inter_dim or max(dim // 2, 1)
        self.w_g = nn.Conv2d(dim, inter_dim, kernel_size=1)   # 门控信号 g（解码器特征）
        self.w_x = nn.Conv2d(dim, inter_dim, kernel_size=1)   # skip 特征 x（编码器）
        self.psi = nn.Conv2d(inter_dim, 1, kernel_size=1)     # -> 单通道注意力 logit
        self.gamma = nn.Parameter(torch.zeros(dim))           # per-channel 门控强度，初值 0

    def forward(self, g, x):
        a = F.relu(self.w_g(g) + self.w_x(x), inplace=True)
        alpha = torch.sigmoid(self.psi(a))                    # [B,1,H,W] ∈ (0,1)
        gate = 1.0 - self.gamma.reshape(1, -1, 1, 1) * (1.0 - alpha)
        return x * gate                                       # γ=0 时恒等返回 x


class StripPooling(nn.Module):
    r"""轻量条带池化上下文模块（Strip Pooling, Hou et al. CVPR2020 思路的精简残差版）。

    放在瓶颈（最低分辨率、最强语义处）补长程带状上下文：
      - 水平条池化：沿宽 W 求均值 -> 每行一个描述子 (B,C,H)，深度 1D 卷积建模沿 H 的长程依赖；
      - 垂直条池化：沿高 H 求均值 -> 每列一个描述子 (B,C,W)，深度 1D 卷积建模沿 W 的长程依赖；
      - 二者广播回 H×W 相加 -> 1×1 卷积跨通道融合得到上下文 ctx。
    以 per-channel 门控 γ 做残差注入：out = x + γ · ctx；γ 初值 0 ⇒ 起步 out=x，
    严格等价 baseline（与 BIM/AG 同一安全套路），γ 训练中自学、上下文渐进生效。
    深度 1D 卷积近乎免费，参数主要在两个 1×1 融合卷积（dim² 量级）。
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = LayerNorm2d(dim)                              # 分支输入归一化，稳住尺度
        self.conv_h = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)  # 沿 H 长程(depthwise)
        self.conv_w = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)  # 沿 W 长程(depthwise)
        self.fuse = nn.Sequential(                               # 跨通道融合 H/W 两路上下文
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
        )
        self.gamma = nn.Parameter(torch.zeros(dim))              # per-channel 门控强度，初值 0

    def forward(self, x):
        y = self.norm(x)
        h = self.conv_h(y.mean(dim=3)).unsqueeze(3)              # (B,C,H,1) 沿宽池化 + 1D 卷积
        w = self.conv_w(y.mean(dim=2)).unsqueeze(2)              # (B,C,1,W) 沿高池化 + 1D 卷积
        ctx = self.fuse(h + w)                                   # 广播相加 -> (B,C,H,W) -> 融合
        g = self.gamma.reshape(1, -1, 1, 1)
        if SP_MODE == 'gate':
            # 乘法门：作用在 x 上，背景 x≈0 处恒≈0 -> 不凭空造远端假阳性；γ=0 -> tanh 项系数为 0 -> 恒等
            return x * (1.0 + g * torch.tanh(ctx))
        # 加法上下文（原行为）：γ=0 时恒等返回 x
        return x + g * ctx


class ConvNeXtUpStage(nn.Module):
    r"""解码器 stage（镜像）：2× 上采样 ->（可选注意力门控 skip）-> 拼接融合 -> 若干 ConvNeXtBIMBlock。

    blocks 命名对齐 up_stages.{j}.blocks.{i}.*，使其 norm/mlp/gamma 可加载
    编码器对应 stage 的同一套 ConvNeXt 预训练权重（镜像）。
    upsample / fuse 从头训练（对应 Swin-Unet 的 PatchExpand / concat_back_dim）。
    use_ag=True 时在拼接前用注意力门(ag)过滤 skip。
    """
    def __init__(self, in_dim, out_dim, depth, drop_path_rates,
                 ls_init_value=1e-6, bias=True, use_bim=True, use_ag=False):
        super().__init__()
        # 2× 上采样：in_dim @ HxW -> out_dim @ 2H x 2W
        self.upsample = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2)
        # 注意力门控 skip（可选）：用上采样后的解码器特征门控 skip
        self.use_ag = use_ag
        if use_ag:
            self.ag = AttentionGate(out_dim)
        # 与 skip(out_dim) 拼接得 2*out_dim，再 1×1 融合回 out_dim
        self.fuse = nn.Conv2d(2 * out_dim, out_dim, kernel_size=1)
        self.blocks = nn.ModuleList([
            ConvNeXtBIMBlock(out_dim, drop_path=drop_path_rates[i],
                             ls_init_value=ls_init_value, bias=bias, use_bim=use_bim)
            for i in range(depth)
        ])

    def forward(self, x, skip):
        x = self.upsample(x)
        if self.use_ag:
            skip = self.ag(x, skip)       # g=解码器特征, x=skip
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class SwinTransformerSys(nn.Module):
    r"""ConvNeXt-T 对称编/解码器 U-Net，空间混合层用 BIM 替换。

    保留原 Swin-Unet 的类名与构造签名（多余的 swin 专用参数被 **kwargs 吞掉），
    使 networks/vision_transformer.py 与上层 train/test 脚本无需改动。

    - 编码器：ConvNeXt-T（stem 4× patchify + 4 个 stage，depths [3,3,9,3]，
      dims [96,192,384,768]，分辨率 56/28/14/7）。
    - 解码器：3 个上采样 stage（dims [384,192,96]，depths [9,3,3]）镜像编码器
      stage 2/1/0，与编码器共享同一套 ConvNeXt 预训练 block 权重。
    - BIM（AttModule）作为每个 block 的空间混合层，从头训练。
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=9,
                 embed_dim=96, depths=None, drop_path_rate=0.1,
                 ls_init_value=1e-6, conv_bias=True,
                 final_upsample="expand_first", **kwargs):
        super().__init__()

        # ConvNeXt-T 配置，硬编码以对齐预训练（不读 config 的 swin DEPTHS=[2,2,6,2]）
        depths = [3, 3, 9, 3]
        dims = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]  # [96,192,384,768]

        print("ConvNeXt-T(BIM) U-Net init----depths:{};dims:{};drop_path_rate:{};num_classes:{}".format(
            depths, dims, drop_path_rate, num_classes))
        print("HEAD={} | BIM master USE_BIM={} | BIM_STAGES(encoder idx)={} | BIM_DECODER={} | USE_AG={} | USE_SP={}".format(
            HEAD, USE_BIM, sorted(BIM_STAGES) if USE_BIM else [], BIM_DECODER, USE_AG, USE_SP))
        if USE_SP:
            print("USE_SP=1 | SP_MODE={} (add=加法上下文 / gate=乘法门)".format(SP_MODE))

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = dims[-1]
        self.final_upsample = final_upsample
        self.img_size = img_size
        self.patch_size = patch_size

        # 随机深度（stochastic depth）按编码器各 stage 线性分配
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        enc_dpr, c = [], 0
        for d in depths:
            enc_dpr.append(dpr[c:c + d])
            c += d

        # stem：4× patchify（对齐 timm stem.0/1）
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=patch_size, stride=patch_size),
            LayerNorm2d(dims[0]),
        )

        # 编码器各 stage（对齐 timm stages.{i}.downsample / blocks）
        self.stages = nn.ModuleList()
        for i in range(self.num_layers):
            in_dim = dims[i - 1] if i > 0 else dims[0]
            stage_use_bim = USE_BIM and (i in BIM_STAGES)
            self.stages.append(
                ConvNeXtStage(in_dim, dims[i], depths[i], enc_dpr[i],
                              downsample=(i > 0), ls_init_value=ls_init_value, bias=conv_bias,
                              use_bim=stage_use_bim))

        # 瓶颈上下文模块（可选）：作用于编码器最末 768c@7×7 特征，进解码器前注入长程带状上下文
        self.sp = StripPooling(dims[3]) if USE_SP else None

        # 解码器各上采样 stage（镜像编码器 stage 2/1/0）
        dec_in = [dims[3], dims[2], dims[1]]     # 768 -> / 384 -> / 192 ->
        dec_out = [dims[2], dims[1], dims[0]]    # 384 / 192 / 96
        dec_depth = [depths[2], depths[1], depths[0]]   # 9 / 3 / 3
        dec_dpr = [enc_dpr[2], enc_dpr[1], enc_dpr[0]]  # 镜像编码器对应 stage 的 drop_path
        self.up_stages = nn.ModuleList()
        for j in range(3):
            # up_stage j 镜像编码器 stage (2-j)，BIM 开关随对应编码器 stage 联动
            up_use_bim = USE_BIM and BIM_DECODER and ((2 - j) in BIM_STAGES)
            self.up_stages.append(
                ConvNeXtUpStage(dec_in[j], dec_out[j], dec_depth[j], dec_dpr[j],
                                ls_init_value=ls_init_value, bias=conv_bias,
                                use_bim=up_use_bim, use_ag=USE_AG))

        # 解码末端归一化
        self.norm_up = LayerNorm2d(dims[0])

        # 输出头（HEAD 切换，消融用）
        c0 = dims[0]
        if HEAD == 'resize':
            # 两级 resize-conv 上采样(56→112→224) + 从输入原图取全分辨率 skip 融合。
            # 与服务器产出 83.91/84.37 的版本逐字一致（命名 skip112/skip224/up_fuse1/up_fuse2/output）。
            self.skip112 = nn.Sequential(
                nn.Conv2d(in_chans, c0, kernel_size=3, stride=2, padding=1),  # 224 -> 112
                LayerNorm2d(c0), nn.GELU(),
            )
            self.skip224 = nn.Sequential(
                nn.Conv2d(in_chans, c0, kernel_size=3, stride=1, padding=1),  # 224 -> 224
                LayerNorm2d(c0), nn.GELU(),
            )
            self.up_fuse1 = nn.Sequential(   # 112 级：56→112 上采样后融合 skip112
                nn.Conv2d(2 * c0, c0, kernel_size=3, padding=1), LayerNorm2d(c0), nn.GELU(),
            )
            self.up_fuse2 = nn.Sequential(   # 224 级：112→224 上采样后融合 skip224
                nn.Conv2d(2 * c0, c0, kernel_size=3, padding=1), LayerNorm2d(c0), nn.GELU(),
            )
            self.output = nn.Conv2d(c0, num_classes, kernel_size=1, bias=True)
        else:
            # 一发 4× ConvTranspose 还原 stem 的 patchify(56→224)，= 75.96 SGD baseline 版。
            self.up = nn.ConvTranspose2d(c0, c0, kernel_size=patch_size, stride=patch_size)
            self.output = nn.Conv2d(c0, num_classes, kernel_size=1, bias=False)

        self.apply(self._init_weights)

        # BIM 增强分支的输出经 gamma_bim 缩放后直接相加（无后置 LayerNorm 归一化），
        # 故其幅值会真实影响残差贡献：若用全局 std=0.02 初始化，四阶级联乘法输出会
        # 衰减到 ~1e-13，一旦 gamma_bim 学出非零值，BIM 分支仍贡献微弱、梯度消失而学不动。
        # 改用方差保持初始化让 BIM 输出维持 O(1)，保证 gamma_bim 增长后 BIM 可正常训练。
        # 必须放在 self.apply(self._init_weights) 之后，避免被其覆盖。
        for m in self.modules():
            if isinstance(m, AttModule):
                m.reset_bim_parameters()

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'gamma'}

    # 编码器：返回 bottleneck 特征 + 各 stage 的 skip 特征
    def forward_features(self, x):
        x = self.stem(x)
        skips = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)   # 96@56, 192@28, 384@14, 768@7
        return x, skips

    # 解码器 + skip 连接
    def forward_up_features(self, x, skips):
        x = self.up_stages[0](x, skips[2])  # -> 384@14
        x = self.up_stages[1](x, skips[1])  # -> 192@28
        x = self.up_stages[2](x, skips[0])  # ->  96@56
        x = self.norm_up(x)
        return x

    def up_x4(self, x, img):
        if HEAD == 'resize':
            s112 = self.skip112(img)   # c0@112
            s224 = self.skip224(img)   # c0@224
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # c0@112
            x = self.up_fuse1(torch.cat([x, s112], dim=1))                              # c0@112
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # c0@224
            x = self.up_fuse2(torch.cat([x, s224], dim=1))                              # c0@224
            x = self.output(x)         # num_classes@224
        else:
            x = self.up(x)             # c0@224
            x = self.output(x)         # num_classes@224
        return x

    def forward(self, x):
        img = x                    # 保留全分辨率输入供 resize 头的高分辨率 skip 使用
        x, skips = self.forward_features(x)
        if self.sp is not None:    # 瓶颈处注入条带池化长程上下文（γ=0 起步等价 baseline）
            x = self.sp(x)
        x = self.forward_up_features(x, skips)
        x = self.up_x4(x, img)
        return x
