# flowmatching_train.py
"""
Flow Matching（整流流 / Rectified Flow）训练脚本。

核心思想：学习一个速度场（向量场）u，把标准正态噪声 x_0 ~ N(0,1) 沿直线路径
    x_t = (1-t) * x_0 + t * x_1,   t ∈ [0,1]
运输到真实数据分布（这里是 CelebA 人脸图像 x_1）。

对上述路径求导可得真实速度场：
    u_target = d(x_t)/dt = x_1 - x_0

训练时随机采样 t，让 UNet 预测的向量场逼近 u_target（MSE 损失）；
采样（生成）时从纯噪声 x_0 出发，沿学到的向量场做欧拉积分 t: 0 → 1，得到生成的图像。
"""
import os
import math
from datetime import datetime
import glob
import kagglehub
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils

# ---------------- 用户配置 ----------------
# 通过 KaggleHub 下载 CelebA 数据集，返回值为数据集缓存目录。
data_root = kagglehub.dataset_download("jessicali9530/celeba-dataset")
print("Path to dataset files:", data_root)

save_dir = "./flowmatch_checkpoints"                 # 模型权重 / 采样图的保存目录
os.makedirs(save_dir, exist_ok=True)
batch_size = 32          # 每批样本数
lr = 1e-4                # 学习率
num_epochs = 100         # 训练轮数
image_size = 104         # 图像缩放到 104x104
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 有 GPU 用 GPU
num_workers = 4          # 数据加载的并行进程数
pin_memory = True        # 锁页内存，加速数据搬运到 GPU
sample_every = 1         # 每隔多少个 epoch 采样一次生成图像
num_sample_images = 9    # 每次采样生成的图像数量
base_ch = 128            # UNet 基础通道数
flow_steps = 200         # 采样时的积分步数（步数越多越精确，越慢）

# ---------------- 数据 ----------------
transform = transforms.Compose([
    transforms.Resize(image_size),      # 缩放到指定尺寸
    transforms.CenterCrop(image_size),  # 中心裁剪成正方形
    transforms.ToTensor(),              # 转为 Tensor，值域 [0,1]
])

class CelebADataset(Dataset):
    """CelebA 人脸数据集封装：读取图片 → 预处理 → 缩放到 [-1,1]（与噪声分布对齐）。"""
    def __init__(self, root, transform=None):
        self.root = root
        # KaggleHub 数据集通常包含嵌套目录，因此递归查找所有 JPG 文件。
        self.paths = sorted(glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True))
        if not self.paths:
            raise FileNotFoundError(f"No JPG images found under Kaggle dataset path: {root}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # 从 [0,1] 缩放到 [-1,1]，与训练时采样的标准正态噪声 x_0 ~ N(0,1) 范围一致
        img = img * 2.0 - 1.0
        return img

dataset = CelebADataset(root=data_root, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=pin_memory)

# -- 模型定义开始 --
class SinusoidalPosEmb(nn.Module):
    """时间步 t 的正弦位置编码。

    把标量时间 t 映射为 dim 维向量，正余弦周期性编码能让模型更好地区分不同时间步，
    是扩散模型 / Flow Matching 中注入时间条件的常用做法。
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (B,) 浮点数（支持连续时间 t，不要求是整数）
        device = t.device
        half = self.dim // 2
        # 频率按指数递减：freq = 1 / (10000^(i/(half-1)))
        emb = torch.exp(torch.arange(half, device=device) * -(math.log(10000) / (half - 1)))
        # t 与频率外积，得到 (B, half)，再分别取 sin / cos 拼接成 (B, dim)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))  # 若 dim 为奇数，补一个 0 对齐维度
        return emb  # (B, dim)

class ResidualBlock(nn.Module):
    """带时间条件注入的残差块。

    结构：x → conv1 → 加上时间嵌入的投影 → conv2 → 与残差连接相加。
    时间嵌入 t_emb 通过线性层投影到通道数后广播到 (B, C, 1, 1) 加到特征图上，
    从而把“当前时间步”的信息注入每一层。
    """
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        )
        # 把时间嵌入投影到 out_ch 维，用于在通道上加到特征图上
        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        )
        # 输入输出通道不一致时用 1x1 卷积对齐维度，否则恒等
        self.residual_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        residual = self.residual_conv(x)  # 残差分支
        h = self.conv1(x)
        t_emb = self.time_emb_proj(t_emb)
        h = h + t_emb[:, :, None, None]   # (B, C) → (B, C, 1, 1) 广播到空间维
        h = self.conv2(h)
        return h + residual

class SelfAttention2D(nn.Module):
    """二维特征图上的多头自注意力。

    把 (B, C, H, W) 特征图 reshape 成 (B, num_heads, C/num_heads, H*W) 的序列，
    在空间位置间做自注意力，用于建模长距离依赖（UNet 底层 / 中间层使用）。
    """
    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, in_channels)
        # 一次 1x1 卷积同时生成 Q、K、V（3 倍通道）
        self.qkv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)  # 拆出 Q、K、V
        # reshape 成多头序列：(B, heads, head_dim, H*W)
        q = q.view(B, self.num_heads, C // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W)

        # 缩放点积注意力：softmax(Q^T K / sqrt(d))
        attn = torch.softmax(torch.matmul(q.transpose(-2, -1), k) / math.sqrt(C // self.num_heads), dim=-1)
        out = torch.matmul(attn, v.transpose(-2, -1)).transpose(-2, -1)
        out = out.contiguous().view(B, C, H, W)  # 还原回 (B, C, H, W)
        out = self.proj_out(out)
        return x + out  # 残差连接

class DownBlock(nn.Module):
    """UNet 编码器（下采样）块：若干残差块 + 可选注意力 + 下采样。"""
    def __init__(self, in_ch, out_ch, time_emb_dim, num_blocks=2, downsample=True, use_attention=False):
        super().__init__()
        # 第一个残差块把通道从 in_ch 提升到 out_ch，后续保持 out_ch
        self.blocks = nn.ModuleList([
            ResidualBlock(in_ch if i == 0 else out_ch, out_ch, time_emb_dim)
            for i in range(num_blocks)
        ])
        self.attn = SelfAttention2D(out_ch) if use_attention else nn.Identity()
        # stride=2 的卷积实现 2 倍下采样
        self.downsample = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1) if downsample else nn.Identity()

    def forward(self, x, t_emb):
        skips = []  # 记录各残差块输出，作为上采样阶段的 skip 连接
        for block in self.blocks:
            x = block(x, t_emb)
            skips.append(x)
        x = self.attn(x)
        x = self.downsample(x)
        return x, skips

class UpBlock(nn.Module):
    """UNet 解码器（上采样）块：上采样 + 若干残差块（拼接 skip 连接）+ 可选注意力。"""
    def __init__(self, in_ch, out_ch, time_emb_dim, num_blocks=2, upsample=True, use_attention=False):
        super().__init__()
        # 转置卷积实现 2 倍上采样
        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1) if upsample else nn.Identity()
        # 上采样后拼接来自下采样阶段的 skip 特征，所以输入通道 = in_ch(已变 out_ch) + out_ch
        self.blocks = nn.ModuleList([
            ResidualBlock(in_ch + out_ch, out_ch, time_emb_dim)
            for _ in range(num_blocks)
        ])
        self.attn = SelfAttention2D(out_ch) if use_attention else nn.Identity()

    def forward(self, x, skips, t_emb):
        x = self.upsample(x)
        for block in self.blocks:
            if skips:
                # 逐个弹出（从最深层开始）的 skip 特征在通道维拼接
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, t_emb)
        x = self.attn(x)
        return x

class MidBlock(nn.Module):
    """UNet 瓶颈（最底层）块：若干残差块 + 自注意力，通道数不变。"""
    def __init__(self, channels, time_emb_dim, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualBlock(channels, channels, time_emb_dim)
            for _ in range(num_blocks)
        ])
        self.attn = SelfAttention2D(channels)

    def forward(self, x, t_emb):
        for block in self.blocks:
            x = block(x, t_emb)
        x = self.attn(x)
        return x

class EnhancedUNet(nn.Module):
    """Flow Matching 的主干网络：带时间条件与自注意力的 UNet。

    结构：时间编码 MLP → 初始卷积 → 4 层下采样（通道逐级翻倍）→ 中间瓶颈 → 4 层上采样 → 输出卷积。
    输入：图像 x 与时间 t；输出：预测的速度场（与输入图像同尺寸）。
    """
    def __init__(self, in_ch=3, base_ch=128, time_emb_dim=512, num_res_blocks=2):
        super().__init__()

        # 时间 t → 正弦编码 → MLP → time_emb_dim 维嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_ch),
            nn.Linear(base_ch, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.init_conv = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)

        # 下采样：通道 base_ch → 2x → 4x → 8x，空间尺寸逐层减半
        self.down1 = DownBlock(base_ch, base_ch, time_emb_dim, num_res_blocks, downsample=False)
        self.down2 = DownBlock(base_ch, base_ch * 2, time_emb_dim, num_res_blocks)
        self.down3 = DownBlock(base_ch * 2, base_ch * 4, time_emb_dim, num_res_blocks)
        self.down4 = DownBlock(base_ch * 4, base_ch * 8, time_emb_dim, num_res_blocks, use_attention=True)

        self.mid = MidBlock(base_ch * 8, time_emb_dim, num_res_blocks * 2)

        # 上采样：通道 8x → 4x → 2x → base_ch，空间尺寸逐层恢复
        self.up4 = UpBlock(base_ch * 8, base_ch * 4, time_emb_dim, num_res_blocks, use_attention=True)
        self.up3 = UpBlock(base_ch * 4, base_ch * 2, time_emb_dim, num_res_blocks)
        self.up2 = UpBlock(base_ch * 2, base_ch, time_emb_dim, num_res_blocks)
        self.up1 = UpBlock(base_ch, base_ch, time_emb_dim, num_res_blocks, upsample=False)

        # 输出层：归一化 + 激活 + 卷积到 3 通道（RGB 速度场）
        self.final = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, in_ch, kernel_size=3, padding=1)
        )

    def forward(self, x, t):
        # t: (B,) 浮点数，值域 [0,1]
        t_emb = self.time_mlp(t)
        x = self.init_conv(x)

        skips = []  # 收集各下采样层的 skip 特征，供上采样层拼接
        x, s1 = self.down1(x, t_emb); skips.extend(s1)
        x, s2 = self.down2(x, t_emb); skips.extend(s2)
        x, s3 = self.down3(x, t_emb); skips.extend(s3)
        x, s4 = self.down4(x, t_emb); skips.extend(s4)

        x = self.mid(x, t_emb)

        x = self.up4(x, skips, t_emb)
        x = self.up3(x, skips, t_emb)
        x = self.up2(x, skips, t_emb)
        x = self.up1(x, skips, t_emb)

        return self.final(x)

# -- 模型定义结束 --

# ---------------- 准备模型、优化器 ----------------
model = EnhancedUNet(in_ch=3, base_ch=base_ch, time_emb_dim=512, num_res_blocks=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

mse = nn.MSELoss()  # 损失函数：预测速度场与真实速度场的 MSE

# ---------------- 工具函数：保存采样图像网格 ----------------
def save_samples(x, epoch):
    # x: 值域 [-1,1] 的张量，形状 (N,3,H,W)
    out = (x.clamp(-1,1) + 1.0) / 2.0  # 缩回 [0,1]
    grid = utils.make_grid(out, nrow=int(math.sqrt(out.shape[0]) + 0.999), padding=2)  # 拼成一张网格图
    filename = os.path.join(save_dir, f"sample_epoch_{epoch:03d}.png")
    utils.save_image(grid, filename)
    print(f"[saved] {filename}")

# ---------------- 采样函数（欧拉积分求解 ODE） ----------------
@torch.no_grad()
def sample_flow(model, n_samples=8, steps=200, device=device):
    """从纯噪声出发，沿学到的速度场欧拉积分生成图像。

    即求解 ODE  dx/dt = u(x, t)，t 从 0 积到 1，初始 x(0) ~ N(0,1)。
    """
    model.eval()
    # 初始状态：标准正态噪声 x ~ N(0,1)
    x = torch.randn(n_samples, 3, image_size, image_size, device=device)
    dt = 1.0 / steps  # 固定步长
    for i in range(steps):
        # 当前时间 t = i/steps，值域 [0,1)
        t = torch.full((n_samples,), float(i) / steps, device=device, dtype=torch.float32)
        u = model(x, t)  # 预测当前状态的速度场
        x = x + u * dt    # 欧拉一步：x_{t+dt} = x_t + u * dt
    model.train()
    return x.clamp(-1,1)

# ---------------- 训练循环 ----------------
print("Starting training... device:", device)
total_loss = 0.0   # 累计损失，用于统计平均
step_count = 0     # 累计步数，用于统计平均
global_step = 0    # 全局步数（跨 epoch）
for epoch in range(num_epochs):
    for z in loader:
        z = z.to(device)  # 真实图像 x_1，值域 [-1,1]，形状 (B,3,H,W)
        B = z.shape[0]

        # 采样初始噪声 x_0 ~ N(0,1)
        x_0 = torch.randn_like(z)

        # 采样时间 t ~ Uniform(0,1)
        t = torch.rand(B, device=device, dtype=torch.float32)

        # 构造插值点 x_t = t*z + (1-t)*x_0（Flow Matching 的直线路径）
        t_broadcast = t.view(B, 1, 1, 1)
        x_t = t_broadcast * z + (1.0 - t_broadcast) * x_0

        # 真实速度场 u_target = d(x_t)/dt = z - x_0（对直线路径求导）
        u_target = (z - x_0).detach()

        optimizer.zero_grad()

        pred = model(x_t, t)  # 预测速度场，t 形状 (B,)
        loss = mse(pred, u_target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪，防止梯度爆炸
        optimizer.step()
        total_loss += loss.item()
        step_count += 1
        if global_step % 500 == 0:
            avg_loss = total_loss / step_count
            print(f"Epoch {epoch:03d} Step {global_step:06d} Average Loss: {avg_loss:.6f}")
            total_loss = 0.0
            step_count = 0

        global_step += 1
    # 每个采样周期生成一批图像并保存
    if epoch % sample_every == 0:
        samples = sample_flow(model, n_samples=num_sample_images, steps=flow_steps, device=device)
        save_samples(samples, epoch)
    # 每个 epoch 结束保存一次 checkpoint
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "epoch": epoch
    }
    ckpt_path = os.path.join(save_dir, f"flowmatch_ckpt_epoch_{epoch:03d}.pt")
    torch.save(ckpt, ckpt_path)
    print(f"[saved checkpoint] {ckpt_path}")

print("Training finished.")