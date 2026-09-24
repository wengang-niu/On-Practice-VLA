# draw_unet.py
"""
可视化 EnhancedUNet（Flow Matching 主干网络）的 U-Net 结构，输出 PNG。
纯绘图脚本，不依赖训练逻辑。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# 注册中文字体，避免中文显示为方框
for p in ["/usr/share/fonts/truetype/Alibaba-PuHuiTi-Regular.ttf",
          "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"]:
    fm.fontManager.addfont(p)
plt.rcParams["font.sans-serif"] = ["Alibaba PuHuiTi", "Droid Sans Fallback", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 配色 ----------
C_DOWN   = "#3B6FB0"   # 编码器（下采样）蓝
C_DOWN_D = "#2A4F7E"   # 深蓝（底层）
C_MID    = "#7B4FA6"   # 瓶颈 紫
C_UP     = "#3F9E6B"   # 解码器（上采样）绿
C_UP_D   = "#2E7A51"   # 深绿
C_IO     = "#6B7280"   # 输入/输出 灰
C_TIME   = "#C05A4E"   # 时间条件 红
C_SKIP   = "#9AA3AF"   # skip 连接 浅灰
TXT_W    = "white"

fig, ax = plt.subplots(figsize=(18, 13.5))
ax.set_xlim(0, 14.6)
ax.set_ylim(0, 12.4)
ax.axis("off")

# ---------- 工具 ----------
def box(x, y, w, h, title, lines, fc, ec="none"):
    b = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle="round,pad=0.06,rounding_size=0.16",
                       linewidth=1.6, edgecolor=ec, facecolor=fc, zorder=4)
    ax.add_patch(b)
    ax.text(x, y + h / 2 - 0.42, title, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=TXT_W, zorder=5)
    for i, ln in enumerate(lines):
        ax.text(x, y + h / 2 - 0.92 - i * 0.42, ln, ha="center", va="center",
                fontsize=8.0, color=TXT_W, zorder=5, alpha=0.95)

def arrow(p1, p2, color=C_SKIP, lw=1.8, ls="-", style="-|>", ms=15):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=ms,
                        color=color, lw=lw, linestyle=ls, zorder=3,
                        shrinkA=2, shrinkB=2)
    ax.add_patch(a)

# ---------- 几何布局 ----------
xL, xR = 3.4, 9.0           # 左/右列中心
W, H = 2.5, 1.35            # 盒子宽高
y_init = 9.7                # init_conv / final
y1, y2, y3, y4 = 8.1, 6.5, 4.9, 3.3   # down1..4 / up1..4
y_mid = 1.7

# ---------- 输入 / 输出 ----------
box(6.2, 11.1, 3.0, 1.0, "输入 x_t = (1-t)·x_0 + t·x_1",
    ["[B, 3, 1024, 1024]"], C_IO)
box(6.2, 9.7, 2.5, 1.35, "init_conv  (3×3, pad=1)",
    ["3 → 128 ch", "1024 × 1024"], C_DOWN)
box(11.9, 9.7, 2.6, 1.35, "final  (GN→SiLU→Conv 3×3)",
    ["128 → 3 ch", "1024 × 1024"], C_IO)
box(11.9, 11.1, 3.2, 1.0, "输出 速度场 u(x, t)",
    ["[B, 3, 1024, 1024]"], C_IO)

# ---------- 下采样路径 ----------
box(xL, y1, W, H, "down1  DownBlock", ["128 → 128 ch", "1024 × 1024", "ResBlock ×2"], C_DOWN)
box(xL, y2, W, H, "down2  DownBlock", ["128 → 256 ch", "1024 → 512  (↓2×)", "ResBlock ×2"], C_DOWN)
box(xL, y3, W, H, "down3  DownBlock", ["256 → 512 ch", "512 → 256  (↓2×)", "ResBlock ×2"], C_DOWN)
box(xL, y4, W, H, "down4  DownBlock", ["512 → 1024 ch", "256 → 128  (↓2×)", "ResBlock ×2 + Attn"], C_DOWN_D)

# ---------- 瓶颈 ----------
box(6.2, y_mid, W, H, "mid  MidBlock", ["1024 ch", "128 × 128", "ResBlock ×4 + Attn"], C_MID)

# ---------- 上采样路径 ----------
box(xR, y4, W, H, "up4  UpBlock", ["1024 → 512 ch", "128 → 256  (↑2×)", "ResBlock ×2 + Attn"], C_UP_D)
box(xR, y3, W, H, "up3  UpBlock", ["512 → 256 ch", "256 → 512  (↑2×)", "ResBlock ×2"], C_UP)
box(xR, y2, W, H, "up2  UpBlock", ["256 → 128 ch", "512 → 1024  (↑2×)", "ResBlock ×2"], C_UP)
box(xR, y1, W, H, "up1  UpBlock", ["128 → 128 ch", "1024 × 1024", "ResBlock ×2"], C_UP)

# ---------- 主通路箭头 ----------
# 编码器（向下）
arrow((6.2, y_init - H / 2), (xL, y1 + H / 2), color=C_SKIP, lw=2.2)
arrow((xL, y1 - H / 2), (xL, y2 + H / 2), color=C_SKIP, lw=2.2)
arrow((xL, y2 - H / 2), (xL, y3 + H / 2), color=C_SKIP, lw=2.2)
arrow((xL, y3 - H / 2), (xL, y4 + H / 2), color=C_SKIP, lw=2.2)
arrow((xL, y4 - H / 2), (6.2, y_mid + H / 2), color=C_SKIP, lw=2.2)
# 解码器（向上）
arrow((6.2, y_mid + H / 2), (xR, y4 + H / 2), color=C_SKIP, lw=2.2)
arrow((xR, y4 - H / 2), (xR, y3 + H / 2), color=C_SKIP, lw=2.2)
arrow((xR, y3 - H / 2), (xR, y2 + H / 2), color=C_SKIP, lw=2.2)
arrow((xR, y2 - H / 2), (xR, y1 + H / 2), color=C_SKIP, lw=2.2)
arrow((xR, y1 - H / 2), (11.9, y_init + H / 2), color=C_SKIP, lw=2.2)

# ---------- skip 连接（水平虚线） ----------
for yl, yr, ch in [(y1, y1, "128 ch  @1024"), (y2, y2, "256 ch  @1024"),
                   (y3, y3, "512 ch  @512"), (y4, y4, "1024 ch @256")]:
    arrow((xL + W / 2, yl), (xR - W / 2, yr), color=C_SKIP, lw=1.5, ls=(0, (5, 4)))
ax.text((xL + xR) / 2, y1 + 0.52, "skip (跳跃连接)：保留高分辨率细节", ha="center",
        fontsize=9, color="#5A6472", style="italic", zorder=6)

# ---------- 时间条件分支 ----------
box(0.75, 11.1, 2.1, 1.3, "时间步 t", ["标量 [B,]", "t ∈ [0,1]"], C_TIME)
box(0.75, 9.7, 2.1, 1.3, "time_mlp", ["Sin 位置编码(128)", "→ Linear → SiLU", "→ Linear(512)"], C_TIME)
arrow((0.75, 9.7 - H / 2), (0.75, 4.0), color=C_TIME, lw=1.6, ls=(0, (4, 3)))
ax.text(0.55, 5.6, "时间条件\nt_emb [512]\n注入每个\nResidualBlock",
        ha="center", va="center", fontsize=8.5, color=C_TIME, zorder=6)
# 从时间线分叉到每个 DownBlock / UpBlock / Mid
for y in [y1, y2, y3, y4, y_mid]:
    arrow((0.75, y), (xL - W / 2, y), color=C_TIME, lw=1.0, ls=(0, (3, 3)), ms=9)
    arrow((0.75, y), (xR + W / 2, y), color=C_TIME, lw=1.0, ls=(0, (3, 3)), ms=9)

# ---------- 图例 / 说明 ----------
ax.text(6.2, 0.45,
        "UNet 沙漏结构：编码器逐层「空间减半 + 通道翻倍」，最底层用自注意力捕捉全局结构；\n"
        "解码器逐层恢复分辨率，并拼接 skip 细节。t_emb 作为全局条件注入每个残差块。",
        ha="center", va="center", fontsize=10, color="#3A4350",
        bbox=dict(boxstyle="round,pad=0.5", fc="#F3F4F6", ec="#D1D5DB", lw=1))

plt.tight_layout()
out = "/home/root123/workspace/source/On-Practice-VLA/basic_knowledge/flowmatching/enhanced_unet_architecture.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"[saved] {out}")
