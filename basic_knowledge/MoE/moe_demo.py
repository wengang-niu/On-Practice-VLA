# moe_demo.py
"""
Mixture-of-Experts (MoE, 混合专家) 从零实现 + 可视化学习 demo。

==============================================================================
为什么需要 MoE？
==============================================================================
普通网络（dense 网络）：每个输入都要经过全部参数。参数越多越聪明，但计算量也线性增长。
于是出现一个矛盾：想提升模型容量（参数量），但又不希望 FLOPs 同步暴涨。

MoE 的核心思想是“条件计算 / 稀疏激活”：
  - 把一个大网络拆成多个并行的“专家（expert）”子网络；
  - 每个输入 token 只激活其中 top-k 个专家（k 远小于专家总数）；
  - 由一个轻量“路由器 / 门控（router / gate）”决定该把当前输入送给哪个专家。

这样：参数总量 = 所有专家参数之和（容量大），但每个 token 的计算量 ≈ k 个专家（小）。
即“以稀疏激活的方式，用更大的参数量换取更好的效果，而几乎不增加推理 FLOPs”。

在 Transformer 里，MoE 通常用来替换 FFN 层：原本每层一个 FFN，改成 N 个 FFN 专家 + router。

==============================================================================
本 demo 的学习路线
==============================================================================
1. 实现 MoE 层的核心三件套：专家（MLP）、路由器（线性层）、Top-k 选择 + 加权融合；
2. 实现负载均衡辅助损失（Switch Transformer 的 aux loss），防止“只用一两个专家”的坍缩；
3. 给一个 MoE 版 Transformer FFN，展示它如何“换掉”普通 FFN；
4. 玩具实验：构造 4 种模式、每种模式需要不同的映射函数的数据，
   训练 MoE 后，可视化“哪种模式被路由到哪个专家”——直观看到专家特化现象。
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 让中文能在图里正常显示：显式注册系统里的 CJK 字体文件，
# 避免 matplotlib 回退到 DejaVu Sans（缺中文字形→图上中文变方块）。
_CJK_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",   # Linux 常见
    "/usr/share/fonts/truetype/Alibaba-PuHuiTi-Regular.ttf",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "C:/Windows/Fonts/msyh.ttc",                                  # Windows
    "/System/Library/Fonts/PingFang.ttc",                         # macOS
]
_font_name = None
for _p in _CJK_CANDIDATES:
    if os.path.exists(_p):
        try:
            font_manager.fontManager.addfont(_p)
            _font_name = font_manager.FontProperties(fname=_p).get_name()
            break
        except Exception:
            continue
matplotlib.rcParams["font.sans-serif"] = ([_font_name] if _font_name else []) + \
                                        ["SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ==================== 全局配置 ====================
torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_MODEL = 16          # 每个 token 的特征维度（类比 transformer 的 d_model）
N_EXPERTS = 8         # 专家数量（多于模式数，演示稀疏 + 负载均衡的作用）
TOP_K = 2             # 每个 token 激活的专家数（Top-2，GShard/DeepSeek 等常用配置）
N_MODES = 4           # 玩具数据的模式数（4 种不同的映射函数）
EXPERT_HIDDEN = 64    # 每个专家 MLP 的隐藏层宽度
SAVE_DIR = "./moe_out"  # 输出目录（图片 / 路由统计）
os.makedirs(SAVE_DIR, exist_ok=True)


# ==================== 核心一：专家（Expert）====================
class Expert(nn.Module):
    """单个专家：就是一个最普通的两层 FFN（SwiGLU 风格的简化版）。

    在真实 MoE Transformer 里，专家的结构和原 FFN 完全一样，
    只是把“一个 FFN”换成了“N 个 FFN + router 选 N 个里的 k 个”。
    """
    def __init__(self, d_in, d_hidden, d_out=None):
        super().__init__()
        d_out = d_out or d_in
        self.w1 = nn.Linear(d_in, d_hidden)
        self.w2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        # x: (..., d_in) -> (..., d_out)
        # 注意：这里每个专家看到的是“路由到自己这一批 token 的子集”，
        # 所以专家内部不需要任何条件逻辑，就是个纯 FFN。
        return self.w2(F.gelu(self.w1(x)))


# ==================== 核心二：MoE 层（路由 + Top-k + 加权融合 + 负载均衡）====================
class MoELayer(nn.Module):
    """稀疏 Top-k MoE 层。

    前向流程（对每个 token）：
      1) router 算出该 token 对每个专家的“偏好分数” logits  (E,)
      2) 取 Top-k 个分数最高的专家，并在 Top-k 内部做 softmax 得到权重  (k,)
         ——注意：softmax 只在选中的 k 个上做，不在全部 E 个上做。
         这是个关键细节：被淘汰的专家权重直接为 0，梯度也只回传到被选中的专家。
      3) 把 token 送进这 k 个专家，按权重加权求和，得到输出。
      4) 额外算一个“负载均衡损失”返回，训练时加到总 loss 里，防止专家使用不均。

    实现上为了清晰（也为了保留稀疏计算的意义），我们按“每个专家各自收集
    路由到自己的 token 子集”的方式做：专家 e 只对“选中它”的那些 token 前向一次，
    而不是对所有 token 都跑一遍——这正是 MoE 省算力的地方。
    """
    def __init__(self, d_model, n_experts, top_k, d_hidden=64):
        super().__init__()
        assert top_k <= n_experts
        self.n_experts = n_experts
        self.top_k = top_k
        # 路由器：一个线性层 d_model -> n_experts，没有非线形激活，
        # 直接输出 logits（选谁完全由它决定，参数量极小）。
        self.router = nn.Linear(d_model, n_experts, bias=False)
        # 专家池：N 个并行 FFN。用 ModuleList 保证各自有独立参数。
        self.experts = nn.ModuleList(
            [Expert(d_model, d_hidden) for _ in range(n_experts)]
        )

    def forward(self, x):
        """
        x: (..., d_model) —— 任意前缀，最后一维是特征。
        return: out (同 x 形状), aux_loss (标量), probs (路由概率, 用于统计)
        """
        orig_shape = x.shape
        d = x.shape[-1]
        # 把任意前缀的 token 拍平成一维序列 (T, d)，便于路由与收集
        x_flat = x.reshape(-1, d)
        T = x_flat.shape[0]

        # ---- 1. 路由打分 ----
        logits = self.router(x_flat)            # (T, E)
        probs = F.softmax(logits, dim=-1)       # (T, E) 全量 softmax 概率，仅用于统计 aux loss

        # ---- 2. Top-k 选择 + 在 k 个内 softmax 得到权重 ----
        topk_logits, topk_idx = logits.topk(self.top_k, dim=-1)   # (T, k)
        topk_weights = F.softmax(topk_logits, dim=-1)             # (T, k)

        # 把 (T,k) 的稀疏选择，转成 (T,E) 的“每个 token 给每个专家的权重”
        # （没被选中的位置为 0）。这一步用 one-hot 做向量化，比写循环更清晰。
        #   topk_idx: (T, k)  ->  one_hot: (T, k, E)
        #   topk_weights: (T, k) -> (T, k, 1) 广播
        one_hot = F.one_hot(topk_idx, self.n_experts).float()      # (T, k, E)
        token_expert_weights = (one_hot * topk_weights.unsqueeze(-1)).sum(dim=1)  # (T, E)

        # ---- 3. 稀疏分发：每个专家只处理“选中它”的那批 token ----
        out = torch.zeros(T, d, device=x.device, dtype=x.dtype)
        for e in range(self.n_experts):
            w_e = token_expert_weights[:, e]      # (T,) 该专家在每个 token 上的权重（0 表示没被选中）
            mask = w_e > 0                         # 选了该专家的 token
            if not mask.any():
                continue
            idx = torch.nonzero(mask, as_tuple=True)[0]
            expert_out = self.experts[e](x_flat[idx])               # (n, d)，n = 选中该专家的 token 数
            # 用 index_add 把加权结果累加回对应位置（autograd 友好，且支持重复 index）
            out = out.index_add(0, idx, w_e[idx].unsqueeze(-1) * expert_out)

        out = out.reshape(orig_shape)

        # ---- 4. 负载均衡辅助损失（Switch Transformer 形式）----
        aux_loss = self._load_balance_loss(probs, topk_idx, T)

        return out, aux_loss, probs

    def _load_balance_loss(self, probs, topk_idx, T):
        """负载均衡损失，鼓励“专家被均匀使用”。

        定义（Switch Transformer / GShard）：
            f_i = 被路由到专家 i 的 token 占比（基于硬选择，取 Top-1 统计）
            P_i = 路由器给专家 i 的平均概率（软概率，probs[:, i].mean()）
            aux = N * Σ_i ( f_i * P_i )

        直觉：
          - f_i 衡量“专家 i 实际分到了多少活”（离散的、不可导）；
          - P_i 衡量“路由器主观上多想把活给专家 i”（连续的、可导，梯度通过它回传）；
          - 两者越均匀（都接近 1/N），Σ f_i*P_i 越小，aux 越接近 1；
          - 若全部 token 都挤给某一个专家，aux ≈ N（最大，被惩罚）。
        所以把 aux 加进总 loss，梯度会推动 router 把概率摊开，避免少数专家过载、其余闲置。

        注：这里 f 用 Top-1 统计是常见简化；严格 Top-2 版本还会考虑“被选为第 2 个”的负载，
        但学习 demo 用 Switch 形式足够说明问题。
        """
        with torch.no_grad():
            top1 = topk_idx[:, 0]                              # 每个 token 的“第一志愿”专家
            f = torch.bincount(top1, minlength=self.n_experts) # 每个专家作为第一志愿的 token 数
            f = f.float() / T                                  # 归一化为占比 (E,)
        P = probs.mean(dim=0)                                  # 每个专家的平均被选概率 (E,)
        return self.n_experts * (f * P).sum()


# ==================== 核心三：MoE 版 Transformer FFN（集成示意）====================
class MoEFFN(nn.Module):
    """把 Transformer 里的 FFN 层换成 MoE：演示“替换”这件事本身。

    标准 Transformer block:  x -> Attn -> Add -> FFN -> Add
    MoE 版本:                x -> Attn -> Add -> MoE(替换FFN) -> Add

    这里只写 FFN 这一段（带残差连接），重点在于展示 MoE 如何无缝顶替 FFN：
    一个 token 进来，MoE 内部选 k 个专家算出结果，再走残差 + LayerNorm。
    aux_loss 由 forward 一并返回，调用方（每层）把它累加进总 loss 即可。
    """
    def __init__(self, d_model, n_experts, top_k, d_hidden=64, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.moe = MoELayer(d_model, n_experts, top_k, d_hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        y, aux, _ = self.moe(x)
        return residual + self.drop(y), aux


# ==================== 玩具数据：4 种模式，各自需要不同函数 ====================
def make_toy_data(n_per_mode=256, d_model=D_MODEL, n_modes=N_MODES):
    """构造一批分簇的玩具数据，用来“逼出”专家特化。

    每种模式 m：
      - 输入 x 来自一个独立的簇（均值向量 mu_m 不同，让 router 有可辨识的输入特征）；
      - 目标 y = 一个该模式专属的非线性函数 f_m(x)（4 个模式用 4 种不同函数）。
    直觉：不同模式需要“不同的变换”，于是 MoE 有动力把不同模式分给不同专家，
    让每个专家专注学一种函数——这就是专家特化。
    """
    # 4 个簇的均值向量（间隔拉大，让 router 容易区分）
    means = [torch.randn(d_model) * 4 for _ in range(n_modes)]

    # 4 种不同的目标函数：同一个 x，不同模式算出不同 y
    def target_fn(m, x):  # x: (n, d)
        if m == 0:   # 模式0：先过 tanh 再线性（“挤压”型）
            return torch.tanh(x @ Wm[0]) @ Wm2[0]
        elif m == 1: # 模式1：二次型（“放大”型）
            return (x ** 2) @ Wm[1] * 0.3
        elif m == 2: # 模式2：取符号 + relu（“分段”型）
            return F.relu(x @ Wm[2] - 1) @ Wm2[2]
        else:        # 模式3：正弦周期（“振荡”型）
            return torch.sin(x @ Wm[3]) @ Wm2[3]

    # 给每个模式随机但固定的线性映射，保证 4 个函数彼此真正不同
    torch.manual_seed(42)
    Wm = [torch.randn(d_model, d_model) for _ in range(n_modes)]
    Wm2 = [torch.randn(d_model, d_model) for _ in range(n_modes)]

    xs, ys, labels = [], [], []
    for m in range(n_modes):
        x = torch.randn(n_per_mode, d_model) * 0.5 + means[m]   # 高斯簇
        y = target_fn(m, x)
        xs.append(x); ys.append(y)
        labels += [m] * n_per_mode
    X = torch.cat(xs, dim=0)
    Y = torch.cat(ys, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return X.to(device), Y.to(device), labels.to(device)


# ==================== 训练循环：让 MoE 学会这 4 种映射 ====================
def train():
    print("=" * 70)
    print(f"训练 MoE：{N_EXPERTS} 个专家，Top-{TOP_K}，{N_MODES} 种模式，特征维 {D_MODEL}")
    print(f"设备：{device}")
    print("=" * 70)

    X, Y, labels = make_toy_data(n_per_mode=512)
    n = X.shape[0]
    print(f"样本数：{n}（每模式 {n // N_MODES} 个）\n")

    # 待训练模型：一个 MoE 层 + 一个输出投影（把 d_model 映回 d_model 做回归）
    moe = MoELayer(D_MODEL, N_EXPERTS, TOP_K, d_hidden=EXPERT_HIDDEN).to(device)
    head = nn.Linear(D_MODEL, D_MODEL).to(device)
    params = list(moe.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)

    # 负载均衡损失权重。太大→只追求均匀、学不好任务；太小→专家坍缩。
    # Switch Transformer 用 0.01 这个量级，这里也沿用。
    AUX_W = 0.05
    EPOCHS = 200
    BS = 256

    step = 0
    for ep in range(EPOCHS):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, BS):
            idx = perm[i:i + BS]
            x, y = X[idx], Y[idx]
            moe_out, aux, _ = moe(x)        # MoE 稀疏前向，顺便拿到 aux loss
            pred = head(moe_out)            # 输出投影
            task_loss = F.mse_loss(pred, y) # 主任务：回归 4 种函数
            loss = task_loss + AUX_W * aux  # 总 loss = 任务 + 负载均衡
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
        if (ep + 1) % 40 == 0 or ep == 0:
            print(f"epoch {ep+1:3d} | task {task_loss.item():.4f} | "
                  f"aux {aux.item():.4f} | total {loss.item():.4f}")

    print("\n训练结束。\n")
    return moe, X, Y, labels


# ==================== 可视化：专家路由分布（看“特化”）====================
@torch.no_grad()
def visualize_routing(moe, X, labels):
    """统计并画出「模式 × 专家」的路由矩阵。

    矩阵 M[m, e] = 模式 m 的 token 里，被路由到（Top-1）专家 e 的比例。
    若出现“每行都被某一个特定专家主导、且不同行对应不同专家”的排列结构，
    就说明专家发生了特化：每种模式稳定地交给某个专家处理。
    """
    moe.eval()
    _, _, probs = moe(X)            # probs: (T, E)
    # 用 router 全量概率里 argmax 作为“第一志愿”（与 aux loss 的 f 统计口径一致）
    top1 = probs.argmax(dim=-1)     # (T,)

    # 统计 模式×专家 的计数矩阵
    counts = torch.zeros(N_MODES, N_EXPERTS, device=device)
    for m in range(N_MODES):
        mask = labels == m
        sel = top1[mask]
        for e in range(N_EXPERTS):
            counts[m, e] = (sel == e).float().sum()
    # 每行归一化为占比
    freq = counts / counts.sum(dim=1, keepdim=True).clamp(min=1)
    freq_np = freq.cpu().numpy()

    # ---- 文本表格（无 matplotlib 也能看）----
    header = "模式\\专家 " + " ".join(f" E{e:1d}" for e in range(N_EXPERTS))
    print("「模式 × 专家」Top-1 路由占比（每行和为 1）：")
    print(header)
    for m in range(N_MODES):
        row = "  ".join(f"{freq_np[m, e]*100:5.1f}%" for e in range(N_EXPERTS))
        print(f"  模式{m}   {row}")
    # 找每行的主导专家
    print("\n各模式的主导专家（Top-1 占比最高的那个）：")
    for m in range(N_MODES):
        e_star = int(freq_np[m].argmax())
        print(f"  模式{m} -> 专家{e_star}  ({freq_np[m, e_star]*100:.1f}%)")
    print("\n若 4 个模式各自指向不同的专家，说明发生了“专家特化”。")

    # ---- 热力图 ----
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(freq_np, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(N_EXPERTS))
    ax.set_xticklabels([f"E{e}" for e in range(N_EXPERTS)])
    ax.set_yticks(range(N_MODES))
    ax.set_yticklabels([f"模式{m}" for m in range(N_MODES)])
    ax.set_xlabel("专家编号")
    ax.set_ylabel("数据模式")
    ax.set_title(f"MoE 路由分布（{N_EXPERTS} 专家, Top-{TOP_K}）：看是否出现专家特化")
    for m in range(N_MODES):
        for e in range(N_EXPERTS):
            v = freq_np[m, e]
            ax.text(e, m, f"{v*100:.0f}", ha="center", va="center",
                    color="white" if v > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label="该模式被路由到该专家的比例")
    plt.tight_layout()
    out = os.path.join(SAVE_DIR, "routing_heatmap.png")
    plt.savefig(out, dpi=130)
    print(f"\n热力图已保存：{out}")

    # ---- 额外：专家整体负载（看是否均衡）----
    overall = probs.mean(dim=0).cpu().numpy()
    target = round(1 / N_EXPERTS, 3)
    print(f"\n专家整体平均路由概率 P_e（完全均衡时应为 1/N={target}）：")
    for e in range(N_EXPERTS):
        bar = "#" * int(overall[e] * N_EXPERTS * 40)
        print(f"  E{e}: {overall[e]:.3f} {bar}")
    print("\n注：这里负载并不完美均衡——这是 aux loss 与“专家特化”之间天然的拉扯：")
    print("    任务本身希望每种模式稳定交给少数专家（特化），而 aux loss 希望摊平（均衡）。")
    print("    aux loss 是“软”压力（权重很小），能把负载从“一个专家吃掉 100%”拉回到")
    print("    “几个专家分担”，但不会、也不应该把特化完全抹掉。若想更均衡，可调大 AUX_W。")


# ==================== 额外：把 MoEFFN 当作 transformer 一层跑一下，证明能跑 ====================
@torch.no_grad()
def smoke_test_moe_ffn():
    print("\n" + "=" * 70)
    print("冒烟测试：把 MoEFFN 当成 Transformer 一层，跑一个 (B, T, D) 的前向")
    print("=" * 70)
    layer = MoEFFN(D_MODEL, N_EXPERTS, TOP_K, d_hidden=EXPERT_HIDDEN).to(device)
    # 模拟一个 batch 的序列：B=2 条样本，每条 seq_len=10 个 token，每 token D_MODEL 维
    x = torch.randn(2, 10, D_MODEL, device=device)
    y, aux = layer(x)
    print(f"输入 shape: {tuple(x.shape)}")
    print(f"输出 shape: {tuple(y.shape)}  （应与输入一致：残差连接不改变维度）")
    print(f"本层 aux loss: {aux.item():.4f}（训练时把它累加进总 loss 即可）")
    print("→ MoE 可以无缝替换 Transformer 里的 FFN 层。")


if __name__ == "__main__":
    moe, X, Y, labels = train()
    visualize_routing(moe, X, labels)
    smoke_test_moe_ffn()
    print("\nDone. 所有输出图片/日志已就绪，可打开 ./moe_out/routing_heatmap.png 查看。\n")
