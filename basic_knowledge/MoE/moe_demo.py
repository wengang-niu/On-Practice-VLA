"""
MoE（Mixture of Experts）最小教学示例
=====================================

一句话核心：多个「专家」子网络 + 一个「路由器」——输入来了，由路由器决定交给
哪个专家，让不同专家各司其职。

这里用人造的 1D 分段函数当拟合目标：
      x ∈ [0, 1/3)   →  y = 3x        （上升段）
      x ∈ [1/3, 2/3) →  y = 1         （水平段）
      x ∈ [2/3, 1]   →  y = -3x + 3   （下降段）
三段行为各不相同（升 / 平 / 降），区间互不重叠，谁也替代不了谁，所以必须分工。
（每段都是直线，任何小网络都能稳稳拟合——demo 的重点是「路由分工」，
而不是「拟合难曲线」。）

训练分两步（warm-start）：
  第 1 步：分别用三段数据训练 3 个专家（各自学自己那一段的回归）；
  第 2 步：冻结专家，把路由器当 3 分类器训练——造数据时知道每个样本属于哪一段，
          直接让它学「输入 x 该交给哪个专家」。

（真实端到端 MoE 是专家+路由器一起训、路由器输出 softmax 软路由做加权组合，
还要加「负载均衡」aux loss 防止所有样本涌向同一个专家。这里为了简单稳定，
路由器用有监督方式训练，分工更干净、更好可视化。）

运行：python3 moe_demo.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ==================== 1. 目标函数 + 造数据 ====================
def target(x):
    """分段函数。返回 (y, 阶段标签)；标签只用于训练路由器/评估/画图。"""
    x = np.asarray(x, np.float64)
    label = (x >= 1 / 3).astype(int) + (x >= 2 / 3).astype(int)   # 0 / 1 / 2
    y = np.where(label == 0, 3 * x,
        np.where(label == 1, 1.0, -3 * x + 3))
    return y, label


def make_data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    y, label = target(x)
    y = y + rng.normal(0, 0.02, n)   # 加一点噪声，变成真正的回归问题
    return x.astype(np.float32), y.astype(np.float32), label


# ==================== 2. 专家 + 路由器 ====================
class Expert(nn.Module):
    """专家：小 MLP，1 维输入 → 1 维输出。"""
    def __init__(self, hidden=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x)


class MoE(nn.Module):
    """MoE = 若干专家 + 一个路由器（软路由：输出 = Σ p_e · expert_e(x)）。"""
    def __init__(self, num_experts=3, hidden=8):
        super().__init__()
        self.experts = nn.ModuleList([Expert(hidden) for _ in range(num_experts)])
        self.gate = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, num_experts))

    def forward(self, x):
        probs, out = self.route(x)
        return (out * probs).sum(dim=-1, keepdim=True)   # [B, 1]

    def route(self, x):
        """返回 (路由分布 [B,E], 各专家输出 [B,E])，画图用。"""
        probs = F.softmax(self.gate(x), dim=-1)               # [B, E]
        out = torch.cat([e(x) for e in self.experts], dim=-1) # [B, E]
        return probs, out


# ==================== 3. 训练 ====================
def fit_reg(model, xs, ys, epochs=300, lr=1e-2, batch_size=64, params=None):
    """回归训练（MSE）。params=None 训练全部参数；否则只训给定参数。"""
    x_t = torch.from_numpy(xs[:, None])
    y_t = torch.from_numpy(ys[:, None])
    opt = torch.optim.Adam(params if params is not None else model.parameters(), lr=lr)
    n = len(xs)
    for _ in range(epochs):
        perm = torch.randperm(n)                       # 每个 epoch 重新 shuffle
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            loss = F.mse_loss(model(x_t[idx]), y_t[idx])
            loss.backward()
            opt.step()


def fit_router(moe, xs, labels, epochs=300, lr=1e-2, batch_size=64):
    """把路由器当 3 分类器训练（交叉熵），输入 x → 预测该交给哪个专家。"""
    x_t = torch.from_numpy(xs[:, None])
    y_t = torch.from_numpy(labels)                     # 阶段标签 0/1/2
    opt = torch.optim.Adam(moe.gate.parameters(), lr=lr)
    n = len(xs)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            loss = F.cross_entropy(moe.gate(x_t[idx]), y_t[idx])   # gate 输出 logits
            loss.backward()
            opt.step()


# ==================== 4. 主流程 ====================
def main():
    torch.manual_seed(0)
    x, y, label = make_data(seed=0)
    n = int(len(x) * 0.8)
    x_tr, y_tr, l_tr = x[:n], y[:n], label[:n]
    x_te, y_te = x[n:], y[n:]

    # --- 第 1 步：每个专家只用自己那段数据训练（warm-start）---
    moe = MoE(3)
    for e in range(3):
        m = l_tr == e
        fit_reg(moe.experts[e], x_tr[m], y_tr[m])

    # --- 第 2 步：冻结专家，只训练路由器（3 分类）---
    for e in moe.experts:
        for p in e.parameters():
            p.requires_grad = False
    fit_router(moe, x_tr, l_tr)

    # --- 测试集表现 ---
    with torch.no_grad():
        pred = moe(torch.from_numpy(x_te[:, None])).numpy().ravel()
    print(f"测试集 MSE: {np.mean((pred - y_te) ** 2):.4f}")

    # --- 画图：左=拟合曲线，右=路由分布 ---
    xs = np.linspace(0, 1, 300, dtype=np.float32)
    with torch.no_grad():
        probs, _ = moe.route(torch.from_numpy(xs[:, None]))
        probs = probs.numpy()
        pred_xs = moe(torch.from_numpy(xs[:, None])).numpy().ravel()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("MoE: 3 experts + 1 router", fontsize=13)

    # 左：训练数据（按真实阶段着色）+ 目标函数 + MoE 拟合
    ax = axes[0]
    for e, c in zip(range(3), ["tab:blue", "tab:orange", "tab:green"]):
        m = l_tr == e
        ax.scatter(x_tr[m], y_tr[m], s=10, c=c, label=f"stage {e}")
    yt, _ = target(xs)
    ax.plot(xs, yt, "k--", lw=2, label="target")
    ax.plot(xs, pred_xs, "r-", lw=2, label="MoE")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("MoE fits a piecewise function")
    ax.legend()

    # 右：路由器学到的软路由分布 p_e(x)
    ax = axes[1]
    for e, c in enumerate(["tab:blue", "tab:orange", "tab:green"]):
        ax.plot(xs, probs[:, e], c=c, lw=2, label=f"p(expert {e})")
    ax.set_xlabel("x"); ax.set_ylabel("routing probability")
    ax.set_title("Router learned to split x into 3 regions")
    ax.legend()

    plt.tight_layout()
    plt.savefig("moe_demo.png", dpi=120, bbox_inches="tight")
    print("图片已保存到 moe_demo.png")
    plt.show()


if __name__ == "__main__":
    main()
