import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import robosuite as suite

# ============================================================
# 场景：用 robosuite 的 Lift 任务生成轨迹数据
# 输入：机器人本体状态（关节位置+速度+夹爪状态）+ 物体状态（方块位置）
# 输出：7维动作（末端位姿 xyz + 旋转 rpy + 夹爪开合）
# MoE作用：不同阶段（接近/抓取/抬升）激活不同专家
# ============================================================

# ==================== 1. 从 robosuite 采集数据 ====================
def collect_robosuite_data(n_episodes=50, steps_per_episode=200, seed=42):
    """
    在 Lift 环境中用随机动作采集轨迹数据
    返回: states [N, state_dim], actions [N, action_dim]
    """
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=False,         # 关闭可视化，加速采集
        has_offscreen_renderer=False,
        use_camera_obs=False,       # 只用低维状态，不用图像
        control_freq=20,
    )

    all_states = []
    all_actions = []

    np.random.seed(seed)
    for ep in range(n_episodes):
        env.reset()
        for step in range(steps_per_episode):
            # 获取观测：机器人本体状态 + 物体状态
            obs = env._get_observations()

            # 拼接状态向量：关节位置(7) + 关节速度(7) + 夹爪(2) + 物体位置(3) + 物体姿态(4)
            state = np.concatenate([
                obs["robot0_joint_pos"],       # 7维
                obs["robot0_joint_vel"],       # 7维
                obs["robot0_gripper_qpos"],    # 2维
                obs["object-state"],           # 14维（方块位置+姿态）
            ])

            # 随机动作：7维（OSC控制器：dx,dy,dz,dr,dp,dy, gripper）
            action = env.action_space.sample()

            all_states.append(state)
            all_actions.append(action)

            env.step(action)

    env.close()
    states = np.array(all_states, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.float32)
    print(f"采集完成: {len(states)} 条样本, 状态维度={states.shape[1]}, 动作维度={actions.shape[1]}")
    return states, actions


# ==================== 2. 专家网络 ====================
class Expert(nn.Module):
    """每个专家专精某一类动作模式（如接近/抓取/抬升）"""
    def __init__(self, in_dim, out_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# ==================== 3. 路由器 ====================
class Router(nn.Module):
    """根据当前状态决定激活哪 K 个专家"""
    def __init__(self, in_dim, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(in_dim, num_experts, bias=False)

    def forward(self, x):
        logits = self.gate(x)                        # [B, E]
        top_k_vals, top_k_idx = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(top_k_vals, dim=-1)      # [B, K]
        return weights, top_k_idx


# ==================== 4. MoE 层 ====================
class MoELayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = Router(in_dim, num_experts, top_k)
        self.experts = nn.ModuleList([
            Expert(in_dim, out_dim) for _ in range(num_experts)
        ])
        self.aux_loss_coef = 0.01  # 负载均衡系数

    def forward(self, x):
        batch_size = x.shape[0]
        weights, top_k_idx = self.router(x)

        # 负载均衡辅助损失：防止所有样本路由到同一个专家
        gate_probs = F.softmax(self.router.gate(x), dim=-1)
        aux_loss = self.aux_loss_coef * gate_probs.mean(dim=0).var()

        # 加权组合 Top-K 专家输出
        out_dim = self.experts[0].net[-1].out_features
        output = torch.zeros(batch_size, out_dim, device=x.device, dtype=x.dtype)

        for k in range(self.top_k):
            expert_idx = top_k_idx[:, k]              # [B]
            weight = weights[:, k]                     # [B]
            for i in range(batch_size):
                eid = expert_idx[i].item()
                output[i] += weight[i] * self.experts[eid](x[i:i+1]).squeeze(0)

        return output, aux_loss, top_k_idx


# ==================== 5. 训练 ====================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 采集数据 ---
    print("\n=== 正在从 robosuite 采集数据 ===")
    states_np, actions_np = collect_robosuite_data(n_episodes=50, steps_per_episode=200)

    # 划分训练/测试集（80/20）
    n = len(states_np)
    split = int(n * 0.8)
    states_train = torch.FloatTensor(states_np[:split])
    actions_train = torch.FloatTensor(actions_np[:split])
    states_test = torch.FloatTensor(states_np[split:])
    actions_test = torch.FloatTensor(actions_np[split:])

    train_dataset = TensorDataset(states_train, actions_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # --- 初始化模型 ---
    state_dim = states_np.shape[1]   # 约33维
    action_dim = actions_np.shape[1] # 7维
    moe = MoELayer(state_dim, action_dim, num_experts=4, top_k=2).to(device)
    optimizer = torch.optim.Adam(moe.parameters(), lr=1e-3)

    # --- 训练 ---
    n_epochs = 50
    print(f"\n=== 开始训练 (状态{state_dim}维 → 动作{action_dim}维) ===")
    for epoch in range(n_epochs):
        total_loss = 0
        total_aux = 0
        for batch_states, batch_actions in train_loader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)

            pred, aux_loss, _ = moe(batch_states)
            mse_loss = F.mse_loss(pred, batch_actions)
            loss = mse_loss + aux_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += mse_loss.item()
            total_aux += aux_loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | "
                  f"MSE: {total_loss/len(train_loader):.4f} | "
                  f"Aux: {total_aux/len(train_loader):.4f}")

    # --- 测试集评估 ---
    moe.eval()
    with torch.no_grad():
        states_test = states_test.to(device)
        actions_test = actions_test.to(device)
        pred, _, _ = moe(states_test)
        test_mse = F.mse_loss(pred, actions_test).item()
        print(f"\n测试集 MSE: {test_mse:.4f}")

    # --- 路由观察 ---
    print("\n=== 路由观察（不同状态激活的专家）===")
    with torch.no_grad():
        for i in range(5):
            sample = states_test[i:i+1]
            _, _, top_k_idx = moe(sample)
            print(f"样本{i} → 激活专家: {top_k_idx[0].tolist()}")


if __name__ == '__main__':
    main()