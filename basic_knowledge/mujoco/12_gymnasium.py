"""学习目标：学会用 Gymnasium 的标准强化学习环境接口（reset / step / action_space）。

Gymnasium 是强化学习环境的统一接口。它的 MuJoCo 内置环境（如 HalfCheetah）底层
用的就是前面 11 个示例里的 mjModel / mjData / mj_step。这一步你只学"接口怎么用"。

运行方式：
    pip install gymnasium     # 若还没装（本目录 requirements.txt 已包含）
    python 12_gymnasium.py
（会弹出查看器窗口，看到半猎豹跑动。）
"""

import gymnasium as gym   # 注意是 gymnasium，不是老的 gym

# -- 1. 创建环境 ----------------------------------------------------------------
# HalfCheetah（半猎豹）是一个经典 MuJoCo 连续控制任务。
# render_mode="human" 会弹出交互式查看器；改成 "rgb_array" 则返回图像帧。
env = gym.make("HalfCheetah-v5", render_mode="human")

# -- 2. 看动作空间和观测空间 ----------------------------------------------------
print("动作空间 action_space      :", env.action_space)        # 6 维连续 [-1,1]
print("观测空间 observation_space :", env.observation_space)   # 17 维连续

# -- 3. reset：重置环境，返回 (观测, 信息) ----------------------------------------
# 注意：新版 Gymnasium 的 reset 返回二元组 (obs, info)，不是只返回 obs。
obs, info = env.reset(seed=42)
print("\n初始观测 obs shape =", obs.shape)

# -- 4. 交互循环：随机动作跑几步，看 step 的 5 元组返回值 ---------------------------
# step 返回 (观测, 奖励, 是否终止, 是否截断, 信息)：
#   terminated：任务"成功/失败"导致的结束（如猎豹摔倒）
#   truncated ：时间上限到了被"截断"（与 terminated 语义不同，但常一起判断）
total_reward = 0.0
for t in range(200):
    action = env.action_space.sample()                     # 随机采样一个动作
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        obs, info = env.reset()                            # 结束后重置
        break

print("随机策略跑完，累计奖励 =", round(total_reward, 2))

env.close()   # 关闭环境，释放查看器
print("\n到这里你已经知道：Gymnasium 环境 = reset 给初始观测，step 喂动作拿下一帧观测。")
