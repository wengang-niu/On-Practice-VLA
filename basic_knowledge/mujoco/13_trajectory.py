"""学习目标：把前面的能力串起来，按固定频率采集一条"轨迹"（状态 + 动作 + 图像）并落盘。

机器人学习的数据就是一条条轨迹：每个时间步记录"观测到什么（状态/图像）+ 做了什么（动作）"。
这里我们用两杆臂 + 固定相机，每 25 个物理子步记一帧（timestep=0.002 → 20 fps），
把字段名对齐 LeRobot 数据集（observation.state / action / observation.image / obj_init）。

运行方式：
    python 13_trajectory.py
（会在当前目录生成 trajectory.npz。）
"""

import mujoco
import numpy as np

# -- 1. 场景：两杆臂 + 一台相机 --------------------------------------------------
XML = """
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <camera name="front" pos="0 -2 1.5" euler="45 0 0" fovy="50"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.7 0.7 0.7 1"/>
    <body name="link1" pos="0 0 0.5">
      <joint name="shoulder" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.4 0 0" size="0.03" rgba="0.8 0.2 0.2 1"/>
      <body name="link2" pos="0.4 0 0">
        <joint name="elbow" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.4 0 0" size="0.03" rgba="0.2 0.2 0.8 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="m1" joint="shoulder"/>
    <motor name="m2" joint="elbow"/>
  </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

# -- 2. 采集参数 -----------------------------------------------------------------
FPS          = 20          # 数据帧率（和 LeRobot 教程一致）
SUBSTEPS     = 25          # 每帧之间的物理子步数（timestep=0.002，25*0.002=0.05s=1/20s）
TOTAL_SECONDS = 5          # 采集 5 秒
TOTAL_FRAMES  = FPS * TOTAL_SECONDS   # 共 100 帧

renderer = mujoco.Renderer(model, height=256, width=256)

# -- 3. 一段演示用的"策略"（这里用正弦波代替真实策略） ------------------------------
def scripted_action(t):
    # 返回两个关节的"目标角"。真实的 VLA 里，这一步是神经网络输出的动作（目标关节角）。
    a1 = 0.5 * np.sin(2 * np.pi * 0.5 * t)   # 肩关节目标角
    a2 = 0.3 * np.sin(2 * np.pi * 0.8 * t)   # 肘关节目标角
    return np.array([a1, a2])

# -- 4. 采集循环 -----------------------------------------------------------------
# 每条轨迹记录：
#   observation.state : 观测到的状态（这里用当前关节角 qpos）
#   action            : 本步采取的动作（这里用策略输出的目标关节角）
#   observation.image : 相机图像（256x256x3）
#   obj_init          : 物体初始状态（reset 后只记录一次）
states, actions, images = [], [], []
obj_init = data.qpos.copy()      # 只在开始记录一次

for frame in range(TOTAL_FRAMES):
    t = frame / FPS
    action = scripted_action(t)
    # 简单 PD：把"目标角"转成力矩再写入 ctrl（真正的 VLA 里这一步由底层控制器完成，见示例 08）
    data.ctrl[:] = 20.0 * (action - data.qpos) - 1.0 * data.qvel

    # 每帧之间跑 SUBSTEPS 个物理子步：仿真跑得快，但我们只按 20 fps 采样记录
    for _ in range(SUBSTEPS):
        mujoco.mj_step(model, data)

    # 记录这一帧
    states.append(data.qpos.copy())          # (2,)  关节角
    actions.append(action.copy())            # (2,)  动作
    renderer.update_scene(data, camera="front")
    images.append(renderer.render())         # (256,256,3) uint8

# -- 5. 落盘 ---------------------------------------------------------------------
states  = np.stack(states)                   # (T, 2)
actions = np.stack(actions)                  # (T, 2)
images  = np.stack(images)                   # (T, 256, 256, 3)
obj_init = np.asarray(obj_init)              # (2,)

np.savez_compressed(
    "trajectory.npz",
    observation_state=states,
    action=actions,
    observation_image=images,
    obj_init=obj_init,
)

print("已保存 trajectory.npz")
print("  observation.state  shape =", states.shape)
print("  action             shape =", actions.shape)
print("  observation.image  shape =", images.shape)
print("  obj_init           shape =", obj_init.shape)

# -- 6. 和 LeRobot 数据格式的对应 ------------------------------------------------
# 目标仓库 lerobot-mujoco-tutorial 采集的数据就是这套东西，只是：
#   - 把每帧数组按行写进 parquet 文件（episode_XXXXXX.parquet）；
#   - 用 meta/info.json 描述每个字段的 dtype/shape；
#   - 它的 observation.state 是末端 6 维位姿(x,y,z,roll,pitch,yaw)，action 是 6 关节+夹爪=7 维。
# 概念完全一致：一段轨迹 = 状态序列 + 动作序列 + 图像序列。
print("\n到这里你已经把 01~12 的知识串成了一条完整的轨迹数据。")
