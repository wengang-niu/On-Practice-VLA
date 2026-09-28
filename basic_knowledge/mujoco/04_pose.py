"""学习目标：理解"自由体"的位姿表示——位置（x,y,z）加四元数（qw,qx,qy,qz）。

有 freejoint 的物体，其位置向量 qpos 是 7 维：
    [x, y, z, qw, qx, qy, qz]
其中后 4 个是表示姿态的四元数，注意 w 在第一个（不是常见的 xyz-w 顺序）。
这一步为后面读懂目标仓库 transforms.py 里的 quat2r / rpy2r 打基础。

运行方式：
    python 04_pose.py
"""

import mujoco
import numpy as np

# -- 1. 一个带 freejoint 的小球，初始就有位置和姿态 -------------------------------
# pos 给位置，quat 给姿态（四元数）。这里 "0.7071 0 0 0.7071" 表示绕 z 轴转 90°。
XML = """
<mujoco>
  <worldbody>
    <body name="ball" pos="1 2 0.5" quat="0.7071 0 0 0.7071">
      <freejoint/>
      <geom type="sphere" size="0.1" rgba="0.8 0.2 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

# -- 2. 看 7 维位置向量 ---------------------------------------------------------
qpos = data.qpos
print("qpos (7 维) =", qpos)
print("  -> 位置 [x, y, z]           =", qpos[0:3])
print("  -> 姿态四元数 [qw,qx,qy,qz] =", qpos[3:7])

# -- 3. 四元数的含义 ------------------------------------------------------------
# 四元数表示"绕某根轴转多少度"。上面 quat 表示绕 z 轴转 90°：
#   qw = cos(45°), qz = sin(45°) ≈ 0.7071
# 单位四元数（不旋转）是 [1, 0, 0, 0]。MuJoCo 里 w 在第一个，很多库是 [x,y,z,w]，别搞混。
print("\n四元数顺序是 [w,x,y,z]，w 在前；不旋转时是 [1,0,0,0]。")

# -- 4. 世界坐标里的"几何中心"（派生量） ------------------------------------------
# data.geom_xpos 是每个几何形状在世界坐标系下的中心坐标（3 维）。
# 注意它是"派生量"：由 qpos 计算而来，必须先 mj_forward 才被填上（否则是 0）。
# 想移动物体要改 qpos，而不是直接改 geom_xpos。
mujoco.mj_forward(model, data)   # 先做一次 forward，派生量才会被计算
print("\ndata.geom_xpos (球的世界坐标) =", data.geom_xpos[0])

# -- 5. 用 MuJoCo 自带的函数在四元数和旋转矩阵之间互转 -----------------------------
# 这一步"只看一眼"，知道有这么个工具即可；后面读目标仓库 transforms.py 会真正用上。
quat = qpos[3:7].copy()
R = np.zeros(9)
mujoco.mju_quat2Mat(R, quat)      # 四元数 -> 3x3 旋转矩阵（按行展开成 9 个数）
print("\n四元数 -> 旋转矩阵 (3x3)：")
print(R.reshape(3, 3).round(3))

print("\n到这里你已经知道：位姿 = 位置(3) + 四元数(4)，共 7 个数。")
