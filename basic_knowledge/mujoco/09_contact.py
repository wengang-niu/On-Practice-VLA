"""学习目标：理解接触检测的原始数据——data.ncon 与 data.contact。

MuJoCo 每步会检测物体之间的接触，结果存在 data.contact 数组里（data.ncon 是接触点个数）。
你会看到小球落地前 ncon=0，落地后变成 1，并能读出接触距离、接触点位置。

运行方式：
    python 09_contact.py
"""

import mujoco
import numpy as np

# -- 1. 自由落体小球 + 地板 -----------------------------------------------------
XML = """
<mujoco>
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="ball" pos="0 0 0.3">
      <freejoint/>
      <geom type="sphere" size="0.1" rgba="0.8 0.2 0.2 1"/>
    </body>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.7 0.7 0.7 1"/>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

print("  步  | ncon(接触点数) | 球高 z(m)")
for step in range(1, 61):
    mujoco.mj_step(model, data)
    if step <= 5 or step % 10 == 0:
        print(f"{step:5d} | {data.ncon}              | {data.qpos[2]:.4f}")

# -- 2. 落地后，读接触细节 -------------------------------------------------------
# 重要：访问 data.contact[i] 之前必须先判断 data.ncon > 0，否则会越界报错。
print("\n落地后的接触信息：")
if data.ncon > 0:
    c = data.contact[0]
    print("  dist     (接触距离，负值=已穿透) =", round(c.dist, 4))
    print("  pos      (接触点世界坐标)        =", np.round(c.pos, 4))
    print("  geom1/geom2 (接触的两个几何 id)  =", c.geom1, c.geom2)
else:
    print("  还没有接触。")

# -- 3. 讲解 ------------------------------------------------------------------
# - data.ncon：当前接触点个数（一个球压在地板上通常 ncon=1）。
# - dist：接触距离；>0 表示还有间隙，<0 表示发生穿透，=0 表示刚好接触。
# - pos：接触点在"世界坐标系"下的位置（米）。
# - geom1/geom2：参与这次接触的两个几何形状的 id。
print("\n到这里你已经知道：接触信息 = ncon 计数 + contact 数组，读之前先看 ncon。")
