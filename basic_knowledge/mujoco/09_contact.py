"""学习目标：理解接触检测的原始数据——data.ncon 与 data.contact。

MuJoCo 每步会检测物体之间的接触，结果存在 data.contact 数组里（data.ncon 是接触点个数）。
你会在窗口里看到小球落地，终端打印 ncon 从 0 变成 1，落地后能读出接触距离、接触点位置。

运行方式：
    python 09_contact.py
（弹出窗口：小球落下、落地；终端打印接触点数；关窗后打印接触细节。）
"""

import time
import mujoco
import mujoco.viewer  # 查看器子模块，必须显式 import
import numpy as np

# -- 1. 自由落体小球 + 地板 -----------------------------------------------------
XML = """
<mujoco>
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="ball" pos="0 0 1">
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
step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(model, data)
        step += 1
        if step == 1 or step % 10 == 0:   # 每 10 步（0.1 秒）打印一次
            print(f"{step:5d} | {data.ncon}              | {data.qpos[2]:.4f}")
        viewer.sync()                     # 把最新状态渲染到窗口
        time_until_next = model.opt.timestep - (time.time() - step_start)
        if time_until_next > 0:
            time.sleep(time_until_next)

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
