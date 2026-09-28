"""学习目标：理解关节（hinge）与执行器（motor），并知道 data.ctrl 施加的是"力矩"。

一个 hinge 关节 = 一个旋转自由度（绕某根轴转）。motor 执行器能给这个关节施加力矩。
你会在终端里看到：给常数力矩后，角度 qpos 越来越快地变化（角加速度）。

运行方式：
    python 05_single_joint.py
"""

import mujoco

# -- 1. 一个"单摆"：一根杆绕 y 轴旋转 ---------------------------------------------
# <joint type="hinge" axis="0 1 0"/>：铰链关节，绕 y 轴旋转，1 个自由度。
# <geom type="capsule" fromto="..."/>：胶囊体（两端半球+圆柱），fromto 给两个端点。
# <motor joint="shoulder"/>：作用在 shoulder 关节上的力矩执行器。
XML = """
<mujoco>
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="link" pos="0 0 0.5">
      <joint name="shoulder" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.03" rgba="0.8 0.2 0.2 1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="m1" joint="shoulder"/>
  </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

print("维度：nq =", model.nq, "（1 个关节角度）")
print("      nv =", model.nv, "（1 个关节角速度）")
print("      nu =", model.nu, "（1 个执行器）")

# -- 2. 施加常数力矩，观察角度变化 -------------------------------------------------
# data.ctrl 是控制向量，长度等于 nu。这里 ctrl[0] 就是施加在 shoulder 上的力矩（N·m）。
# 力矩 -> 角加速度 -> 角速度 -> 角度。所以角度会"越来越快"地变化。
data.ctrl[0] = 1.0   # 给 1 N·m 的力矩

print("\n  步   time(s)   角度 qpos(rad)  角速度 qvel(rad/s)")
for step in range(1, 21):
    mujoco.mj_step(model, data)
    if step % 2 == 0:
        print(f"{step:4d}   {data.time:.2f}      {data.qpos[0]:+.4f}          {data.qvel[0]:+.4f}")

print("\n注意：qpos 的单位是弧度（rad），不是度。ctrl 的单位是力矩（N·m），不是角度。")
