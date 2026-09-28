"""学习目标：理解多刚体/父子 body 的嵌套，以及多个关节如何在 qpos 里排列。

一个 body 可以"挂"在另一个 body 下面（父子关系）。这里搭一个两杆机械臂：
肩关节 + 肘关节，共 2 个自由度。你还会学会用 mj_name2id 按名字定位某个关节。

运行方式：
    python 06_two_link_arm.py
（弹出窗口：两杆臂在力矩下摆动；终端打印两个关节角；关窗退出。）
"""

import time
import mujoco
import mujoco.viewer  # 查看器子模块，必须显式 import

# -- 1. 两杆臂 -----------------------------------------------------------------
# link1 挂在世界里（肩关节 shoulder），link2 挂在 link1 末端（肘关节 elbow）。
# 注意：link2 的 pos 是"相对父 body link1 的坐标系"，(0.4,0,0) 表示在 link1 的 x 正方向 0.4m 处。
XML = """
<mujoco>
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <worldbody>
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

print("维度：nq =", model.nq, "，nv =", model.nv, "，nu =", model.nu)
print("qpos 初始 =", data.qpos)   # [肩角度, 肘角度]

# -- 2. 按名字定位关节在 qpos 里的位置 -------------------------------------------
# mj_name2id 把"名字"换成"id"（索引）。mjOBJ_JOINT 表示我们要找的是关节。
shoulder_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder")
elbow_id    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "elbow")
print("shoulder 的 id =", shoulder_id, "，elbow 的 id =", elbow_id)

# model.jnt_qposadr[joint_id] 给出该关节的角度在 qpos 里的"起始下标"。
# 对我们这种每个关节 1 个自由度的场景，下标正好等于关节 id。
print("shoulder 角度在 qpos 的下标 =", model.jnt_qposadr[shoulder_id])
print("elbow    角度在 qpos 的下标 =", model.jnt_qposadr[elbow_id])

# -- 3. 给两个关节各施加一个力矩，看它在窗口里摆动 ----------------------------------
data.ctrl[0] = 0.5   # 肩关节 0.5 N·m
data.ctrl[1] = -0.3  # 肘关节 -0.3 N·m
print("\n  步   肩角度(rad)  肘角度(rad)")
step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(model, data)
        step += 1
        if step == 1 or step % 25 == 0:   # 每 25 步（0.25 秒）打印一次
            print(f"{step:4d}   {data.qpos[0]:+.4f}     {data.qpos[1]:+.4f}")
        viewer.sync()                     # 把最新状态渲染到窗口
        time_until_next = model.opt.timestep - (time.time() - step_start)
        if time_until_next > 0:
            time.sleep(time_until_next)
