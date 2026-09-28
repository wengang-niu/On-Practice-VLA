"""学习目标：区分两种执行器——"位置伺服"（position）与"力矩"（motor）。

- <position joint="..." kp="..." kv="..."/>：你给它一个"目标角度"，它内部用 PD 伺服
  自己算力矩去追这个角度（是 MuJoCo 内置的控制器，有微小的稳态误差）。
- <motor joint="..."/>：你直接给它"力矩"，它原样施加，没有任何"目标"概念。

这一步对应机器人 / VLA 里一个关键问题：策略输出的是"目标"还是"力矩"？
（目标仓库 y_env.py 的 action_type 就是在选这两种。）

运行方式：
    python 07_actuator_modes.py
（弹出窗口：左摆自己追到目标角停住，右摆不动；终端打印两个关节角；关窗退出。）
"""

import time
import mujoco
import mujoco.viewer  # 查看器子模块，必须显式 import

# -- 1. 两个完全相同的单摆，一个用 position 执行器，一个用 motor ---------------------
XML = """
<mujoco>
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <worldbody>
    <!-- 左摆：position 执行器 -->
    <body name="link_pos" pos="-0.5 0 0.5">
      <joint name="j_pos" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.03" rgba="0.8 0.2 0.2 1"/>
    </body>
    <!-- 右摆：motor 执行器 -->
    <body name="link_motor" pos="0.5 0 0.5">
      <joint name="j_motor" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.03" rgba="0.2 0.2 0.8 1"/>
    </body>
  </worldbody>
  <actuator>
    <!-- position：ctrl 给的是"目标角度"，kp 是弹簧刚度，kv 是阻尼（没有 kv 会一直震荡） -->
    <position name="a_pos" joint="j_pos" kp="100" kv="5"/>
    <!-- motor：ctrl 给的是"力矩" -->
    <motor name="a_motor" joint="j_motor"/>
  </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

# -- 2. 命令：position 给目标角 0.3，motor 给 0 力矩 ---------------------------------
data.ctrl[0] = 0.3   # position 执行器：把 0.3 当作"目标角"，它会自己追过去
data.ctrl[1] = 0.0   # motor 执行器：0 力矩，摆就因重力自然垂着不动

print("  步  | position 关节角(rad) | motor 关节角(rad)")
step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(model, data)
        step += 1
        if step == 1 or step % 25 == 0:   # 每 25 步（0.25 秒）打印一次
            print(f"{step:5d} | {data.qpos[0]:+.4f}             | {data.qpos[1]:+.4f}")
        viewer.sync()                     # 把最新状态渲染到窗口
        time_until_next = model.opt.timestep - (time.time() - step_start)
        if time_until_next > 0:
            time.sleep(time_until_next)

# -- 3. 观察到的现象 ------------------------------------------------------------
# position 执行器把关节追到了 0.3 附近并停住（因为重力，会停在略小于 0.3 的位置）。
# motor 执行器没有收到力矩，就一直在初始位置（竖直下垂的平衡点）不动。
print("\n观察：position 会自己'追'目标角；motor 只是忠实施加力矩，你不给力它就不动。")
print("（position 内部其实就是个 PD 控制器——这正是下一个示例要手写的东西。）")
