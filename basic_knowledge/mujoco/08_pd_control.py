"""学习目标：手写一个最简单的 PD 反馈控制器，把关节稳定到目标角度。

PD 控制器：力矩 = kp*(目标角 - 当前角) - kv*当前角速度
它是机器人控制里最基础、最重要的闭环控制器。VLA 里"策略输出目标 → 底层控制器
把目标变成力矩"正是这个思想（目标仓库 y_env.py 里动作执行的本质）。

运行方式：
    python 08_pd_control.py
（弹出窗口：单摆从下垂位置被拉到 0.5 rad 并停住；终端打印角度/力矩；关窗退出。）
"""

import time
import mujoco
import mujoco.viewer  # 查看器子模块，必须显式 import

# -- 1. 一个单摆（motor 执行器，供我们写入 PD 算出的力矩） ---------------------------
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

# -- 2. PD 参数与目标 -----------------------------------------------------------
q_des = 0.5    # 目标角度（rad）
kp    = 20.0   # 比例增益：差得越多，拉得越用力
kv    = 2.0    # 微分增益：提供阻尼，避免来回震荡

# -- 3. 控制循环：每步算误差 -> 算力矩 -> 写入 data.ctrl -> 推进仿真 -----------------
print("  步  | 当前角 q(rad) | 目标(rad) | 力矩 tau(N·m)")
step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        q  = data.qpos[0]                  # 当前角度
        qd = data.qvel[0]                  # 当前角速度
        tau = kp * (q_des - q) - kv * qd   # PD 控制律
        data.ctrl[0] = tau                 # 把算出的力矩交给 motor
        mujoco.mj_step(model, data)
        step += 1
        if step == 1 or step % 25 == 0:    # 每 25 步（0.25 秒）打印一次
            print(f"{step:5d} | {data.qpos[0]:+.4f}       | {q_des:+.4f}   | {data.ctrl[0]:+.4f}")
        viewer.sync()                      # 把最新状态渲染到窗口
        time_until_next = model.opt.timestep - (time.time() - step_start)
        if time_until_next > 0:
            time.sleep(time_until_next)

# -- 4. 结果说明 ----------------------------------------------------------------
# 角度从 0 逐渐逼近 0.5 rad：误差大时力矩大、把关节快速拉过去，接近时力矩减小。
# 由于重力一直往下拉，最终会停在略小于 0.5 的位置（这叫"稳态误差"，kp 越大误差越小）。
print("\n到这里你已经知道：闭环反馈 = 看误差、算力、写入 ctrl、推进仿真，循环往复。")
