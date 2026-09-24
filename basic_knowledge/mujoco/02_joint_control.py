"""关节、执行器和控制输入：用位置执行器让摆杆跟踪目标角度。"""

import time

import mujoco
import mujoco.viewer


XML = """
<mujoco model="joint_control">
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <default>
    <!-- damping 消耗速度，armature 增加等效转动惯量，有助于稳定数值仿真。 -->
    <joint damping="0.2" armature="0.01"/>
    <geom rgba="0.25 0.55 0.85 1"/>
  </default>
  <worldbody>
    <light pos="0 0 2"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
    <body name="arm" pos="0 0 0.6">
      <!-- hinge 绕 y 轴转动，range 单位是弧度。 -->
      <joint name="hinge" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.5" size="0.06"/>
    </body>
  </worldbody>
  <actuator>
    <!-- position 的 ctrl 是目标角度，不是直接施加的力矩；kp 是位置伺服增益。 -->
    <!-- ctrlrange 限制策略可以发送的目标角度范围。 -->
    <position name="hinge_position" joint="hinge" kp="8" ctrlrange="-1.5 1.5"/>
  </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
# ID 是模型对象的整数查找句柄；用名字查找比假设数组下标更可靠。
joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hinge")
actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hinge_position")
# qpos 和 qvel 的布局不同：关节角查 qposadr，自由度速度查 dofadr。
qpos_index = model.jnt_qposadr[joint_id]

target = 1.2  # 目标角度，单位是弧度，约 68.8°。
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and data.time < 8.0:
        # 控制循环的顺序是：写入 action -> 推进仿真 -> 同步可视化。
        data.ctrl[actuator_id] = target
        mujoco.mj_step(model, data)
        viewer.sync()
        # 仿真本身可以跑得很快；sleep 让窗口更新速度接近真实时间。
        time.sleep(model.opt.timestep)

print(f"joint angle: {data.qpos[qpos_index]:.3f} rad")
print(f"joint velocity: {data.qvel[model.jnt_dofadr[joint_id]]:.3f} rad/s")
