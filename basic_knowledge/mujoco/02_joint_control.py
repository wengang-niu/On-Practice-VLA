"""关节、执行器和控制输入：用位置执行器让摆杆跟踪目标角度。"""

import time

import mujoco
import mujoco.viewer


XML = """
<mujoco model="joint_control">
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <default>
    <joint damping="0.2" armature="0.01"/>
    <geom rgba="0.25 0.55 0.85 1"/>
  </default>
  <worldbody>
    <light pos="0 0 2"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
    <body name="arm" pos="0 0 0.6">
      <joint name="hinge" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.5" size="0.06"/>
    </body>
  </worldbody>
  <actuator>
    <position name="hinge_position" joint="hinge" kp="8" ctrlrange="-1.5 1.5"/>
  </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hinge")
actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hinge_position")
qpos_index = model.jnt_qposadr[joint_id]

target = 0.8
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and data.time < 8.0:
        data.ctrl[actuator_id] = target
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)

print(f"joint angle: {data.qpos[qpos_index]:.3f} rad")
print(f"joint velocity: {data.qvel[model.jnt_dofadr[joint_id]]:.3f} rad/s")
