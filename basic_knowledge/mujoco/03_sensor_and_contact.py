"""接触与传感器可视化：观察球落地时的接触数量和传感器读数。"""

import time

import mujoco
import mujoco.viewer


XML = """
<mujoco model="contact_sensor">
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <!-- geom 参与碰撞；site 只是无碰撞的标记，适合挂载传感器。 -->
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <site name="floor_site" pos="0 0 0.08" size="0.01"/>
    <body name="ball" pos="0 0 1">
      <freejoint/>
      <geom name="ball_geom" type="sphere" size="0.08" mass="0.1" rgba="1 0 0 1"/>
    </body>
  </worldbody>
  <sensor>
    <!-- touch 传感器读取 site 附近的接触载荷，输出位于 sensordata。 -->
    <touch name="floor_touch" site="floor_site"/>
  </sensor>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
# sensor ID 用于索引 sensordata，不是 geom ID，也不是 contact 列表中的编号。
touch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "floor_touch")

max_force = 0.0
contact_step = None
# viewer 让你能看到“传感器什么时候开始有读数”；统计仍按每个仿真步采样。
with mujoco.viewer.launch_passive(model, data) as viewer:
    for step in range(1500):
        if not viewer.is_running():
            break
        # mj_step 会完成本步动力学、碰撞检测和约束求解，然后更新传感器输出。
        mujoco.mj_step(model, data)
        viewer.sync()
        sensor_force = float(data.sensordata[touch_id])
        # 这是离散采样到的峰值，不一定等于连续时间上的严格最大值。
        max_force = max(max_force, sensor_force)
        # ncon 表示当前时刻的 active contacts 数量，不是整个 episode 的累计数量。
        if data.ncon > 0 and contact_step is None:
            contact_step = step
        time.sleep(model.opt.timestep)

print(f"first contact step: {contact_step}")
print(f"maximum touch sensor value: {max_force:.4f}")
print(f"active contacts at end: {data.ncon}")
