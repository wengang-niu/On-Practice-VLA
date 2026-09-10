"""接触与传感器：观察球落地时的接触数量和法向力。"""

import mujoco


XML = """
<mujoco model="contact_sensor">
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <site name="floor_site" pos="0 0 0.08" size="0.01"/>
    <body name="ball" pos="0 0 1">
      <freejoint/>
      <geom name="ball_geom" type="sphere" size="0.08" mass="0.1" rgba="1 0 0 1"/>
    </body>
  </worldbody>
  <sensor>
    <touch name="floor_touch" site="floor_site"/>
  </sensor>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
touch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "floor_touch")

max_force = 0.0
contact_step = None
for step in range(1500):
    mujoco.mj_step(model, data)
    sensor_force = float(data.sensordata[touch_id])
    max_force = max(max_force, sensor_force)
    if data.ncon > 0 and contact_step is None:
        contact_step = step

print(f"first contact step: {contact_step}")
print(f"maximum touch sensor value: {max_force:.4f}")
print(f"active contacts at end: {data.ncon}")
