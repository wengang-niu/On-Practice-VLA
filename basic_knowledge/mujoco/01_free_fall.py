"""最小 MuJoCo 示例：创建 XML、推进仿真并读取自由落体状态。"""

import mujoco


XML = """
<mujoco model="free_fall">
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
    <body name="ball" pos="0 0 1">
      <freejoint/>
      <geom name="ball_geom" type="sphere" size="0.08" mass="0.1" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

for _ in range(1000):
    mujoco.mj_step(model, data)

print(f"simulation time: {data.time:.3f}s")
print(f"ball position: {data.qpos[:3]}")
print(f"ball velocity: {data.qvel[:3]}")
