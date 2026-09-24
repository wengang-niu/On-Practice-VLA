import mujoco
import mujoco.viewer

XML = """
<mujoco model="free_fall">
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.30" rgba="0.8 0.8 0.8 1"/>
    <body name="ball" pos="0 0 5">
      <freejoint/>
      <geom name="ball_geom" type="sphere" size="0.08" mass="0.1" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

# 阻塞模式：自动推进仿真，无需手动调用 mj_step
mujoco.viewer.launch(model, data=data)