import mujoco
import mujoco.viewer


XML = """
<mujoco>
  <option gravity="0 0 -9.81"/>

  <worldbody>
    <light pos="0 0 3"/>

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

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
