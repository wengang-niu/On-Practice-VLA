"""离屏渲染：把 MuJoCo 相机画面保存成 PNG，理解图像观测的来源。"""

from pathlib import Path

import imageio.v2 as imageio
import mujoco


XML = """
<mujoco model="camera_render">
  <option gravity="0 0 -9.81"/>
  <visual><global offwidth="256" offheight="256"/></visual>
  <worldbody>
    <light pos="0 -1 2"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.75 0.75 0.75 1"/>
    <body name="box" pos="0 0 0.2">
      <freejoint/>
      <geom type="box" size="0.2 0.2 0.2" mass="0.5" rgba="0.1 0.5 0.9 1"/>
    </body>
    <camera name="front" pos="2 -2 1.5" xyaxes="1 1 0 -0.25 0.25 0.5"/>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=256, width=256)
mujoco.mj_forward(model, data)
renderer.update_scene(data, camera="front")

output = Path(__file__).with_name("camera_frame.png")
imageio.imwrite(output, renderer.render())
renderer.close()
print(f"saved {output}")
