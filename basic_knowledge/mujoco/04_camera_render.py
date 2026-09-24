"""相机可视化：保存一帧离屏 RGB 图像，同时用 viewer 观察场景运动。"""

import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import mujoco.viewer


XML = """
<mujoco model="camera_render">
  <option gravity="0 0 -9.81"/>
  <!-- offwidth/offheight 设置离屏渲染缓冲区的默认分辨率。 -->
  <visual><global offwidth="256" offheight="256"/></visual>
  <worldbody>
    <light pos="0 -1 2"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.75 0.75 0.75 1"/>
    <body name="box" pos="0 0 0.2">
      <freejoint/>
      <geom type="box" size="0.2 0.2 0.2" mass="0.5" rgba="0.1 0.5 0.9 1"/>
    </body>
    <!-- pos 是相机位置；xyaxes 用两组轴向量定义相机朝向。 -->
    <camera name="front" pos="2 -2 1.5" xyaxes="1 1 0 -0.25 0.25 0.5"/>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
# Renderer 管理离屏图形缓冲区，生成的 RGB 数组可以作为机器人视觉 observation。
renderer = mujoco.Renderer(model, height=256, width=256)
output = Path(__file__).with_name("camera_frame.png")

# 这里保存初始状态的相机画面；mj_forward 只更新派生量，不推进仿真时间。
mujoco.mj_forward(model, data)
renderer.update_scene(data, camera="front")
imageio.imwrite(output, renderer.render())
renderer.close()
print(f"saved {output}")

# Renderer 负责生成 PNG，passive viewer 负责打开实时窗口；两者是不同的可视化出口。
# 下面推进一小段时间，让窗口中的自由箱体运动可见。
with mujoco.viewer.launch_passive(model, data) as viewer:
    for _ in range(300):
        if not viewer.is_running():
            break
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
