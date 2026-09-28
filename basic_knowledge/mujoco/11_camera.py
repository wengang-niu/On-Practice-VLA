"""学习目标：理解相机（camera）与离屏渲染，得到一张 RGB 图像（就是 VLA 的 observation.image）。

MuJoCo 可以在场景里定义相机，用 mujoco.Renderer 离屏渲染成 numpy 数组。
这张 (H, W, 3) 的 uint8 RGB 图，正是 VLA 模型"看"到的东西。

运行方式：
    python 11_camera.py
（会在当前目录生成 camera_front.png 一张渲染图。）
"""

import mujoco
from PIL import Image

# -- 1. 带相机和灯光的场景 ------------------------------------------------------
# <light>：光源（没有光场景会全黑）。
# <camera name="front" .../>：一台固定相机，pos 是位置，euler 是朝向（单位：度）。
XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <camera name="front" pos="0 -1.5 1.5" euler="45 0 0" fovy="50"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.7 0.7 0.7 1"/>
    <body name="ball" pos="0 0 0.1">
      <freejoint/>
      <geom type="sphere" size="0.1" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

# 先仿真几步，让小球落到地板上（否则它悬空，画面里位置不自然）
for _ in range(50):
    mujoco.mj_step(model, data)

# -- 2. 离屏渲染器 -------------------------------------------------------------
# Renderer 不需要显示器，直接在内存里画出图像。
renderer = mujoco.Renderer(model, height=480, width=640)

# update_scene：把当前状态按指定相机的视角"摆进"渲染器。
# 参数 camera 直接传相机的名字字符串即可。
renderer.update_scene(data, camera="front")

# render()：真正画出来，返回 (H, W, 3) 的 uint8 数组，0~255 的 RGB。
img = renderer.render()

print("图像 shape =", img.shape, "，dtype =", img.dtype)
print("像素值范围 =", img.min(), "~", img.max(), "（uint8 的 0~255）")

# -- 3. 保存成 PNG -------------------------------------------------------------
# Pillow 的 fromarray 直接吃 RGB 的 uint8 数组。
Image.fromarray(img).save("camera_front.png")
print("已保存 camera_front.png，打开看看——这张图就是机器人'看到'的画面。")

print("\n到这里你已经知道：相机 + 渲染器，就能把仿真场景变成一张可以喂给模型的图像。")
