"""学习目标：会用 MuJoCo 自带的交互式查看器，让仿真"持续跑起来"并实时显示。

前面我们是手动调用了几次 mj_step。真实的机器人仿真/训练里，是一个"循环"：
每步 mj_step 推进物理，再用 viewer.sync() 把最新画面刷到窗口上。

运行方式：
    python 03_viewer.py
（会弹出一个窗口，小球持续下落；按 ESC 或点关闭按钮退出。）
"""

import mujoco
import mujoco.viewer  # 注意：viewer 是一个子模块，必须显式 import，否则会报错

# -- 1. 场景：和 02 一样的自由落体小球 -------------------------------------------
XML = """
<mujoco>
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="ball" pos="0 0 2">
      <freejoint/>
      <geom type="sphere" size="0.1" rgba="0.8 0.2 0.2 1"/>
    </body>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.7 0.7 0.7 1"/>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

# -- 2. 打开查看器，进入持续仿真循环 ----------------------------------------------
# launch_passive 会弹出一个窗口，并把窗口对象交给我们。
# while viewer.is_running()：只要窗口还开着，就继续循环。
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)   # 推进一个物理步
        viewer.sync()                 # 把最新状态渲染到窗口

# 循环退出（窗口被关闭）后，脚本自然结束。
print("查看器已关闭，程序结束。")
