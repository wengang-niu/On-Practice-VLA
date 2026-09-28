"""学习目标：会用 MuJoCo 自带的交互式查看器，让仿真"持续跑起来"并实时显示。

真实的机器人仿真/训练里，是一个"循环"：每步 mj_step 推进物理，再用 viewer.sync()
把最新画面刷到窗口上。下面这段循环是"标准实时仿真循环"，后面每个会动的示例都会复用它。

运行方式：
    python 03_viewer.py
（会弹出一个窗口，小球按真实速度下落；按住鼠标左键拖动可旋转视角；按 ESC 或关窗退出。）
"""

import time
import mujoco
import mujoco.viewer  # 注意：viewer 是一个子模块，必须显式 import，否则会报错

# -- 1. 场景：自由落体小球 ------------------------------------------------------
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

# -- 2. 打开查看器，进入"实时"仿真循环 ----------------------------------------------
# launch_passive 弹出一个窗口，把窗口对象交给我们。
# while viewer.is_running()：只要窗口还开着，就继续循环（关窗/ESC 即退出）。
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()          # 记下本步开始的时间
        mujoco.mj_step(model, data)       # 推进一个物理步
        viewer.sync()                     # 把最新状态渲染到窗口
        # 对齐到实时：这一步应该用掉 model.opt.timestep 秒，睡掉没用完的部分。
        # 不加这段的话，循环会以最快速度刷帧，小球瞬间就落完了，根本看不清。
        time_until_next = model.opt.timestep - (time.time() - step_start)
        if time_until_next > 0:
            time.sleep(time_until_next)

# 循环退出（窗口被关闭）后，脚本自然结束。
print("查看器已关闭，程序结束。")
