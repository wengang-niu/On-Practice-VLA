"""学习目标：理解仿真循环的核心——`mj_step` 如何"前进一个时间步"并更新状态。

这一步给小球加一个 freejoint（自由关节，可平移+旋转），让它自由下落。
你会在终端里看到 data.time 逐步增加、小球高度 data.qpos[2] 逐步下降，
同时窗口里看到小球从高处落到地板上。

运行方式：
    python 02_free_fall.py
（弹出一个窗口，小球实时下落；终端同时打印时间与高度；关窗退出。）
"""

import time
import mujoco
import mujoco.viewer  # 查看器子模块，必须显式 import

# -- 1. 场景：一个带 freejoint 的小球 + 地板 -------------------------------------
# <freejoint/> 表示这个 body 相对世界可以自由平移和旋转（共 6 个自由度）。
# 因此它的位置需要 7 个数描述：x,y,z + 四元数 qw,qx,qy,qz（下一步会细讲）。
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

print("初始时刻：time =", data.time, "，小球高度 z =", data.qpos[2])

# -- 2. 对照：mj_forward 只"计算"，不"前进时间" -----------------------------------
# mj_forward(model, data) 只根据当前 qpos/qvel 计算派生量（世界坐标、接触力等），
# 但不会推进时间，也不会改变 qpos/qvel。
# 想"让世界向前演化"用 mj_step；只想"读某个时刻的受力/接触"用 mj_forward。
t_before = data.time
mujoco.mj_forward(model, data)
print("mj_forward 之后 time 是否变化？", data.time != t_before, "（False 表示时间没动）")

# -- 3. 打开查看器，实时推进仿真 -------------------------------------------------
# mj_step(model, data) 做了两件事：
#   1) 根据当前的位置/速度，计算所有的力（重力、接触力……）；
#   2) 用半隐式欧拉积分把状态"往前推一个 timestep"。
# 每调用一次，data.time 就增加 model.opt.timestep（这里是 0.01 秒）。
# while viewer.is_running()：只要窗口还开着就继续，所以你能一直看到小球落下、落地、停住。
print("\n  步   time(s)   高度 z(m)")
step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(model, data)
        step += 1
        if step == 1 or step % 25 == 0:   # 每 25 步（0.25 秒）打印一次，避免刷屏
            print(f"{step:4d}   {data.time:.2f}     {data.qpos[2]:.4f}")
        viewer.sync()                     # 把最新状态渲染到窗口
        # 对齐到实时：这一步应该用掉 timestep 秒，睡掉没用完的部分，让画面按真实速度播放
        time_until_next = model.opt.timestep - (time.time() - step_start)
        if time_until_next > 0:
            time.sleep(time_until_next)

print("\n到这里你已经知道：mj_step 是'让世界前进一格'的开关，每步走一个 timestep。")
