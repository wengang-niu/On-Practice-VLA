"""学习目标：理解仿真循环的核心——`mj_step` 如何"前进一个时间步"并更新状态。

这一步给小球加一个 freejoint（自由关节，可平移+旋转），让它自由下落。
你会在终端里看到 data.time 逐步增加、小球高度 data.qpos[2] 逐步下降。

运行方式：
    python 02_free_fall.py
"""

import mujoco

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

# -- 2. 手动推进若干仿真步 ------------------------------------------------------
# mj_step(model, data) 做了两件事：
#   1) 根据当前的位置/速度，计算所有的力（重力、接触力……）；
#   2) 用半隐式欧拉积分把状态"往前推一个 timestep"。
# 每调用一次，data.time 就增加 model.opt.timestep（这里是 0.01 秒）。
for step in range(1, 21):
    mujoco.mj_step(model, data)
    # 只打印前几步 + 每 5 步一次，避免刷屏
    if step <= 5 or step % 5 == 0:
        print(f"第 {step:2d} 步: time={data.time:.2f}s, 高度 z={data.qpos[2]:.3f} m")

# -- 3. 对照：mj_forward 只"计算"，不"前进时间" -----------------------------------
# mj_forward(model, data) 只根据当前 qpos/qvel 计算派生量（世界坐标、接触力等），
# 但不会推进时间，也不会改变 qpos/qvel。
# 想"让世界向前演化"用 mj_step；只想"读某个时刻的受力/接触"用 mj_forward。
t_before = data.time
mujoco.mj_forward(model, data)
print("\nmj_forward 之后 time 是否变化？", data.time != t_before, "（False 表示时间没动）")

print("\n到这里你已经知道：mj_step 是'让世界前进一格'的开关，每步走一个 timestep。")
