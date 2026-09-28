"""学习目标：理解触觉/力传感器 <touch>，并用 data.sensordata 读出接触力。

touch 传感器挂在某个 site（参考点）上，测该点处的"接触法向力"。
你会看到小球落地前 sensordata≈0，落地后≈球的重力 mg。

运行方式：
    python 10_touch_sensor.py
"""

import mujoco

# -- 1. 小球底部贴一个 site，再挂一个 touch 传感器 ---------------------------------
# <site> 是"参考点"，可以放在 body 上的任意位置（局部坐标）。
# <touch site="bottom"/>：测量 site 所在位置处的接触力。
# 注意：site 必须正好落在"会发生接触的位置"，否则测到 0。这里放在球面最低点 z=-0.1。
XML = """
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="ball" pos="0 0 0.3">
      <freejoint/>
      <geom type="sphere" size="0.1" rgba="0.8 0.2 0.2 1"/>
      <!-- 球底部（z=-0.1，即球面最低点）放一个 site -->
      <site name="bottom" pos="0 0 -0.1" size="0.005"/>
    </body>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.7 0.7 0.7 1"/>
  </worldbody>
  <sensor>
    <touch site="bottom" name="contact_force"/>
  </sensor>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

# -- 2. 球的质量（用于对照 touch 读数） -------------------------------------------
ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
mass = model.body_mass[ball_id]
print(f"小球质量 = {mass:.4f} kg，理论重力 mg = {mass * 9.81:.4f} N")

print("\n  步  | touch 读数 (N)")
for step in range(1, 301):
    mujoco.mj_step(model, data)
    if step in (1, 50, 100, 150, 200, 300):
        print(f"{step:5d} | {data.sensordata[0]:.4f}")

# -- 3. 讲解 ------------------------------------------------------------------
# 落地前 touch≈0（没有接触）；落地后 touch≈mg（地面对球的支持力 = 重力）。
# 这就是力传感器的意义：不用看几何，直接读一个数值就知道有没有接触、接触多重。
print(f"\n落地后 touch ≈ {data.sensordata[0]:.4f} N，与理论重力 {mass * 9.81:.4f} N 接近。")

print("\n提示：如果把 site 放在球心（pos='0 0 0'），那个点永远碰不到地面，就会一直读到 0。")
print("     —— 所以 sensor 要贴在真正会发生接触的地方。")
