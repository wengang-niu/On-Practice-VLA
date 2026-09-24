"""最小 MuJoCo viewer 示例：从 XML 创建模型并观察自由落体。"""

import mujoco
import mujoco.viewer


# XML 描述的是模型结构：worldbody 放置场景中的物体，geom 定义碰撞和外观。
# body 建立局部坐标系；freejoint 给小球 3 个平移和 3 个转动自由度。
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


# MjModel 是编译后的静态模型（关节、几何体、时间步等）；
# MjData 保存会随仿真变化的 qpos、qvel、接触和时间等运行时状态。
model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

# passive viewer 不会替我们推进仿真；每次 mj_step 前进一个 model.opt.timestep。
# viewer.sync() 把最新的 MjData 同步到窗口。关闭窗口后循环才会结束。
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
