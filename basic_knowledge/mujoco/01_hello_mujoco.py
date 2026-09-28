"""学习目标：认识 MuJoCo 的两个核心对象——模型 `mjModel` 与状态 `mjData`，并会读模型维度。

这一步先"只加载、不仿真"。你只需要建立两个直觉：
1. `mjModel`：不可变的物理定义（几何、质量、关节、执行器等），一次加载后基本不再改。
2. `mjData`：可变的状态（位置、速度、接触等），仿真循环每步都在更新它。

运行方式：
    python 01_hello_mujoco.py
（终端打印模型信息，最后会弹出一个窗口展示这个静态场景。）
"""

import mujoco
import mujoco.viewer  # 查看器是一个子模块，必须显式 import

# -- 1. 用 MJCF（MuJoCo XML）字符串描述一个最简单的静态场景 ------------------------
# MuJoCo 的场景用 MJCF（MuJoCo XML）格式描述。
# 这里只有一个灰色地板，和一个悬在半空的红色小方块（没有关节，所以是"固定"的）。
XML = """
<mujoco>
  <worldbody>
    <!-- 地板：type="plane" 是无限大平面，size 里前两个是半长半宽 -->
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.7 0.7 0.7 1"/>
    <!-- 一个固定在 (0,0,0.5) 的小方块，没有 joint，所以不会动 -->
    <body name="box" pos="0 0 0.5">
      <geom name="box_geom" type="box" size="0.1 0.1 0.1" rgba="0.8 0.2 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""

# -- 2. 从 XML 字符串构建"模型" -----------------------------------------------
# from_xml_string 把 MJCF 文本解析成 mjModel。
# mjModel 里存的是"物理世界长什么样"：有几个刚体、每个多重、关节怎么连……
model = mujoco.MjModel.from_xml_string(XML)

# -- 3. 用模型"分配"出一份"状态"，即 mjData --------------------------------------
# 模型是"定义"，状态是"当前时刻的数据"。同一个 model 可以对应多份 data（并行仿真），
# 但入门阶段先记住"一个模型 + 一份状态"即可。
data = mujoco.MjData(model)

# -- 4. 读模型维度：自由度 ------------------------------------------------------
# nq：广义坐标个数（可理解成"位置"维度）
# nv：广义速度个数（"速度"维度）
# nu：执行器/控制量个数（"能施加多少路力"）
# 这个场景里没有任何关节，所以三者都是 0。
print("== 模型维度 ==")
print("nq (位置维度) =", model.nq)
print("nv (速度维度) =", model.nv)
print("nu (控制维度) =", model.nu)

# -- 5. 模型里有多少"物体" -----------------------------------------------------
# nbody：刚体个数。注意 MuJoCo 会隐式地多加一个"世界(world)"刚体，所以是 1(世界)+1(方块)=2。
# ngeom：几何形状个数。这里地板算 1 个，方块算 1 个，共 2。
print("\n== 物体数量 ==")
print("nbody (刚体数，含隐式的 world) =", model.nbody)
print("ngeom (几何形状数)             =", model.ngeom)

# -- 6. 模型里的物理参数 -------------------------------------------------------
# 模型还保存了仿真参数，例如时间步长 timestep（秒）和重力 gravity（m/s^2，z 轴向下为负）。
print("\n== 物理参数 ==")
print("timestep (时间步长, 秒) =", model.opt.timestep)
print("gravity  (重力, m/s^2)  =", model.opt.gravity)

# -- 7. 读"当前状态" -----------------------------------------------------------
# 因为还没有任何自由度，qpos（位置）和 qvel（速度）都是空数组。
# 下一例我们给小球加一个 freejoint 让它能自由运动，届时 nq 会变成 7。
print("\n== 当前状态 ==")
print("data.time (仿真时钟) =", data.time)
print("data.qpos (位置)     =", data.qpos)
print("data.qvel (速度)     =", data.qvel)

# -- 8. 用查看器看一眼这个静态场景 ----------------------------------------------
# 这个场景没有任何关节、物体固定，所以不推进仿真（不 mj_step），
# 只反复 viewer.sync() 把当前状态渲染到窗口。按住鼠标左键拖动可以旋转视角。
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        viewer.sync()

print("\n到这里你已经知道：模型(mjModel)定义世界，状态(mjData)记录当前时刻。")
