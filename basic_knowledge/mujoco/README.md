# MuJoCo 渐进式学习

这组示例的目标不是马上训练机器人，而是先把 MuJoCo 中最常用的概念拆开：

`XML 建模 -> 仿真状态 -> 动作控制 -> 接触/传感器 -> 相机观测 -> 环境接口 -> 轨迹数据`

学完后，再去阅读 [lerobot-mujoco-tutorial](https://github.com/jeongeun980906/lerobot-mujoco-tutorial)，就能把其中的仿真环境、数据采集 notebook 和 ACT/PI0/SmolVLA 部署串起来。

## 1. 环境准备

建议使用 Python 3.10 的虚拟环境。MuJoCo 3.1.6 是目标教程明确使用的版本，本目录允许同一大版本内更新：

```bash
cd basic_knowledge/mujoco
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

先检查安装：

```bash
python -c "import mujoco; print(mujoco.__version__)"
```

## 2. 示例顺序

| 顺序 | 文件 | 重点 | 可视化与输出 |
|---|---|---|---|
| 1 | `01_free_fall.py` | `MjModel`、`MjData`、XML、`mj_step` | viewer 观察小球下落，并打印位置和速度 |
| 2 | `02_joint_control.py` | joint、actuator、`data.ctrl`、viewer | viewer 观察摆杆跟踪目标角度 |
| 3 | `03_sensor_and_contact.py` | site、touch sensor、`data.ncon` | viewer 观察落地接触，并打印传感器统计 |
| 4 | `04_camera_render.py` | camera、renderer、RGB 图像 | viewer 观察运动，同时生成 `camera_frame.png` |
| 5 | `05_gymnasium_env.py` | `reset`、`step`、`render`、action/observation space | Gymnasium viewer 观察环境，并打印返回值 |
| 6 | `06_trajectory_contract.py` | 状态、动作、时间戳、轨迹 | viewer 观察采集过程，并生成 `trajectory.npz` |

每个文件都可以从当前目录直接运行：

```bash
python 01_free_fall.py
python 02_joint_control.py
python 03_sensor_and_contact.py
python 04_camera_render.py
python 05_gymnasium_env.py
python 06_trajectory_contract.py
```

所有示例默认都会尝试打开 MuJoCo 交互窗口；运行期间可以直接观察物体或关节运动。`04_camera_render.py` 同时展示实时 viewer 和离屏相机渲染，并生成 `camera_frame.png`。窗口关闭后对应程序会结束可视化循环，终端仍会打印已经采集到的统计或轨迹信息。

如果在没有桌面环境的服务器、容器或 SSH 会话中运行，实时 viewer 可能因 GLFW/OpenGL 初始化失败。此时可以设置 `MUJOCO_GL=egl` 或 `MUJOCO_GL=osmesa` 使用离屏渲染；需要完整实时窗口时，请在带图形桌面的本机运行。

## 3. 每一步应该弄懂什么

### 第 1 步：模型和数据

- `MjModel` 是不随时间变化的模型描述：关节数量、执行器、相机、几何体等。
- `MjData` 是当前时刻的状态：`qpos`（位置）、`qvel`（速度）、`ctrl`（控制输入）、`time`（仿真时间）。
- `mujoco.mj_step(model, data)` 推进一个时间步。
- XML 中的 `body` 是坐标系层级，`geom` 是碰撞/可视化几何体，`joint` 决定自由度，`actuator` 接收控制量。

先修改 `01_free_fall.py` 的初始高度、重力和时间步，预测输出后再运行。

### 第 2 步：从动作到运动

`02_joint_control.py` 中的 `position` 执行器内部实现了一个位置控制器。重点追踪：

1. 用 `mj_name2id` 找到 joint 和 actuator 的整数 ID。
2. 用 `model.jnt_qposadr` / `model.jnt_dofadr` 找到状态数组中的地址。
3. 每步写入 `data.ctrl[actuator_id]`，再调用 `mj_step`。

这就是后续机器人策略部署的基本循环：读取 observation，计算 action，写入控制接口。

### 第 3 步：接触和传感器

MuJoCo 的接触检测由几何体完成；`site` 是适合挂载传感器和定义观测位置的无碰撞标记。`data.ncon` 表示当前接触数量，`data.sensordata` 保存传感器输出。

尝试改变球的质量、地面摩擦和初始高度，观察接触时刻和传感器值变化。

### 第 4 步：相机就是图像观测

`04_camera_render.py` 使用 `mujoco.Renderer` 做离屏渲染。真实项目通常在每个环境 step 中渲染多个 camera，并把数组放进 observation 字典。

目标教程中的 `observation.image`、`observation.wrist_image` 等字段，本质上就是不同 MuJoCo camera 在指定分辨率下的 RGB 帧。

### 第 5 步：环境接口

`05_gymnasium_env.py` 把前面的模型包装成 Gymnasium 环境：

- `action_space` 描述策略可以输出什么。
- `observation_space` 描述策略会收到什么。
- `reset()` 开始一个 episode。
- `step(action)` 推进仿真并返回 observation、reward、结束标记和额外信息。

先理解这个小环境，再阅读目标教程里的机器人环境类，会更容易识别哪些代码是 MuJoCo 逻辑，哪些代码只是数据集或策略适配。

### 第 6 步：轨迹和 LeRobot 数据

`06_trajectory_contract.py` 用 NumPy 保存一个简化轨迹：

- `observation.state`：机器人状态，例如关节角、末端位姿。
- `action`：策略输出，例如关节目标或夹爪值。
- `timestamp`：采样时间。

目标教程进一步把这些字段写入 LeRobot 数据集，并增加 `observation.image`、`observation.wrist_image`、episode 索引和元数据。先运行本例检查 shape，再看教程的 parquet/meta 目录，会比较容易。

## 4. 对照目标教程的阅读顺序

建议按下面的顺序读，不要一上来从训练 notebook 开始：

1. `mujoco_env/`：先找 XML 加载、`MjModel` / `MjData`、动作写入和 camera 渲染。
2. `1.collect_data.ipynb`：对应本目录第 2、4、5、6 步，重点看键盘动作如何变成 action，以及每一帧如何保存。
3. `2.visualize_data.ipynb`：对应状态回放、动作回放和多相机图像显示。
4. `3.train.ipynb` 与 `4.deploy.ipynb`：确认 ACT 的输入字段和 action chunk 如何回到环境。
5. `5.language_env.ipynb`、`6.visualize_data.ipynb`：理解语言任务如何进入 episode 元数据和任务条件。
6. `7.pi0.ipynb`、`8.smolvla.ipynb`：最后再看更大的视觉语言策略部署。

## 5. 从示例迁移到教程时的检查清单

- 模型加载：XML / asset 路径是否正确，MuJoCo 版本是否匹配。
- 控制频率：仿真 timestep、环境控制频率、数据采样频率是否一致。
- 状态维度：`observation.state` 的顺序、单位和 shape 是否与 action policy 配置一致。
- 图像字段：camera 名称、分辨率、RGB/BGR 顺序和 dtype 是否一致。
- 动作语义：action 是关节角、关节速度、力矩还是夹爪开合，不要只看 shape。
- episode 边界：reset、成功、超时和失败是否正确写入数据集。
- 数据回放：把录制的 action 重新喂给环境，确认轨迹与采集时一致。

## 6. 常见问题

### viewer 不显示

先确认当前环境是否有桌面显示和可用的 GLFW/OpenGL。服务器环境没有图形桌面时，实时 viewer 无法打开；可优先运行 `04_camera_render.py` 的离屏渲染部分，并设置 `MUJOCO_GL=egl` 或 `MUJOCO_GL=osmesa`。

### `GLFW` 或 OpenGL 报错

这是图形环境问题，不一定是 MuJoCo 安装失败。可先确认 `import mujoco` 成功，再配置本机图形驱动或使用 EGL/OSMesa 的无窗口渲染环境。

### 目标教程安装失败

先只安装本目录的 MuJoCo 依赖并完成六个示例。LeRobot、策略训练和机器人资产是第二阶段依赖，按目标教程要求单独创建环境，避免把初学环境弄得过重。
