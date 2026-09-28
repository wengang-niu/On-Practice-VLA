# MuJoCo 渐进式学习教程

这是一份面向 **MuJoCo 初学者** 的中文教程：用 13 个**互不依赖、可独立运行**的小示例，从"认识模型"一路走到"采集一条机器人轨迹数据"。每学完一个，你就掌握一个**新概念**，最终目标是能读懂并复用 [lerobot-mujoco-tutorial](https://github.com/jeongeun980906/lerobot-mujoco-tutorial) 这个仓库（LeRobot + MuJoCo 的 VLA 教程）。

## 学习路线图

```text
认识模型 → 仿真步 → 查看器 → 位姿 → 关节与执行器 → 接触与传感器 → 相机 → Gymnasium 环境 → 轨迹数据
     01        02       03       04        05~08          09~10          11         12            13
```

每个示例只引入**一个**真正的新概念，重复内容只做最小回顾。每个示例文件开头的一行 docstring 就是它的**学习目标**，先读那一句再动手。

## 环境要求

- **Python 3.10**（与根项目 `project.toml` 一致）
- **有显示器**：示例 **01–10、12** 会弹出交互式查看器窗口。没有显示器（SSH / 容器 / 云主机）时，示例 **11、13** 仍可用离屏渲染跑通，其余需要自行跳过或配置虚拟显示（如 `xvfb`）。

## 安装

```bash
cd basic_knowledge/mujoco
python -m venv .venv
source .venv/bin/activate        # Windows：.venv\Scripts\activate
pip install -r requirements.txt
```

依赖说明（见 [requirements.txt](./requirements.txt)）：

- `mujoco==3.1.6`：锁定，与目标仓库一致（本机若已是 3.x，示例代码同样兼容）。
- `numpy`：锁定 1.x，避免与 mujoco 3.1.6 的 ABI 意外。
- `gymnasium`：仅示例 12 需要。
- `Pillow`：示例 11 保存渲染出的 PNG。

## 如何运行

每个文件独立运行：`python 0X_xxx.py`。逐行对照下表：

| 文件 | 学习目标 | 运行命令 | 应看到什么 |
| --- | --- | --- | --- |
| `01_hello_mujoco.py` | 认识 `mjModel` 与 `mjData` 的分离 | `python 01_hello_mujoco.py` | 打印模型维度 `nq/nv/nu`、刚体/几何数量；弹窗展示静态场景 |
| `02_free_fall.py` | 理解 `mj_step` 如何推进时间与状态 | `python 02_free_fall.py` | 窗口里小球实时下落；终端 `data.time` 递增、高度逐减 |
| `03_viewer.py` | 会用交互式查看器看持续仿真 | `python 03_viewer.py` | 弹窗，小球按真实速度下落，可拖拽视角 |
| `04_pose.py` | 理解自由体的位姿（位置 + 四元数） | `python 04_pose.py` | 打印 7 维 `qpos`（四元数 w 在前）；弹窗展示初始姿态 |
| `05_single_joint.py` | 理解关节（hinge）与力矩执行器（motor） | `python 05_single_joint.py` | 窗口里单摆在力矩下加速旋转；终端 `qpos` 单调变化 |
| `06_two_link_arm.py` | 理解刚体嵌套与多关节，按名取关节角 | `python 06_two_link_arm.py` | `nq==nv==nu==2`，按名找到下标；弹窗看两杆臂摆动 |
| `07_actuator_modes.py` | 区分"位置伺服"与"力矩"执行器 | `python 07_actuator_modes.py` | 左摆自己追目标停住、右摆不动，并排对比 |
| `08_pd_control.py` | 手写 PD 控制器锁定目标角 | `python 08_pd_control.py` | 窗口里单摆被拉到目标角并稳定 |
| `09_contact.py` | 理解接触检测 `data.ncon` / `data.contact` | `python 09_contact.py` | 小球落地，终端 `ncon` 由 0 变 1；关窗后打印接触细节 |
| `10_touch_sensor.py` | 理解触觉传感器 `<touch>` | `python 10_touch_sensor.py` | 小球落地，终端 `sensordata` 升到 ≈mg |
| `11_camera.py` | 理解相机与离屏渲染，得到 RGB 图像 | `python 11_camera.py` | 生成 `camera_front.png`（离屏，无需显示器） |
| `12_gymnasium.py` | 学会 Gymnasium 标准接口 | `python 12_gymnasium.py` | 弹出 HalfCheetah 查看器窗口 |
| `13_trajectory.py` | 按固定频率采集轨迹并落盘 | `python 13_trajectory.py` | 生成 `trajectory.npz`（离屏，无需显示器） |

## 示例详解

> 示例 **01–10** 会弹出交互式查看器窗口（关窗后脚本打印总结），每个窗口里都能按住鼠标左键拖动旋转视角；**11、13** 是离屏渲染，不弹窗、无需显示器。

### 01 `hello_mujoco` — 认识模型与状态

MuJoCo 把仿真拆成两个对象：

- **`mjModel`（模型）**：世界"长什么样"——刚体、关节、几何、执行器、物理参数。加载后基本不变。
- **`mjData`（状态）**：世界"此刻如何"——时间、位置 `qpos`、速度 `qvel` 等会随时间变化的量。

示例加载一个静态场景（不仿真），打印 `model.nq/nv/nu`、`nbody/ngeom` 和 `data.time/qpos/qvel`，让你先分清这两样东西。

### 02 `free_fall` — 仿真步

`mujoco.mj_step(model, data)` 是"让世界前进一格"的开关：推进一个 `timestep`，更新位置、速度、时间。对比 `mj_forward`（只计算派生量、不推进时间）。示例里小球带着 `freejoint` 自由落体，你看到 `data.time` 递增、`qpos[2]`（高度）逐减。

### 03 `viewer` — 交互式查看器

`mujoco.viewer.launch_passive(model, data)` 打开一个窗口，配合 `viewer.is_running()` + `viewer.sync()` 持续渲染。注意必须先 `import mujoco.viewer`，否则会报 `module 'mujoco' has no attribute 'viewer'`。

### 04 `pose` — 位姿与四元数

自由体（`freejoint`）的 `qpos` 是 7 维：`[x, y, z, qw, qx, qy, qz]`——前 3 个是位置，后 4 个是姿态四元数，**w 在前**。示例演示不旋转时 `[1,0,0,0]`、绕 z 转 90° 时如何，并用 `mju_quat2Mat` 转成旋转矩阵。这是读懂目标仓库 `transforms.py`（`quat2r`/`rpy2r`）的基础。

### 05 `single_joint` — 关节与力矩执行器

`<joint type="hinge">` 是单自由度转动关节；`<motor joint="...">` 是力矩执行器。`data.ctrl[0]` 写入的就是**力矩**（单位 N·m），而 `data.qpos` 是**角度**（单位弧度）。给一个恒定力矩，看角度/角速度如何变化。

### 06 `two_link_arm` — 多刚体嵌套

body 可以嵌套 child body 形成"链"，每个 hinge 是一个自由度。用 `mujoco.mj_name2id` 按名字找到 body/joint 的 id，再用 `model.jnt_qposadr` 找到某个关节在 `qpos` 数组里的下标。

### 07 `actuator_modes` — 位置伺服 vs 力矩

- `<position joint="..." kp="..." kv="..."/>`：你给**目标角**，它内部用 PD 伺服自己算力矩追过去（有微小稳态误差）。
- `<motor joint="..."/>`：你给**力矩**，它原样施加，没有"目标"概念。

这对应机器人/VLA 的核心问题：**策略输出的是"目标"还是"力矩"？** 目标仓库 `y_env.py` 的 `action_type` 就在选这个。

### 08 `pd_control` — 手写 PD

`tau = kp*(q_des - q) - kv*qd`，每步算误差→算力矩→写入 `data.ctrl`→推进仿真。这就是把"目标角"变成"力矩"的闭环反馈，也是 VLA 里"策略输出目标 → 底层控制器执行"的本质。

### 09 `contact` — 接触检测

MuJoCo 每步检测接触，结果在 `data.contact` 数组（`data.ncon` 是接触点个数）。落地前 `ncon=0`、落地后 `ncon=1`。**访问 `data.contact[i]` 前必须先判 `data.ncon > 0`**，否则越界。可读 `dist`（接触距离）、`pos`（接触点世界坐标）、`geom1/geom2`（接触的两个几何）。

### 10 `touch_sensor` — 触觉传感器

`<touch site="...">` 挂在某个 `site`（参考点）上，测该点处的接触法向力，读数在 `data.sensordata`。小球落地后读数 ≈ `mg`。**坑：site 必须落在真正会发生接触的位置**，放在球心就永远读 0。注意 `<force>/<torque>` 传感器是净力、静平衡时为 0，不能用它来"检测接触"。

### 11 `camera` — 相机与渲染

`<camera>` 定义相机，`<light>` 提供光源（没光会全黑）。`mujoco.Renderer` 离屏渲染，`update_scene(data, camera="名字")` + `render()` 得到 `(H, W, 3)` 的 **uint8 RGB 0–255** 数组——这张图就是 VLA 的 `observation.image`。

### 12 `gymnasium` — 标准 RL 接口

Gymnasium 是强化学习环境的统一接口，`gymnasium.make("HalfCheetah-v5", render_mode="human")` 底层就是前面 11 个示例的 `mjModel/mjData/mj_step`。学会 `reset`（返回 `(obs, info)`）、`step`（返回 5 元组）、`action_space/observation_space`，并分清 `terminated`（任务失败结束）与 `truncated`（时间上限截断）。

### 13 `trajectory` — 轨迹采集与落盘

把前面的能力串起来：两杆臂 + 固定相机，按 **20 fps**（`timestep=0.002`，每 25 个子步记一帧）采集"状态 + 动作 + 图像"，字段名对齐 LeRobot（`observation.state`/`action`/`observation.image`/`obj_init`），用 `numpy` 堆成 `(T, ...)` 后 `np.savez`。关键概念：**仿真频率 ≠ 数据频率**，`obj_init` 只在 reset 后记录一次。

## FAQ

**Q：运行 03 报 `module 'mujoco' has no attribute 'viewer'`？**
A：查看器是子模块，必须显式 `import mujoco.viewer` 才能用 `mujoco.viewer.launch_passive`。

**Q：查看器打不开 / `GLFW error` / 黑屏？**
A：通常是没显示器（SSH/容器/云主机）。示例 11、13 用离屏渲染不受影响；01–10、12 需要显示器或虚拟显示（如 `xvfb`）。

**Q：`gymnasium` 装不上 / 找不到环境？**
A：`HalfCheetah-v5` 需要安装 `gymnasium`（本目录 `requirements.txt` 已包含）。若版本冲突，优先保住 `mujoco==3.1.6`，再下调 gymnasium 的补丁版本。

**Q：`reset` / `step` 的返回值怎么和网上老教程不一样？**
A：新版 Gymnasium 的 `reset` 返回 `(obs, info)` 二元组，`step` 返回 `(obs, reward, terminated, truncated, info)` 五元组。老教程（旧 `gym`）是 `reset` 只返回 `obs`、`step` 返回四元组。以本仓库示例 12 为准。

**Q：角度单位是度还是弧度？**
A：`data.qpos` / `data.ctrl` 里关节量都是**弧度**，力矩是 **N·m**。写目标角时别用 30（度），要用 `math.radians(30)`。

**Q：touch 传感器一直读 0？**
A：`site` 必须正好落在会发生接触的位置（例如球面最低点 `pos="0 0 -0.1"`）。放在球心那个点永远碰不到地面，自然读 0。

**Q：需要安装 LeRobot 吗？**
A：不需要。本教程只用 `mujoco + numpy + gymnasium + Pillow`，不引入 LeRobot。示例 13 只是把字段名对齐 LeRobot 的数据格式，方便你日后衔接。

## 如何读懂 lerobot-mujoco-tutorial

学完 13 个示例后，用下面这张表对应到目标仓库的源码：

| 教程概念（示例） | 目标仓库对应文件 / 机制 |
| --- | --- |
| 模型加载与状态（01/02） | `mujoco_env/mujoco_parser.py`：`from_xml_string/path` 构造 `mjModel`/`mjData` |
| 位姿与四元数（04） | `mujoco_env/transforms.py`：`t2p/t2r/pr2t/rpy2r/r2rpy/quat2r` |
| 关节与动作控制（05–08） | `mujoco_env/ik.py`（`solve_ik`、雅可比伪逆）+ `y_env.py` 的 `action_type`/`state_type` |
| 接触与传感器（09/10） | `mujoco_parser.py` 对 `data.sensordata`/`data.ncon` 的读取 |
| 相机观测（11） | `mujoco_parser.py` 的 `mujoco.Renderer` + `update_scene`；腕部相机 ↔ `observation.wrist_image` |
| Gymnasium 环境（12） | `mujoco_env/y_env.py` 的 `SimpleEnv`（`reset(seed)`/`step`） |
| 轨迹数据（13） | LeRobot 格式：parquet 单集（`observation.image/state`、`action`、`obj_init`）+ meta（`info.json/episodes.jsonl/stats.json/tasks.jsonl`）；20 fps 采集 |

## 进一步阅读

- [MuJoCo 官方文档](https://mujoco.readthedocs.io/)（XML 元素、API 手册）
- [yet-another-mujoco-tutorial](https://github.com/kevinzakka/mujoco-starter)（另一个入门教程）
- [lerobot-mujoco-tutorial](https://github.com/jeongeun980906/lerobot-mujoco-tutorial)（本教程的最终目标仓库）
- [LeRobot](https://github.com/huggingface/lerobot)（数据格式与训练框架）
