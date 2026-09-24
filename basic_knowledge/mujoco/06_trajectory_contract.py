"""轨迹数据契约：可视化采集过程，并保存 LeRobot 风格的简化序列。"""

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


XML = """
<mujoco model="trajectory_contract">
  <option gravity="0 0 -9.81" timestep="0.01"/>
  <worldbody>
    <geom type="plane" size="2 2 0.1"/>
    <body pos="0 0 0.5">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.5" size="0.06"/>
    </body>
  </worldbody>
  <actuator><motor joint="hinge" gear="1" ctrlrange="-1 1"/></actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
frames = []
# viewer 让轨迹采集中的控制动作和关节运动可以同时被观察。
with mujoco.viewer.launch_passive(model, data) as viewer:
    for step in range(100):
        if not viewer.is_running():
            break
        # action 的 shape/dtype 必须和 actuator 数量及数据集约定一致。
        action = np.array([0.5 * np.sin(step / 10)], dtype=np.float32)
        data.ctrl[:] = action
        mujoco.mj_step(model, data)
        viewer.sync()
        # 这一帧记录的是“施加 action 后、完成本次 step”的状态。
        # copy() 防止把指向可变 MuJoCo 数组的视图保存到后续帧中。
        frames.append(
            {
                "observation.state": np.array(data.qpos, dtype=np.float32).copy(),
                "action": action.copy(),
                "timestamp": data.time,
            }
        )
        time.sleep(model.opt.timestep)

# stack 沿 axis 0 建立时间维度，要求每一帧的 shape 都相同。
trajectory = {
    key: np.stack([frame[key] for frame in frames])
    for key in ("observation.state", "action")
}
# timestamp 是以秒为单位的仿真时间；它也沿时间维度排列。
trajectory["timestamp"] = np.array([frame["timestamp"] for frame in frames], dtype=np.float32)
output = Path(__file__).with_name("trajectory.npz")
# NPZ 的关键字会成为文件字段；这里只是简化示例，不是完整 LeRobot 数据集。
np.savez(output, **trajectory)
print(f"saved {len(frames)} frames at {1 / model.opt.timestep:.0f} Hz to {output}")
for key, value in trajectory.items():
    print(f"{key}: shape={value.shape}, dtype={value.dtype}")
