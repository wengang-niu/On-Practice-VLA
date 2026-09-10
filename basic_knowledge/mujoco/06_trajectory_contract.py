"""轨迹数据契约：把 MuJoCo 的状态和控制量整理成 LeRobot 风格的序列。"""

from pathlib import Path

import mujoco
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
for step in range(100):
    action = np.array([0.5 * np.sin(step / 10)], dtype=np.float32)
    data.ctrl[:] = action
    mujoco.mj_step(model, data)
    frames.append(
        {
            "observation.state": np.array(data.qpos, dtype=np.float32).copy(),
            "action": action.copy(),
            "timestamp": data.time,
        }
    )

trajectory = {
    key: np.stack([frame[key] for frame in frames])
    for key in ("observation.state", "action")
}
trajectory["timestamp"] = np.array([frame["timestamp"] for frame in frames], dtype=np.float32)
output = Path(__file__).with_name("trajectory.npz")
np.savez(output, **trajectory)
print(f"saved {len(frames)} frames at {1 / model.opt.timestep:.0f} Hz to {output}")
for key, value in trajectory.items():
    print(f"{key}: shape={value.shape}, dtype={value.dtype}")
