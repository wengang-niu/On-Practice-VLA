"""把 MuJoCo 包装成 Gymnasium 环境：理解 reset/step/observation/action 契约。"""

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


XML = """
<mujoco model="reach_point">
  <option gravity="0 0 -9.81" timestep="0.01"/>
  <worldbody>
    <geom type="plane" size="2 2 0.1"/>
    <body name="arm" pos="0 0 0.5">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.5" size="0.06"/>
    </body>
  </worldbody>
  <actuator><motor name="hinge_motor" joint="hinge" gear="1" ctrlrange="-1 1"/></actuator>
</mujoco>
"""


class ReachEnv(gym.Env):
    """单关节环境；action 是力矩，observation 是角度和角速度。"""

    metadata = {"render_modes": []}

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)

    def _observation(self):
        return np.array([self.data.qpos[0], self.data.qvel[0]], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = self.np_random.uniform(-0.2, 0.2)
        mujoco.mj_forward(self.model, self.data)
        return self._observation(), {}

    def step(self, action):
        self.data.ctrl[0] = np.clip(action, -1.0, 1.0)
        mujoco.mj_step(self.model, self.data)
        observation = self._observation()
        reward = -abs(float(observation[0])) - 0.01 * abs(float(observation[1]))
        terminated = False
        truncated = self.data.time >= 5.0
        return observation, reward, terminated, truncated, {}


if __name__ == "__main__":
    env = ReachEnv()
    observation, _ = env.reset(seed=0)
    for _ in range(10):
        observation, reward, _, _, _ = env.step(env.action_space.sample())
    print(f"observation={observation}, reward={reward:.4f}")
    env.close()
