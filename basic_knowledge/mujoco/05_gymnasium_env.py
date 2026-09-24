"""把 MuJoCo 包装成 Gymnasium 环境：理解 reset/step/render 契约。"""

import time

import gymnasium as gym
import mujoco
import mujoco.viewer
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
  <!-- motor 的 ctrl 是力/力矩式输入；它和位置执行器的目标角度语义不同。 -->
  <actuator><motor name="hinge_motor" joint="hinge" gear="1" ctrlrange="-1 1"/></actuator>
</mujoco>
"""


class ReachEnv(gym.Env):
    """单关节环境；action 是力矩，observation 是角度和角速度。"""

    metadata = {"render_modes": ["human"], "render_fps": 100}

    def __init__(self, render_mode=None):
        if render_mode not in self.metadata["render_modes"] + [None]:
            raise ValueError(f"unsupported render_mode: {render_mode}")
        self.render_mode = render_mode
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self._viewer_context = None
        # space 同时声明范围、shape 和 dtype，必须与实际传入/返回的数组一致。
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)

    def _observation(self):
        # observation 的两个分量依次是关节角（rad）和角速度（rad/s）。
        return np.array([self.data.qpos[0], self.data.qvel[0]], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        # super().reset 建立 Gymnasium 管理的 np_random，保证 seed 可复现。
        super().reset(seed=seed)
        # 清空时间、位置、速度和控制状态，再随机化初始关节角。
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = self.np_random.uniform(-0.2, 0.2)
        # 手动改 qpos 后重新计算派生量，下一次 step 才从一致状态开始。
        mujoco.mj_forward(self.model, self.data)
        return self._observation(), {}

    def step(self, action):
        # 裁剪可防止外部策略越过 action_space；广播结果仍应保持一个动作维度。
        self.data.ctrl[0] = np.clip(action, -1.0, 1.0)
        mujoco.mj_step(self.model, self.data)
        observation = self._observation()
        # 奖励鼓励角度接近 0，同时轻微惩罚过大的角速度；这是教学用示例目标。
        reward = -abs(float(observation[0])) - 0.01 * abs(float(observation[1]))
        # terminated 表示任务自然结束；本环境没有成功/失败条件，因此始终为 False。
        terminated = False
        # truncated 表示外部时间限制结束，而不是任务本身完成。
        truncated = self.data.time >= 5.0
        return observation, reward, terminated, truncated, {}

    def render(self):
        """将当前 MjData 同步到实时 viewer；viewer 不会自动推进仿真。"""
        if self.render_mode != "human":
            return
        # 用 context manager 的底层 enter/exit 保持 viewer 跨多次 render 存活。
        if self.viewer is None:
            self._viewer_context = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer = self._viewer_context.__enter__()
        if self.viewer.is_running():
            self.viewer.sync()
            time.sleep(1 / self.metadata["render_fps"])

    def close(self):
        """释放 viewer；重复调用 close 也不会报错。"""
        if self._viewer_context is not None:
            self._viewer_context.__exit__(None, None, None)
            self._viewer_context = None
            self.viewer = None


if __name__ == "__main__":
    # 使用 human 模式运行较长的一段，窗口中才能看清关节运动。
    env = ReachEnv(render_mode="human")
    observation, _ = env.reset(seed=0)
    for _ in range(300):
        observation, reward, _, _, _ = env.step(env.action_space.sample())
        env.render()
        if env.viewer is not None and not env.viewer.is_running():
            break
    print(f"observation={observation}, reward={reward:.4f}")
    env.close()
