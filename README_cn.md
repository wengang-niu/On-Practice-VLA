# On-Practice-VLA

On-Practice-VLA 是一个面向实践驱动迭代的研究项目仓库，用于整理 VLA 及相关基础组件的学习内容、经典方案参考与实践材料。


## basic_knowledge

`basic_knowledge/` 主要放 VLA 相关基础知识和常见技术栈的学习内容。

| 方向 | 示例 |
|---|---|
| 视觉与表征 | ViT、ResNet、CLIP、JEPA |
| 序列建模 | Transformer、SSM |
| 生成建模 | VAE、Diffusion |
| 强化学习 | DPO、RLHF |
| 训练与分布式 | DeepSpeed、FSDP、LORA |
| 数据集与格式 | QEGO、UMI、Lerobot、RLDS |




## 经典方案参考
ACT
OpenVLA
Diffusion Policy 
StarVLA
PI系列
GROOT
eepmind.google/models/gemini-robotics/
generalistai.com/blog/gen-1.5

## MuJoCo 学习

从 [basic_knowledge/mujoco/README.md](basic_knowledge/mujoco/README.md) 的渐进式示例开始，依次学习 XML 建模、状态读取、关节控制、接触传感器、相机渲染、Gymnasium 环境和轨迹数据，再阅读 [lerobot-mujoco-tutorial](https://github.com/jeongeun980906/lerobot-mujoco-tutorial)。