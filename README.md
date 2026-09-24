# On-Practice-VLA

On-Practice-VLA is a research repository focused on practice-driven iteration. It organizes learning materials, references to classic approaches, and practical resources related to Vision-Language-Action (VLA) models and their foundational components.

> This project is still under active development. Example code and documentation will be added progressively as experiments advance.

## Repository Structure

| Directory | Contents |
| --- | --- |
| [`basic_knowledge/`](./basic_knowledge/) | Learning examples covering VLA fundamentals and common technology stacks |
| [`basic_knowledge/mujoco/`](./basic_knowledge/mujoco/) | Progressive examples ranging from basic MuJoCo simulation to trajectory data |
| [`basic_knowledge/transformer/`](./basic_knowledge/transformer/) | A PyTorch-based English-to-Chinese translation example using a Transformer |
| [`basic_knowledge/vae/`](./basic_knowledge/vae/) | A PyTorch-based convolutional variational autoencoder example |
| [`basic_knowledge/resnet/`](./basic_knowledge/resnet/) | ResNet-related learning code |
| [`source/`](./source/) | Experimental code for VLA data loading, training, and inference |
| [`pic/`](./pic/) | Project images and experiment records |

## Hardware

The current hardware records are shown below. Device models, VRAM, controllers, and experimental purposes will be documented in more detail later.

![Hardware records](./pic/测试.jpg)

## Foundational Learning Roadmap

The following learning path is recommended:

- **Vision and representation**: ViT, ResNet, CLIP, JEPA
- **Sequence modeling**: Transformer, SSM
- **Generative modeling**: VAE, Diffusion
- **Policy optimization**: behavioral cloning, DPO, RLHF, and reinforcement learning
- **Training and distributed systems**: DeepSpeed, FSDP, LoRA
- **Datasets and formats**: LeRobot, RLDS, UMI, and more

The README in each subdirectory describes the dependencies, configuration, and steps required to run the corresponding examples.

## References to Classic Approaches

### VLA and Robotic Policies

- [ACT](https://tonyzhaozh.github.io/aloha/)
- [OpenVLA](https://openvla.github.io/)
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
- StarVLA
- The PI series (Physical Intelligence)
- GROOT

### Further Resources

- [Gemini Robotics — Google DeepMind](https://deepmind.google/models/gemini-robotics/)
- [Gen-1.5 — Generalist AI](https://generalistai.com/blog/gen-1.5)
- [LeRobot MuJoCo Tutorial](https://github.com/jeongeun980906/lerobot-mujoco-tutorial)

## Learning MuJoCo

Start with the [progressive MuJoCo learning documentation](./basic_knowledge/mujoco/README.md) and follow this sequence:

```text
XML modeling → simulation state → action control → contacts and sensors → camera observations → Gymnasium environments → trajectory data
```

This directory contains six independently runnable examples covering free fall, joint control, contact sensors, camera rendering, Gymnasium environments, and trajectory data. For detailed environment setup, run commands, frequently asked questions, and the relationship to the LeRobot tutorial, please refer to the README in that directory.

## Requirements

The root project's [`project.toml`](./project.toml) specifies the Python `3.10` environment and the main dependencies required for the VLA experiments. Different learning modules may have their own dependencies and Python version requirements. Please follow the instructions in the corresponding directory first when creating a virtual environment and installing dependencies.

For example, to run the MuJoCo examples:

```bash
cd basic_knowledge/mujoco
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python 01_free_fall.py
```

For detailed installation and usage instructions for the Transformer and VAE modules, see:

- [Transformer README](./basic_knowledge/transformer/README.md)
- [VAE README](./basic_knowledge/vae/README.md)
