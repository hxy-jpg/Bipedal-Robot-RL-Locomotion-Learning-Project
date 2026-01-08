# 双足机器人强化学习运动学习项目

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)

## demo

- 穿越多种复杂地形

![play_isaaclab](pictures/demo.gif)

---

## 1.概述

该仓库用于训练和仿真双足机器人，例如[limxdynamics TRON1](https://www.limxdynamics.com/en/tron1)。
借助[Isaac Lab](https://github.com/isaac-sim/IsaacLab)框架，我们可以训练双足机器人在不同环境中行走，包括平地、粗糙地形和楼梯等。

This repository is used to train and simulate bipedal robots, such as [limxdynamics TRON1](https://www.limxdynamics.com/en/tron1).
With the help of [Isaac Lab](https://github.com/isaac-sim/IsaacLab), we can train the bipedal robots to walk in different environments, such as flat, rough, and stairs.

**关键词 / Keywords:** isaaclab, locomotion, bipedal, pointfoot, TRON1

---

## 2.安装

### 2.1 环境准备
- 在 GradMotion 上创建 **Ubuntu 22.04 + IsaacSim 4.5.0 + IsaacLab 2.1.0** 的环境。


### 2.2 安装 VS Code
```bash
sudo apt update
sudo apt install software-properties-common apt-transport-https wget -y
wget -q https://packages.microsoft.com/keys/microsoft.asc -O- | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main"
sudo apt update
sudo apt install code
```

```bash
code --no-sandbox --user-data-dir=/tmp/vscode-root
```

### 2.3 仓库克隆与环境准备

- 将仓库克隆到 Isaac Lab 安装目录之外的独立位置（即在 `IsaacLab` 目录外），将文件夹改名为：`bipedal_locomotion_isaaclab`  

### 2.4 进入仓库并安装相关依赖
- 进入conda环境
```bash
conda activate isaaclab
```
- 安装相关依赖
```bash
cd bipedal_locomotion_isaaclab
python -m pip install -e exts/bipedal_locomotion

# remove rsl_rl
rm -rf /root/miniconda3/envs/env_isaaclab/lib/python3.10/site-packages/rsl_rl
rm -rf /root/miniconda3/envs/env_isaaclab/lib/python3.10/site-packages/rsl_rl-*.egg-info

# install rsl_rl
cd bipedal_locomotion_isaaclab/rsl_rl
python -m pip install -e .
```

---
## 3.训练双足机器人智能体

- 使用`scripts/rsl_rl/train.py`脚本直接训练机器人，指定任务：

```bash
python3 scripts/rsl_rl/train.py --task=Isaac-Limx-PF-Blind-Flat-v0 --headless
```

- 以下参数可用于自定义训练：

| 参数 / Argument      | 说明 / Description                                         |
|---------------------|------------------------------------------------------------|
| `--headless`        | 以无渲染模式运行仿真，适合在服务器或远程环境下运行 |
| `--num_envs`        | 并行环境数量，即同时运行多少个训练环境 |
| `--max_iterations`  | 最大训练迭代次数，训练将在达到该次数后停止 |
| `--save_interval`   | 模型保存间隔，每隔指定迭代次数保存一次模型 |
| `--seed`            | 随机种子，用于确保训练可复现 |
---

## 4.运行训练好的模型

- 要运行训练好的模型：

```bash
python3 scripts/rsl_rl/play.py --task=Isaac-Limx-PF-Blind-Flat-Play-v0 --checkpoint_path=path/to/checkpoint
```

- 以下参数可用于自定义运行：

| 参数        | 说明                                        |
|------------------------|------------------------------------------------------------|
| `--num_envs`           | 并行环境数量，即同时运行多少个环境 |
| `--headless`           | 以无渲染模式运行仿真，适合在服务器或远程环境下运行 |
| `--checkpoint_path`    | 要加载的模型检查点路径 |

---

## 5.如何查看 TensorBoard 日志？

- 训练完成后，日志文件会保存在 log_dir 中（e.g., logs/rsl_rl/experiment_name/ 子目录）。
- 运行以下命令启动 TensorBoard 查看器：
```bash
tensorboard --logdir logs/rsl_rl/
```

- 之后，直接在浏览器打开：
http://localhost:6006

- 训练指标示例 / Example Training Metrics

| 指标 / Metric | 说明 / Description |
|---------------|------------------|
| Learning Iteration | 当前训练迭代 |
| Computation | 仿真步长 / 计算速度 |
| Value function loss | 值函数损失，用于评估 critic 网络拟合情况 |
| Surrogate loss | 代理损失，衡量策略更新稳定性 |
| Mean action noise std | 平均动作噪声标准差 |
| Learning rate | 学习率，控制网络参数更新速度 |
| Mean reward | 每条轨迹平均获得的奖励 |
| Mean episode length | 每个回合的平均长度 |

- Episode

| 指标 / Metric | 说明 / Description |
|---------------|------------------|
| Episode_Reward/keep_balance | 保持机器人平衡的奖励，鼓励机身姿态稳定 |
| Episode_Reward/rew_lin_vel_xy | 机器人沿 XY 平面线速度跟随指令的奖励 |
| Episode_Reward/rew_ang_vel_z | 机器人绕 Z 轴角速度（转向）跟随指令的奖励 |
| Episode_Reward/pen_base_height | 基座高度偏离目标高度的惩罚，过低表示机器人容易摔倒 |
| Episode_Reward/pen_lin_vel_z | 垂直方向速度偏差的惩罚 |
| Episode_Reward/pen_ang_vel_xy | roll/pitch 角速度过大，表示身体晃动，给予惩罚 |
| Episode_Reward/pen_joint_torque | 关节力矩过大或控制不平滑的惩罚 |
| Episode_Reward/pen_joint_accel | 关节加速度过大的惩罚，防止动作过猛 |
| Episode_Reward/pen_action_rate | 动作变化过快的惩罚，鼓励平滑动作 |
| Episode_Reward/pen_joint_pos_limits | 关节位置超出限制的惩罚 |
| Episode_Reward/pen_joint_vel_l2 | 关节速度过大的惩罚 |
| Episode_Reward/pen_joint_powers | 关节功率过大的惩罚 |
| Episode_Reward/pen_undesired_contacts | 非期望部位接触地面的惩罚 |
| Episode_Reward/pen_action_smoothness | 动作不平滑的惩罚 |
| Episode_Reward/pen_flat_orientation | 身体在平地上的倾斜惩罚，鼓励平衡姿态 |
| Episode_Reward/pen_feet_distance | 足部间距不合理的惩罚，影响步态质量 |
| Episode_Reward/pen_feet_regulation | 步态是否像自然行走的惩罚 |
| Episode_Reward/foot_landing_vel | 落脚速度过快的惩罚，防止摔倒或损伤 |
| Episode_Reward/test_gait_reward | 步态测试奖励，用于整体步态评价 |

### Curriculum / 课程指标
| 指标 / Metric | 说明 / Description |
|---------------|------------------|
| Curriculum/terrain_levels | 当前环境地形难度等级，用于课程学习 |

### Metrics / 速度指标
| 指标 / Metric | 说明 / Description |
|---------------|------------------|
| Metrics/base_velocity/error_vel_xy | 实际 XY 平面速度与指令速度的误差 |
| Metrics/base_velocity/error_vel_yaw | 实际 Yaw 角速度与指令角速度的误差 |

### Episode Termination / 回合终止条件
| 指标 / Metric | 说明 / Description |
|---------------|------------------|
| Episode_Termination/time_out | 回合是否因超时而终止 |
| Episode_Termination/base_contact | 回合是否因基座非法接触地面而终止 |

