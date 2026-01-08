"""RSL-RL智能体检查点播放脚本 / Script to play a checkpoint of an RL agent from RSL-RL."""

"""首先启动Isaac Sim仿真器 / Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# 添加argparse参数 / Add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Relative path to checkpoint file.")
# [新增] 添加键盘控制开关参数
parser.add_argument("--use_keyboard", action="store_true", default=True, help="Use keyboard control.") 

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import os
import torch

def quat_to_euler(quat):
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = torch.where(torch.abs(sinp) >= 1, torch.sign(sinp) * (torch.pi / 2), torch.asin(sinp))

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

from isaaclab.devices import Se2Keyboard 

from rsl_rl.runner import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg,DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
# Import extensions to set up environment tasks
import bipedal_locomotion  # noqa: F401
from bipedal_locomotion.utils.wrappers.rsl_rl import RslRlPpoAlgorithmMlpCfg, export_mlp_as_onnx, export_policy_as_jit


def main():
    """使用RSL-RL智能体进行测试 / Play with RSL-RL agent."""
    # 解析配置 / Parse configuration
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    env_cfg.episode_length_s = 1000.0
    agent_cfg: RslRlPpoAlgorithmMlpCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    env_cfg.seed = agent_cfg.seed

    
    # 指定日志实验目录 / Specify directory for logging experiments
    if args_cli.checkpoint_path is None:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    else:
        resume_path = args_cli.checkpoint_path
    log_dir = os.path.dirname(resume_path)

    # 创建isaac环境 / Create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
    # load previously trained model
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    encoder = ppo_runner.get_inference_encoder(device=env.unwrapped.device)

    # 导出策略到onnx / Export policy to onnx
    if EXPORT_POLICY:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(
            ppo_runner.alg.actor_critic, export_model_dir
        )
        print("Exported policy as jit script to: ", export_model_dir)
        export_mlp_as_onnx(
            ppo_runner.alg.actor_critic.actor, 
            export_model_dir, 
            "policy",
            ppo_runner.alg.actor_critic.num_actor_obs,
        )
        export_mlp_as_onnx(
            ppo_runner.alg.encoder,
            export_model_dir, 
            "encoder",
            ppo_runner.alg.encoder.num_input_dim,
        )
    
    # reset environment
    obs, obs_dict = env.get_observations()
    obs_history = obs_dict["observations"].get("obsHistory")
    obs_history = obs_history.flatten(start_dim=1)
    commands = obs_dict["observations"].get("commands") 

    # [新增] 初始化键盘监听器
    if args_cli.use_keyboard:
        keyboard = Se2Keyboard()
        # 获取底层的 Command Manager
        vel_cmd_term = env.unwrapped.command_manager.get_term("base_velocity")

    log_data = {
        "cmd_vel": [],    # 指令速度 [vx, vy, wz]
        "meas_vel": [],   # 实际速度 [vx, vy, wz]
        "attitude": [],   # 姿态角 [roll, pitch]
        "dones": []       # 摔倒/结束标记
    }
    
    max_steps = 30000 # 设定测试时长，比如 1000 步 (约20秒)
    current_step = 0
    
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            
            # [新增] 键盘指令注入逻辑
            if args_cli.use_keyboard:
                # 1. 读取键盘输入 (返回 [vx, vy, wz])
                delta_cmd = keyboard.advance()

                # 2. 转换为 Tensor
                cmd_tensor = torch.tensor(delta_cmd, device=env.unwrapped.device)
                
                # 3. 更新 commands 变量
                commands[:, :3] = cmd_tensor.repeat(env.unwrapped.num_envs, 1)

                # 4. 更新环境内部的 command buffer
                vel_cmd_term.vel_command_b[:] = commands[:, :3]

            # agent stepping
            est = encoder(obs_history)
            actions = policy(torch.cat((est, obs, commands), dim=-1).detach())
            
            # env stepping
            obs, _, dones, infos = env.step(actions)
            
            # ================= [新增] 数据记录逻辑开始 =================
            # 获取底层环境数据 (Ground Truth)
            # 1. 先获取机器人资产对象
            robot = env.unwrapped.scene["robot"]
            
            # 2. 从机器人资产的数据中读取状态
            base_lin_vel = robot.data.root_lin_vel_b  # 基座线速度 (Body Frame)
            base_ang_vel = robot.data.root_ang_vel_b  # 基座角速度 (Body Frame)
            base_quat = robot.data.root_quat_w        # 基座姿态 (World Frame)

            # 计算欧拉角
            roll, pitch, _ = quat_to_euler(base_quat)

            # 整理数据 (只取第0个环境，如果你跑的是单环境)
            # 指令: Vx, Vy, Wz
            cmd_vec = commands[0, :3].cpu().numpy()
            # 实际: Vx, Vy, Wz
            meas_vec = torch.cat([base_lin_vel[0, :2], base_ang_vel[0, 2:3]]).cpu().numpy()
            # 姿态: Roll, Pitch
            att_vec = torch.stack([roll[0], pitch[0]]).cpu().numpy()

            # 存入列表
            log_data["cmd_vel"].append(cmd_vec)
            log_data["meas_vel"].append(meas_vec)
            log_data["attitude"].append(att_vec)
            log_data["dones"].append(dones[0].item())
            
            current_step += 1
            # ================= [新增] 数据记录逻辑结束 =================

            # --- 4. 检查是否结束测试 ---
            # 如果机器人摔倒 (dones=True) 或者达到最大步数
            if dones[0].item() or current_step >= max_steps:
                print(f"测试结束! (原因: {'摔倒' if dones[0].item() else '时间到'})")
                break


            # 更新下一帧的数据
            obs_history = infos["observations"].get("obsHistory")
            obs_history = obs_history.flatten(start_dim=1)
            commands = infos["observations"].get("commands")

    import numpy as np

    print("\n" + "="*40)
    print("       性能评估报告 (Performance Report)       ")
    print("="*40)

    # 1. 数据转换
    cmd_arr = np.array(log_data["cmd_vel"])   # Shape: (N, 3)
    meas_arr = np.array(log_data["meas_vel"]) # Shape: (N, 3)
    att_arr = np.array(log_data["attitude"])  # Shape: (N, 2)
    
    # 2. 计算速度追踪误差 (MSE)
    # 分别计算 Vx, Vy, Wz 的误差
    vel_error = (cmd_arr - meas_arr) ** 2
    mse_vx = np.mean(vel_error[:, 0])
    mse_vy = np.mean(vel_error[:, 1])
    mse_wz = np.mean(vel_error[:, 2])
    total_mse = np.mean(vel_error)

    print(f"[1] 速度追踪误差 (MSE):")
    print(f"    - Total MSE:  {total_mse:.4f}")
    print(f"    - Vx MSE:     {mse_vx:.4f}")
    print(f"    - Vy MSE:     {mse_vy:.4f}")
    print(f"    - Wz MSE:     {mse_wz:.4f}")

    # 3. 计算姿态稳定性 (Roll/Pitch 震荡幅度)
    # 我们计算绝对值的平均值 (Mean Absolute Error from 0) 或者 标准差
    avg_abs_roll = np.mean(np.abs(att_arr[:, 0]))  # 弧度
    avg_abs_pitch = np.mean(np.abs(att_arr[:, 1])) # 弧度
    
    # 转换为角度显示更直观
    print(f"[2] 姿态稳定性 (平均偏离度):")
    print(f"    - Roll  Avg:  {np.degrees(avg_abs_roll):.2f}°")
    print(f"    - Pitch Avg:  {np.degrees(avg_abs_pitch):.2f}°")

    # 4. 存活率/步数
    is_fallen = log_data["dones"][-1]
    survival_time = current_step * env_cfg.sim.dt * env_cfg.decimation # 估算秒数
    print(f"[3] 存活情况:")
    print(f"    - 状态:       {'❌ 摔倒 (Failed)' if is_fallen else '✅ 存活 (Success)'}")
    print(f"    - 持续时长:   {survival_time:.2f} s / {current_step} steps")
    
    print("="*40 + "\n")
    # =======================================================

    # close the simulator
    env.close()


if __name__ == "__main__":
    EXPORT_POLICY = True # 如果不需要每次导出，可以设为 False
    # run the main execution
    main()
    # close sim app
    simulation_app.close()