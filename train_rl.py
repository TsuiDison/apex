import os
import sys
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

# 确保能导入 uav_search 模块
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from uav_search.train_code.uav_env_multi import AirSimDroneEnv

def make_env(rank):
    """
    创建环境的工厂函数
    """
    def _init():
        # AirSimDroneEnv 内部会根据 worker_index 处理端口偏移和任务分配
        env = AirSimDroneEnv(worker_index=rank)
        return env
    return _init

def train():
    # 基础配置
    num_envs = 4
    total_timesteps = 100000
    model_dir = os.path.join(ROOT_DIR, "uav_search", "models")
    log_dir = os.path.join(ROOT_DIR, "ppo_uav_tensorboard")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"正在启动 {num_envs} 个并行训练环境...")

    # 1. 创建多进程矢量化环境
    # 注意：在 Linux 上使用 SubprocVecEnv 通常比 DummyVecEnv 快
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    
    # 2. 加入归一化包装器 (VecNormalize)
    # 这是 APEX 运行脚本中加载模型前必须经过的步骤
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 3. 初始化 PPO 模型
    # 使用 MultiInputPolicy 因为观测空间是 Dict (包含地图和障碍物信息)
    model = PPO(
        "MultiInputPolicy", 
        env, 
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1, 
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log=log_dir
    )

    # 4. 设置保存点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // num_envs, 
        save_path=model_dir,
        name_prefix="f_ppo_num_4"
    )

    # 5. 开始学习
    print(f"开始训练，目标总步数: {total_timesteps}")
    try:
        model.learn(
            total_timesteps=total_timesteps, 
            callback=checkpoint_callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("训练被用户中断，正在尝试保存当前权重...")

    # 6. 保存最终模型和归一化统计数据 (STATS_PATH)
    model_path = os.path.join(model_dir, "f_ppo_num_4_final_100000")
    stats_path = os.path.join(model_dir, "vec_normalize_f_ppo_num_4_final.pkl")
    
    model.save(model_path)
    env.save(stats_path)
    
    print(f"训练完成！")
    print(f"模型已保存至: {model_path}.zip")
    print(f"归一化数据已保存至: {stats_path}")


if __name__ == "__main__":
    train()

