import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from uav_search.train_code.uav_env_multi import AirSimDroneEnv

def make_env(rank):
    def _init():
        env = AirSimDroneEnv(worker_index=rank)
        return env
    return _init

if __name__ == "__main__":
    # 1. 开启多进程环境（对应论文中的 Parallel Explorer）
    num_cpu = 8 
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    
    # 2. 包装 VecNormalize（这会生成最终缺失的 .pkl 文件）
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 3. 定义 PPO 模型
    model = PPO("MultiInputPolicy", env, verbose=1, device="cuda")

    # 4. 两段式训练策略逻辑：
    # 第一阶段：专注于探索（可以在 env 中调整奖励权重，先训练探索）
    model.learn(total_timesteps=200000)
    
    # 第二阶段：加入吸引力奖励进行微调
    model.learn(total_timesteps=200000)

    # 5. 保存模型（生成最终缺失的 .zip 文件）
    model.save("uav_search/models/f_ppo_num_4_final_400000")
    env.save("uav_search/models/vec_normalize_f_ppo_num_4_final.pkl")