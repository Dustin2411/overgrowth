import og_env
import gymnasium as gym
import numpy as np
import time
import psutil
import os
import sys
import matplotlib.pyplot as plt
import wandb
from stable_baselines3 import PPO

# Create directories
os.makedirs('logs/videos', exist_ok=True)
os.makedirs('logs/tb', exist_ok=True)
os.makedirs('logs/plots', exist_ok=True)
os.makedirs('models/ppo_overgrowth', exist_ok=True)

def test_reset():
    env = og_env.OvergrowthEnv()
    env = gym.wrappers.RecordVideo(env, video_folder='logs/videos', episode_trigger=lambda ep: ep % 10 == 0)
    obs, info = env.reset(seed=42)
    assert 'seed_used' in info and obs.shape == (24,) and obs.dtype == np.float32
    print(f"Reset OK, seed: {info['seed_used']}")
    return env, obs, info

def benchmark_steps(env, num_steps=300):
    start = time.perf_counter()
    actions_taken = []
    rewards = []
    for i in range(num_steps):
        act = env.action_space.sample()
        obs_, r, term, trunc, info_ = env.step(act)
        actions_taken.append(act)
        rewards.append(r)
        assert not np.any(np.isnan(obs_)) and not np.any(np.isinf(obs_))
        if term or trunc:
            break
    sps = (len(actions_taken)) / (time.perf_counter() - start)
    assert sps > 100
    print(f"Benchmark: {sps:.1f} SPS")
    return sps, rewards, actions_taken

def train_rl(env):
    model_dir = 'models/ppo_overgrowth'
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log='logs/tb', device='cpu')
    model.learn(total_timesteps=10000)
    model.save(model_dir + '/final_model')
    print("Training complete")
    return model

def test_reproducibility(env):
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    assert np.allclose(obs1, obs2)
    print("Reproducibility OK")

def visualize(rewards, actions_taken):
    # Plot reward curve
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Reward Curve')
    plt.xlabel('Step')
    plt.ylabel('Reward')

    # Plot action histogram
    plt.subplot(1, 2, 2)
    plt.hist(actions_taken, bins=20)
    plt.title('Action Histogram')
    plt.xlabel('Action')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('logs/plots/trajectories.png')
    plt.close()
    print("Visualization saved")

def monitor_memory():
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    # Simulate some operations
    env = og_env.OvergrowthEnv()
    for _ in range(100):
        env.reset(seed=42)
        env.step(env.action_space.sample())
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    assert mem_after - mem_before < 50  # Assume no major leaks
    print(f"Memory before: {mem_before:.1f} MB, after: {mem_after:.1f} MB")

def main():
    wandb.init(project='overgrowth-rl')

    try:
        env, obs, info = test_reset()
        sps, rewards, actions_taken = benchmark_steps(env)
        train_rl(env)
        test_reproducibility(env)
        visualize(rewards, actions_taken)
        monitor_memory()

        # Vectorization prep (example)
        num_envs = 4
        envs = gym.vector.SyncVectorEnv([lambda: og_env.OvergrowthEnv() for _ in range(num_envs)])

        # Log to wandb
        wandb.log({'sps': sps, 'episode_return': np.sum(rewards)})

        print("API compliance verified, ready for advanced RL")

    except Exception as e:
        with open('logs/error.log', 'w') as f:
            f.write(str(e))
        raise
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()