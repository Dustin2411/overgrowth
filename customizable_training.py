import json
import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch
from og_env_wrapper import OvergrowthGymEnv

def load_config(config_path="agent_config.json"):
    with open(config_path, 'r') as f:
        return json.load(f)

def train_custom_agent():
    # 1. Load Configuration
    config = load_config()
    print("Loaded Configuration:")
    print(json.dumps(config, indent=4))

    # 2. Setup Environment
    env_config = config["environment"]
    env = OvergrowthGymEnv(render_mode=env_config.get("render_mode", "rgb_array"))
    
    # Apply Environment Settings
    if "time_scale" in env_config:
        print(f"Setting game speed to {env_config['time_scale']}x")
        env.set_game_speed(env_config["time_scale"])
        
    if "gravity_y" in env_config:
        print(f"Setting gravity to (0, {env_config['gravity_y']}, 0)")
        env.set_gravity(0, env_config["gravity_y"], 0)

    # 3. Setup Network Architecture
    net_config = config["network"]
    # Parse activation function
    activation_fn = nn.Tanh
    if net_config.get("activation_fn") == "relu":
        activation_fn = nn.ReLU
    elif net_config.get("activation_fn") == "elu":
        activation_fn = nn.ELU

    policy_kwargs = dict(
        activation_fn=activation_fn,
        net_arch=net_config.get("net_arch", [64, 64])
    )

    # 4. Setup Agent
    train_config = config["training"]
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=train_config.get("learning_rate", 0.0003),
        ent_coef=train_config.get("ent_coef", 0.0),
        batch_size=train_config.get("batch_size", 64),
        n_epochs=train_config.get("n_epochs", 10),
        policy_kwargs=policy_kwargs
    )

    # 5. Train
    print(f"Starting training for {train_config['total_timesteps']} timesteps...")
    model.learn(total_timesteps=train_config["total_timesteps"])
    print("Training complete.")

    # 6. Save
    save_path = "custom_agent_model"
    model.save(save_path)
    print(f"Model saved to {save_path}.zip")

if __name__ == "__main__":
    train_custom_agent()
