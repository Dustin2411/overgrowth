import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from og_env_wrapper import OvergrowthGymEnv
import os

def train_self_play():
    # Create environment
    env = OvergrowthGymEnv()
    
    # Initialize models
    champion_model_path = "champion_model"
    challenger_model_path = "challenger_model"
    
    if os.path.exists(champion_model_path + ".zip"):
        print("Loading existing Champion model...")
        champion_model = PPO.load(champion_model_path)
    else:
        print("Creating new Champion model...")
        champion_model = PPO("MlpPolicy", env, verbose=0)
        champion_model.save(champion_model_path)

    print("Creating Challenger model...")
    challenger_model = PPO("MlpPolicy", env, verbose=1)

    # Self-Play Training Loop
    episodes = 100
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # 1. Get Action for Challenger (Main Agent)
            challenger_action, _ = challenger_model.predict(obs)
            
            # 2. Get Action for Champion (Enemy Agent)
            # We need the observation from the enemy's perspective
            enemy_obs = env.get_enemy_observation()
            champion_action, _ = champion_model.predict(enemy_obs)
            
            # 3. Apply Enemy Action
            env.set_enemy_action(champion_action)
            
            # 4. Step Environment with Challenger Action
            obs, reward, terminated, truncated, info = env.step(challenger_action)
            done = terminated or truncated
            total_reward += reward
            
            # Note: In a real training loop, you would call challenger_model.learn() here
            # or collect transitions into a buffer. SB3's learn() manages the loop internally,
            # so for true self-play with SB3, you'd need a custom Callback that sets the enemy action.
            
        print(f"Episode {episode + 1}: Reward = {total_reward}")

        # Simple League Logic: If Challenger wins enough, it becomes the new Champion
        if total_reward > 5.0: # Arbitrary threshold
            print("Challenger defeated Champion! Promoting...")
            challenger_model.save(champion_model_path)
            champion_model = PPO.load(champion_model_path)

    print("Self-play training demonstration complete.")

if __name__ == "__main__":
    train_self_play()
