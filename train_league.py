import gymnasium as gym
import numpy as np
import os
import random
import shutil
from stable_baselines3 import PPO
from og_env_wrapper import OvergrowthGymEnv

def train_league():
    # Create environment
    env = OvergrowthGymEnv()
    
    # League Directory
    league_dir = "league_models"
    os.makedirs(league_dir, exist_ok=True)
    
    # Initialize Champion
    champion_path = os.path.join(league_dir, "champion_0")
    if not os.path.exists(champion_path + ".zip"):
        print("Initializing League with first Champion...")
        model = PPO("MlpPolicy", env, verbose=0)
        model.save(champion_path)
    
    current_agent_path = "current_agent"
    if os.path.exists(current_agent_path + ".zip"):
        current_agent = PPO.load(current_agent_path, env=env)
    else:
        current_agent = PPO("MlpPolicy", env, verbose=1)

    # Training Loop
    episodes = 1000
    champion_idx = 0
    
    for episode in range(episodes):
        # 1. Domain Randomization (Sim2Real)
        # Randomize gravity between 0.8g and 1.2g
        g_mult = random.uniform(0.8, 1.2)
        env.set_gravity(0, -9.8 * g_mult, 0)
        
        # 2. League Matchmaking
        # 80% chance to fight a random past champion, 20% chance to fight the latest
        opponent_files = [f for f in os.listdir(league_dir) if f.endswith(".zip")]
        if random.random() < 0.8 and opponent_files:
            opponent_name = random.choice(opponent_files)
            opponent_path = os.path.join(league_dir, opponent_name[:-4]) # remove .zip
            print(f"Episode {episode}: Fighting {opponent_name} (Gravity: {g_mult:.2f}x)")
        else:
            opponent_path = os.path.join(league_dir, f"champion_{champion_idx}")
            print(f"Episode {episode}: Fighting Current Champion (Gravity: {g_mult:.2f}x)")
            
        opponent_model = PPO.load(opponent_path)
        
        # 3. Self-Play Episode
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Agent Action
            action, _ = current_agent.predict(obs)
            
            # Opponent Action
            enemy_obs = env.get_enemy_observation()
            enemy_action, _ = opponent_model.predict(enemy_obs)
            env.set_enemy_action(enemy_action)
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # Learn (simplified, usually done via model.learn callback)
            # current_agent.collect_rollouts(...) 
        
        print(f"Result: Reward = {total_reward}")
        
        # 4. Promotion Logic
        if total_reward > 10.0: # Strong win
            print("Promoting Agent to League!")
            champion_idx += 1
            new_champ_path = os.path.join(league_dir, f"champion_{champion_idx}")
            current_agent.save(new_champ_path)
            current_agent.save(current_agent_path) # Save checkpoint

if __name__ == "__main__":
    train_league()
