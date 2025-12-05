
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from og_env_wrapper import OvergrowthGymEnv
import imageio
import os

def main():
    print("Initializing environment...")
    env = OvergrowthGymEnv(render_mode="rgb_array")
    
    # Check if environment follows Gym API
    print("Checking environment compatibility...")
    check_env(env)
    print("Environment is compatible!")

    # Train a simple agent
    print("\nTraining PPO agent...")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1000) # Short training for demonstration
    print("Training complete!")

    # Save the model
    model.save("ppo_overgrowth")

    # Run and render
    print("\nRunning trained agent and capturing video...")
    obs, info = env.reset(seed=42)
    frames = []
    
    for i in range(100): # Run for 100 steps
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render frame
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        
        if terminated or truncated:
            obs, info = env.reset()

    env.close()

    # Save GIF
    if frames:
        print(f"Saving {len(frames)} frames to agent_demo.gif...")
        imageio.mimsave("agent_demo.gif", frames, fps=30)
        print("Video saved successfully!")
    else:
        print("No frames captured.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
