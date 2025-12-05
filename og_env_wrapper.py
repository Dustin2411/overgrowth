
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import og_env

class OvergrowthGymEnv(gym.Env):
    """
    Gymnasium wrapper for the C++ OvergrowthEnv.
    """
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        self.env = og_env.OvergrowthEnv.getInstance()
        self.render_mode = render_mode
        
        # Define spaces based on C++ env
        # Action space is Discrete(12)
        self.action_space = spaces.Discrete(12)
        
        # Observation space is Box(10,)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(10,), 
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # C++ reset takes optional seed
        # Ensure seed is None or a valid non-negative integer
        cpp_seed = int(seed) if seed is not None else None
        obs, info = self.env.reset(seed=cpp_seed)
        
        return np.array(obs, dtype=np.float32), info

    def step(self, action):
        # C++ step takes int
        obs, reward, terminated, truncated, info = self.env.step(int(action))
        return np.array(obs, dtype=np.float32), float(reward), bool(terminated), bool(truncated), info

    def render(self):
        return self.env.render(self.render_mode if self.render_mode else "console")

    def close(self):
        self.env.close()

if __name__ == "__main__":
    # Test the wrapper
    env = OvergrowthGymEnv()
    print("Wrapper created successfully")
    
    obs, info = env.reset(seed=42)
    print(f"Reset obs shape: {obs.shape}")
    
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    print(f"Step result: reward={reward}, term={term}")
    print("Wrapper verification passed!")
