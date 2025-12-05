
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
        
        # Observation space is Box(18,)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(18,), 
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

    def set_game_speed(self, speed):
        """
        Sets the game simulation speed (e.g., 2.0 for 2x speed).
        """
        self.env.set_game_speed(float(speed))

    def set_enemy_property(self, name, value):
        """
        Sets a float property on the enemy's script (e.g., 'aggression').
        """
        self.env.set_enemy_script_float(name, float(value))

    def set_enemy_action(self, action):
        """
        Sets the action for the enemy agent (Self-Play).
        """
        self.env.set_enemy_action(int(action))

    def get_enemy_observation(self):
        """
        Gets the observation from the enemy's perspective (Self-Play).
        Returns numpy array of shape (18,).
        """
        obs = self.env.get_enemy_observation()
        return np.array(obs, dtype=np.float32)

    def start_recording(self, filename):
        """
        Starts recording transitions to a file (Imitation Learning).
        """
        self.env.start_recording(filename)

    def stop_recording(self):
        """
        Stops recording and saves the dataset.
        """
        self.env.stop_recording()

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
