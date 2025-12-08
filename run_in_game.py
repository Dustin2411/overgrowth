
import sys
import os
import numpy as np
from stable_baselines3 import PPO

# Global model variable
model = None

def init_model(model_path):
    """
    Initialize the model from the given path.
    """
    global model
    print(f"[Python] Loading model from: {model_path}")
    try:
        # Check if file exists
        if not os.path.exists(model_path):
            print(f"[Python] Error: Model file not found at {model_path}")
            return False
            
        model = PPO.load(model_path)
        print("[Python] Model loaded successfully.")
        return True
    except Exception as e:
        print(f"[Python] Exception loading model: {e}")
        return False

def get_action_from_obs(obs_list):
    """
    Get action from observation list.
    Called from C++.
    """
    global model
    if model is None:
        return 0 # Default action if model not loaded

    try:
        # Convert list to numpy array
        # Expecting 10 dimensions for the "Self" observation
        obs = np.array(obs_list, dtype=np.float32)
        
        # Predict
        action, _states = model.predict(obs, deterministic=True)
        
        # Ensure it's a scalar int
        return int(action)
    except Exception as e:
        print(f"[Python] Error during prediction: {e}")
        return 0
