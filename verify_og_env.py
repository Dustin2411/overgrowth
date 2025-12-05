
import og_env
import numpy as np
import sys

def verify():
    print("Verifying og_env module...")
    
    # Test 1: Instantiation
    print("\nTest 1: Instantiation")
    try:
        env = og_env.OvergrowthEnv.getInstance()
        print("  Success: Environment instance obtained")
    except Exception as e:
        print(f"  Failed: {e}")
        return

    # Test 2: Reset
    print("\nTest 2: Reset")
    try:
        obs, info = env.reset(seed=42)
        print(f"  Success: Reset complete. Obs shape: {obs.shape}, Info: {info}")
    except Exception as e:
        print(f"  Failed: {e}")
        return

    # Test 3: Step
    print("\nTest 3: Step")
    try:
        action = 0 # Discrete action
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Success: Step complete.")
        print(f"    Reward: {reward}")
        print(f"    Terminated: {terminated}")
        print(f"    Truncated: {truncated}")
        print(f"    Info: {info}")
    except Exception as e:
        print(f"  Failed: {e}")
        return

    print("\nVerification passed!")

if __name__ == "__main__":
    verify()
