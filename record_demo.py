import time
from og_env_wrapper import OvergrowthGymEnv

def record_demo():
    # Create environment
    env = OvergrowthGymEnv()
    
    # Start recording
    dataset_file = "expert_demo.json"
    print(f"Starting recording to {dataset_file}...")
    env.start_recording(dataset_file)
    
    obs, _ = env.reset()
    
    # Run for 1000 steps
    # In a real scenario, you would let the built-in AI fight,
    # or control the character manually while recording.
    # Here we assume the built-in AI is active or we are just logging states.
    for i in range(1000):
        # For demo purposes, we take random actions, 
        # but in the real game, the 'action' logged would be what the AI/Player actually did.
        # Since we are in the loop, we need to pass *some* action to step().
        # If we want to record the BUILT-IN AI, we might need to modify C++ to return the action it took.
        # For now, this script demonstrates the recording *mechanism*.
        action = env.action_space.sample() 
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, _ = env.reset()
            
        if i % 100 == 0:
            print(f"Recorded {i} steps...")
            
    # Stop recording and save
    env.stop_recording()
    print("Recording complete.")

if __name__ == "__main__":
    record_demo()
