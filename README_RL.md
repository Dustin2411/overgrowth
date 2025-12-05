## Advanced Features (New!)
The environment now supports advanced combat training:

### 1. Enhanced Observation Space (18 Dimensions)
The agent now "sees" the enemy:
- **Self (10)**: Position (3), Velocity (3), Health (1), Facing (3).
- **Enemy (8)**: Relative Position (3), Relative Velocity (3), Distance (1), Enemy Health (1).

### 2. Combat Rewards
The reward function encourages fighting:
- **Damage Dealt**: Positive reward for hurting the enemy.
- **Victory**: Large bonus for winning.
### 3. Expert Features
- **Time Scaling**: Speed up training with `env.set_game_speed(5.0)`.
- **Dynamic Difficulty**: Adjust enemy AI on the fly with `env.set_enemy_property("aggression", 1.0)`.

### 4. Grandmaster Features (Self-Play)
- **Multi-Agent Control**: Control both the agent and the enemy with `env.set_enemy_action(action)`.
- **Symmetric Vision**: Get the enemy's perspective with `env.get_enemy_observation()`.
- **Self-Play Training**: Run `python train_self_play.py` to train a "Challenger" model against a "Champion" model.

### 5. Godlike Features (Imitation Learning)
- **Headless Mode**: Run the game with `--disable-rendering` for 100x speedup.
- **Recording**: Run `python record_demo.py` to capture gameplay data.
- **Behavior Cloning**: Run `python pretrain_agent.py` to learn from the recorded data.
- Python 3.7+
- C++ Compiler (MSVC on Windows)
- CMake (for fetching dependencies)

### Building the Module
The module is built using `pip`, which handles the compilation of C++ sources.

```bash
pip install -e .
```

This command compiles `og_env_bindings.cpp` and `overgrowth_env.cpp` and installs the `og_env` module in editable mode.

## Usage

### Basic Python Usage
You can use the raw C++ environment directly:

```python
import og_env

env = og_env.OvergrowthEnv()
obs, info = env.reset(seed=42)
action = 0
obs, reward, terminated, truncated, info = env.step(action)
```

### Using with Gymnasium / Stable Baselines3
For standard RL workflows, use the provided wrapper:

```python
from og_env_wrapper import OvergrowthGymEnv
from stable_baselines3 import PPO

# Create environment
env = OvergrowthGymEnv(render_mode="rgb_array")

# Train agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Save model
model.save("ppo_overgrowth")
```

## Demos
- **Verification**: Run `python verify_og_env.py` to test basic functionality.
- **Training Demo**: Run `python train_agent.py` to train a PPO agent and generate a GIF of the agent in action (`agent_demo.gif`).

## Project Structure
- `overgrowth-main/Source/RL/`: C++ source files (`overgrowth_env.cpp`, `overgrowth_env.hpp`, `og_env_bindings.cpp`).
- `og_env_wrapper.py`: Gymnasium wrapper for the environment.
- `setup.py`: Build configuration script.
- `train_agent.py`: Example script for training and rendering.

## Game Integration (Next Steps)
The current environment is a **standalone simulation** for testing the RL pipeline. To train an NPC to fight in the actual game:
1.  Read the `integration_plan.md` artifact I created.
2.  You must build the full Overgrowth game engine (not just this module).
3.  You need to modify `overgrowth_env.cpp` to connect to the `MovementObject` class (the game's character controller).

See `integration_plan.md` for the specific code changes required.

## Linux Support
For instructions on building and running on Linux, see `linux_setup.md`.
