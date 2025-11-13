# Overgrowth RL Interface

A comprehensive reinforcement learning interface for the Overgrowth game engine, providing a Gymnasium-compatible environment for training AI agents in physics-based combat scenarios.

## Overview

The Overgrowth RL Interface enables seamless integration of reinforcement learning algorithms with the Overgrowth game engine. This project implements a complete RL environment that exposes Overgrowth's physics simulation engine, allowing researchers and developers to train AI agents in realistic, dynamic combat scenarios without disrupting the main game's functionality.

### Key Features

- **Full Gymnasium API Compatibility**: Standard `reset()`, `step()`, and `render()` methods
- **Thread-Safe Multi-threading**: Advanced safeguards for stable RL training loops
- **Optimized Performance**: Efficient data handling and real-time computations
- **Rich State Representation**: Comprehensive observations including agent/opponent positions, health, LIDAR scans
- **Modular Reward System**: Configurable reward structures for diverse training objectives
- **PyBind11 Integration**: Seamless C++/Python bindings for high performance

### Architecture

The RL interface is implemented through five interconnected phases:

1. **Environment Setup & Build Integration**: CMake/PyBind11 integration with compilation guards
2. **C++ Environment Class Implementation**: Gymnasium API wrapper with thread-safe singleton pattern
3. **Performance and Stability Controls**: Multi-threading safeguards and resource management
4. **State/Action/Reward Logic**: High-quality data handling for RL training
5. **Python Testing Scripts**: Comprehensive validation and integration testing

## Prerequisites

### System Requirements
- **Operating System**: Windows 10/11, Linux (Ubuntu 18.04+), or macOS 10.14+
- **RAM**: Minimum 8GB, Recommended 16GB+ for RL training
- **Storage**: 10GB+ free space
- **GPU**: Optional but recommended for accelerated training (CUDA-compatible)

### Software Dependencies

#### Core Requirements
- **C++ Compiler**: GCC 9.0+ / Clang 10.0+ / MSVC 2019+ with C++17 support
- **Python**: 3.8+ (3.10+ recommended)
- **CMake**: 3.20+ for build system management
- **Git**: For repository cloning and submodule management

#### Python Packages
```bash
pip install gymnasium>=0.29.0 numpy>=1.21.0
```

#### Optional Dependencies
- **PyTorch**: `pip install torch>=2.0` (for neural network training)
- **CUDA**: Version 11.0+ (for GPU acceleration)
- **Boost**: For enhanced C++ utilities

## Installation

### Step 1: Clone Repository
```bash
git clone --recursive https://github.com/WolfireGames/overgrowth.git
cd overgrowth
```

**Important**: The `--recursive` flag is required to initialize all submodules including Bullet Physics and other dependencies.

### Step 2: Verify Submodules
```bash
git submodule update --init --recursive
```

### Step 3: Create Build Directory
```bash
mkdir build && cd build
```

### Step 4: Configure with CMake
```bash
# Enable RL interface and configure build
cmake -S .. -B . -DOG_RL_BUILD=ON -DCMAKE_BUILD_TYPE=Release

# Optional: Enable CUDA support if available
cmake -S .. -B . -DOG_RL_BUILD=ON -DCUDA=ON -DCMAKE_BUILD_TYPE=Release
```

### Step 5: Build the Project
```bash
# Build with available CPU cores
cmake --build . --config Release -j$(nproc)

# Alternative: Use Ninja if preferred
cmake --build . --config Release --target og_rl_interface
```

### Step 6: Install Python Package
```bash
cd ../python
pip install -e .
```

### Alternative: Using setup.py (Legacy)
```bash
cd python
python setup.py build_ext --inplace
pip install .
```

## Building

### CMake Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `-DOG_RL_BUILD=ON` | Enable RL interface compilation | OFF |
| `-DCMAKE_BUILD_TYPE=Release` | Optimized release build | Debug |
| `-DCUDA=ON` | Enable CUDA acceleration | OFF |
| `-DBUILD_SHARED_LIBS=ON` | Build shared libraries | ON |

### Build Targets

- `og_rl_interface`: Main RL interface library
- `all`: Complete project build
- `install`: Install built components

### Platform-Specific Notes

#### Windows (Visual Studio)
```bash
cmake -S .. -B . -G "Visual Studio 16 2019" -DOG_RL_BUILD=ON
cmake --build . --config Release --target og_rl_interface
```

#### Linux/macOS
```bash
cmake -S .. -B . -DOG_RL_BUILD=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) og_rl_interface
```

## Running Tests

### Basic Environment Test
```python
cd python

# Run the test scripts from the tests directory
python -m pytest tests/ -v

# Or run individual test scripts
python tests/test_rl.py
python tests/test_og_env.py
```

### Expected Output
```
Testing Overgrowth RL Environment...
✓ Environment creation successful
✓ Reset successful, observation shape: (128,)
✓ Step successful, reward: -0.05
✓ Render successful, shape: (480, 640, 3)
✓ Reward accumulation test passed, total: -4.23
All tests completed!
```

### Advanced Testing
```bash
# Install development dependencies
pip install -e .[dev]

# Run pytest suite from tests directory
pytest tests/

# With coverage
pytest --cov=og_rl_interface --cov-report=html

# Run specific test files
python tests/test_rl.py
python tests/test_og_env.py
```

## Usage Examples

### Basic Training Loop
```python
import gymnasium as gym
import numpy as np

# Create environment
env = gym.make('Overgrowth-v0')

# Training parameters
episodes = 1000
max_steps = 1000

for episode in range(episodes):
    obs, info = env.reset()
    episode_reward = 0
    
    for step in range(max_steps):
        # Random policy for demonstration
        action = env.action_space.sample()
        
        # Execute action
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Accumulate reward
        episode_reward += reward
        
        # Check termination
        if terminated or truncated:
            break
            
        obs = next_obs
    
    print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

env.close()
```

### Integration with Training Frameworks

#### PyTorch Example
```python
import torch
import torch.nn as nn
import gymnasium as gym
from torch.optim import Adam

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.net(x)

# Setup
env = gym.make('Overgrowth-v0')
policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
optimizer = Adam(policy.parameters(), lr=1e-3)

# Training loop
for episode in range(1000):
    obs, _ = env.reset()
    episode_reward = 0
    
    while True:
        # Get action probabilities
        action_probs = policy(torch.FloatTensor(obs))
        
        # Sample action
        action = torch.multinomial(action_probs, 1).item()
        
        # Execute action
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Training logic here...
        
        episode_reward += reward
        obs = next_obs
        
        if terminated or truncated:
            break
    
    print(f"Episode {episode}: {episode_reward}")

env.close()
```

### Jupyter Notebook Training
```python
# For interactive training and visualization
import matplotlib.pyplot as plt
from IPython.display import clear_output

env = gym.make('Overgrowth-v0')
obs, _ = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    
    # Render and display
    rgb = env.render(mode='rgb_array')
    plt.imshow(rgb)
    plt.title(f"Step: {_}, Reward: {reward:.2f}")
    clear_output(wait=True)
    plt.show()
    
    if term or trunc:
        break

env.close()
```

## API Documentation

### Core Classes

#### OvergrowthEnv

The main environment class implementing the Gymnasium interface.

##### Constructor
```python
env = og_rl_interface.OvergrowthEnv()
```

##### Methods

###### reset(seed=None)
Reset the environment to initial state.

**Parameters:**
- `seed` (int, optional): Random seed for reproducible episodes

**Returns:**
- `observation` (np.ndarray): Initial observation vector
- `info` (dict): Additional reset information

**Example:**
```python
obs, info = env.reset(seed=42)
```

###### step(action_id)
Execute one timestep in the environment.

**Parameters:**
- `action_id` (int): Action identifier (0-11 for discrete actions)

**Returns:**
- `observation` (np.ndarray): Next observation vector
- `reward` (float): Reward for the action
- `terminated` (bool): Whether episode ended due to terminal state
- `truncated` (bool): Whether episode was truncated (time limit)
- `info` (dict): Additional step information

**Example:**
```python
obs, reward, terminated, truncated, info = env.step(5)
```

###### render(mode='rgb_array')
Render the current environment state.

**Parameters:**
- `mode` (str): Render mode ('rgb_array' for RGB image)

**Returns:**
- `rgb_array` (np.ndarray): RGB image array (H, W, 3)

**Example:**
```python
rgb = env.render()
plt.imshow(rgb)
```

###### close()
Clean up environment resources.

**Example:**
```python
env.close()
```

##### Attributes

###### action_space
Discrete action space with 12 possible actions:
- 0: Move Forward
- 1: Move Backward
- 2: Move Left
- 3: Move Right
- 4: Turn Left
- 5: Turn Right
- 6: Jump
- 7: Crouch
- 8: Light Attack
- 9: Heavy Attack
- 10: Block
- 11: Idle

###### observation_space
Box observation space with 128 dimensions including:
- Agent position (x, y, z)
- Agent rotation (pitch, yaw, roll)
- Agent velocity (x, y, z)
- Agent health and stamina
- Opponent position, rotation, velocity, health, stamina
- LIDAR distance readings (16 rays)
- Action cooldowns and states

###### metadata
Environment metadata dictionary containing render modes and timestep limits.

### Spaces

#### Action Space
```python
# Discrete action space
action_space = gym.spaces.Discrete(12)

# Action mapping
actions = {
    0: "move_forward",
    1: "move_backward",
    2: "strafe_left",
    3: "strafe_right",
    4: "turn_left",
    5: "turn_right",
    6: "jump",
    7: "crouch",
    8: "light_attack",
    9: "heavy_attack",
    10: "block",
    11: "idle"
}
```

#### Observation Space
```python
# Continuous observation space
observation_space = gym.spaces.Box(
    low=-np.inf, high=np.inf, 
    shape=(128,), dtype=np.float32
)

# Observation structure (example)
obs_structure = {
    'agent_pos': obs[0:3],      # x, y, z
    'agent_rot': obs[3:6],      # pitch, yaw, roll
    'agent_vel': obs[6:9],      # vx, vy, vz
    'agent_health': obs[9],     # 0-100
    'agent_stamina': obs[10],   # 0-100
    'opponent_pos': obs[11:14], # x, y, z
    'opponent_rot': obs[14:17], # pitch, yaw, roll
    'opponent_vel': obs[17:20], # vx, vy, vz
    'opponent_health': obs[20], # 0-100
    'opponent_stamina': obs[21],# 0-100
    'lidar_distances': obs[22:38], # 16 rays
    'action_states': obs[38:128]   # cooldowns and flags
}
```

### Reward System

The reward function is computed based on multiple factors:

```python
reward = (
    health_delta_reward * reward_weights['health'] +
    stamina_delta_reward * reward_weights['stamina'] +
    distance_reward * reward_weights['distance'] +
    damage_dealt_reward * reward_weights['damage'] +
    exploration_reward * reward_weights['exploration'] +
    survival_bonus * reward_weights['survival'] +
    idle_penalty * reward_weights['idle']
)
```

#### Reward Components

- **Health Delta**: Reward for maintaining/increasing health
- **Stamina Management**: Efficient stamina usage
- **Distance Control**: Optimal positioning relative to opponent
- **Damage Dealt**: Combat effectiveness
- **Exploration**: Action diversity bonus
- **Survival**: Time-based survival rewards
- **Idle Penalty**: Discourages inaction

#### Termination Conditions

Episodes terminate when:
- Agent health <= 0 (death)
- Opponent health <= 0 (victory)
- Episode exceeds maximum timesteps (1000 steps)
- Agent falls out of arena bounds

## Troubleshooting

### Build Issues

#### CMake Configuration Errors
```
Error: Could not find Python development headers
```
**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install python3-dev python3-numpy

# macOS
brew install python3 numpy

# Windows
pip install --upgrade pip setuptools wheel
```

#### PyBind11 Not Found
```
CMake Error: PyBind11 not found
```
**Solution:**
```bash
pip install pybind11[global]>=2.10.0
```

#### Compiler Errors
```
error: C++17 standard required
```
**Solution:**
- GCC: Add `-std=c++17` flag
- Clang: Add `-std=c++17` flag
- MSVC: Use Visual Studio 2019+ or add `/std:c++17`

### Runtime Issues

#### Import Errors
```
ModuleNotFoundError: No module named 'og_rl_interface'
```
**Solution:**
```bash
# Ensure library is built and in Python path
cd build
cmake --build . --config Release --target og_rl_interface
cd ../python
pip install -e .
```

#### Threading Conflicts
```
RuntimeError: Thread safety violation
```
**Solution:**
- Ensure single environment instance per process
- Use environment in main thread only
- Implement proper process isolation for parallel training

#### Memory Issues
```
std::bad_alloc during training
```
**Solution:**
- Reduce episode length
- Implement experience replay with memory limits
- Monitor system resources during training
- Use smaller batch sizes

### Performance Issues

#### Slow Training
- Use GPU acceleration if available (`-DCUDA=ON`)
- Optimize observation space (reduce LIDAR rays if needed)
- Implement frame skipping
- Use parallel environments with `gym.vector`

#### Frame Rate Synchronization
- The environment targets 10ms per step (100 FPS)
- Training loops may run slower due to neural network inference
- Consider asynchronous training methods

## Integration Notes

### Build System Integration

#### CMakeLists.txt Modifications
```cmake
# In root CMakeLists.txt
option(OG_RL_BUILD "Build RL interface" OFF)

if(OG_RL_BUILD)
    add_subdirectory(Projects/RL)
    # Additional RL-specific configurations
endif()
```

#### Compilation Guards
All RL code is wrapped in `#ifdef OG_RL_BUILD` to prevent conflicts with production builds:

```cpp
#ifdef OG_RL_BUILD
// RL-specific code here
class OvergrowthEnv { ... };
#endif
```

### Game Loop Integration

The RL interface runs a separate simulation loop independent of the main game:
- Uses dedicated physics simulation
- Maintains separate state from gameplay
- No interference with main game mechanics
- Thread-safe operation with GIL management

### Asset Requirements

The RL environment requires minimal assets:
- Basic physics collision shapes
- No texture or mesh dependencies
- Procedural generation for visual elements
- Lightweight rendering for observation generation

## Contributing

### Development Setup
```bash
# Fork and clone
git clone https://github.com/yourusername/overgrowth.git
cd overgrowth

# Create feature branch
git checkout -b feature/rl-improvement

# Install development dependencies
cd python
pip install -e .[dev]
```

### Code Style
```bash
# Format Python code
black .

# Lint Python code
flake8 .

# Format C++ code (if clang-format available)
clang-format -i Projects/RL/*.cpp Projects/RL/*.hpp
```

### Testing
```bash
# Run Python tests from tests directory
pytest tests/

# Run with coverage
pytest tests/ --cov=og_rl_interface --cov-report=term-missing

# Build and test C++ components
cd build
cmake --build . --config Debug --target og_rl_interface
ctest

# Run individual test scripts
python tests/test_rl.py
python tests/test_og_env.py
```

### Pull Request Guidelines
1. Ensure all tests pass
2. Update documentation for API changes
3. Add tests for new features
4. Follow existing code style
5. Provide clear commit messages

## Appendices

### Phase 1: Environment Setup & Build Integration

**Objectives:**
- Seamless CMake/PyBind11 integration
- Compilation guards with `OG_RL_BUILD` flag
- Dedicated `og_rl_interface` target
- Python package structure in `python/` subdirectory

**Key Deliverables:**
- Modified root `CMakeLists.txt` with RL build options
- PyBind11-based Python extension module
- `setup.py` and `pyproject.toml` for package management
- Build scripts supporting multiple platforms

**Integration Points:**
- Root CMakeLists.txt: `add_subdirectory(Projects/RL)`
- Build target: `og_rl_interface`
- Python package: `og_rl_interface`

### Phase 2: C++ Environment Class Implementation

**Objectives:**
- Full Gymnasium API compatibility
- Thread-safe singleton pattern implementation
- PyBind11 bindings for Python exposure
- Comprehensive state management

**Key Components:**
- `OvergrowthEnv` class with core methods (`reset`, `step`, `render`)
- Thread-safe singleton access pattern
- PyBind11 module definition
- Exception handling and error recovery

**Technical Details:**
- Thread safety via `std::shared_mutex`
- RNG seeding with `std::mt19937_64`
- Profiling infrastructure with `spdlog`
- Memory management with smart pointers

### Phase 3: Performance and Stability Controls

**Objectives:**
- Multi-threading safeguards for RL training
- Efficient resource management
- Stability enhancements for training loops
- Crash prevention in multi-threaded contexts

**Key Features:**
- GIL management for Python thread safety
- Atomic operations for shared state
- Exception-safe resource handling
- Frame rate synchronization (10ms target)
- Memory leak prevention
- Proper cleanup on environment destruction

**Threading Model:**
- Main simulation thread for physics
- Python GIL management for bindings
- Atomic counters for metrics
- Lock-free observation generation

### Phase 4: State/Action/Reward Logic

**Objectives:**
- High-quality data handling for RL training
- Precise state representations
- Action mappings with realistic timing
- Termination conditions and reward structures

**State Representation:**
- Agent/opponent position, rotation, velocity (9 floats each)
- Health and stamina values (2 floats each)
- LIDAR distance readings (16 floats)
- Action states and cooldowns (remaining floats)

**Action System:**
- 12 discrete actions with realistic physics
- Attack cooldowns (100ms)
- Movement constraints based on stamina
- Collision detection and response

**Reward Structure:**
- Health preservation (+0.1 per point maintained)
- Damage dealt (+1.0 per point)
- Optimal positioning (distance-based)
- Action diversity (+0.05 for unique actions)
- Survival bonus (+0.01 per step alive)
- Idle penalty (-0.05 for repeated inaction)

### Phase 5: Python Testing Scripts

**Objectives:**
- Comprehensive validation suite
- Environment instantiation testing
- Action sampling and execution
- Rendering and observation validation
- Reward accumulation verification

**Test Coverage:**
- Environment creation and destruction
- Reset functionality with seeding
- Step execution across all actions
- Rendering in multiple modes
- Reward computation accuracy
- Termination condition handling
- Multi-episode stability
- Memory leak detection
- Threading safety verification

**Test Scripts:**
- `tests/test_rl.py`: Basic functionality validation
- `tests/test_og_env.py`: Pytest-based comprehensive testing
- Integration tests with Gymnasium
- Performance benchmarking
- Stress testing for stability

---

## License

Apache License 2.0 - See [LICENSE](../LICENSE) for details.

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines.

## FAQ

**Q: Can I use this for commercial projects?**
A: Yes, the Apache 2.0 license allows commercial use.

**Q: Does this modify the main Overgrowth game?**
A: No, the RL interface is completely separate and uses compilation guards to prevent interference.

**Q: What platforms are supported?**
A: Windows, Linux, and macOS with appropriate compilers.

**Q: Can I train multiple agents simultaneously?**
A: Yes, but each environment instance should run in a separate process for thread safety.

**Q: What's the performance impact on training?**
A: The environment targets 100 FPS (10ms per step) and is optimized for RL training workloads.