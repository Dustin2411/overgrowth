import pytest
import og_env
import gymnasium as gym
import numpy as np
import time

def test_invalid_seeds():
    env = og_env.OvergrowthEnv()
    with pytest.raises(ValueError):
        env.reset(seed=-1)

def test_out_of_range_actions():
    env = og_env.OvergrowthEnv()
    env.reset(seed=42)
    # Assuming action space is continuous or discrete, test invalid actions
    # For example, if action space is Box, test values outside bounds
    try:
        env.step(np.array([10.0] * env.action_space.shape[0]))  # Assuming high > 1
    except ValueError:
        pass  # Expected
    else:
        pytest.fail("Expected ValueError for out-of-range action")

def test_termination_logic():
    env = og_env.OvergrowthEnv()
    obs, info = env.reset(seed=42)
    terminated = False
    steps = 0
    while not terminated and steps < 1000:
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        if term:
            terminated = True
            assert info.get('terminated_reason') in ['health_zero', 'max_steps']
        steps += 1
    assert terminated, "Environment should terminate"

def test_state_reproducibility():
    env = og_env.OvergrowthEnv()
    obs_list = []
    for _ in range(100):
        obs, _ = env.reset(seed=42)
        obs_list.append(obs)
    for i in range(1, len(obs_list)):
        assert np.allclose(obs_list[0], obs_list[i])

def test_performance_benchmarks():
    env = og_env.OvergrowthEnv()
    env.reset(seed=42)
    start = time.perf_counter()
    for _ in range(1000):
        env.step(env.action_space.sample())
    end = time.perf_counter()
    sps = 1000 / (end - start)
    assert sps > 100, f"SPS too low: {sps}"

def test_edge_cases():
    env = og_env.OvergrowthEnv()
    # Close without reset
    env.close()  # Should not raise error
    # Reset after close
    with pytest.raises(Exception):  # Assuming close invalidates env
        env.reset(seed=42)

def test_gaussian_noise_consistency():
    env = og_env.OvergrowthEnv()
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    # Assuming some noise is Gaussian, should be reproducible
    assert np.allclose(obs1, obs2)