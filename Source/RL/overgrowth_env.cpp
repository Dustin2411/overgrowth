/**
 * @file overgrowth_env.cpp
 * @brief Implementation of Overgrowth RL Environment
 */

#ifdef OG_RL_BUILD
#include "overgrowth_env.hpp"
#include "Objects/movementobject.h"
#include "Main/engine.h"

#include <cmath>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <unordered_set>

using namespace py::literals;

// Static member definitions
std::shared_ptr<OvergrowthEnv> OvergrowthEnv::instance_ = nullptr;
std::once_flag OvergrowthEnv::once_flag_;
const ActionSpace OvergrowthEnv::action_space_ = ActionSpace();
const ObservationSpace OvergrowthEnv::observation_space_ = ObservationSpace();

/**
 * @brief Constructor for OvergrowthEnv
 */
OvergrowthEnv::OvergrowthEnv() {
    // Initialize RNG
    std::random_device rd;
    rng_.seed(rd());

    // Initialize state
    current_obs_ = std::vector<float>(10, 0.0f); // 10-dimensional observation
    cum_reward_ = 0.0f;
    terminated_ = false;
    truncated_ = false;
    step_count_ = 0;
    deterministic_ = false;
    seed_used_ = 0;

    // Set up Gymnasium-compatible spaces
    action_space = py::cast(12); // Discrete(12)
    // 10 (self) + 3 (rel pos) + 3 (rel vel) + 1 (dist) + 1 (enemy health) = 18
    observation_space = py::dict("shape"_a = py::make_tuple(18), "dtype"_a = py::dtype::of<float>());

    // Set up metadata
    metadata = py::dict();
    metadata["render_modes"] = py::cast(std::vector<std::string>{"rgb_array", "human"});
    metadata["render_fps"] = 30;
    metadata["action_space_type"] = "Discrete";
    metadata["observation_space_type"] = "Box";
    metadata["reward_range"] = py::make_tuple(-10.0f, 10.0f);
    metadata["max_episode_steps"] = 1000;

    // Set unwrapped reference
    unwrapped = std::shared_ptr<OvergrowthEnv>(this, [](OvergrowthEnv*) {});
}

/**
 * @brief Destructor
 */
OvergrowthEnv::~OvergrowthEnv() {}

/**
 * @brief Thread-safe singleton accessor
 */
std::shared_ptr<OvergrowthEnv> OvergrowthEnv::getInstance(py::kwargs kwargs) {
    std::cout << "[DEBUG] getInstance called with kwargs size: " << kwargs.size() << std::endl;
    std::call_once(once_flag_, []() {
        std::cout << "[DEBUG] Creating singleton OvergrowthEnv instance" << std::endl;
        instance_ = std::shared_ptr<OvergrowthEnv>(new OvergrowthEnv());
    });
    return instance_;
}

/**
 * @brief Gets action space metadata
 */
ActionSpace OvergrowthEnv::getActionSpace() {
    return action_space_;
}

/**
 * @brief Gets observation space metadata
 */
ObservationSpace OvergrowthEnv::getObservationSpace() {
    return observation_space_;
}

/**
 * @brief Resets the environment
 */
std::tuple<py::array_t<float>, py::dict> OvergrowthEnv::reset(std::optional<uint64_t> seed) {
    if (seed.has_value()) {
        seed_used_ = seed.value();
        rng_.seed(seed_used_);
    } else {
        std::random_device rd;
        seed_used_ = rd();
        rng_.seed(seed_used_);
    }

    // Reset state
    current_obs_ = std::vector<float>(18, 0.0f);
    cum_reward_ = 0.0f;
    terminated_ = false;
    truncated_ = false;
    step_count_ = 0;

    // Initialize observation with some random values
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& val : current_obs_) {
        val = dist(rng_);
    }

    auto observation = get_observation();
    py::dict info;
    info["seed_used"] = seed_used_;

    return {observation, info};
}

/**
 * @brief Takes a step in the environment
 */
std::tuple<py::array_t<float>, float, bool, bool, py::dict> OvergrowthEnv::step(int action_id) {
    if (action_id < 0 || action_id >= 12) {
        throw std::invalid_argument("Action ID must be between 0 and 11");
    }

    // Update state based on action
    update_state(action_id);

    // Compute reward
    float reward = compute_reward(action_id);

    // Compute new observation
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& val : current_obs_) {
        val += dist(rng_) * 0.1f; // Small random changes
        val = std::max(-1.0f, std::min(1.0f, val)); // Clamp to [-1, 1]
    }

    // Check termination conditions
    terminated_ = check_termination();
    truncated_ = check_truncation();

    step_count_++;

    auto observation = get_observation();
    py::dict info;
    info["step_count"] = step_count_;
    info["cum_reward"] = cum_reward_;

    return {observation, reward, terminated_, truncated_, info};
}

/**
 * @brief Gets action mask
 */
py::array_t<bool> OvergrowthEnv::get_action_mask() const {
    py::array_t<bool> mask(12);
    auto mask_buf = mask.request();
    bool* mask_ptr = static_cast<bool*>(mask_buf.ptr);
    std::fill(mask_ptr, mask_ptr + 12, true); // All actions available for now
    return mask;
}

/**
 * @brief Renders the environment
 */
py::array_t<uint8_t> OvergrowthEnv::render(const std::string& mode) {
    if (mode != "rgb_array") {
        throw std::runtime_error("Only 'rgb_array' render mode is supported");
    }

    const int width = 64, height = 64;
    py::array_t<uint8_t> rgb_array({height, width, 3});
    auto buf = rgb_array.request();
    uint8_t* ptr = static_cast<uint8_t*>(buf.ptr);

    // Simple gradient
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            ptr[(y * width + x) * 3 + 0] = static_cast<uint8_t>((x * 255) / width);
            ptr[(y * width + x) * 3 + 1] = static_cast<uint8_t>((y * 255) / height);
            ptr[(y * width + x) * 3 + 2] = 128;
        }
    }

    return rgb_array;
}

/**
 * @brief Closes the environment
 */
void OvergrowthEnv::close() {
    terminated_ = true;
}

/**
 * @brief Gets profiling data
 */
py::dict OvergrowthEnv::get_profiling_data() const {
    py::dict result;
    for (const auto& [key, value] : profiling_data_) {
        result[key.c_str()] = value;
    }
    return result;
}

/**
 * @brief Sets deterministic mode
 */
void OvergrowthEnv::set_deterministic(bool deterministic) {
    deterministic_ = deterministic;
    if (deterministic) {
        rng_.seed(42);
    } else {
        std::random_device rd;
        rng_.seed(rd());
    }
}

/**
 * @brief Sets log level
 */
void OvergrowthEnv::set_log_level(const std::string& level) {
    // Placeholder - no logging implemented
}

/**
 * @brief Sets the Enemy to track
 */
void OvergrowthEnv::set_enemy(MovementObject* enemy) {
    enemy_ = enemy;
}

/**
 * @brief Sets game speed (Time Scaling)
 */
void OvergrowthEnv::set_game_speed(float speed) {
    if (Engine::Instance()) {
        Engine::Instance()->SetGameSpeed(speed, true);
    }
}

/**
 * @brief Sets a float variable in the enemy's AngelScript (Dynamic Difficulty)
 */
void OvergrowthEnv::set_enemy_script_float(const std::string& var_name, float value) {
    if (enemy_) {
        enemy_->ASSetFloatVar(var_name, value);
    }
}

/**
 * @brief Gets current observation
 */
py::array_t<float> OvergrowthEnv::get_observation() const {
    if (npc_) {
        std::vector<float> obs;
        obs.reserve(18);
        
        // 1. Self State (10 dims)
        obs.push_back(npc_->position.x);
        obs.push_back(npc_->position.y);
        obs.push_back(npc_->position.z);
        obs.push_back(npc_->velocity.x);
        obs.push_back(npc_->velocity.y);
        obs.push_back(npc_->velocity.z);
        obs.push_back(npc_->GetTempHealth());
        obs.push_back(npc_->facing.x);
        obs.push_back(npc_->facing.y);
        obs.push_back(npc_->facing.z);

        // 2. Enemy State (8 dims)
        if (enemy_) {
            // Relative Position
            obs.push_back(enemy_->position.x - npc_->position.x);
            obs.push_back(enemy_->position.y - npc_->position.y);
            obs.push_back(enemy_->position.z - npc_->position.z);
            
            // Relative Velocity
            obs.push_back(enemy_->velocity.x - npc_->velocity.x);
            obs.push_back(enemy_->velocity.y - npc_->velocity.y);
            obs.push_back(enemy_->velocity.z - npc_->velocity.z);

            // Distance
            float dist = glm::distance(npc_->position, enemy_->position);
            obs.push_back(dist);

            // Enemy Health
            obs.push_back(enemy_->GetTempHealth());
        } else {
            // Padding if no enemy
            for(int i=0; i<8; ++i) obs.push_back(0.0f);
        }
        
        return py::cast(obs);
    }

    // Fallback to placeholder if no NPC
    py::array_t<float> observation(current_obs_.size());
    auto buf = observation.request();
    float* ptr = static_cast<float*>(buf.ptr);
    std::copy(current_obs_.begin(), current_obs_.end(), ptr);
    return observation;
}

/**
 * @brief Computes reward for the given action
 */
float OvergrowthEnv::compute_reward(int action_id) {
    float reward = 0.0f;

    if (npc_ && enemy_) {
        // 1. Damage Dealt (Positive)
        // Note: This requires tracking previous health. For now, we assume direct access or event-based.
        // Simplified: Reward for being close and attacking
        float dist = glm::distance(npc_->position, enemy_->position);
        if (dist < 2.0f && action_id == 6) { // Attack action
            reward += 0.5f;
        }

        // 2. Proximity Reward (Shaping)
        if (dist < 5.0f) {
            reward += 0.01f * (5.0f - dist);
        }

        // 3. Health Check (Victory/Defeat)
        if (enemy_->GetTempHealth() <= 0.0f) reward += 10.0f; // Win
        if (npc_->GetTempHealth() <= 0.0f) reward -= 10.0f;   // Lose
        
        return reward;
    }

    // Fallback random reward for testing
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    reward += dist(rng_);
    cum_reward_ += reward;
    return reward;
}

/**
 * @brief Updates environment state based on action
 */
void OvergrowthEnv::update_state(int action_id) {
    if (npc_) {
        // Map actions to game inputs
        // This is a simplified mapping
        switch(action_id) {
            case 0: npc_->InputFromAngelScript("forward"); break;
            case 1: npc_->InputFromAngelScript("back"); break;
            case 2: npc_->InputFromAngelScript("left"); break;
            case 3: npc_->InputFromAngelScript("right"); break;
            case 4: npc_->InputFromAngelScript("jump"); break;
            case 5: npc_->InputFromAngelScript("crouch"); break;
            case 6: npc_->InputFromAngelScript("attack"); break;
            case 7: npc_->InputFromAngelScript("block"); break;
            case 8: npc_->InputFromAngelScript("throw"); break;
            case 9: npc_->InputFromAngelScript("grab"); break;
            // ... add more mappings
        }
        return;
    }

    // Simple state update - modify observation based on action
    float delta = (action_id % 2 == 0) ? 0.05f : -0.05f;
    for (size_t i = 0; i < current_obs_.size(); ++i) {
        current_obs_[i] += delta * (i + 1) * 0.01f;
        current_obs_[i] = std::max(-1.0f, std::min(1.0f, current_obs_[i]));
    }
}

/**
 * @brief Checks if episode should terminate
 */
bool OvergrowthEnv::check_termination() const {
    // Terminate after 100 steps or if sum of observation exceeds threshold
    float obs_sum = std::accumulate(current_obs_.begin(), current_obs_.end(), 0.0f);
    return step_count_ >= 100 || std::abs(obs_sum) > 5.0f;
}

/**
 * @brief Checks if episode should be truncated
 */
bool OvergrowthEnv::check_truncation() const {
    return step_count_ >= 1000;
}

#endif // OG_RL_BUILD