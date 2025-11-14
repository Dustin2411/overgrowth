#ifdef OG_RL_BUILD
#include "overgrowth_env.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#ifdef _WIN32
#include <windows.h>
#endif
#ifdef __linux__
#include <sys/resource.h>
#endif

std::shared_ptr<OvergrowthEnv> OvergrowthEnv::instance_;
std::once_flag OvergrowthEnv::once_flag_;

// Singleton implementation
std::shared_ptr<OvergrowthEnv> OvergrowthEnv::getInstance(py::kwargs kwargs) {
    std::call_once(once_flag_, [&kwargs]() {
        instance_ = std::shared_ptr<OvergrowthEnv>(new OvergrowthEnv(kwargs));
    });
    return instance_;
}

OvergrowthEnv::OvergrowthEnv(py::kwargs kwargs) {
    // Headless mode setup - disable GUI
#ifdef _WIN32
    // Create offscreen HDC for Windows headless mode
    HDC offscreenDC = CreateCompatibleDC(NULL);
    if (!offscreenDC) {
        throw pybind11::runtime_error("OvergrowthEnv Error: Failed to create offscreen DC for headless mode");
    }
#else
    // On Linux/Mac, assume xvfb-run is used externally or handle via environment
#endif

    try {
        // Initialize kwargs
        if (kwargs.contains("reward_weights")) {
            reward_weights_ = kwargs["reward_weights"].cast<py::dict>();
        } else {
            reward_weights_["sparse"] = 1.0f;
            reward_weights_["dense"] = 1.0f;
            reward_weights_["shaping"] = 1.0f;
            reward_weights_["curiosity"] = 1.0f;
            reward_weights_["punishment"] = 1.0f;
        }

        if (kwargs.contains("lidar_rays")) {
            lidar_rays_ = kwargs["lidar_rays"].cast<int>();
            lidar_rays_ = std::clamp(lidar_rays_, 0, 16);
        }

        initialize_spaces();
        initialize_metadata();
        unwrapped = getInstance(); // Self-reference for Gym compatibility

        // Initialize logging
        logger_ = spdlog::basic_logger_mt("overgrowth_rl", "overgrowth_rl.log");
        logger_->set_level(spdlog::level::info);
        logger_->flush_on(spdlog::level::warn);

        // Initialize timing
        simulation_start_time_ = std::chrono::high_resolution_clock::now();
        last_frame_time_ = simulation_start_time_;

        SPDLOG_LOGGER_INFO(logger_, "OvergrowthEnv initialized in headless mode");

    } catch (const std::exception& e) {
        throw pybind11::runtime_error(std::string("OvergrowthEnv Error: ") + e.what());
    }
}

void OvergrowthEnv::initialize_spaces() {
    // Import gym.spaces dynamically
    py::module_ gym = py::module_::import("gymnasium");
    py::module_ spaces = gym.attr("spaces");

    // Expanded action space: Discrete(35) for enriched actions
    action_space = spaces.attr("Discrete")(35);

    // Extended observation space: Box with normalized values [-1, 1], expanded dimensions for enriched features
    int obs_size = 60 + lidar_rays_; // Expanded for new sensory and state features
    observation_space = spaces.attr("Box")(
        py::cast(std::vector<float>(obs_size, -1.0f)),
        py::cast(std::vector<float>(obs_size, 1.0f)),
        py::make_tuple(obs_size),
        py::dtype("float32")
    );
}

void OvergrowthEnv::initialize_metadata() {
    metadata["render_modes"] = py::make_tuple("rgb_array");
    metadata["render_fps"] = 30;
}

std::tuple<py::array_t<float>, py::dict> OvergrowthEnv::reset(std::optional<uint64_t> seed) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    try {
        episode_id_++;
        SPDLOG_LOGGER_INFO(logger_, "Episode {}: Reset called", episode_id_);

        reset_episode(seed);

        auto observation = get_observation();
        auto info = create_info_dict();

        SPDLOG_LOGGER_INFO(logger_, "Episode {}: Reset completed successfully", episode_id_);
        return {observation, info};
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(logger_, "Episode {}: Reset failed: {}", episode_id_, e.what());
        throw pybind11::runtime_error(std::string("OvergrowthEnv Error: ") + e.what());
    }
}

std::tuple<py::array_t<float>, float, bool, bool, py::dict> OvergrowthEnv::step(int action_id) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    try {
        auto step_start = std::chrono::high_resolution_clock::now();

        validate_action(action_id);
        inject_action(action_id);

        // Custom loop with fixed dt=0.01s, decoupled from wall-clock
        auto current_time = std::chrono::high_resolution_clock::now();
        double wall_dt = std::chrono::duration<double>(current_time - last_frame_time_).count();

        // Accumulate time for fixed simulation steps
        accumulated_time_ += wall_dt;

        double tick_start = accumulated_time_;
        while (accumulated_time_ >= target_dt_) {
            auto tick_time = std::chrono::high_resolution_clock::now();

            update_physics(static_cast<float>(target_dt_));
            update_ai();

            auto tick_end = std::chrono::high_resolution_clock::now();
            double tick_duration = std::chrono::duration<double>(tick_end - tick_time).count();

            profiling_data_.emplace_back("physics_tick", tick_duration);

            accumulated_time_ -= target_dt_;
        }
        double tick_total = accumulated_time_ - tick_start;
        profiling_data_.emplace_back("step_total", tick_total);

        last_frame_time_ = current_time;

        step_count_++;
        episode_steps_++;

        auto observation = get_observation();
        float reward = compute_reward();
        cum_reward_ += reward;
        bool terminated = check_termination();
        bool truncated = check_truncation();
        auto info = create_info_dict();

        // Check for NaN/inf in observation and reward
        check_array_for_nans(observation);
        if (!is_finite(reward)) {
            SPDLOG_LOGGER_WARN(logger_, "Episode {}: Invalid reward detected, clamping to 0.0", episode_id_);
            reward = 0.0f;
        }

        auto step_end = std::chrono::high_resolution_clock::now();
        double step_duration = std::chrono::duration<double>(step_end - step_start).count();
        profiling_data_.emplace_back("step_duration", step_duration);

        SPDLOG_LOGGER_INFO(logger_, "Episode {}: Step {} completed in {:.3f}ms", episode_id_, step_count_, step_duration * 1000.0);

        return {observation, reward, terminated, truncated, info};
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(logger_, "Episode {}: Step failed: {}", episode_id_, e.what());
        throw pybind11::runtime_error(std::string("OvergrowthEnv Error: ") + e.what());
    }
}

py::array_t<uint8_t> OvergrowthEnv::render(const std::string& mode) {
    // Return mock RGB array (480x640x3)
    const int height = 480;
    const int width = 640;
    const int channels = 3;

    py::array_t<uint8_t> rgb_array({height, width, channels});
    auto buf = rgb_array.mutable_unchecked<3>();

    // Fill with zeros (black screen)
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            for (int c = 0; c < channels; ++c) {
                buf(h, w, c) = 0;
            }
        }
    }

    return rgb_array;
}

void OvergrowthEnv::close() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    try {
        SPDLOG_LOGGER_INFO(logger_, "OvergrowthEnv closing");
        // Cleanup resources if needed
        initialized_ = false;
        profiling_data_.clear();
        episode_id_ = 0;
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(logger_, "Error during OvergrowthEnv close: {}", e.what());
    }
}

py::dict OvergrowthEnv::get_profiling_data() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    py::dict result;

    try {
        // Calculate averages and percentiles
        std::unordered_map<std::string, std::vector<double>> timings;

        for (const auto& [name, time] : profiling_data_) {
            timings[name].push_back(time);
        }

        for (const auto& [name, values] : timings) {
            if (values.empty()) continue;

            std::sort(values.begin(), values.end());
            double sum = 0.0;
            for (double v : values) sum += v;

            double avg = sum / values.size();
            double p50 = values[values.size() / 2];
            double p95 = values[values.size() * 95 / 100];

            py::dict timing_data;
            timing_data["average"] = avg;
            timing_data["median"] = p50;
            timing_data["95th_percentile"] = p95;
            timing_data["count"] = static_cast<int>(values.size());

            result[name.c_str()] = timing_data;
        }
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(logger_, "Error getting profiling data: {}", e.what());
    }

    return result;
}

void OvergrowthEnv::set_deterministic(bool deterministic) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    deterministic_ = deterministic;
    SPDLOG_LOGGER_INFO(logger_, "Deterministic mode set to: {}", deterministic);
}

void OvergrowthEnv::set_log_level(const std::string& level) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    try {
        if (level == "DEBUG") {
            logger_->set_level(spdlog::level::debug);
        } else if (level == "INFO") {
            logger_->set_level(spdlog::level::info);
        } else if (level == "WARN") {
            logger_->set_level(spdlog::level::warn);
        } else if (level == "ERROR") {
            logger_->set_level(spdlog::level::err);
        } else {
            throw std::invalid_argument("Invalid log level: " + level);
        }
        SPDLOG_LOGGER_INFO(logger_, "Log level set to: {}", level);
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(logger_, "Error setting log level: {}", e.what());
        throw pybind11::runtime_error(std::string("OvergrowthEnv Error: ") + e.what());
    }
}

py::array_t<float> OvergrowthEnv::get_observation() const {
    // Enhanced state vector with enriched sensory and state features
    int obs_size = 60 + lidar_rays_;
    std::vector<float> obs(obs_size, 0.0f);

    // Core agent state (0-9): pos[3], rot[3], vel[3], health, stamina
    for (size_t i = 0; i < 3; ++i) {
        obs[i] = std::clamp(agent_pos_[i] / arena_radius_, -1.0f, 1.0f);
        obs[3 + i] = std::clamp(agent_rot_[i] / static_cast<float>(M_PI), -1.0f, 1.0f);
        obs[6 + i] = std::clamp(agent_vel_[i] / max_velocity_, -1.0f, 1.0f);
    }
    obs[9] = health_ / 100.0f * 2.0f - 1.0f;
    obs[10] = stamina_ / 100.0f * 2.0f - 1.0f;

    // Core opponent state (11-21): pos[3], rot[3], vel[3], health, stamina
    for (size_t i = 0; i < 3; ++i) {
        obs[11 + i] = std::clamp(opponent_pos_[i] / arena_radius_, -1.0f, 1.0f);
        obs[14 + i] = std::clamp(opponent_rot_[i] / static_cast<float>(M_PI), -1.0f, 1.0f);
        obs[17 + i] = std::clamp(opponent_vel_[i] / max_velocity_, -1.0f, 1.0f);
    }
    obs[20] = opponent_health_ / 100.0f * 2.0f - 1.0f;
    obs[21] = opponent_stamina_ / 100.0f * 2.0f - 1.0f;

    // Combat and timing state (22-29)
    float dist = calculate_distance(agent_pos_, opponent_pos_);
    obs[22] = std::clamp(dist / max_dist_, -1.0f, 1.0f);
    obs[23] = action_cooldown_timer_ / attack_cooldown_duration_ * 2.0f - 1.0f;
    obs[24] = parry_timer_ / 0.5f * 2.0f - 1.0f; // Parry window normalized
    obs[25] = dodge_cooldown_ / 1.0f * 2.0f - 1.0f; // Dodge cooldown normalized
    obs[26] = punch_active_ ? 1.0f : -1.0f;
    obs[27] = kick_active_ ? 1.0f : -1.0f;
    obs[28] = riposte_window_ ? 1.0f : -1.0f;

    // Weapon and stance state (29-35)
    obs[29] = current_weapon_ == 0 ? -1.0f : 1.0f; // -1: melee, 1: gun
    obs[30] = current_stance_ == 0 ? 1.0f : (current_stance_ == 1 ? -1.0f : 0.0f); // Offensive:1, Defensive:-1, Balanced:0
    obs[31] = stamina_regen_rate_ / 1.2f * 2.0f - 1.0f; // Regen rate normalized
    obs[32] = fatigue_accumulation_ / 10.0f * 2.0f - 1.0f; // Fatigue level
    obs[33] = locomotion_mode_ == 0 ? -1.0f : (locomotion_mode_ == 1 ? 1.0f : 0.0f); // Walk:-1, Sprint:1, Stealth:0

    // Ally and environmental state (34-45)
    obs[34] = ally_status_ * 2.0f - 1.0f;
    obs[35] = ally_shared_health_ * 2.0f - 1.0f;
    obs[36] = ally_cooldown_ / 5.0f * 2.0f - 1.0f;
    obs[37] = climbing_active_ ? 1.0f : -1.0f;
    obs[38] = swinging_active_ ? rope_tension_ * 2.0f - 1.0f : -1.0f;
    obs[39] = grapple_active_ ? grapple_momentum_ / max_velocity_ * 2.0f - 1.0f : -1.0f;
    obs[40] = wall_running_ ? wall_velocity_taper_ * 2.0f - 1.0f : -1.0f;
    obs[41] = ledge_grabbed_ ? ledge_recovery_anim_ * 2.0f - 1.0f : -1.0f;
    obs[42] = object_held_ ? 1.0f : -1.0f;
    obs[43] = throwable_trajectory_ * 2.0f - 1.0f;
    obs[44] = in_water_ ? buoyancy_force_ * 2.0f - 1.0f : -1.0f;
    obs[45] = path_progress_ * 2.0f - 1.0f;

    // Sensory expansions (46-69)
    obs[46] = vibration_feedback_ * 2.0f - 1.0f;
    obs[47] = audio_fft_spectrum_[0] * 2.0f - 1.0f; // Low freq band
    obs[48] = audio_fft_spectrum_[1] * 2.0f - 1.0f; // Mid freq band
    obs[49] = audio_fft_spectrum_[2] * 2.0f - 1.0f; // High freq band
    obs[50] = (audio_fft_spectrum_[3] + audio_fft_spectrum_[4] + audio_fft_spectrum_[5]) / 3.0f * 2.0f - 1.0f; // Avg upper bands
    obs[51] = (audio_fft_spectrum_[6] + audio_fft_spectrum_[7]) / 2.0f * 2.0f - 1.0f; // Highest bands
    obs[52] = gas_dispersion_[0] * 2.0f - 1.0f; // Gas type 1
    obs[53] = gas_dispersion_[1] * 2.0f - 1.0f; // Gas type 2
    obs[54] = gas_dispersion_[2] * 2.0f - 1.0f; // Gas type 3
    obs[55] = gas_dispersion_[3] * 2.0f - 1.0f; // Gas type 4

    // Depth sensing point cloud (56-71)
    for (size_t i = 0; i < depth_point_cloud_.size(); ++i) {
        obs[56 + i] = depth_point_cloud_[i];
    }

    // LIDAR rays (72+)
    if (lidar_rays_ > 0) {
        std::vector<float> lidar_data = perform_lidar_scan();
        for (int i = 0; i < lidar_rays_; ++i) {
            obs[72 + i] = std::clamp(lidar_data[i] / arena_radius_, -1.0f, 1.0f);
        }
    }

    py::array_t<float> result(obs_size);
    auto buf = result.mutable_unchecked<1>();
    for (size_t i = 0; i < obs_size; ++i) {
        buf(i) = obs[i];
    }

    return result;
}

float OvergrowthEnv::compute_reward() {
    float reward = 0.0f;
    py::dict reward_breakdown;

    // Enhanced sparse reward: win/loss with combat bonuses
    float sparse_reward = 0.0f;
    if (check_termination()) {
        if (opponent_health_ <= 0.0f && health_ > 0.0f) {
            sparse_reward = 20.0f; // Win
            if (riposte_window_ && last_action_id_ == 11) { // Successful riposte finish
                sparse_reward += 10.0f;
            }
        } else if (health_ <= 0.0f && opponent_health_ > 0.0f) {
            sparse_reward = -20.0f; // Loss
        }
    }
    reward_breakdown["sparse"] = sparse_reward;

    // Enhanced dense reward: health, stamina, and new mechanics
    float dense_reward = 0.0f;
    float health_delta = health_ - prev_agent_health_;
    dense_reward += health_delta * 0.2f;

    // Stamina management reward
    float stamina_delta = stamina_ - (prev_agent_health_ * 0.5f); // Approximate previous stamina
    dense_reward += stamina_delta * 0.1f;

    // Ally interaction reward
    if (ally_shared_health_ > prev_agent_health_ * 0.1f) { // Ally healing
        dense_reward += 0.5f;
    }
    reward_breakdown["dense"] = dense_reward;

    // Enhanced shaping reward: proximity, environmental, and tactical incentives
    float shaping_reward = 0.0f;
    float current_dist = calculate_distance(agent_pos_, opponent_pos_);
    if (current_dist < prev_dist_to_opponent_) {
        shaping_reward += 0.1f; // Closer to opponent
    } else {
        shaping_reward -= 0.05f; // Idle penalty
    }

    // Environmental interaction bonuses
    if (climbing_active_ && bone_ik_solvers_.size() > 0) {
        shaping_reward += 0.2f; // Climbing progress
    }
    if (grapple_active_ && grapple_momentum_ > 0) {
        shaping_reward += 0.3f; // Successful grapple
    }
    if (ledge_grabbed_ && ledge_recovery_anim_ > 0) {
        shaping_reward += 0.1f; // Ledge recovery
    }
    if (wall_running_ && wall_velocity_taper_ > 0.7f) {
        shaping_reward += 0.15f; // Wall running
    }
    if (swinging_active_ && rope_tension_ > 0.5f) {
        shaping_reward += 0.1f; // Rope swinging
    }

    // Tactical bonuses
    if (parry_active_ && parry_timer_ > 0) {
        shaping_reward += 0.05f; // Active parry
    }
    if (riposte_window_ && last_action_id_ != 11) {
        shaping_reward += 0.1f; // Riposte opportunity available
    }
    if (current_stance_ == 1 && stamina_ > 80.0f) { // Good defensive positioning
        shaping_reward += 0.05f;
    }

    // Locomotion efficiency
    if (locomotion_mode_ == 2 && fatigue_accumulation_ < 2.0f) { // Stealth without fatigue
        shaping_reward += 0.05f;
    }
    if (locomotion_mode_ == 1 && fatigue_accumulation_ > 5.0f) { // Sprinting with high fatigue
        shaping_reward -= 0.1f;
    }
    reward_breakdown["shaping"] = shaping_reward;

    // Enhanced curiosity reward: exploration of new mechanics
    float curiosity_reward = 0.0f;
    if (last_action_id_ >= 0 && unique_actions_taken_.find(last_action_id_) == unique_actions_taken_.end()) {
        unique_actions_taken_.insert(last_action_id_);
        curiosity_reward += 0.01f;
    }

    // Bonus for exploring enriched actions
    if (last_action_id_ >= 10 && last_action_id_ <= 34) {
        curiosity_reward += 0.005f; // Extra exploration bonus for new actions
    }

    // Navigation and sensory exploration
    if (!a_star_path_.empty()) {
        curiosity_reward += 0.02f; // Pathfinding usage
    }
    if (audio_fft_spectrum_[0] > 0.5f) { // Audio cues detected
        curiosity_reward += 0.01f;
    }
    if (vibration_feedback_ > 0.3f) { // Tactile feedback
        curiosity_reward += 0.01f;
    }
    reward_breakdown["curiosity"] = curiosity_reward;

    // Enhanced punishment: collisions, failed actions, and hazards
    float punishment_reward = collision_count_ * -0.1f;

    // Failed parry/riposte attempts
    if (last_action_id_ == 10 && parry_timer_ <= 0) {
        punishment_reward -= 0.2f; // Failed parry
    }
    if (last_action_id_ == 11 && !riposte_window_) {
        punishment_reward -= 0.5f; // Invalid riposte
    }

    // Environmental hazards
    for (float gas : gas_dispersion_) {
        if (gas > 0.7f) { // High gas concentration
            punishment_reward -= 0.1f;
        }
    }

    // Weapon switching inefficiency
    if (last_action_id_ == 20 && stamina_ < 20.0f) {
        punishment_reward -= 0.1f; // Switching when low stamina
    }

    // Failed environmental interactions
    if (last_action_id_ == 22 && !climbing_active_) {
        punishment_reward -= 0.1f; // Failed climb
    }
    if (last_action_id_ == 25 && grapple_momentum_ <= 0) {
        punishment_reward -= 0.2f; // Failed grapple
    }
    reward_breakdown["punishment"] = punishment_reward;

    // Apply weights
    float weighted_reward = sparse_reward * reward_weights_["sparse"].cast<float>() +
                            dense_reward * reward_weights_["dense"].cast<float>() +
                            shaping_reward * reward_weights_["shaping"].cast<float>() +
                            curiosity_reward * reward_weights_["curiosity"].cast<float>() +
                            punishment_reward * reward_weights_["punishment"].cast<float>();

    // Store breakdown in info dict (will be accessed by create_info_dict)
    reward_breakdown_ = reward_breakdown;

    // Update previous values
    prev_agent_health_ = health_;
    prev_opponent_health_ = opponent_health_;
    prev_dist_to_opponent_ = current_dist;

    // Reset idle penalty timer if action taken
    if (last_action_id_ != 9) { // Not idle
        idle_penalty_timer_ = 0.0f;
    }

    // Clamp to [-50, 50]
    return std::clamp(weighted_reward, -50.0f, 50.0f);
}

py::array_t<bool> OvergrowthEnv::get_action_mask() const {
    std::vector<bool> mask(35, true); // All actions available by default

    // Prune invalid actions based on stamina and state
    if (stamina_ < 10.0f) {
        // Disable high-stamina actions (original attacks)
        mask[6] = false; // Punch
        mask[7] = false; // Block
        mask[8] = false; // Kick
        // Disable new high-stamina actions
        mask[10] = false; // Parry
        mask[11] = false; // Riposte
        mask[12] = false; // Evade Dodge
        mask[20] = false; // Weapon Switch
        mask[25] = false; // Grapple Hook
    }

    if (action_cooldown_timer_ > 0.0f) {
        // Disable attack actions during cooldown
        mask[6] = false; // Punch
        mask[8] = false; // Kick
        mask[10] = false; // Parry
        mask[11] = false; // Riposte
        mask[12] = false; // Evade Dodge
    }

    if (dodge_cooldown_ > 0.0f) {
        mask[12] = false; // Evade Dodge
    }

    if (parry_timer_ > 0.0f) {
        mask[11] = false; // Riposte (must be within parry window)
    }

    if (ally_cooldown_ > 0) {
        mask[21] = false; // Ally Interact
    }

    // Disable environmental actions based on state
    if (!climbing_active_ && !ledge_grabbed_) {
        mask[26] = false; // Environmental Interact
    }

    if (object_held_) {
        mask[27] = false; // Object Pickup (already holding)
    }

    if (grapple_active_) {
        mask[25] = false; // Grapple Hook
    }

    py::array_t<bool> result(35);
    auto buf = result.mutable_unchecked<1>();
    for (size_t i = 0; i < 35; ++i) {
        buf(i) = mask[i];
    }

    return result;
}

void OvergrowthEnv::update_physics(float dt) {
    // Enhanced physics update with enriched mechanics
    try {
        // Update action cooldown
        if (action_cooldown_timer_ > 0.0f) {
            action_cooldown_timer_ -= dt;
            if (action_cooldown_timer_ < 0.0f) action_cooldown_timer_ = 0.0f;
        }

        // Update parry/riposte timing windows
        if (parry_timer_ > 0.0f) {
            parry_timer_ -= dt;
            if (parry_timer_ <= 0.0f) {
                parry_active_ = false;
                riposte_window_ = false; // Window expired
            } else if (parry_timer_ > 0.3f) { // First 0.2s is active parry
                riposte_window_ = true; // Last 0.3s is riposte window
            }
        }

        // Update dodge cooldown
        if (dodge_cooldown_ > 0.0f) {
            dodge_cooldown_ -= dt;
            if (dodge_cooldown_ < 0.0f) dodge_cooldown_ = 0.0f;
        }

        // Update ally interaction cooldown
        if (ally_cooldown_ > 0) {
            ally_cooldown_--;
        }

        // Update attack flags
        if (punch_active_ || kick_active_) {
            punch_active_ = false;
            kick_active_ = false;
        }

        // Enhanced agent movement with environmental physics
        float movement_modifier = 1.0f;
        if (climbing_active_) {
            movement_modifier = 0.7f; // Slower climbing
            // Update IK solvers for climbing
            for (auto& ik : bone_ik_solvers_) {
                // Simple IK simulation - move joints toward target
                for (size_t i = 0; i < 3; ++i) {
                    if (ik.joint_positions.size() <= i) ik.joint_positions.push_back(agent_pos_[i]);
                    ik.joint_positions[i] += (ik.target_pos[i] - ik.joint_positions[i]) * 0.1f;
                }
            }
        } else if (swinging_active_) {
            movement_modifier = 1.2f; // Momentum from swinging
            rope_tension_ = std::max(0.1f, rope_tension_ - 0.01f); // Tension decay
        } else if (grapple_active_) {
            movement_modifier = 1.5f; // Grapple momentum
            grapple_momentum_ = std::max(0.0f, grapple_momentum_ - 0.5f * dt);
        } else if (wall_running_) {
            movement_modifier = 1.3f; // Wall running speed
            wall_velocity_taper_ = std::max(0.5f, wall_velocity_taper_ - 0.02f * dt);
        } else if (ledge_grabbed_) {
            movement_modifier = 0.0f; // Stationary while grabbing ledge
        } else if (in_water_) {
            movement_modifier = 0.6f; // Water resistance
            // Apply buoyancy
            agent_vel_[1] += buoyancy_force_ * dt;
            // Apply water resistance
            for (auto& vel : agent_vel_) {
                vel *= (1.0f - water_resistance_ * dt);
            }
        }

        // Apply locomotion fatigue
        if (locomotion_mode_ == 1) { // Sprinting
            fatigue_accumulation_ += dt * 0.2f;
            movement_modifier *= 1.4f;
        } else if (locomotion_mode_ == 2) { // Stealth
            movement_modifier *= 0.8f;
        }

        // Simulate agent movement with modifiers
        std::uniform_real_distribution<float> vel_dist(-max_velocity_ * 0.1f * movement_modifier, max_velocity_ * 0.1f * movement_modifier);
        for (size_t i = 0; i < 3; ++i) {
            agent_vel_[i] += vel_dist(rng_) * dt;
            agent_pos_[i] += agent_vel_[i] * dt * movement_modifier;
            clamp_values(agent_pos_[i], -arena_radius_, arena_radius_);
        }

        // Simulate opponent movement
        std::uniform_real_distribution<float> opp_vel_dist(-max_velocity_ * 0.15f, max_velocity_ * 0.15f);
        for (size_t i = 0; i < 3; ++i) {
            opponent_vel_[i] += opp_vel_dist(rng_) * dt;
            opponent_pos_[i] += opponent_vel_[i] * dt;
            clamp_values(opponent_pos_[i], -arena_radius_, arena_radius_);
        }

        // Enhanced health/stamina with new mechanics
        float health_decay = -1.0f * dt * 10.0f;
        if (current_stance_ == 1) { // Defensive stance reduces decay
            health_decay *= 0.8f;
        }
        if (ally_shared_health_ > 0.0f) { // Ally health sharing
            health_ += ally_shared_health_ * dt * 5.0f;
            ally_shared_health_ = std::max(0.0f, ally_shared_health_ - dt * 0.1f);
        }

        if (deterministic_) {
            health_ += health_decay;
            stamina_ += -1.5f * dt * 10.0f;
            opponent_health_ += -0.8f * dt * 10.0f;
        } else {
            std::uniform_real_distribution<float> health_dist(-2.0f, 1.0f);
            health_ += health_dist(rng_) * dt * 10.0f + health_decay;

            std::uniform_real_distribution<float> stamina_dist(-3.0f, 2.0f);
            stamina_ += stamina_dist(rng_) * dt * 10.0f;

            std::uniform_real_distribution<float> opp_health_dist(-1.5f, 0.5f);
            opponent_health_ += opp_health_dist(rng_) * dt * 10.0f;
        }

        // Apply fatigue effects
        stamina_ -= fatigue_accumulation_ * dt * 2.0f;
        fatigue_accumulation_ = std::max(0.0f, fatigue_accumulation_ - dt * 0.05f);

        clamp_values(health_, 0.0f, 100.0f);
        clamp_values(stamina_, 0.0f, 100.0f);
        clamp_values(opponent_health_, 0.0f, 100.0f);

        // Enhanced collision detection with environmental interactions
        float dist = calculate_distance(agent_pos_, opponent_pos_);
        if (dist < 2.0f) { // Collision threshold
            collision_count_++;
            health_ -= 5.0f;
            opponent_health_ -= 5.0f;
            vibration_feedback_ = 0.5f;

            // Simulate gas dispersion on impact (hazardous areas)
            gas_dispersion_[0] += 0.3f; // Increase gas level
        }

        // Update throwable trajectory prediction
        if (throwable_trajectory_ > 0.0f) {
            throwable_trajectory_ -= dt * 0.5f;
            if (throwable_trajectory_ <= 0.0f) {
                throwable_trajectory_ = 0.0f;
            }
        }

        // Update ledge recovery animation
        if (ledge_recovery_anim_ > 0.0f) {
            ledge_recovery_anim_ -= dt;
            if (ledge_recovery_anim_ <= 0.0f) {
                climbing_active_ = false; // Recovery complete
                ledge_recovery_anim_ = 0.0f;
            }
        }

        // Update path progress
        if (!a_star_path_.empty()) {
            path_progress_ += dt * 0.1f; // Slow progress along path
            if (path_progress_ >= a_star_path_.size()) {
                a_star_path_.clear(); // Path complete
                path_progress_ = 0.0f;
            }
        }

        // Update metrics
        physics_ticks_++;

        // Log enhanced anomalies
        if (health_ < 10.0f) {
            SPDLOG_LOGGER_WARN(logger_, "Episode {}: Low health detected: {:.2f}", episode_id_, health_);
        }
        if (stamina_ < 10.0f) {
            SPDLOG_LOGGER_WARN(logger_, "Episode {}: Low stamina detected: {:.2f}", episode_id_, stamina_);
        }
        if (fatigue_accumulation_ > 5.0f) {
            SPDLOG_LOGGER_WARN(logger_, "Episode {}: High fatigue detected: {:.2f}", episode_id_, fatigue_accumulation_);
        }

    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(logger_, "Episode {}: Physics update failed: {}", episode_id_, e.what());
        throw pybind11::runtime_error(std::string("OvergrowthEnv Error: Physics update failed - ") + e.what());
    }
}

void OvergrowthEnv::update_ai() {
    // Update opponent AI state and populate opponent data
    std::uniform_int_distribution<int> state_dist(0, 4);
    opponent_ai_state_ = state_dist(rng_);

    // Adjust skill level based on seed
    std::uniform_real_distribution<float> skill_dist(0.3f, 0.8f);
    opponent_skill_level_ = skill_dist(rng_);

    // Update opponent rotation to face agent
    std::vector<float> to_agent = {agent_pos_[0] - opponent_pos_[0], agent_pos_[1] - opponent_pos_[1], 0.0f};
    float dist_to_agent = std::sqrt(to_agent[0] * to_agent[0] + to_agent[1] * to_agent[1]);
    if (dist_to_agent > 1e-6f) {
        opponent_rot_[0] = std::atan2(to_agent[1], to_agent[0]); // Yaw
        opponent_rot_[1] = 0.0f; // Pitch
        opponent_rot_[2] = 0.0f; // Roll
    }

    // Simulate opponent actions
    std::uniform_int_distribution<int> action_dist(0, 9);
    opponent_last_action_ = action_dist(rng_);

    // Apply opponent action effects
    switch (opponent_last_action_) {
        case 6: // Attack
            opponent_stamina_ -= 10.0f;
            break;
        case 7: // Block
            opponent_stamina_ -= 6.0f;
            break;
    }
    clamp_values(opponent_stamina_, 0.0f, 100.0f);

    // Opponent health regen
    if (opponent_health_ < 100.0f) {
        opponent_health_ += 0.5f;
        opponent_health_regen_++;
    }
}

bool OvergrowthEnv::check_termination() const {
    auto now = std::chrono::steady_clock::now();
    auto episode_duration = std::chrono::duration_cast<std::chrono::seconds>(
        now - episode_start_time_).count();

    bool win_condition = (opponent_health_ <= 0.0f && health_ > 0.0f);
    bool loss_condition = (health_ <= 0.0f && opponent_health_ > 0.0f);
    bool timeout_condition = (episode_duration >= 20);

    return win_condition || loss_condition || timeout_condition;
}

bool OvergrowthEnv::check_truncation() const {
    if (step_count_ >= 300) {
        timeouts_++;
        return true;
    }
    return false;
}

void OvergrowthEnv::reset_episode(std::optional<uint64_t> seed) {
    if (seed.has_value()) {
        seed_used_ = seed.value();
    } else {
        std::random_device rd;
        seed_used_ = rd();
    }

    rng_.seed(seed_used_);
    step_count_ = 0;
    health_ = 100.0f;
    stamina_ = 100.0f;
    episode_start_time_ = std::chrono::steady_clock::now();
    last_step_time_ = episode_start_time_;

    // Reset agent state
    agent_pos_ = {0.0f, 0.0f, 5.0f};
    agent_rot_ = {0.0f, 0.0f, 0.0f};
    agent_vel_ = {0.0f, 0.0f, 0.0f};

    // Reset opponent state
    std::uniform_real_distribution<float> pos_dist(-arena_radius_ * 0.8f, arena_radius_ * 0.8f);
    opponent_pos_ = {pos_dist(rng_), pos_dist(rng_), 5.0f};
    opponent_rot_ = {0.0f, 0.0f, 0.0f};
    opponent_vel_ = {0.0f, 0.0f, 0.0f};
    opponent_health_ = 100.0f;
    opponent_stamina_ = 100.0f;

    // Reset action system
    action_cooldown_timer_ = 0.0f;
    punch_active_ = false;
    kick_active_ = false;
    last_action_id_ = -1;

    // Reset enhanced action states
    parry_timer_ = 0.0f;
    parry_active_ = false;
    riposte_window_ = false;
    dodge_cooldown_ = 0.0f;
    current_weapon_ = 0;
    current_stance_ = 0;
    stamina_regen_rate_ = 1.0f;

    // Reset ally and environmental states
    ally_status_ = 1.0f;
    ally_shared_health_ = 0.0f;
    ally_cooldown_ = 0;
    climbing_active_ = false;
    swinging_active_ = false;
    rope_tension_ = 0.0f;
    object_held_ = false;
    throwable_trajectory_ = 0.0f;
    grapple_active_ = false;
    grapple_momentum_ = 0.0f;
    wall_running_ = false;
    wall_velocity_taper_ = 0.0f;
    ledge_grabbed_ = false;
    ledge_recovery_anim_ = 0.0f;
    locomotion_mode_ = 0;
    fatigue_accumulation_ = 0.0f;
    in_water_ = false;
    buoyancy_force_ = 0.0f;
    water_resistance_ = 1.0f;

    // Reset sensory expansions
    audio_fft_spectrum_ = std::vector<float>(8, 0.0f); // 8 frequency bands
    vibration_feedback_ = 0.0f;
    gas_dispersion_ = std::vector<float>(4, 0.0f); // 4 gas types
    depth_point_cloud_ = std::vector<float>(16, 0.0f); // 16 depth points

    // Reset navigation
    a_star_path_.clear();
    crowd_sourced_waypoints_ = std::vector<float>(6, 0.0f); // 6 waypoints
    path_progress_ = 0.0f;

    // Reset IK solvers
    bone_ik_solvers_.clear();

    // Reset opponent AI
    std::uniform_real_distribution<float> skill_dist(0.2f, 0.9f);
    opponent_skill_level_ = skill_dist(rng_);
    opponent_ai_state_ = 0;
    opponent_last_action_ = -1;

    // Reset reward system
    cum_reward_ = 0.0f;
    prev_agent_health_ = health_;
    prev_opponent_health_ = opponent_health_;
    prev_dist_to_opponent_ = calculate_distance(agent_pos_, opponent_pos_);
    unique_actions_taken_.clear();
    collision_count_ = 0;
    idle_penalty_timer_ = 0.0f;

    // Reset metrics
    episode_steps_ = 0;
    physics_ticks_ = 0;
    timeouts_ = 0;
    opponent_health_regen_ = 0;

    initialized_ = true;
}

void OvergrowthEnv::validate_action(int action_id) const {
    if (action_id < 0 || action_id >= 35) {
        throw std::invalid_argument("Action ID must be between 0 and 34");
    }
}

void OvergrowthEnv::inject_action(int action_id) {
    // Enhanced action mapping with enriched mechanics
    last_action_id_ = action_id;

    // Update action cooldown for attacks
    if (action_id == 6 || action_id == 8 || action_id == 13 || action_id == 14) { // Punch, Kick, Blade, Gun attacks
        action_cooldown_timer_ = attack_cooldown_duration_;
    }

    // Apply fatigue accumulation based on locomotion mode
    float fatigue_cost = 1.0f;
    if (locomotion_mode_ == 1) { // Sprinting
        fatigue_cost = 1.5f;
    } else if (locomotion_mode_ == 2) { // Stealth
        fatigue_cost = 0.8f;
    }

    switch (action_id) {
        // Original movement actions (0-9)
        case 0: // MoveForward
            stamina_ -= 5.0f * fatigue_cost;
            agent_vel_[2] += max_velocity_ * 0.5f;
            fatigue_accumulation_ += 0.1f;
            break;
        case 1: // MoveBackward
            stamina_ -= 3.0f * fatigue_cost;
            agent_vel_[2] -= max_velocity_ * 0.3f;
            break;
        case 2: // StrafeLeft
            stamina_ -= 4.0f * fatigue_cost;
            agent_vel_[0] -= max_velocity_ * 0.4f;
            break;
        case 3: // StrafeRight
            stamina_ -= 4.0f * fatigue_cost;
            agent_vel_[0] += max_velocity_ * 0.4f;
            break;
        case 4: // Jump
            stamina_ -= 8.0f * fatigue_cost;
            agent_vel_[1] += max_velocity_ * 0.6f;
            break;
        case 5: // Crouch
            stamina_ -= 2.0f;
            locomotion_mode_ = 2; // Enter stealth mode
            break;
        case 6: // Punch
            stamina_ -= 10.0f * (current_weapon_ == 0 ? 1.0f : 1.2f);
            punch_active_ = true;
            opponent_health_ -= 8.0f * (current_stance_ == 0 ? 1.2f : 0.8f); // Offensive stance bonus
            vibration_feedback_ = 0.8f; // Impact feedback
            break;
        case 7: // Block
            stamina_ -= 6.0f;
            health_ -= 2.0f; // Reduced damage taken
            current_stance_ = 1; // Defensive stance
            break;
        case 8: // Kick
            stamina_ -= 7.0f * (current_weapon_ == 0 ? 1.0f : 1.2f);
            kick_active_ = true;
            opponent_health_ -= 6.0f;
            vibration_feedback_ = 0.6f;
            break;
        case 9: // Idle
            stamina_ += 2.0f * stamina_regen_rate_;
            fatigue_accumulation_ = std::max(0.0f, fatigue_accumulation_ - 0.05f);
            locomotion_mode_ = 0; // Normal locomotion
            break;

        // New enriched actions (10-34)
        case 10: // Parry
            if (parry_timer_ <= 0.0f) {
                stamina_ -= 8.0f;
                parry_active_ = true;
                parry_timer_ = 0.5f; // 0.5s parry window
                riposte_window_ = false;
            }
            break;
        case 11: // Riposte
            if (riposte_window_) {
                stamina_ -= 12.0f;
                opponent_health_ -= 15.0f; // High damage riposte
                riposte_window_ = false;
                vibration_feedback_ = 1.0f;
            }
            break;
        case 12: // Evade Dodge
            if (dodge_cooldown_ <= 0.0f) {
                stamina_ -= 15.0f;
                dodge_cooldown_ = 1.0f;
                // Quick directional dodge
                agent_vel_[0] += max_velocity_ * 0.8f * (rng_() % 2 == 0 ? 1.0f : -1.0f);
                agent_vel_[2] += max_velocity_ * 0.6f;
            }
            break;
        case 13: // Blade Combo Attack
            if (current_weapon_ == 0) {
                stamina_ -= 12.0f;
                opponent_health_ -= 10.0f;
                punch_active_ = true; // Visual effect
                vibration_feedback_ = 0.7f;
            }
            break;
        case 14: // Gun Shot
            if (current_weapon_ == 1) {
                stamina_ -= 5.0f;
                opponent_health_ -= 12.0f;
                kick_active_ = true; // Recoil effect
                vibration_feedback_ = 1.2f;
            }
            break;
        case 15: // Switch to Offensive Stance
            current_stance_ = 0;
            stamina_regen_rate_ = 0.8f;
            break;
        case 16: // Switch to Defensive Stance
            current_stance_ = 1;
            stamina_regen_rate_ = 1.2f;
            break;
        case 17: // Switch to Balanced Stance
            current_stance_ = 2;
            stamina_regen_rate_ = 1.0f;
            break;
        case 18: // Sprint Mode
            locomotion_mode_ = 1;
            stamina_ -= 2.0f; // Continuous cost
            break;
        case 19: // Stealth Mode
            locomotion_mode_ = 2;
            stamina_regen_rate_ = 0.6f;
            break;
        case 20: // Weapon Switch (Melee<->Gun)
            stamina_ -= 3.0f;
            current_weapon_ = 1 - current_weapon_;
            break;
        case 21: // Ally Interact
            if (ally_cooldown_ <= 0) {
                stamina_ -= 5.0f;
                ally_cooldown_ = 5; // 5 step cooldown
                ally_shared_health_ = std::min(1.0f, ally_shared_health_ + 0.2f); // Healing share
            }
            break;
        case 22: // Climb
            if (!climbing_active_) {
                stamina_ -= 10.0f;
                climbing_active_ = true;
                // Initialize IK solvers for climbing
                bone_ik_solvers_.push_back({{agent_pos_[0], agent_pos_[1] + 2.0f, agent_pos_[2]}, {}, 1.0f});
            }
            break;
        case 23: // Swing on Rope
            if (!swinging_active_) {
                swinging_active_ = true;
                rope_tension_ = 0.8f;
            }
            break;
        case 24: // Dive/Swim
            if (!in_water_) {
                in_water_ = true;
                buoyancy_force_ = 0.5f;
                water_resistance_ = 1.5f;
            }
            break;
        case 25: // Grapple Hook
            if (!grapple_active_) {
                stamina_ -= 8.0f;
                grapple_active_ = true;
                grapple_momentum_ = max_velocity_ * 1.2f;
            }
            break;
        case 26: // Environmental Interact (Climb/Ledge)
            if (ledge_grabbed_) {
                ledge_recovery_anim_ = 1.0f;
                ledge_grabbed_ = false;
            }
            break;
        case 27: // Object Pickup/Manipulate
            if (!object_held_) {
                stamina_ -= 4.0f;
                object_held_ = true;
                throwable_trajectory_ = 0.0f;
            }
            break;
        case 28: // Throw Object
            if (object_held_) {
                stamina_ -= 6.0f;
                object_held_ = false;
                throwable_trajectory_ = 1.0f; // Predictive trajectory overlay active
                opponent_health_ -= 5.0f; // Throw damage
            }
            break;
        case 29: // Wall Run
            if (!wall_running_) {
                wall_running_ = true;
                wall_velocity_taper_ = 0.9f;
            }
            break;
        case 30: // Ledge Grab
            if (!ledge_grabbed_) {
                ledge_grabbed_ = true;
                ledge_recovery_anim_ = 0.0f;
                stamina_ -= 3.0f;
            }
            break;
        case 31: // A* Path Navigate
            // Simulate pathfinding computation
            a_star_path_ = {{{0,0,10}, {5,0,10}, {5,0,5}}}; // Simple path
            path_progress_ = 0.0f;
            break;
        case 32: // Update Crowd Waypoints
            // Simulate crowdsourced waypoint update
            crowd_sourced_waypoints_[0] += 0.1f;
            break;
        case 33: // Audio FFT Scan
            // Simulate FFT computation
            for (auto& band : audio_fft_spectrum_) {
                band = (rng_() % 100) / 100.0f;
            }
            break;
        case 34: // Depth Sensing Scan
            // Simulate point cloud generation
            for (auto& point : depth_point_cloud_) {
                point = (rng_() % 200 - 100) / 100.0f; // -1 to 1 range
            }
            break;
    }

    clamp_values(stamina_, 0.0f, 100.0f);
    clamp_values(health_, 0.0f, 100.0f);
    clamp_values(opponent_health_, 0.0f, 100.0f);

    // Update vibration decay
    vibration_feedback_ = std::max(0.0f, vibration_feedback_ - 0.1f);

    // Update gas dispersion
    for (auto& gas : gas_dispersion_) {
        gas = std::max(0.0f, gas - 0.05f);
    }
}

py::dict OvergrowthEnv::create_info_dict() const {
    py::dict info;
    info["seed_used"] = seed_used_;
    info["map_loaded"] = std::string("rl_arena.map");
    info["reset_time_ms"] = 0.0; // Mock value
    info["initial_health"] = 100.0f;
    info["agent_positions"] = py::make_tuple(agent_pos_[0], agent_pos_[1], agent_pos_[2]);
    info["rng_state"] = py::int_(rng_()); // Proper RNG state
    info["episode_start_time"] = py::cast(std::chrono::duration_cast<std::chrono::milliseconds>(
        episode_start_time_.time_since_epoch()).count());
    info["step_time_ms"] = 10.0; // Mock 10ms step time
    info["framerate_est"] = 100.0f; // Mock 100 FPS
    info["health_delta"] = health_ - prev_agent_health_;
    info["stamina_delta"] = stamina_ - 100.0f; // Assume start at 100
    info["opponent_ai_state"] = opponent_ai_state_;
    info["action_cooldown"] = static_cast<int>(action_cooldown_timer_ * 1000.0f); // ms
    info["physics_substeps"] = 1; // Mock
    info["deterministic_mode"] = deterministic_;

    // New info fields
    info["cum_reward"] = cum_reward_;
    info["action_id_last"] = last_action_id_;
    info["opponent_action_taken"] = opponent_last_action_;

    // Memory usage (via getrusage if available)
#ifdef __linux__
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    info["memory_peak_kb"] = static_cast<long>(usage.ru_maxrss);
#else
    info["memory_peak_kb"] = 0L; // Not available on Windows
#endif

    info["episode_steps"] = static_cast<int>(episode_steps_);
    info["collision_count"] = collision_count_;
    info["physics_ticks"] = static_cast<int>(physics_ticks_);
    info["timeouts"] = static_cast<int>(timeouts_);
    info["opponent_health_regen"] = static_cast<int>(opponent_health_regen_);
    info["rng_checksum"] = compute_crc32({health_, stamina_, opponent_health_, static_cast<float>(step_count_)});
    info["episode_id"] = episode_id_;

    // Include reward breakdown if available
    if (!reward_breakdown_.empty()) {
        info["reward_breakdown"] = reward_breakdown_;
    }

    return info;
}

void OvergrowthEnv::clamp_values(float& value, float min_val, float max_val) const {
    if (value < min_val) value = min_val;
    if (value > max_val) value = max_val;
}

bool OvergrowthEnv::is_finite(float value) const {
    return std::isfinite(value);
}

void OvergrowthEnv::check_array_for_nans(const py::array_t<float>& arr) const {
    auto buf = arr.unchecked<1>();
    for (size_t i = 0; i < arr.size(); ++i) {
        if (!is_finite(buf(i))) {
            throw std::runtime_error("NaN or Inf detected in observation array");
        }
    }
}

std::vector<float> OvergrowthEnv::perform_lidar_scan() const {
    std::vector<float> distances(lidar_rays_, arena_radius_);

    // Mock LIDAR implementation - simulate raycasts in a circle around agent
    for (int i = 0; i < lidar_rays_; ++i) {
        float angle = 2.0f * M_PI * i / lidar_rays_;
        float ray_dir_x = std::cos(angle);
        float ray_dir_z = std::sin(angle);

        // Check distance to opponent (simplified 2D collision)
        float opp_dx = opponent_pos_[0] - agent_pos_[0];
        float opp_dz = opponent_pos_[2] - agent_pos_[2];
        float opp_dist = std::sqrt(opp_dx * opp_dx + opp_dz * opp_dz);

        // If opponent is in ray direction, return distance
        float dot_product = ray_dir_x * (opp_dx / opp_dist) + ray_dir_z * (opp_dz / opp_dist);
        if (dot_product > 0.9f) { // Within ~25 degrees
            distances[i] = std::min(opp_dist, distances[i]);
        }

        // Add some random obstacles
        std::uniform_real_distribution<float> obstacle_dist(5.0f, arena_radius_ * 0.8f);
        distances[i] = std::min(distances[i], obstacle_dist(rng_));
    }

    return distances;
}

float OvergrowthEnv::calculate_distance(const std::vector<float>& pos1, const std::vector<float>& pos2) const {
    float dx = pos1[0] - pos2[0];
    float dy = pos1[1] - pos2[1];
    float dz = pos1[2] - pos2[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

uint32_t OvergrowthEnv::compute_crc32(const std::vector<float>& data) const {
    uint32_t crc = 0xFFFFFFFF;
    for (float val : data) {
        uint32_t int_val = *reinterpret_cast<uint32_t*>(&val);
        for (int i = 0; i < 32; ++i) {
            if ((crc ^ int_val) & 1) {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
            int_val >>= 1;
        }
    }
    return ~crc;
}

#endif // OG_RL_BUILD