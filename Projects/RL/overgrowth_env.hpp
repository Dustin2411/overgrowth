#ifdef OG_RL_BUILD
#ifndef OVERGROWTH_ENV_HPP
#define OVERGROWTH_ENV_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <memory>
#include <shared_mutex>
#include <mutex>
#include <optional>
#include <random>
#include <vector>
#include <tuple>
#include <chrono>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <atomic>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>

namespace py = pybind11;

class OvergrowthEnv {
public:
    // Thread-safe singleton access
    static std::shared_ptr<OvergrowthEnv> getInstance();

    // Gymnasium-compatible methods
    std::tuple<py::array_t<float>, py::dict> reset(std::optional<uint64_t> seed = std::nullopt);
    std::tuple<py::array_t<float>, float, bool, bool, py::dict> step(int action_id);
    py::array_t<bool> get_action_mask() const;

    // Core attributes
    py::object action_space;
    py::object observation_space;
    py::object spec;
    std::shared_ptr<OvergrowthEnv> unwrapped;
    py::dict metadata;

    // Additional methods
    py::array_t<uint8_t> render(const std::string& mode = "rgb_array");
    void close();
    py::dict get_profiling_data() const;
    void set_deterministic(bool deterministic);
    void set_log_level(const std::string& level);

private:
    // Singleton pattern
    static std::shared_ptr<OvergrowthEnv> instance_;
    static std::once_flag once_flag_;

    // Thread safety
    mutable std::shared_mutex mutex_;

    // RNG and state
    std::mt19937_64 rng_;
    uint64_t seed_used_ = 0;
    bool initialized_ = false;
    bool deterministic_ = false;

    // Logging
    std::shared_ptr<spdlog::logger> logger_;

    // Profiling
    std::vector<std::pair<std::string, double>> profiling_data_;
    std::chrono::high_resolution_clock::time_point last_frame_time_;

    // Headless timing
    std::chrono::high_resolution_clock::time_point simulation_start_time_;
    double accumulated_time_ = 0.0;
    const double target_dt_ = 0.01; // 10ms per step

    // Episode ID for logging
    int episode_id_ = 0;

    // Episode state
    int step_count_ = 0;
    float health_ = 100.0f;
    float stamina_ = 100.0f;
    std::chrono::steady_clock::time_point episode_start_time_;
    std::chrono::steady_clock::time_point last_step_time_;

    // Agent state (position, rotation, velocity)
    std::vector<float> agent_pos_ = {0.0f, 0.0f, 5.0f};
    std::vector<float> agent_rot_ = {0.0f, 0.0f, 0.0f};
    std::vector<float> agent_vel_ = {0.0f, 0.0f, 0.0f};

    // Opponent state
    std::vector<float> opponent_pos_ = {10.0f, 0.0f, 5.0f};
    std::vector<float> opponent_rot_ = {0.0f, 0.0f, 0.0f};
    std::vector<float> opponent_vel_ = {0.0f, 0.0f, 0.0f};
    float opponent_health_ = 100.0f;
    float opponent_stamina_ = 100.0f;

    // Environment constants
    const float arena_radius_ = 50.0f;
    const float max_velocity_ = 20.0f;
    const float max_dist_ = arena_radius_ * 2.0f;

    // Action system
    float action_cooldown_timer_ = 0.0f;
    const float attack_cooldown_duration_ = 0.1f;
    int last_action_id_ = -1;

    // Attack flags
    bool punch_active_ = false;
    bool kick_active_ = false;

    // Opponent AI state
    float opponent_skill_level_ = 0.5f;
    int opponent_ai_state_ = 0;
    int opponent_last_action_ = -1;

    // Reward system
    py::dict reward_weights_;
    float prev_agent_health_ = 100.0f;
    float prev_opponent_health_ = 100.0f;
    float prev_dist_to_opponent_ = 0.0f;
    float cum_reward_ = 0.0f;
    std::unordered_set<int> unique_actions_taken_;
    int collision_count_ = 0;
    float idle_penalty_timer_ = 0.0f;

    // LIDAR system
    int lidar_rays_ = 16;
    std::vector<float> lidar_distances_;

    // Enhanced action and state variables
    float parry_timer_ = 0.0f; // Parry/riposte timing window
    bool parry_active_ = false;
    bool riposte_window_ = false;
    float dodge_cooldown_ = 0.0f; // Evasive dodge cooldown
    int current_weapon_ = 0; // 0: melee, 1: gun
    int current_stance_ = 0; // 0: offense, 1: defense, 2: balanced
    float stamina_regen_rate_ = 1.0f; // Stamina regeneration rate

    // Ally interaction system
    float ally_status_ = 1.0f; // Ally health/status (0-1 normalized)
    float ally_shared_health_ = 0.0f; // Status sharing mechanism
    int ally_cooldown_ = 0; // Ally interaction cooldown

    // Environmental interactions
    bool climbing_active_ = false;
    bool swinging_active_ = false;
    float rope_tension_ = 0.0f;
    bool object_held_ = false;
    float throwable_trajectory_ = 0.0f;
    bool grapple_active_ = false;
    float grapple_momentum_ = 0.0f;
    bool wall_running_ = false;
    float wall_velocity_taper_ = 0.0f;
    bool ledge_grabbed_ = false;
    float ledge_recovery_anim_ = 0.0f;
    int locomotion_mode_ = 0; // 0: walk, 1: sprint, 2: stealth
    float fatigue_accumulation_ = 0.0f;
    bool in_water_ = false;
    float buoyancy_force_ = 0.0f;
    float water_resistance_ = 1.0f;

    // Sensory expansions
    std::vector<float> audio_fft_spectrum_; // FFT-based audio spectral cues
    float vibration_feedback_ = 0.0f; // Impact vibration feedback
    std::vector<float> gas_dispersion_; // Olfactory gas dispersion model
    std::vector<float> depth_point_cloud_; // Visual depth sensing point cloud

    // Pathfinding and navigation
    std::vector<std::vector<float>> a_star_path_; // A* pathfinding waypoints
    std::vector<float> crowd_sourced_waypoints_; // Crowdsourced navigation data
    float path_progress_ = 0.0f;

    // Inverse kinematics for climbing/environmental interactions
    struct BoneIK {
        std::vector<float> target_pos;
        std::vector<float> joint_positions;
        float ik_weight = 1.0f;
    };
    std::vector<BoneIK> bone_ik_solvers_;

    // Metrics accumulators (thread-safe)
    std::atomic<int> episode_steps_{0};
    std::atomic<int> physics_ticks_{0};
    std::atomic<int> timeouts_{0};
    std::atomic<int> opponent_health_regen_{0};

    // Private constructor for singleton
    OvergrowthEnv(py::kwargs kwargs = py::kwargs());

    // Helper methods
    void initialize_spaces();
    void initialize_metadata();
    py::array_t<float> get_observation() const;
    float compute_reward();
    void update_physics(float dt);
    void update_ai();
    std::vector<float> perform_lidar_scan() const;
    bool check_termination() const;
    bool check_truncation() const;
    void reset_episode(std::optional<uint64_t> seed);
    void validate_action(int action_id) const;
    void inject_action(int action_id);
    py::dict create_info_dict() const;
    float calculate_distance(const std::vector<float>& pos1, const std::vector<float>& pos2) const;
    uint32_t compute_crc32(const std::vector<float>& data) const;

    // Safety checks
    void clamp_values(float& value, float min_val, float max_val) const;
    bool is_finite(float value) const;
    void check_array_for_nans(const py::array_t<float>& arr) const;
};

#endif // OVERGROWTH_ENV_HPP
#endif // OG_RL_BUILD