#ifdef OG_RL_BUILD
#ifndef OVERGROWTH_ENV_HPP
#define OVERGROWTH_ENV_HPP

#define PYBIND11_KEYWORD_ARGS
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <memory>
#include <optional>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <string>
#include <random>
#include <chrono>

namespace py = pybind11;

// Forward declaration
class MovementObject;

/**
 * @brief Custom exception class for RL-related errors.
 */
class RLException : public std::runtime_error {
public:
    explicit RLException(const std::string& message) : std::runtime_error(message) {}
};

/**
 * @brief Type alias for action representation.
 */
using action_t = std::vector<float>;

/**
 * @brief Type alias for observation representation.
 */
using obs_t = std::vector<float>;

/**
 * @brief Struct to mimic Python Gymnasium tuple for step() return.
 */
struct tuple_t {
    obs_t observation;
    float reward;
    bool terminated;
    bool truncated;
    std::unordered_map<std::string, float> info;

    tuple_t() : reward(0.0f), terminated(false), truncated(false) {}
};

/**
 * @brief Struct describing the action space.
 */
struct ActionSpace {
    std::vector<size_t> shape;
    std::string dtype;

    ActionSpace() : shape({12}), dtype("int32") {}
};

/**
 * @brief Struct describing the observation space.
 */
struct ObservationSpace {
    std::vector<size_t> shape;
    std::string dtype;

    ObservationSpace() : shape({10}), dtype("float32") {}
};

/**
 * @brief Thread-safe singleton class for Overgrowth RL Environment.
 */
class OvergrowthEnv {
public:
    /**
     * @brief Gets the singleton instance of OvergrowthEnv.
     */
    static std::shared_ptr<OvergrowthEnv> getInstance(py::kwargs kwargs = py::kwargs());

    // Gymnasium-compatible methods
    std::tuple<py::array_t<float>, py::dict> reset(std::optional<uint64_t> seed = std::nullopt);
    std::tuple<py::array_t<float>, float, bool, bool, py::dict> step(int action_id);

    static ActionSpace getActionSpace();
    static ObservationSpace getObservationSpace();
    py::array_t<bool> get_action_mask() const;

    // Additional methods
    py::array_t<uint8_t> render(const std::string& mode = "rgb_array");
    void close();
    py::dict get_profiling_data() const;
    void set_deterministic(bool deterministic);
    void set_log_level(const std::string& level);
    void set_npc(MovementObject* npc);
    void set_enemy(MovementObject* enemy);
    
    // Expert methods
    void set_game_speed(float speed);
    void set_enemy_script_float(const std::string& var_name, float value);

    // Grandmaster methods (Self-Play)
    void set_enemy_action(int action_id);
    py::array_t<float> get_enemy_observation() const;

    // Core attributes
    py::object action_space;
    py::object observation_space;
    py::object spec;
    std::shared_ptr<OvergrowthEnv> unwrapped;
    py::dict metadata;

private:
    OvergrowthEnv();

public:
    ~OvergrowthEnv();

private:
    OvergrowthEnv(const OvergrowthEnv&) = delete;
    OvergrowthEnv& operator=(const OvergrowthEnv&) = delete;

    static std::shared_ptr<OvergrowthEnv> instance_;
    static std::once_flag once_flag_;
    static const ActionSpace action_space_;
    static const ObservationSpace observation_space_;

    // Implementation members
    std::vector<float> current_obs_;
    float cum_reward_;
    bool terminated_;
    bool truncated_;
    int step_count_;
    std::mt19937 rng_;
    uint64_t seed_used_;
    bool deterministic_;
    std::vector<std::pair<std::string, double>> profiling_data_;
    MovementObject* npc_ = nullptr;
    MovementObject* enemy_ = nullptr;

    // Helper methods
    py::array_t<float> get_observation() const;
    float compute_reward(int action_id);
    void update_state(int action_id);
    bool check_termination() const;
    bool check_truncation() const;
};

#endif // OVERGROWTH_ENV_HPP
#endif // OG_RL_BUILD