#ifdef OG_RL_BUILD
#ifndef OG_RL_INTERFACE_HPP
#define OG_RL_INTERFACE_HPP

#include <memory>
#include <functional>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <any>
#include <filesystem>
#include <Eigen/Dense>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <cmath>

// Conditional compilation for neural network backends
// Define OG_USE_LIBTORCH, OG_USE_TENSORRT, OG_USE_ONNX_RUNTIME as needed in build system
#if defined(OG_USE_LIBTORCH)
#include <torch/torch.h>
#endif
#if defined(OG_USE_TENSORRT)
#include <NvInfer.h>
#endif
#if defined(OG_USE_ONNX_RUNTIME)
#include <onnxruntime_cxx_api.h>
#endif

namespace og_rl {

// Typedefs for RL components
using State = Eigen::VectorXd;
using Action = Eigen::VectorXd;
using Reward = double;
using Done = bool;

// Struct for additional information in transitions
struct Info {
    std::unordered_map<std::string, std::any> data;
    Info() = default;
    Info(const std::unordered_map<std::string, std::any>& d) : data(d) {}
};

// Extended Transition tuple: (state, action, reward, next_state, done, info)
using Transition = std::tuple<State, Action, Reward, State, Done, Info>;

// Network callback for neural network interactions
using NetworkCallback = std::function<std::vector<double>(const std::vector<double>& inputs, const std::unordered_map<std::string, std::any>& kwargs)>;

/**
 * @brief Abstract base class for RL policies.
 * Supports on-policy (PPO) and off-policy (SAC, DQN) algorithms.
 */
class Policy {
public:
    virtual ~Policy() = default;

    /**
     * @brief Evaluate the policy for a given observation.
     * @param obs Current observation
     * @param extra_info Additional information (e.g., log probabilities for PPO)
     * @return Action to take
     */
    virtual Action evaluate(const State& obs, const Info& extra_info = {}) const = 0;

    /**
     * @brief Train the policy using transitions.
     * @param transitions Batch of transitions
     * @param learning_rate Learning rate for optimization
     * @param hyperparameters Additional hyperparameters (e.g., clip ratio for PPO)
     */
    virtual void train(const std::vector<Transition>& transitions, double learning_rate, const std::unordered_map<std::string, double>& hyperparameters = {}) = 0;

    /**
     * @brief Save the policy to disk.
     * @param path File path to save to
     */
    virtual void save(const std::filesystem::path& path) const = 0;

    /**
     * @brief Load the policy from disk.
     * @param path File path to load from
     */
    virtual void load(const std::filesystem::path& path) = 0;

    /**
     * @brief Get policy entropy for exploration metrics.
     * @param obs Observation to compute entropy for
     * @return Entropy value
     */
    virtual double get_entropy(const State& obs) const {
        // Default implementation: return 0.0
        std::cerr << "Warning: get_entropy not implemented in derived class\n";
        return 0.0;
    }

    /**
     * @brief Update hyperparameters for sweeps.
     * @param new_hyperparams New hyperparameter values
     */
    virtual void update_hyperparameters(const std::unordered_map<std::string, double>& new_hyperparams) {
        // Default empty implementation
    }

    // Placeholder for neural network integration
    NetworkCallback network_forward;
    NetworkCallback network_backward;
};

/**
 * @brief Abstract base class for value functions.
 * Used in actor-critic methods and Q-learning variants.
 */
class ValueFunction {
public:
    virtual ~ValueFunction() = default;

    /**
     * @brief Predict the value for a given observation.
     * @param obs Current observation
     * @return Predicted value
     */
    virtual double predict_value(const State& obs) const = 0;

    /**
     * @brief Update the value function using transitions.
     * @param transitions Batch of transitions
     * @param learning_rate Learning rate for optimization
     */
    virtual void update_value(const std::vector<Transition>& transitions, double learning_rate) = 0;

    /**
     * @brief Save the value function to disk.
     * @param path File path to save to
     */
    virtual void save_value(const std::filesystem::path& path) const = 0;

    /**
     * @brief Load the value function from disk.
     * @param path File path to load from
     */
    virtual void load_value(const std::filesystem::path& path) = 0;

    // Placeholder for neural network integration
    NetworkCallback value_network_forward;
    NetworkCallback value_network_backward;
};

// Utility functions

/**
 * @brief Compute discounted returns for a trajectory.
 * @param rewards Vector of rewards
 * @param done Terminal flag
 * @param gamma Discount factor
 * @param last_value Bootstrap value (0.0 for terminal)
 * @return Vector of discounted returns
 */
inline std::vector<double> compute_discounted_returns(const std::vector<Reward>& rewards, Done done, double gamma, double last_value = 0.0) {
    std::vector<double> returns(rewards.size());
    double discounted_sum = done ? 0.0 : last_value;
    for (int i = static_cast<int>(rewards.size()) - 1; i >= 0; --i) {
        discounted_sum = rewards[i] + gamma * discounted_sum;
        returns[i] = discounted_sum;
    }
    return returns;
}

/**
 * @brief Compute generalized advantage estimation (GAE).
 * @param values Vector of value predictions
 * @param rewards Vector of rewards
 * @param done Terminal flag
 * @param gamma Discount factor
 * @param lambda GAE lambda parameter
 * @return Vector of advantages
 */
inline std::vector<double> compute_advantage(const std::vector<double>& values, const std::vector<Reward>& rewards, Done done, double gamma, double lambda) {
    size_t T = rewards.size();
    std::vector<double> advantages(T);
    std::vector<double> deltas(T);

    // Compute TD residuals
    for (size_t t = 0; t < T - 1; ++t) {
        deltas[t] = rewards[t] + gamma * values[t + 1] - values[t];
    }
    deltas[T - 1] = rewards[T - 1] + gamma * (done ? 0.0 : values[T]) - values[T - 1];

    // Compute GAE
    double advantage = 0.0;
    for (int t = static_cast<int>(T) - 1; t >= 0; --t) {
        advantage = deltas[t] + gamma * lambda * advantage;
        advantages[t] = advantage;
    }
    return advantages;
}

/**
 * @brief Normalize advantages for stable training.
 * @param advantages Vector of advantages (modified in-place)
 * @return Normalized advantages
 */
inline std::vector<double>& normalize_advantages(std::vector<double>& advantages) {
    if (advantages.empty()) return advantages;

    double mean = std::accumulate(advantages.begin(), advantages.end(), 0.0) / advantages.size();
    double variance = 0.0;
    for (double adv : advantages) {
        variance += (adv - mean) * (adv - mean);
    }
    variance /= advantages.size();
    double std_dev = std::sqrt(variance + 1e-8); // Add epsilon for numerical stability

    for (double& adv : advantages) {
        adv = (adv - mean) / std_dev;
    }
    return advantages;
}

// Logging utilities
inline void log_info(const std::string& message) {
    std::cout << "[OG_RL INFO] " << message << std::endl;
}

inline void log_warning(const std::string& message) {
    std::cerr << "[OG_RL WARNING] " << message << std::endl;
}

inline void log_error(const std::string& message) {
    std::cerr << "[OG_RL ERROR] " << message << std::endl;
}

// Example concrete implementations

/**
 * @brief Random policy for baseline comparisons.
 */
class RandomPolicy : public Policy {
public:
    RandomPolicy(size_t action_dim) : action_dim_(action_dim), rng_(std::random_device{}()) {}

    Action evaluate(const State& obs, const Info& extra_info = {}) const override {
        Action action(action_dim_);
        std::normal_distribution<double> dist(0.0, 1.0);
        for (size_t i = 0; i < action_dim_; ++i) {
            action[i] = dist(rng_);
        }
        return action;
    }

    void train(const std::vector<Transition>& transitions, double learning_rate, const std::unordered_map<std::string, double>& hyperparameters = {}) override {
        // No training for random policy
        log_info("RandomPolicy: Training skipped as expected.");
    }

    void save(const std::filesystem::path& path) const override {
        // Nothing to save
        log_info("RandomPolicy saved (no-op).");
    }

    void load(const std::filesystem::path& path) override {
        // Nothing to load
        log_info("RandomPolicy loaded (no-op).");
    }

private:
    size_t action_dim_;
    mutable std::mt19937 rng_;
};

/**
 * @brief Simple tabular Q-learning policy.
 */
class TabularQPolicy : public Policy {
public:
    TabularQPolicy(size_t state_dim, size_t action_dim, double epsilon = 0.1)
        : state_dim_(state_dim), action_dim_(action_dim), epsilon_(epsilon), rng_(std::random_device{}()) {
        // Initialize Q-table with zeros
        q_table_ = Eigen::MatrixXd::Zero(state_dim, action_dim);
    }

    Action evaluate(const State& obs, const Info& extra_info = {}) const override {
        // Discretize observation (simple binning for tabular method)
        size_t state_idx = discretize_state(obs);

        // Epsilon-greedy action selection
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        if (dist(rng_) < epsilon_) {
            // Random action
            std::uniform_int_distribution<size_t> action_dist(0, action_dim_ - 1);
            size_t action_idx = action_dist(rng_);
            Action action = Action::Zero(action_dim_);
            action[action_idx] = 1.0; // One-hot encoding
            return action;
        } else {
            // Greedy action
            Eigen::Index max_idx;
            q_table_.row(state_idx).maxCoeff(&max_idx);
            Action action = Action::Zero(action_dim_);
            action[max_idx] = 1.0;
            return action;
        }
    }

    void train(const std::vector<Transition>& transitions, double learning_rate, const std::unordered_map<std::string, double>& hyperparameters = {}) override {
        double gamma = hyperparameters.count("gamma") ? hyperparameters.at("gamma") : 0.99;

        for (const auto& transition : transitions) {
            const auto& [state, action, reward, next_state, done, info] = transition;
            size_t state_idx = discretize_state(state);
            size_t next_state_idx = discretize_state(next_state);

            // Find action index (assuming one-hot)
            Eigen::Index action_idx;
            action.maxCoeff(&action_idx);

            double current_q = q_table_(state_idx, action_idx);
            double max_next_q = done ? 0.0 : q_table_.row(next_state_idx).maxCoeff();
            double target = reward + gamma * max_next_q;
            q_table_(state_idx, action_idx) += learning_rate * (target - current_q);
        }
    }

    void save(const std::filesystem::path& path) const override {
        // Simple text save (extend for binary serialization if needed)
        std::ofstream file(path);
        if (file.is_open()) {
            file << q_table_;
            log_info("TabularQPolicy saved to " + path.string());
        } else {
            log_error("Failed to save TabularQPolicy to " + path.string());
        }
    }

    void load(const std::filesystem::path& path) override {
        std::ifstream file(path);
        if (file.is_open()) {
            file >> q_table_;
            log_info("TabularQPolicy loaded from " + path.string());
        } else {
            log_error("Failed to load TabularQPolicy from " + path.string());
        }
    }

    void update_hyperparameters(const std::unordered_map<std::string, double>& new_hyperparams) override {
        if (new_hyperparams.count("epsilon")) {
            epsilon_ = new_hyperparams.at("epsilon");
        }
    }

private:
    size_t discretize_state(const State& state) const {
        // Simple discretization by flooring (extend for more sophisticated binning)
        size_t idx = 0;
        for (int i = 0; i < state.size(); ++i) {
            idx = idx * 10 + static_cast<size_t>(std::floor(state[i] * 10)); // 10 bins per dimension
        }
        return idx % state_dim_; // Ensure within bounds
    }

    size_t state_dim_;
    size_t action_dim_;
    double epsilon_;
    Eigen::MatrixXd q_table_;
    mutable std::mt19937 rng_;
};

// Notes for extensibility:
// - For CUDA support, use thrust::device_vector and cuda kernels in derived classes.
// - Bindings: Extend with pybind11 for Python integration (e.g., for stable-baselines3).
// - Multi-agent: Add agent_id to State/Action/Info structs and modify evaluate/train signatures.
// - Hierarchical RL: Introduce MetaPolicy class inheriting from Policy with sub-policies.
// - Conditional compilation: Use #ifdef OG_USE_GPU for GPU-specific code paths.
// - Neural networks: Implement concrete classes using LibTorch, TensorRT, or ONNX Runtime.
//   Example: class PPOPolicy : public Policy { torch::nn::Module actor_; /* ... */ };

} // namespace og_rl

#endif // OG_RL_INTERFACE_HPP
#endif // OG_RL_BUILD