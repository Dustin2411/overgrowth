#ifndef OG_RL_BUILD
#define OG_RL_BUILD

#include <mutex>
#include <atomic>
#include <string>
#include <memory>
#include <optional>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <pybind11/pybind11.h>
namespace py = pybind11;

/**
 * @brief Custom exception class for RL-related errors.
 *
 * Derived from std::runtime_error to provide detailed error messages
 * for RL interface operations.
 */
class RLException : public std::runtime_error {
public:
    /**
     * @brief Constructs an RLException with a message.
     *
     * @param message The error message.
     */
    explicit RLException(const std::string& message)
        : std::runtime_error(message) {}
};

/**
 * @brief Type alias for action representation.
 *
 * Represents a vector of floats for continuous action spaces.
 */
using action_t = std::vector<float>;

/**
 * @brief Type alias for observation representation.
 *
 * Can be either a vector of floats or a map of string to float for dict-like observations.
 */
using obs_t = std::variant<std::vector<float>, std::unordered_map<std::string, float>>;

/**
 * @brief Struct to mimic Python Gymnasium tuple for step() return.
 *
 * Contains observation, reward, terminated, truncated, and info.
 */
struct tuple_t {
    obs_t observation;      ///< Next observation (dict or vector).
    float reward;           ///< Reward value.
    bool terminated;        ///< Episode terminated flag.
    bool truncated;         ///< Episode truncated flag.
    std::unordered_map<std::string, float> info;  ///< Additional info map.

    /**
     * @brief Default constructor with default values.
     */
    tuple_t() : reward(0.0f), terminated(false), truncated(false) {}
};

/**
 * @brief Struct describing the action space.
 *
 * Represents a continuous vector space with shape and dtype.
 */
struct ActionSpace {
    std::vector<size_t> shape;  ///< Shape of the action space (e.g., {3} for 3D vector).
    std::string dtype;          ///< Data type (e.g., "float32").

    /**
     * @brief Default constructor.
     */
    ActionSpace() : shape({3}), dtype("float32") {}
};

/**
 * @brief Struct describing the observation space.
 *
 * Represents a dict-like space with keys for various observations.
 */
struct ObservationSpace {
    std::vector<std::string> keys;  ///< Keys for observation dict (e.g., {"player_pos", "health"}).

    /**
     * @brief Default constructor.
     */
    ObservationSpace() : keys({"player_pos", "health", "enemies", "objects"}) {}
};

/**
 * @brief Thread-safe singleton class for Overgrowth RL Environment.
 *
 * Provides Gymnasium-like interface for RL interactions with the Overgrowth engine.
 * Implements lazy initialization with std::atomic and mutex for thread safety.
 */
class OvergrowthEnv {
public:
    /**
     * @brief Gets the singleton instance of OvergrowthEnv.
     *
     * Uses double-checked locking for thread-safe lazy initialization.
     *
     * @param args Keyword arguments for initialization.
     * @return Shared pointer to the singleton instance.
     */
    static std::shared_ptr<OvergrowthEnv> getInstance(pybind11::kwargs args = pybind11::kwargs());

    /**
     * @brief Resets the environment to initial state.
     *
     * @return Initial observation (dict or vector).
     */
    obs_t reset();

    /**
     * @brief Takes a step in the environment.
     *
     * @param action The action to take.
     * @return Tuple containing next observation, reward, terminated, truncated, info.
     */
    tuple_t step(const action_t& action);

    /**
     * @brief Closes the environment and cleans up resources.
     */
    void close();

    /**
     * @brief Renders the environment (placeholder).
     *
     * @param mode Rendering mode (default "human").
     */
    void render(const std::string& mode = "human");

    /**
     * @brief Gets the action space metadata.
     *
     * @return ActionSpace struct.
     */
    static ActionSpace getActionSpace();

    /**
     * @brief Gets the observation space metadata.
     *
     * @return ObservationSpace struct.
     */
    static ObservationSpace getObservationSpace();

private:
    /**
     * @brief Private constructor for singleton.
     */
    OvergrowthEnv();

    /**
     * @brief Private destructor.
     */
    ~OvergrowthEnv();

    /**
     * @brief Deleted copy constructor.
     */
    OvergrowthEnv(const OvergrowthEnv&) = delete;

    /**
     * @brief Deleted assignment operator.
     */
    OvergrowthEnv& operator=(const OvergrowthEnv&) = delete;

    /**
     * @brief Initializes the Overgrowth engine state.
     *
     * Placeholder for actual engine initialization.
     */
    void initializeEngine();

    /**
     * @brief Captures the current scene state.
     *
     * Placeholder for Overgrowth scene capture.
     *
     * @return Observation data.
     */
    obs_t captureScene();

    /**
     * @brief Sets player action in the engine.
     *
     * Placeholder for applying actions to Overgrowth.
     *
     * @param action The action vector.
     */
    void setPlayerAction(const action_t& action);

    /**
     * @brief Computes reward based on current state.
     *
     * Placeholder for reward calculation.
     *
     * @return Reward value.
     */
    float computeReward();

    /**
     * @brief Checks if episode is terminated.
     *
     * Placeholder for termination logic.
     *
     * @return True if terminated.
     */
    bool isTerminated();

    /**
     * @brief Checks if episode is truncated.
     *
     * Placeholder for truncation logic.
     *
     * @return True if truncated.
     */
    bool isTruncated();

    static std::shared_ptr<OvergrowthEnv> instance_;  ///< Shared pointer for singleton instance.
    static std::once_flag once_flag_;                 ///< Flag for call_once.

    bool initialized_;                             ///< Flag for engine initialization.
    obs_t current_obs_;                            ///< Current observation state.
    float current_reward_;                         ///< Current reward.
    bool terminated_;                              ///< Termination flag.
    bool truncated_;                               ///< Truncation flag.
    std::unordered_map<std::string, py::dict> reward_breakdown_;  ///< Reward breakdown dictionary.

    static const ActionSpace action_space_;        ///< Static action space metadata.
    static const ObservationSpace observation_space_;  ///< Static observation space metadata.
};

// Static member definitions
std::shared_ptr<OvergrowthEnv> OvergrowthEnv::instance_;
std::once_flag OvergrowthEnv::once_flag_;
const ActionSpace OvergrowthEnv::action_space_ = ActionSpace();
const ObservationSpace OvergrowthEnv::observation_space_ = ObservationSpace();

#endif // OG_RL_BUILD