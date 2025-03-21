#pragma once

#include "tracking_lib/Misc/logger_setup.h"

#include <type_safe/strong_typedef.hpp>
#include <debug_assert.hpp>

#include <numeric>
#include <eigen3/Eigen/Dense>

#include <chrono>
#include <optional>
#include <map>
#include <boost/functional/hash.hpp>
namespace ttb
{
constexpr auto TTB_EPS = std::numeric_limits<double>::epsilon();
using namespace std::chrono_literals;

namespace ts = type_safe;

template <class SID>
class IDGenerator
{
public:
  [[nodiscard]] SID getID() noexcept
  {
    return SID{ ++_next };
  }

private:
  std::atomic<typename SID::type> _next{ 1 };
};

// #####################################################################################################################
/// Unique Id to distinguish different data sources
struct SourceId : ts::strong_typedef<SourceId, std::string>,
                  ts::strong_typedef_op::output_operator<SourceId>,
                  ts::strong_typedef_op::equality_comparison<SourceId>,   // allow == comparison
                  ts::strong_typedef_op::relational_comparison<SourceId>  // allow <, <=, >, >= -> needed for map
{
  using type = std::string;
  using strong_typedef::strong_typedef;
};

// #####################################################################################################################
/// Unique Id to distinguish State Models
struct StateModelId
  : ts::strong_typedef<StateModelId, std::size_t>,
    ts::strong_typedef_op::output_operator<StateModelId>,
    ts::strong_typedef_op::equality_comparison<StateModelId>,   // allow == comparison
    ts::strong_typedef_op::relational_comparison<StateModelId>  // allow <, <=, >, >= -> needed for map
{
  using type = std::size_t;
  using strong_typedef::strong_typedef;
};

// #####################################################################################################################
/// Unique Id to distinguish Measurement Models
struct MeasModelId : ts::strong_typedef<MeasModelId, std::string>,
                     ts::strong_typedef_op::output_operator<MeasModelId>,
                     ts::strong_typedef_op::equality_comparison<MeasModelId>,   // allow == comparison
                     ts::strong_typedef_op::relational_comparison<MeasModelId>  // allow <, <=, >, >= -> needed for map
{
  using type = std::string;
  using strong_typedef::strong_typedef;
};

// #####################################################################################################################
/// Unique Id to distinguish Distributions
struct DistributionId : ts::strong_typedef<DistributionId, std::size_t>,
                        ts::strong_typedef_op::output_operator<DistributionId>,
                        ts::strong_typedef_op::equality_comparison<DistributionId>,  // allow ==, != comparison
                        ts::strong_typedef_op::relational_comparison<DistributionId>
{
  using type = std::size_t;
  using strong_typedef::strong_typedef;
};
constexpr DistributionId NO_DISTRIBUTION_ID_HISTORY{ 0 };

// #####################################################################################################################
/// Unique Id to distinguish State Distributions
struct StateId : ts::strong_typedef<StateId, std::size_t>,
                 ts::strong_typedef_op::output_operator<StateId>,
                 ts::strong_typedef_op::equality_comparison<StateId>,
                 ts::strong_typedef_op::relational_comparison<StateId>
{
  using type = std::size_t;
  using strong_typedef::strong_typedef;
};

// #####################################################################################################################
/// (Non-) Unique Id to distinguish Multi-Object Densities (unique for every Type)
struct MODistributionId : ts::strong_typedef<MODistributionId, std::size_t>,
                          ts::strong_typedef_op::output_operator<MODistributionId>,
                          ts::strong_typedef_op::equality_comparison<MODistributionId>  // allow ==, != comparison
{
  using type = std::size_t;
  using strong_typedef::strong_typedef;
};
// #####################################################################################################################
/// Unique Id to distinguish Measurements
struct MeasurementId
  : ts::strong_typedef<MeasurementId, std::size_t>,
    ts::strong_typedef_op::output_operator<MeasurementId>,
    ts::strong_typedef_op::equality_comparison<MeasurementId>,   // allow ==, != comparison
    ts::strong_typedef_op::relational_comparison<MeasurementId>  // allow <, <=, >, >= -> needed for map

{
  using type = std::size_t;
  using strong_typedef::strong_typedef;
};
constexpr MeasurementId NOT_DETECTED{ 0 };

// #####################################################################################################################
/// Unique Id used to identify Objects if the Detection provide such an info
struct ObjectId : ts::strong_typedef<ObjectId, std::size_t>,
                  ts::strong_typedef_op::output_operator<ObjectId>,
                  ts::strong_typedef_op::equality_comparison<ObjectId>,   // allow ==, != comparison
                  ts::strong_typedef_op::relational_comparison<ObjectId>  // allow <, <=, >, >=

{
  using type = std::size_t;
  using strong_typedef::strong_typedef;
};

// #####################################################################################################################
/// Unique Id to distinguish Hypotheses
struct HypothesisId : ts::strong_typedef<HypothesisId, std::size_t>,
                      ts::strong_typedef_op::output_operator<HypothesisId>,
                      ts::strong_typedef_op::equality_comparison<HypothesisId>,   // allow ==, != comparison
                      ts::strong_typedef_op::relational_comparison<HypothesisId>  // allow <, <=, >, >=
{
  using type = std::size_t;
  using strong_typedef::strong_typedef;
};
constexpr HypothesisId NO_HYPOTHESIS_ID_HISTORY{ 0 };

// #####################################################################################################################
/// Label of a Track
struct Label : ts::strong_typedef<Label, std::size_t>,
               ts::strong_typedef_op::output_operator<Label>,
               ts::strong_typedef_op::equality_comparison<Label>,   // allow ==, != comparison
               ts::strong_typedef_op::relational_comparison<Label>  // allow <, <=, >, >=

{
  using type = std::size_t;
  using strong_typedef::strong_typedef;
};
}  // namespace ttb

template <>
struct std::hash<std::vector<ttb::StateId>>
{
  std::size_t operator()(std::vector<ttb::StateId> ids) const
  {
    std::size_t seed = 0;
    for (ttb::StateId id : ids)
    {
      boost::hash_combine(seed, id.value_);
    }
    return seed;
  }
};

template <>
struct std::hash<std::vector<ttb::DistributionId>>
{
  std::size_t operator()(std::vector<ttb::DistributionId> ids) const
  {
    std::size_t seed = 0;
    for (ttb::DistributionId id : ids)
    {
      boost::hash_combine(seed, id.value_);
    }
    return seed;
  }
};

template <class T, class V>
struct std::hash<std::pair<T, V>>
{
  std::size_t operator()(std::pair<T, V> ids) const
  {
    std::size_t seed = 0;
    boost::hash_combine(seed, ids.first);
    boost::hash_combine(seed, ids.second);
    return seed;
  }
};

template <>
struct std::hash<ttb::Label> : type_safe::hashable<ttb::Label>
{
};

template <>
struct std::hash<ttb::HypothesisId> : type_safe::hashable<ttb::HypothesisId>
{
};

template <>
struct std::hash<ttb::ObjectId> : type_safe::hashable<ttb::ObjectId>
{
};

template <>
struct std::hash<ttb::MeasurementId> : type_safe::hashable<ttb::MeasurementId>
{
};

template <>
struct std::hash<ttb::StateId> : type_safe::hashable<ttb::StateId>
{
};

template <>
struct std::hash<ttb::DistributionId> : type_safe::hashable<ttb::DistributionId>
{
};

template <>
struct std::hash<ttb::MeasModelId> : type_safe::hashable<ttb::MeasModelId>
{
};

template <>
struct std::hash<ttb::SourceId> : type_safe::hashable<ttb::SourceId>
{
};

namespace ttb
{

using Probability = double;
using Nis = double;

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using RowVector = Eigen::RowVectorXd;
using Matrix33 = Eigen::Matrix3d;
using Vector3 = Eigen::Vector3d;
using Vector2 = Eigen::Vector2d;
using Matrix22 = Eigen::Matrix2d;
using Array = Eigen::ArrayXXd;
using SE3Trafo = Eigen::Transform<double, 3, Eigen::TransformTraits::Isometry>;
using Index = Eigen::Index;
using Indices = Eigen::Array<Index, -1, 1>;

using namespace std::chrono_literals;
using Duration = std::chrono::duration<int64_t, std::nano>;
using Time = std::chrono::time_point<std::chrono::high_resolution_clock, Duration>;

[[nodiscard]] double to_seconds(Duration dur);
[[nodiscard]] double to_milliseconds(Duration dur);
[[nodiscard]] int64_t to_nanoseconds(Duration dur);
[[nodiscard]] std::string to_string(Duration dur);
[[nodiscard]] std::string to_string(Time time);

class Innovation;
class TTBManager;
class State;
class BaseMeasurementModel;
class BaseStateModel;

// #####################################################################################################################
///// PM-Filter truncation strategies in fusion
enum class SELECTION_STRATEGY
{
  ALL_COMBINATIONS,
  K_BEST,
  K_BEST_POISSON,
  GENERALIZED_K_BEST
};
namespace impl
{

const std::map<SELECTION_STRATEGY, std::string> SELECTION_STRATEGY_2_STRING{
  { SELECTION_STRATEGY::ALL_COMBINATIONS, "ALL_COMBINATIONS" },
  { SELECTION_STRATEGY::K_BEST, "K_BEST" },
  { SELECTION_STRATEGY::K_BEST_POISSON, "K_BEST_POISSON" },
  { SELECTION_STRATEGY::GENERALIZED_K_BEST, "GENERALIZED_K_BEST" }
};
}
std::string to_string(SELECTION_STRATEGY type);
std::optional<SELECTION_STRATEGY> to_SELECTION_STRATEGY(std::string type);
// #####################################################################################################################

// #####################################################################################################################
///// GNN-filter track state stages

enum class STAGE
{
  TENTATIVE,
  PRELIMINARY,
  CONFIRMED,
  DEAD
};
namespace impl
{

const std::map<STAGE, std::string> STAGE_2_STRING{ { STAGE::TENTATIVE, "TENTATIVE" },
                                                   { STAGE::PRELIMINARY, "PRELIMINARY" },
                                                   { STAGE::CONFIRMED, "CONFIRMED" },
                                                   { STAGE::DEAD, "DEAD" } };
}
std::string to_string(STAGE type);
std::optional<STAGE> to_STAGE_TYPE(std::string type);

// #####################################################################################################################

enum class BUFFER_MODE
{
  BUFFER,
};

// #####################################################################################################################
enum class CLASSIFICATION_TYPE
{
  SL,
};
namespace impl
{
const std::map<CLASSIFICATION_TYPE, std::string> CLASSIFICATION_TYPE_2_STRING{ { CLASSIFICATION_TYPE::SL, "SL" } };
}
std::string to_string(CLASSIFICATION_TYPE type);
std::optional<CLASSIFICATION_TYPE> to_CLASSIFICATION_TYPE(std::string type);

// #####################################################################################################################
enum class CLASS
{
  UNKNOWN,  ///< ATTENTION: Unknown is treated as any other class and represents a normal classification of object class
            ///< "Unknown"
  PEDESTRIAN,
  BICYCLE,
  MOTORBIKE,
  CAR,
  TRUCK,
  VAN,
  BUS,
  TRAIN,
  ROAD_OBSTACLE,
  ANIMAL,
  TRAFFIC_LIGHT_GREEN,
  TRAFFIC_LIGHT_YELLOW,
  TRAFFIC_LIGHT_RED,
  TRAFFIC_LIGHT_RED_YELLOW,
  TRAFFIC_LIGHT_NONE,
  TRAFFIC_SIGN_NONE,
  CAR_UNION,
  TRUCK_UNION,
  BIKE_UNION,
  TARGET,
  NOT_CLASSIFIED,  ///< the NOT_CLASSIFIED class on the other hand represents non-information, i.e., from a detector
                   ///< which simply does not provide classification information
  NOT_TRACKABLE    ///< explicit notion for non-trackable classes
};

std::vector const unified_classes = { CLASS::UNKNOWN,
                                      CLASS::PEDESTRIAN,
                                      CLASS::CAR_UNION,
                                      CLASS::BIKE_UNION,
                                      CLASS::TRUCK_UNION };

std::vector const trackable_classes = {
  CLASS::UNKNOWN,    CLASS::PEDESTRIAN,  CLASS::BICYCLE, CLASS::MOTORBIKE,     CLASS::CAR,    CLASS::TRUCK,
  CLASS::VAN,        CLASS::BUS,         CLASS::TRAIN,   CLASS::ROAD_OBSTACLE, CLASS::ANIMAL, CLASS::CAR_UNION,
  CLASS::BIKE_UNION, CLASS::TRUCK_UNION, CLASS::TARGET,  CLASS::NOT_CLASSIFIED
};

namespace impl
{
const std::map<CLASS, std::string> CLASS_2_STRING{
  { CLASS::UNKNOWN, "UNKNOWN" },
  { CLASS::PEDESTRIAN, "PEDESTRIAN" },
  { CLASS::BICYCLE, "BICYCLE" },
  { CLASS::MOTORBIKE, "MOTORBIKE" },
  { CLASS::CAR, "CAR" },
  { CLASS::TRUCK, "TRUCK" },
  { CLASS::VAN, "VAN" },
  { CLASS::BUS, "BUS" },
  { CLASS::TRAIN, "TRAIN" },
  { CLASS::ROAD_OBSTACLE, "ROAD_OBSTACLE" },
  { CLASS::ANIMAL, "ANIMAL" },
  { CLASS::TRAFFIC_LIGHT_GREEN, "TRAFFIC_LIGHT_GREEN" },
  { CLASS::TRAFFIC_LIGHT_YELLOW, "TRAFFIC_LIGHT_YELLOW" },
  { CLASS::TRAFFIC_LIGHT_RED, "TRAFFIC_LIGHT_RED" },
  { CLASS::TRAFFIC_LIGHT_RED_YELLOW, "TRAFFIC_LIGHT_RED_YELLOW" },
  { CLASS::TRAFFIC_LIGHT_NONE, "TRAFFIC_LIGHT_NONE" },
  { CLASS::TRAFFIC_SIGN_NONE, "TRAFFIC_SIGN_NONE" },
  { CLASS::CAR_UNION, "CAR_UNION" },
  { CLASS::TRUCK_UNION, "TRUCK_UNION" },
  { CLASS::BIKE_UNION, "BIKE_UNION" },
  { CLASS::TARGET, "TARGET" },
  { CLASS::NOT_TRACKABLE, "NOT_TRACKABLE" },
  { CLASS::NOT_CLASSIFIED, "NOT_CLASSIFIED" },
};

const std::map<CLASS, CLASS> CLASS_2_CLASS_unify{ { CLASS::UNKNOWN, CLASS::UNKNOWN },
                                                  { CLASS::PEDESTRIAN, CLASS::PEDESTRIAN },
                                                  { CLASS::BICYCLE, CLASS::BIKE_UNION },
                                                  { CLASS::MOTORBIKE, CLASS::BIKE_UNION },
                                                  { CLASS::CAR, CLASS::CAR_UNION },
                                                  { CLASS::VAN, CLASS::CAR_UNION },
                                                  { CLASS::TRUCK, CLASS::TRUCK_UNION },
                                                  { CLASS::BUS, CLASS::TRUCK_UNION },
                                                  { CLASS::TRAIN, CLASS::TRUCK_UNION },
                                                  { CLASS::CAR_UNION, CLASS::CAR_UNION },
                                                  { CLASS::TRUCK_UNION, CLASS::TRUCK_UNION },
                                                  { CLASS::BIKE_UNION, CLASS::BIKE_UNION },
                                                  { CLASS::ROAD_OBSTACLE, CLASS::UNKNOWN },
                                                  { CLASS::ANIMAL, CLASS::UNKNOWN },
                                                  { CLASS::TRAFFIC_LIGHT_GREEN, CLASS::NOT_TRACKABLE },
                                                  { CLASS::TRAFFIC_LIGHT_YELLOW, CLASS::NOT_TRACKABLE },
                                                  { CLASS::TRAFFIC_LIGHT_RED, CLASS::NOT_TRACKABLE },
                                                  { CLASS::TRAFFIC_LIGHT_RED_YELLOW, CLASS::NOT_TRACKABLE },
                                                  { CLASS::TRAFFIC_LIGHT_NONE, CLASS::NOT_TRACKABLE },
                                                  { CLASS::TRAFFIC_SIGN_NONE, CLASS::NOT_TRACKABLE },
                                                  { CLASS::TARGET, CLASS::UNKNOWN },
                                                  { CLASS::NOT_CLASSIFIED, CLASS::NOT_CLASSIFIED },
                                                  { CLASS::NOT_TRACKABLE, CLASS::NOT_TRACKABLE } };
}  // namespace impl
std::string to_string(CLASS type);
std::optional<CLASS> to_CLASS(std::string type);
std::optional<CLASS> unify(CLASS type);

// ######################################################################################################################
enum class SENSOR_CALIBRATION_TYPE
{
  EXTRINSIC = 1,
  CAMERA = 2,
  ODOMETRY = 3,
  VEHICLE = 4,
  EXTRINSIC_WITH_COVERAGE = 5
};
namespace impl
{
const std::map<SENSOR_CALIBRATION_TYPE, std::string> SENSOR_CALIBRATION_TYPE_2_STRING{
  { SENSOR_CALIBRATION_TYPE::EXTRINSIC, "EXTRINSIC" },
  { SENSOR_CALIBRATION_TYPE::CAMERA, "CAMERA" },
  { SENSOR_CALIBRATION_TYPE::ODOMETRY, "ODOMETRY" },
  { SENSOR_CALIBRATION_TYPE::VEHICLE, "VEHICLE" },
  { SENSOR_CALIBRATION_TYPE::EXTRINSIC_WITH_COVERAGE, "EXTRINSIC_WITH_COVERAGE" }
};
}
std::string to_string(SENSOR_CALIBRATION_TYPE type);
std::optional<SENSOR_CALIBRATION_TYPE> to_SENSOR_CALIBRATION_TYPE(std::string type);

// #####################################################################################################################
enum class STATE_DISTRIBUTION_EXTRACTION_TYPE
{
  BEST_STATE_MODEL,
  AVERAGE
};
namespace impl
{
const std::map<STATE_DISTRIBUTION_EXTRACTION_TYPE, std::string> STATE_DISTRIBUTION_EXTRACTION_TYPE_2_STRING{
  { STATE_DISTRIBUTION_EXTRACTION_TYPE::BEST_STATE_MODEL, "BEST_STATE_MODEL" },
  { STATE_DISTRIBUTION_EXTRACTION_TYPE::AVERAGE, "AVERAGE" }
};
}
std::string to_string(STATE_DISTRIBUTION_EXTRACTION_TYPE type);
std::optional<STATE_DISTRIBUTION_EXTRACTION_TYPE> to_STATE_DISTRIBUTION_EXTRACTION_TYPE(std::string type);

// #####################################################################################################################
enum class TRANSITION_TYPE
{
  MEAN_TRANSFORM,
  UNSCENTED_TRANSFORM,
};
namespace impl
{
const std::map<TRANSITION_TYPE, std::string> TRANSITION_TYPE_2_STRING{
  { TRANSITION_TYPE::MEAN_TRANSFORM, "MEAN_TRANSFORM" },
  { TRANSITION_TYPE::UNSCENTED_TRANSFORM, "UNSCENTED_TRANSFORM" }
};
}
std::string to_string(TRANSITION_TYPE type);
std::optional<TRANSITION_TYPE> to_TRANSITION_TYPE(std::string type);

// #####################################################################################################################
enum class BUILD_MODE
{
  DEBUG,
  RELEASE,
  NONE
};
namespace impl
{
const std::map<BUILD_MODE, std::string> BUILD_MODE_2_STRING{
  { BUILD_MODE::DEBUG, "DEBUG" },
  { BUILD_MODE::RELEASE, "RELEASE" },
  { BUILD_MODE::NONE, "NONE" },
};
}

std::string to_string(BUILD_MODE type);
std::optional<BUILD_MODE> to_BUILD_MODE(std::string type);

// #####################################################################################################################
enum class TTT_FILTER_TYPE
{
  NO,     ///< No actual filter, just return the to the state converted detections
  TRANS,  ///< Simple track-to-track filter: fuses near tracks, transparently passes all non-fused tracks through
  EVAL    ///< Evaluates other filters based on some external ground-truth, i.e., CAM,
};
namespace impl
{
const std::map<TTT_FILTER_TYPE, std::string> TTT_FILTER_TYPE_2_STRING{ { TTT_FILTER_TYPE::NO, "NO" },
                                                                       { TTT_FILTER_TYPE::TRANS, "TRANS" },
                                                                       { TTT_FILTER_TYPE::EVAL, "EVAL" }

};
}
std::string to_string(TTT_FILTER_TYPE type);
std::optional<TTT_FILTER_TYPE> to_TTT_FILTER_TYPE(std::string type);

// #####################################################################################################################
enum class LMB_UPDATE_METHOD
{
  GLMB,  ///< convert LMB to GLMB and update GLMB
  LBP    ///< perform loopy belief propagation
};
namespace impl
{
const std::map<LMB_UPDATE_METHOD, std::string> LMB_UPDATE_METHOD_2_STRING{
  { LMB_UPDATE_METHOD::GLMB, "GLMB" },
  { LMB_UPDATE_METHOD::LBP, "LBP" },

};
}
std::string to_string(LMB_UPDATE_METHOD type);
std::optional<LMB_UPDATE_METHOD> to_LMB_UPDATE_METHOD(std::string type);

// #####################################################################################################################
enum class TTT_UNCORRELATED_TRACKS_FUSION_STRATEGY
{
  IC_LMB_VERSION_WITHOUT_ASSOCIATION_BEFORE,   ///< uses the IC_LMB update without t2t association before
  FPM_LMB_VERSION_WITHOUT_ASSOCIATION_BEFORE,  ///< uses the FPM_LMB update without t2t association before
  IC_LMB_VERSION_WITH_ASSOCIATION_BEFORE,      ///< uses the IC_LMB update without t2t association before
  FPM_LMB_VERSION_WITH_ASSOCIATION_BEFORE,     ///< uses the FPM_LMB update without t2t association before
};
namespace impl
{
const std::map<TTT_UNCORRELATED_TRACKS_FUSION_STRATEGY, std::string> TTT_UNCORRELATED_TRACKS_FUSION_STRATEGY_2_STRING{
  { TTT_UNCORRELATED_TRACKS_FUSION_STRATEGY::IC_LMB_VERSION_WITHOUT_ASSOCIATION_BEFORE,
    "IC_LMB_VERSION_WITHOUT_ASSOCIATION_BEFORE" },
  { TTT_UNCORRELATED_TRACKS_FUSION_STRATEGY::FPM_LMB_VERSION_WITHOUT_ASSOCIATION_BEFORE,
    "FPM_LMB_VERSION_WITHOUT_ASSOCIATION_BEFORE" },
  { TTT_UNCORRELATED_TRACKS_FUSION_STRATEGY::IC_LMB_VERSION_WITH_ASSOCIATION_BEFORE,
    "IC_LMB_VERSION_WITH_ASSOCIATION_BEFORE" },
  { TTT_UNCORRELATED_TRACKS_FUSION_STRATEGY::FPM_LMB_VERSION_WITH_ASSOCIATION_BEFORE,
    "FPM_LMB_VERSION_WITH_ASSOCIATION_BEFORE" }
};
}
std::string to_string(TTT_UNCORRELATED_TRACKS_FUSION_STRATEGY type);
std::optional<TTT_UNCORRELATED_TRACKS_FUSION_STRATEGY> to_TTT_UNCORRELATED_TRACKS_FUSION_STRATEGY(std::string type);

// #####################################################################################################################
enum class FILTER_TYPE
{
  NO,   ///< No actual filter, just return the to the state converted detections
  NN,   ///< Nearest Neighbor "filter" -> use the innovation with greatest likelihood (could use measurements multiple
        ///< times, no real filter)
  GNN,  ///< Global Nearest Neighbor filter ->
  GLMB_IC,     ///< implements the multi-sensor GLMB update sequentiell (GLMB pred -> single sensor GLMB upd 1 -> single
               ///< sensor GLMB upd 2)
  GLMB_PM,     ///< https://ieeexplore.ieee.org/document/9524466 , https://ieeexplore.ieee.org/document/9841275
  LMB_IC,      ///< implements the multi-sensor LMB update sequentiell (LMB2GLMB -> single sensor upd 1 -> GLMB2LMB ->
               ///< LMB2GLMB -> single sensor upd 2 -> GLMB2LMB)
  LMB_PM,      ///< PM-LMB filter https://ieeexplore.ieee.org/document/10224121
  LMB_FPM,     ///< FPM-LMB filter https://ieeexplore.ieee.org/document/10224189
  ID_TRACKER,  ///< perform the association on the object id only
  PHD          ///< PHD Filter https://ieeexplore.ieee.org/document/1710358
};
namespace impl
{
const std::map<FILTER_TYPE, std::string> FILTER_TYPE_2_STRING{
  { FILTER_TYPE::NO, "NO" },           { FILTER_TYPE::NN, "NN" },           { FILTER_TYPE::GNN, "GNN" },
  { FILTER_TYPE::GLMB_IC, "GLMB_IC" }, { FILTER_TYPE::GLMB_PM, "GLMB_PM" }, { FILTER_TYPE::LMB_IC, "LMB_IC" },
  { FILTER_TYPE::LMB_PM, "LMB_PM" },   { FILTER_TYPE::LMB_FPM, "LMB_FPM" }, { FILTER_TYPE::ID_TRACKER, "ID_TRACKER" },
  { FILTER_TYPE::PHD, "PHD" }
};
}
std::string to_string(FILTER_TYPE type);
std::optional<FILTER_TYPE> to_FILTER_TYPE(std::string type);

enum class EXTENT
{
  NONE,         ///< no extent
  RECTANGULAR,  ///< LENGTH, WIDTH, HEIGHT
  CIRCULAR      ///< RADIUS
};
namespace impl
{
const std::map<EXTENT, std::string> EXTENT_2_STRING{
  { EXTENT::NONE, "NONE" },
  { EXTENT::RECTANGULAR, "RECTANGULAR" },
  { EXTENT::CIRCULAR, "CIRCULAR" },
};
}
std::string to_string(EXTENT type);
std::optional<EXTENT> to_EXTENT(std::string type);

// #####################################################################################################################
enum class COMPONENT
{
  POS_X,  ///< Position x [m]
  POS_Y,  ///< Position y [m]
  POS_Z,  ///< Position z [m]

  VEL_X,    ///< Velocity in x [m/s]
  VEL_Y,    ///< Velocity in y [m/s]
  VEL_Z,    ///< Velocity in z [m/s]
  VEL_ABS,  ///< Absolute velocity [m/s]

  ACC_X,    ///< Acceleration in x [m/s^2]
  ACC_Y,    ///< Acceleration in y [m/s^2]
  ACC_Z,    ///< Acceleration in z [m/s^2]
  ACC_ABS,  ///< Absolute Acceleration [m/s^2]

  JERK_X,    ///< Jerk in x [m/s^3]
  JERK_Y,    ///< Jerk in y [m/s^3]
  JERK_Z,    ///< Jerk in z [m/s^3]
  JERK_ABS,  ///< Absolute Jerk [m/s^3]

  ROT_Z,      ///< Orientation/Rotation around Z axis resp. yaw angle of object [rad]
  VEL_ROT_Z,  ///< Velocity of Orientation/Rotation around Z axis resp YawRate [rad/s]
  ACC_ROT_Z,  ///< Acceleration of Rotation around Z axis [rad/s^2]

  AZIMUTH,    ///< Azimuth angle of the object (angle between the xz-plane and object) [rad] Be careful: This value is
              ///< always treated in sensor coordinates! todo(hermann,scheible): Find a better way to handle this case!
  ELEVATION,  ///< Elevation angle of the object (angle between xy-plane and object)   [rad]

  X_CC_LOWER_LEFT,   ///< Describes a bounding box in image coordinates at z-distance = 1
  X_CC_UPPER_RIGHT,  ///< Describes a bounding box in image coordinates at z-distance = 1
  Y_CC_LOWER_LEFT,   ///< Describes a bounding box in image coordinates at z-distance = 1
  Y_CC_UPPER_RIGHT,  ///< Describes a bounding box in image coordinates at z-distance = 1

  LENGTH,  ///< Length of object [m] // Todo: log space?
  WIDTH,   ///< Width of object [m] // Todo: log space?
  HEIGHT,  ///< Height of object [m] // Todo: log space?
  RADIUS,  ///< Radius of object [m]

  LENGTH_CHANGE,  ///< Change of the Length [m/s]
  WIDTH_CHANGE,   ///< Change of the Width [m/s]
  HEIGHT_CHANGE,  ///< Change of the Height [m/s]
  RADIUS_CHANGE   ///< Change of the Radius [m/s]
};
std::vector const ALL_COMPONENTS{ COMPONENT::POS_X,           COMPONENT::POS_Y,           COMPONENT::POS_Z,
                                  COMPONENT::VEL_X,           COMPONENT::VEL_Y,           COMPONENT::VEL_Z,
                                  COMPONENT::VEL_ABS,         COMPONENT::ACC_X,           COMPONENT::ACC_Y,
                                  COMPONENT::ACC_Z,           COMPONENT::ACC_ABS,         COMPONENT::JERK_X,
                                  COMPONENT::JERK_Y,          COMPONENT::JERK_Z,          COMPONENT::JERK_ABS,
                                  COMPONENT::ROT_Z,           COMPONENT::VEL_ROT_Z,       COMPONENT::ACC_ROT_Z,
                                  COMPONENT::AZIMUTH,         COMPONENT::ELEVATION,       COMPONENT::LENGTH,
                                  COMPONENT::WIDTH,           COMPONENT::HEIGHT,          COMPONENT::RADIUS,
                                  COMPONENT::LENGTH_CHANGE,   COMPONENT::WIDTH_CHANGE,    COMPONENT::HEIGHT_CHANGE,
                                  COMPONENT::RADIUS_CHANGE,   COMPONENT::X_CC_LOWER_LEFT, COMPONENT::X_CC_UPPER_RIGHT,
                                  COMPONENT::Y_CC_LOWER_LEFT, COMPONENT::Y_CC_UPPER_RIGHT };
std::vector const REQUIRED_EGO_MOTION_COMPONENTS{
  COMPONENT::VEL_ABS,
  COMPONENT::VEL_ROT_Z,
};
std::vector const EXTENT_RECTANGULAR{ COMPONENT::LENGTH, COMPONENT::WIDTH, COMPONENT::HEIGHT };
std::vector const EXTENT_RECTANGULAR_NOISE{ COMPONENT::LENGTH_CHANGE,
                                            COMPONENT::WIDTH_CHANGE,
                                            COMPONENT::HEIGHT_CHANGE };
std::vector const EXTENT_CIRCULAR{ COMPONENT::RADIUS };
std::vector const EXTENT_CIRCULAR_NOISE{ COMPONENT::RADIUS_CHANGE };

std::size_t const NUM_COMPONENTS{ ALL_COMPONENTS.size() };

namespace impl
{
const std::map<COMPONENT, std::string> COMPONENT_2_STRING{
  { COMPONENT::POS_X, "POS_X" },
  { COMPONENT::POS_Y, "POS_Y" },
  { COMPONENT::POS_Z, "POS_Z" },

  { COMPONENT::VEL_X, "VEL_X" },
  { COMPONENT::VEL_Y, "VEL_Y" },
  { COMPONENT::VEL_Z, "VEL_Z" },
  { COMPONENT::VEL_ABS, "VEL_ABS" },

  { COMPONENT::ACC_X, "ACC_X" },
  { COMPONENT::ACC_Y, "ACC_Y" },
  { COMPONENT::ACC_Z, "ACC_Z" },
  { COMPONENT::ACC_ABS, "ACC_ABS" },

  { COMPONENT::JERK_X, "JERK_X" },
  { COMPONENT::JERK_Y, "JERK_Y" },
  { COMPONENT::JERK_Z, "JERK_Z" },
  { COMPONENT::JERK_ABS, "JERK_ABS" },

  { COMPONENT::ROT_Z, "ROT_Z" },
  { COMPONENT::VEL_ROT_Z, "VEL_ROT_Z" },
  { COMPONENT::ACC_ROT_Z, "ACC_ROT_Z" },

  { COMPONENT::AZIMUTH, "AZIMUTH" },
  { COMPONENT::ELEVATION, "ELEVATION" },

  { COMPONENT::LENGTH, "LENGTH" },
  { COMPONENT::WIDTH, "WIDTH" },
  { COMPONENT::HEIGHT, "HEIGHT" },
  { COMPONENT::RADIUS, "RADIUS" },

  { COMPONENT::LENGTH_CHANGE, "LENGTH_CHANGE" },
  { COMPONENT::WIDTH_CHANGE, "WIDTH_CHANGE" },
  { COMPONENT::HEIGHT_CHANGE, "HEIGHT_CHANGE" },
  { COMPONENT::RADIUS_CHANGE, "RADIUS_CHANGE" },

  { COMPONENT::X_CC_LOWER_LEFT, "X_CC_LOWER_LEFT" },
  { COMPONENT::X_CC_UPPER_RIGHT, "X_CC_UPPER_RIGHT" },
  { COMPONENT::Y_CC_LOWER_LEFT, "Y_CC_LOWER_LEFT" },
  { COMPONENT::Y_CC_UPPER_RIGHT, "Y_CC_UPPER_RIGHT" },
};
std::map<COMPONENT, COMPONENT> const COMPONENT_DT{
  { COMPONENT::POS_X, COMPONENT::VEL_X },
  { COMPONENT::POS_Y, COMPONENT::VEL_Y },
  { COMPONENT::POS_Z, COMPONENT::VEL_Z },

  { COMPONENT::VEL_X, COMPONENT::ACC_X },
  { COMPONENT::VEL_Y, COMPONENT::ACC_Y },
  { COMPONENT::VEL_Z, COMPONENT::ACC_Z },
  { COMPONENT::VEL_ABS, COMPONENT::ACC_ABS },

  { COMPONENT::ACC_X, COMPONENT::JERK_X },
  { COMPONENT::ACC_Y, COMPONENT::JERK_Y },
  { COMPONENT::ACC_Z, COMPONENT::JERK_Z },
  { COMPONENT::ACC_ABS, COMPONENT::JERK_ABS },

  //            {COMPONENT::JERK_X,    },
  //            {COMPONENT::JERK_Y,    },
  //            {COMPONENT::JERK_Z,    },
  //            {COMPONENT::JERK_ABS,  },

  { COMPONENT::ROT_Z, COMPONENT::VEL_ROT_Z },
  { COMPONENT::VEL_ROT_Z, COMPONENT::ACC_ROT_Z },
  //            {COMPONENT::ACC_ROT_Z, },

  //            {COMPONENT::ELEVATION, "ELEVATION"},

  { COMPONENT::LENGTH, COMPONENT::LENGTH_CHANGE },
  { COMPONENT::WIDTH, COMPONENT::WIDTH_CHANGE },
  { COMPONENT::HEIGHT, COMPONENT::HEIGHT_CHANGE },

  //            {COMPONENT::LENGTH_CHANGE,    },
  //            {COMPONENT::WIDTH_CHANGE,     },
  //            {COMPONENT::HEIGHT_CHANGE,    },
};
}  // namespace impl
[[maybe_unused]] constexpr std::array COMPONENT_SPATIAL_SIZE{ COMPONENT::LENGTH, COMPONENT::WIDTH, COMPONENT::HEIGHT };
[[maybe_unused]] constexpr std::array COMPONENT_SPATIAL_SIZE_CHANGE{ COMPONENT::LENGTH_CHANGE,
                                                                     COMPONENT::WIDTH_CHANGE,
                                                                     COMPONENT::HEIGHT_CHANGE };
// constexpr std::array<COMPONENT, 1> COMPONENT_ANGULAR{ COMPONENT::ROT_Z };

std::string to_string(COMPONENT type);
std::optional<COMPONENT> to_COMPONENT(std::string type);
std::optional<COMPONENT> derivate(COMPONENT comp);

// #####################################################################################################################
enum class MO_DISTRIBUTION_TYPE
{
  GLMB,
  LMB,
};
namespace impl
{
const std::map<MO_DISTRIBUTION_TYPE, std::string> MO_DISTRIBUTION_TYPE_2_STRING{ { MO_DISTRIBUTION_TYPE::GLMB, "GLMB" },
                                                                                 { MO_DISTRIBUTION_TYPE::LMB, "LMB" } };
}
std::string to_string(MO_DISTRIBUTION_TYPE type);
std::optional<MO_DISTRIBUTION_TYPE> to_MO_DISTRIBUTION_TYPE(std::string type);

// #####################################################################################################################
enum class LMB_2_GLMB_CONVERISON_TYPE
{
  SAMPLING,
  K_BEST,
  ALL
};
namespace impl
{
const std::map<LMB_2_GLMB_CONVERISON_TYPE, std::string> LMB_2_GLMB_CONVERISON_TYPE_2_STRING{
  { LMB_2_GLMB_CONVERISON_TYPE::SAMPLING, "SAMPLING" },
  { LMB_2_GLMB_CONVERISON_TYPE::K_BEST, "K_BEST" },
  { LMB_2_GLMB_CONVERISON_TYPE::ALL, "ALL" },
};
}
std::string to_string(LMB_2_GLMB_CONVERISON_TYPE type);
std::optional<LMB_2_GLMB_CONVERISON_TYPE> to_LMB_2_GLMB_CONVERISON_TYPE(std::string type);

// #####################################################################################################################
enum class MO_DISTRIBUTION_EXTRACTION_TYPE
{
  EXISTENCE_PROBABILITY,
  CARDINALITY,
  BEST_HYPOTHESIS
};
namespace impl
{
const std::map<MO_DISTRIBUTION_EXTRACTION_TYPE, std::string> MO_DISTRIBUTION_EXTRACTION_TYPE_2_STRING{
  { MO_DISTRIBUTION_EXTRACTION_TYPE::EXISTENCE_PROBABILITY, "EXISTENCE_PROBABILITY" },
  { MO_DISTRIBUTION_EXTRACTION_TYPE::CARDINALITY, "CARDINALITY" },
  { MO_DISTRIBUTION_EXTRACTION_TYPE::BEST_HYPOTHESIS, "BEST_HYPOTHESIS" }
};
}
std::string to_string(MO_DISTRIBUTION_EXTRACTION_TYPE type);
std::optional<MO_DISTRIBUTION_EXTRACTION_TYPE> to_MO_DISTRIBUTION_EXTRACTION_TYPE(std::string type);

// #####################################################################################################################
/// Bit field flags denoting the position of the reference point on a 3D object.
/// The absence and the combination of both axis enumerators, "0b00" & "0b11", are
/// both interpreted as the center for this axis (default).
enum class REFERENCE_POINT
{
  CENTER = 0,  // 0b000000,
  // x-axis
  FRONT = 1,  // 0b000001,
  BACK = 2,   // 0b000010,
  // y-axis
  LEFT = 4,   // 0b000100,
  RIGHT = 8,  // 0b001000,
  // z-axis
  TOP = 16,     // 0b010000,
  BOTTOM = 32,  // 0b100000,
  // combined ReferencePoints for usability
  FRONT_LEFT = 5,   // 0b000101,
  FRONT_RIGHT = 9,  // 0b001001,
  BACK_LEFT = 6,    // 0b000110,
  BACK_RIGHT = 10,  // 0b001010,
};
namespace impl
{
const std::map<REFERENCE_POINT, std::string> REFERENCE_POINT_2_STRING{ { REFERENCE_POINT::CENTER, "CENTER" },
                                                                       { REFERENCE_POINT::FRONT, "FRONT" },
                                                                       { REFERENCE_POINT::BACK, "BACK" },
                                                                       { REFERENCE_POINT::LEFT, "LEFT" },
                                                                       { REFERENCE_POINT::RIGHT, "RIGHT" },
                                                                       { REFERENCE_POINT::TOP, "TOP" },
                                                                       { REFERENCE_POINT::BOTTOM, "BOTTOM" },
                                                                       { REFERENCE_POINT::FRONT_LEFT, "FRONT_LEFT" },
                                                                       { REFERENCE_POINT::FRONT_RIGHT, "FRONT_RIGHT" },
                                                                       { REFERENCE_POINT::BACK_LEFT, "BACK_LEFT" },
                                                                       { REFERENCE_POINT::BACK_RIGHT, "BACK_RIGHT" } };
}
REFERENCE_POINT inverseRP(REFERENCE_POINT rp);
std::string to_string(REFERENCE_POINT ref);
std::optional<REFERENCE_POINT> to_REFERENCE_POINT(std::string type);

// #####################################################################################################################
enum class BIRTH_MODEL_TYPE
{
  DYNAMIC,
  STATIC
};
namespace impl
{
const std::map<BIRTH_MODEL_TYPE, std::string> BIRTH_MODEL_TYPE_2_STRING{ { BIRTH_MODEL_TYPE::DYNAMIC, "DYNAMIC" },
                                                                         { BIRTH_MODEL_TYPE::STATIC, "STATIC" } };
}
std::string to_string(BIRTH_MODEL_TYPE type);
std::optional<BIRTH_MODEL_TYPE> to_BIRTH_MODEL(std::string type);

// #####################################################################################################################
enum class TRANSITION_MODEL_TYPE
{
  BASE
};
namespace impl
{
const std::map<TRANSITION_MODEL_TYPE, std::string> TRANSITION_MODEL_TYPE_2_STRING{ { TRANSITION_MODEL_TYPE::BASE,
                                                                                     "BASE" } };
}
std::string to_string(TRANSITION_MODEL_TYPE type);
std::optional<TRANSITION_MODEL_TYPE> to_TRANSITION_MODEL(std::string type);

// #####################################################################################################################
enum class OCCLUSION_MODEL_TYPE
{
  NO_OCCLUSION,
  LINE_OF_SIGHT_OCCLUSION,
};
namespace impl
{
const std::map<OCCLUSION_MODEL_TYPE, std::string> OCCLUSION_MODEL_TYPE_2_STRING{
  { OCCLUSION_MODEL_TYPE::NO_OCCLUSION, "NO_OCCLUSION" },
  { OCCLUSION_MODEL_TYPE::LINE_OF_SIGHT_OCCLUSION, "LINE_OF_SIGHT_OCCLUSION" }
};
}
std::string to_string(OCCLUSION_MODEL_TYPE type);
std::optional<OCCLUSION_MODEL_TYPE> to_OCCLUSION_MODEL(std::string type);

// #####################################################################################################################
enum class PERSISTENCE_MODEL_TYPE
{
  CONSTANT,
};
namespace impl
{
const std::map<PERSISTENCE_MODEL_TYPE, std::string> PERSISTENCE_MODEL_TYPE_2_STRING{
  { PERSISTENCE_MODEL_TYPE::CONSTANT, "CONSTANT" },
};
}
std::string to_string(PERSISTENCE_MODEL_TYPE type);
std::optional<PERSISTENCE_MODEL_TYPE> to_PERSISTENCE_MODEL(std::string type);

// #####################################################################################################################
enum class DISTRIBUTION_TYPE
{
  GAUSSIAN,
  PARTICLE,
  MIXTURE,
};
namespace impl
{
const std::map<DISTRIBUTION_TYPE, std::string> DISTRIBUTION_TYPE_2_STRING{ { DISTRIBUTION_TYPE::GAUSSIAN, "GAUSSIAN" },
                                                                           { DISTRIBUTION_TYPE::PARTICLE, "PARTICLE" },
                                                                           { DISTRIBUTION_TYPE::MIXTURE, "MIXTURE" } };
}
std::string to_string(DISTRIBUTION_TYPE type);
std::optional<DISTRIBUTION_TYPE> to_DISTRIBUTION_TYPE(std::string type);
// #####################################################################################################################

enum class DISTRIBUTION_EXTRACTION
{
  BEST_COMPONENT,
  MIXTURE
};
namespace impl
{
const std::map<DISTRIBUTION_EXTRACTION, std::string> DISTRIBUTION_EXTRACTION_2_STRING{
  { DISTRIBUTION_EXTRACTION::BEST_COMPONENT, "BEST_COMPONENT" },
  { DISTRIBUTION_EXTRACTION::MIXTURE, "MIXTURE" }
};
}
std::string to_string(DISTRIBUTION_EXTRACTION type);
std::optional<DISTRIBUTION_EXTRACTION> to_DISTRIBUTION_EXTRACTION(std::string type);

// #####################################################################################################################
enum class STATE_MODEL_TYPE
{
  CP,      //!< Constant position model
  CTP,     ///< Constant Position with Rotation about Z-axis
  CV,      //!< Constant velocity model
  CA,      //!< Constant acceleration model
  CTRV,    //!< Constant turn rate and velocity state model
  CTRA,    //!< Constant turn rate and acceleration state model
  ISCATR,  //!< Independent Split Constant acceleration and turn rate model
};
namespace impl
{
const std::map<STATE_MODEL_TYPE, std::string> STATE_MODEL_TYPE_2_STRING{
  { STATE_MODEL_TYPE::CP, "CP" },        { STATE_MODEL_TYPE::CTP, "CTP" },   { STATE_MODEL_TYPE::CV, "CV" },
  { STATE_MODEL_TYPE::CA, "CA" },        { STATE_MODEL_TYPE::CTRV, "CTRV" }, { STATE_MODEL_TYPE::CTRA, "CTRA" },
  { STATE_MODEL_TYPE::ISCATR, "ISCATR" }
};
}
std::string to_string(STATE_MODEL_TYPE type);
std::optional<STATE_MODEL_TYPE> to_STATE_MODEL_TYPE(std::string type);

// #####################################################################################################################
enum class MEASUREMENT_MODEL_TYPE
{
  GAUSSIAN,
};
namespace impl
{
const std::map<MEASUREMENT_MODEL_TYPE, std::string> MEASUREMENT_MODEL_TYPE_2_STRING{
  { MEASUREMENT_MODEL_TYPE::GAUSSIAN, "GAUSSIAN" },
};
}  // namespace impl
std::string to_string(MEASUREMENT_MODEL_TYPE model);
std::optional<MEASUREMENT_MODEL_TYPE> to_MEASUREMENT_MODEL(std::string name);

// ###################################################################################################################
enum class GLMB_ASSIGNMENT_METHOD
{
  MURTY,
  GIBBS_SAMPLING,
};

namespace impl
{
const std::map<GLMB_ASSIGNMENT_METHOD, std::string> RANKED_ASSIGNMENT_ALGORITHM_TYPE_2_STRING{
  { GLMB_ASSIGNMENT_METHOD::MURTY, "MURTY" },
  { GLMB_ASSIGNMENT_METHOD::GIBBS_SAMPLING, "GIBBS_SAMPLING" },
};
}
std::string to_string(GLMB_ASSIGNMENT_METHOD type);
std::optional<GLMB_ASSIGNMENT_METHOD> to_GLMB_UPDATE_METHOD(std::string type);

enum class MULTI_SENSOR_UPDATE_METHOD
{
  PM,
  FPM,
};

namespace impl
{
const std::map<MULTI_SENSOR_UPDATE_METHOD, std::string> MULTI_SENSOR_UPDATE_METHOD_TYPE_2_STRING{
  { MULTI_SENSOR_UPDATE_METHOD::PM, "PM" },
  { MULTI_SENSOR_UPDATE_METHOD::FPM, "FPM" },
};
}
std::string to_string(MULTI_SENSOR_UPDATE_METHOD type);
std::optional<MULTI_SENSOR_UPDATE_METHOD> to_MULTI_SENSOR_UPDATE_METHOD(std::string type);

namespace impl
{
template <typename T, typename SmartPointer>
auto smartPtrCast(const SmartPointer& ptr, const std::string& msg) -> const T*
{
  const auto* ret = dynamic_cast<const T*>(ptr);
  if (nullptr == ret)
  {
    LOG_ERR(msg);
    throw std::bad_cast();
  }

  return ret;
}
}  // namespace impl

}  // namespace ttb
