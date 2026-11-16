#pragma once

#include "tracking_lib/Misc/logger_setup.h"

#include <type_safe/strong_typedef.hpp>
#include <type_safe/deferred_construction.hpp>
#include <debug_assert.hpp>

#include <numeric>
#include <Eigen/Dense>
#include <chrono>
#include <optional>
#include <map>
#include <format>
#include <stacktrace>
#include <set>
#include <ranges>
#include <subjective_logic_lib/opinions/opinion.hpp>
#include <subjective_logic_lib/opinions/trusted_opinion.hpp>
#include <XAD/XAD.hpp>
#include <XAD/StdCompatibility.hpp>
#include "tracking_lib/Misc/HPXWrapper.h"
#include <type_traits>
namespace ttb
{

/// restrict template to supported number of alternatives
template <typename... Types>
struct TypeList
{
};

template <typename T, typename List>
concept IsAnyOf = []<typename... Types>(TypeList<Types...>) { return (std::is_same_v<Types, T> || ...); }(List());

/// use IsAnyOf to restrict template parameters
using SupportedTypes = TypeList<double, float, std::string, bool>;
static_assert(IsAnyOf<double, SupportedTypes>);
static_assert(not IsAnyOf<char, SupportedTypes>);

/// intentional not defined to get useful type information on errors, see to_val
template <class T>
struct NotDefined;

template <auto MemberPtr>
inline constexpr auto member =
    std::views::transform([]<typename T0>(T0&& elem) -> decltype(auto) { return std::forward<T0>(elem).*MemberPtr; });

class Components;

/// To support automatic differentiation (AD) all operations must use this data type.
/// As using this type introduces some overhead compared to plain double, the whole library is templated on this type.
/// The template paramter is usually called UDT
using AD = xad::AReal<double>;

template <class UDT>
using Probability = UDT;

template <class T>
constexpr bool always_false_v = false;

/// convert potential AD variables to double, i.e., interrupt the AD graph
template <class UDT>
auto to_val(UDT const& val)
{
  if constexpr (std::is_same_v<UDT, double>)
  {
    return val;
  }
  else if constexpr (std::is_convertible_v<UDT, AD>)
  {
    return val.value();
  }
  else if constexpr (std::is_base_of_v<Eigen::EigenBase<UDT>, UDT>)
  {
    return val.unaryExpr([](auto const& x) { return to_val(x); }).eval();
  }
  else
  {
    NotDefined<UDT> this_does_not_compile;
    static_assert(always_false_v<UDT>, "to_val: unsupported type");
  }
}

template <class UDT>
using Matrix = Eigen::Matrix<UDT, Eigen::Dynamic, Eigen::Dynamic>;

template <class UDT>
using Vector = Eigen::Matrix<UDT, Eigen::Dynamic, 1>;

using RowVector = Eigen::Matrix<double, 1, Eigen::Dynamic>;
using Matrix33 = Eigen::Matrix<double, 3, 3, Eigen::DontAlign>;
using Matrix44 = Eigen::Matrix<double, 4, 4, Eigen::DontAlign>;
using Vector3 = Eigen::Matrix<double, 3, 1, Eigen::DontAlign>;
using Vector2 = Eigen::Matrix<double, 2, 1, Eigen::DontAlign>;
using Matrix22 = Eigen::Matrix<double, 2, 2, Eigen::DontAlign>;
using SE3Trafo = Eigen::Transform<double, 3, Eigen::TransformTraits::Isometry, Eigen::DontAlign>;
using Index = Eigen::Index;
using Indices = Eigen::Array<Index, -1, 1>;
using MatrixIndex = Eigen::Matrix<Index, -1, -1>;
using VectorIndex = Eigen::Matrix<Index, -1, 1>;
using Cardinality = std::size_t;

/// Time Definitions
using namespace std::chrono_literals;
using Duration = std::chrono::duration<int64_t, std::nano>;
using Time = std::chrono::time_point<std::chrono::high_resolution_clock, Duration>;

template <class UDT>
class DataSourceContainer;

[[nodiscard]] double to_seconds(Duration dur);
[[nodiscard]] double to_milliseconds(Duration dur);
[[nodiscard]] int64_t to_nanoseconds(Duration dur);

using Opinion = subjective_logic::Opinion<2, double>;

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
  std::atomic<typename SID::type> _next{ 11 };
};

struct GospaResult
{
  double missed_cost;
  double false_cost;
  double loc_cost;
  double gospa;
  std::size_t card_gt;
  std::size_t card_est;
};

struct OspaResult
{
  double ospa;
};

struct Ospa2Result
{
  double ospa2;
};

template <class UDT>
struct ProcessMatrices
{
  Matrix<UDT> Q;
  Matrix<UDT> F;
};

/// Unique Id for one State Model
/// Note that all ids in the library are defined as strong types with a restricted number of allowed operations.
/// To extend the allowed operations, specifiy it in the typedef, see https://github.com/foonathan/type_safe for
/// examples
struct StateModelId
  : ts::strong_typedef<StateModelId, std::size_t>,
    ts::strong_typedef_op::output_operator<StateModelId>,
    ts::strong_typedef_op::equality_comparison<StateModelId>,   // allow == comparison
    ts::strong_typedef_op::relational_comparison<StateModelId>  // allow <, <=, >, >= -> needed for map
{
  using type = std::size_t;
  using strong_typedef::strong_typedef;
};

/// Unique Id for one data source
struct SourceId : ts::strong_typedef<SourceId, std::string>,
                  ts::strong_typedef_op::output_operator<SourceId>,
                  ts::strong_typedef_op::equality_comparison<SourceId>,   // allow == comparison
                  ts::strong_typedef_op::relational_comparison<SourceId>  // allow <, <=, >, >= -> needed for map
{
  using type = std::string;
  using strong_typedef::strong_typedef;
};

const SourceId DUMMY_SOURCE_ID{ "Dummy Data Source" };

/// Unique Id for one data source container, i.e., a collection of measurements from one data source of one time point
struct DataSourceContainerId
  : ts::strong_typedef<DataSourceContainerId, std::size_t>,
    ts::strong_typedef_op::output_operator<DataSourceContainerId>,
    ts::strong_typedef_op::equality_comparison<DataSourceContainerId>,   // allow == comparison
    ts::strong_typedef_op::relational_comparison<DataSourceContainerId>  // allow <, <=, >, >= -> needed for map
{
  using type = std::size_t;
  using strong_typedef::strong_typedef;
};

constexpr DataSourceContainerId DUMMY_CONTAINER{ 0 };

/// Unique Id for a (kinematic state) distribution
/// Any write access on a distribution should change its id
struct DistributionId : ts::strong_typedef<DistributionId, std::size_t>,
                        ts::strong_typedef_op::output_operator<DistributionId>,
                        ts::strong_typedef_op::equality_comparison<DistributionId>,  // allow ==, != comparison
                        ts::strong_typedef_op::relational_comparison<DistributionId>
{
  using type = std::size_t;
  using strong_typedef::strong_typedef;
};
constexpr DistributionId NO_DISTRIBUTION_ID_HISTORY{ 0 };
constexpr DistributionId NO_DISTRIBUTION{ 1 };
inline IDGenerator<DistributionId> _dist_id_Generator{};

/// Unique Id for one State, i.e., the state of one tracked object at one specific time.
/// Note that in general any write access on a State should change its id, e.g., predicting, updating, or
/// post-processing it, changes the id.
struct StateId : ts::strong_typedef<StateId, std::size_t>,
                 ts::strong_typedef_op::output_operator<StateId>,
                 ts::strong_typedef_op::equality_comparison<StateId>,
                 ts::strong_typedef_op::relational_comparison<StateId>
{
  using type = std::size_t;
  using strong_typedef::strong_typedef;
  std::strong_ordering operator<=>(StateId const& other) const
  {
    return value_ <=> other.value_;
  }
};
constexpr StateId NO_PRIOR{ 0 };
constexpr StateId NOT_DETECTED{ 1 };
constexpr StateId EMPTY_STATE{ 2 };

enum class STATE_TYPE
{
  PRIOR,          ///< default state type before update
  UPDATED_PRIOR,  ///< state type after this state has been updated
  BIRTH,          ///< birth track
  POSTERIOR,      ///< state type for the updates
  MEASUREMENT,    ///< a measurement
  ESTIMATION,     ///< a Estimation of an object, i.e., unimodal gaussian distribution
};

template <class UDT>
struct StateDetection
{
  StateId state_id;
  UDT detection_probability = std::numeric_limits<double>::quiet_NaN();
};

template <class UDT>
struct MeasClutter
{
  StateId meas_id;
  UDT clutter_intensity = std::numeric_limits<double>::quiet_NaN();
};

namespace impl
{

const std::map<STATE_TYPE, std::string> STATE_TYPE_2_STRING{ { STATE_TYPE::PRIOR, "PRIOR" },
                                                             { STATE_TYPE::UPDATED_PRIOR, "UPDATED_PRIOR" },
                                                             { STATE_TYPE::BIRTH, "BIRTH" },
                                                             { STATE_TYPE::POSTERIOR, "POSTERIOR" },
                                                             { STATE_TYPE::MEASUREMENT, "MEASUREMENT" },
                                                             { STATE_TYPE::ESTIMATION, "ESTIMATION" } };
}
std::string to_string(STATE_TYPE type);
std::optional<STATE_TYPE> to_STATE_TYPE(std::string type);

/// (Non-) Unique Id to distinguish Multi-Object Densities (unique for every Type)
struct MODistributionId : ts::strong_typedef<MODistributionId, std::size_t>,
                          ts::strong_typedef_op::output_operator<MODistributionId>,
                          ts::strong_typedef_op::equality_comparison<MODistributionId>  // allow ==, != comparison
{
  using type = std::size_t;
  using strong_typedef::strong_typedef;
};

/// Id to distinguish Measurements
/// Since internally a measurement is also represented as State, this is not a strong definition
using MeasurementId = StateId;  ///< not strong!

/// Unique Id used to identify Objects if the Detection provide such an info
struct ObjectId : ts::strong_typedef<ObjectId, std::size_t>,
                  ts::strong_typedef_op::output_operator<ObjectId>,
                  ts::strong_typedef_op::equality_comparison<ObjectId>,   // allow ==, != comparison
                  ts::strong_typedef_op::relational_comparison<ObjectId>  // allow <, <=, >, >=

{
  using type = std::size_t;
  using strong_typedef::strong_typedef;
};

/// Unique Id of one Hypothesis, i.e., a specific assignment from tracks to objects
struct HypothesisId : ts::strong_typedef<HypothesisId, std::size_t>,
                      ts::strong_typedef_op::output_operator<HypothesisId>,
                      ts::strong_typedef_op::equality_comparison<HypothesisId>,   // allow ==, != comparison
                      ts::strong_typedef_op::relational_comparison<HypothesisId>  // allow <, <=, >, >=
{
  using type = std::size_t;
  using strong_typedef::strong_typedef;
};
constexpr HypothesisId NO_HYPOTHESIS_ID_HISTORY{ 0 };

/// Label of a Track
/// Note that the label of a state is consistent over multiple time steps
struct Label : ts::strong_typedef<Label, std::size_t>,
               ts::strong_typedef_op::output_operator<Label>,
               ts::strong_typedef_op::equality_comparison<Label>,   // allow ==, != comparison
               ts::strong_typedef_op::relational_comparison<Label>  // allow <, <=, >, >=

{
  using type = std::size_t;
  using strong_typedef::strong_typedef;
};

}  // namespace ttb

template <class T>
struct std::formatter<ttb::Vector<T>>
{
  constexpr auto parse(std::format_parse_context& ctx)
  {
    return ctx.begin();
  }
  auto format(ttb::Vector<T> const& vector, std::format_context& ctx) const
  {
    std::stringstream ss;
    ss << vector;
    return std::format_to(ctx.out(), "{}", ss.str());
  }
};

template <>
struct std::formatter<ttb::StateModelId>
{
  constexpr auto parse(std::format_parse_context& ctx)
  {
    return ctx.begin();
  }
  auto format(ttb::StateModelId const& id, std::format_context& ctx) const
  {
    return std::format_to(ctx.out(), "{}", id.value_);
  }
};

template <>
struct std::formatter<ttb::AD>
{
  constexpr auto parse(std::format_parse_context& ctx)
  {
    return ctx.begin();
  }
  auto format(ttb::AD const& ad_val, std::format_context& ctx) const
  {
    return std::format_to(ctx.out(), "{}", ttb::to_val(ad_val));
  }
};

template <>
struct std::formatter<ttb::SourceId>
{
  constexpr auto parse(std::format_parse_context& ctx)
  {
    return ctx.begin();
  }
  auto format(ttb::SourceId const& id, std::format_context& ctx) const
  {
    return std::format_to(ctx.out(), "{}", id.value_);
  }
};

template <>
struct std::formatter<ttb::DataSourceContainerId>
{
  constexpr auto parse(std::format_parse_context& ctx)
  {
    return ctx.begin();
  }
  auto format(ttb::DataSourceContainerId const& id, std::format_context& ctx) const
  {
    return std::format_to(ctx.out(), "{}", id.value_);
  }
};

template <>
struct std::formatter<ttb::DistributionId>
{
  constexpr auto parse(std::format_parse_context& ctx)
  {
    return ctx.begin();
  }
  auto format(ttb::DistributionId const& id, std::format_context& ctx) const
  {
    if (id == ttb::NO_DISTRIBUTION)
    {
      return std::format_to(ctx.out(), "{}", "NO_DISTRIBUTION");
    }
    if (id == ttb::NO_DISTRIBUTION_ID_HISTORY)
    {
      return std::format_to(ctx.out(), "{}", "NO_DISTRIBUTION_ID_HISTORY");
    }

    return std::format_to(ctx.out(), "{}", id.value_);
  }
};

template <>
struct std::formatter<ttb::StateId>
{
  static constexpr auto parse(std::format_parse_context& ctx)
  {
    return ctx.begin();
  }
  auto format(ttb::StateId const& id, std::format_context& ctx) const
  {
    if (id == ttb::NOT_DETECTED)
    {
      return std::format_to(ctx.out(), "{}", "NOT_DETECTED");
    }
    if (id == ttb::NO_PRIOR)
    {
      return std::format_to(ctx.out(), "{}", "NO_PRIOR");
    }
    if (id == ttb::EMPTY_STATE)
    {
      return std::format_to(ctx.out(), "{}", "EMPTY_STATE");
    }
    return std::format_to(ctx.out(), "{}", id.value_);
  }
};

template <class T>
  requires ttb::IsAnyOf<T,
                        ttb::TypeList<ttb::StateId,
                                      ttb::Label,
                                      ttb::DistributionId,
                                      ttb::StateModelId>>
struct std::formatter<std::vector<T>>  // NOLINT
{
  constexpr auto parse(std::format_parse_context& ctx)
  {
    return ctx.begin();
  }
  auto format(std::vector<T> const& ts, std::format_context& ctx) const
  {
    auto out = ctx.out();
    ctx.advance_to(std::format_to(out, "["));
    for (T const& t : ts)
    {
      ctx.advance_to(std::format_to(out, "{}, ", t));
    }
    return std::format_to(out, "]");
  }
};

template <class T>
  requires ttb::IsAnyOf<T,
                        ttb::TypeList<ttb::StateId,
                                      ttb::Label,
                                      ttb::DistributionId,
                                      ttb::StateModelId>>
struct std::formatter<std::set<T>>  // NOLINT
{
  constexpr auto parse(std::format_parse_context& ctx)
  {
    return ctx.begin();
  }
  auto format(std::set<T> const& ts, std::format_context& ctx) const
  {
    auto out = ctx.out();
    ctx.advance_to(std::format_to(out, "["));
    for (T const& t : ts)
    {
      ctx.advance_to(std::format_to(out, "{}, ", t));
    }
    return std::format_to(out, "]");
  }
};

template <>
struct std::formatter<ttb::MODistributionId>
{
  constexpr auto parse(std::format_parse_context& ctx)
  {
    return ctx.begin();
  }
  auto format(ttb::MODistributionId const& id, std::format_context& ctx) const
  {
    return std::format_to(ctx.out(), "{}", id.value_);
  }
};

template <>
struct std::formatter<ttb::HypothesisId>
{
  constexpr auto parse(std::format_parse_context& ctx)
  {
    return ctx.begin();
  }
  auto format(ttb::HypothesisId const& id, std::format_context& ctx) const
  {
    if (id == ttb::NO_HYPOTHESIS_ID_HISTORY)
    {
      return std::format_to(ctx.out(), "{}", "NO_HYPOTHESIS_ID_HISTORY");
    }
    return std::format_to(ctx.out(), "{}", id.value_);
  }
};

template <>
struct std::formatter<ttb::Label>
{
  constexpr auto parse(std::format_parse_context& ctx)
  {
    return ctx.begin();
  }
  auto format(ttb::Label const& id, std::format_context& ctx) const
  {
    return std::format_to(ctx.out(), "{}", id.value_);
  }
};

template <>
struct std::formatter<ttb::STATE_TYPE>
{
  constexpr auto parse(std::format_parse_context& ctx)
  {
    return ctx.begin();
  }
  auto format(ttb::STATE_TYPE const& type, std::format_context& ctx) const
  {
    return std::format_to(ctx.out(), "{}", ttb::to_string(type));
  }
};

namespace boost
{
inline std::size_t hash_value(ttb::Time const& time);
inline std::size_t hash_value(ttb::Label const& label);
inline std::size_t hash_value(ttb::StateId const& id);
inline std::size_t hash_value(ttb::DataSourceContainerId const& id);
inline std::size_t hash_value(ttb::SourceId const& id);
inline std::size_t hash_value(ttb::HypothesisId const& id);
inline std::size_t hash_value(ttb::Components const& comps);
inline std::size_t hash_value(ttb::StateModelId const& id);
}  // namespace boost

#include <boost/functional/hash.hpp>

inline std::size_t boost::hash_value(ttb::Time const& time)
{
  boost::hash<ttb::Time::rep> constexpr hasher;
  return hasher(time.time_since_epoch().count());
}
inline std::size_t boost::hash_value(ttb::Label const& label)
{
  boost::hash<ttb::Label::type> constexpr hasher;
  return hasher(label.value_);
}
inline std::size_t boost::hash_value(ttb::StateId const& id)
{
  boost::hash<ttb::StateId::type> constexpr hasher;
  return hasher(id.value_);
}
inline std::size_t boost::hash_value(ttb::DataSourceContainerId const& id)
{
  boost::hash<ttb::DataSourceContainerId::type> constexpr hasher;
  return hasher(id.value_);
}
inline std::size_t boost::hash_value(ttb::SourceId const& id)
{
  boost::hash<ttb::SourceId::type> constexpr hasher;
  return hasher(id.value_);
}
inline std::size_t boost::hash_value(ttb::HypothesisId const& id)
{
  boost::hash<ttb::HypothesisId::type> constexpr hasher;
  return hasher(id.value_);
}
inline std::size_t boost::hash_value(ttb::StateModelId const& id)
{
  boost::hash<ttb::StateModelId::type> constexpr hasher;
  return hasher(id.value_);
}

using TTBTypes = ttb::TypeList<ttb::Label,
                               ttb::DataSourceContainerId,
                               ttb::HypothesisId,
                               ttb::ObjectId,
                               ttb::StateId,
                               ttb::DistributionId,
                               ttb::SourceId>;

template <class T>
  requires ttb::IsAnyOf<T, TTBTypes>
struct std::hash<std::vector<T>>  // NOLINT
{
  std::size_t operator()(std::vector<T> ids) const noexcept
  {
    std::size_t seed = 0;
    for (T const& id : ids)
    {
      boost::hash_combine(seed, id.value_);
    }
    return seed;
  }
};

template <class T, class V>
struct std::hash<std::pair<T, V>>
{
  std::size_t operator()(std::pair<T, V> ids) const noexcept
  {
    std::size_t seed = 0;
    boost::hash_combine(seed, ids.first);
    boost::hash_combine(seed, ids.second);
    return seed;
  }
};

template <class... Ts>
struct std::hash<std::tuple<Ts...>>
{
  std::size_t operator()(std::tuple<Ts...> const& arg) const noexcept
  {
    boost::hash<std::tuple<Ts...>> constexpr hasher;
    return hasher(arg);
  }
};

template <class T>
  requires ttb::IsAnyOf<T,
                        TTBTypes>
struct std::hash<T> : type_safe::hashable<T>  // NOLINT
{
};

namespace ttb
{
template <class UDT>
class TTBManager;
template <class UDT>
class State;
template <class UDT>
class StateModel;

// #####################################################################################################################
///// used for GKBest Algorithm
enum class SELECTION_STRATEGY

{
  GENERALIZED_K_BEST
};
namespace impl
{

const std::map<SELECTION_STRATEGY, std::string> SELECTION_STRATEGY_2_STRING{ { SELECTION_STRATEGY::GENERALIZED_K_BEST,
                                                                               "GENERALIZED_K_BEST" } };
}
std::string to_string(SELECTION_STRATEGY type);
std::optional<SELECTION_STRATEGY> to_SELECTION_STRATEGY(std::string type);

enum class GLMB_POST_PROCESSING_METHOD
{
  STANDARD,              ///< Use simple postprocessing, e.g. remove hypotheses with weight below threshold, etc...>
  DISSERTATION_HERRMANN  //< use postprocessing presented in dissertation of Martin Herrmann >
};
namespace impl
{
const std::map<GLMB_POST_PROCESSING_METHOD, std::string> GLMB_POST_PROCESSING_METHOD_2_STRING{
  { GLMB_POST_PROCESSING_METHOD::STANDARD, "STANDARD" },
  { GLMB_POST_PROCESSING_METHOD::DISSERTATION_HERRMANN, "DISSERTATION_HERRMANN" }
};
}
std::string to_string(GLMB_POST_PROCESSING_METHOD type);
std::optional<GLMB_POST_PROCESSING_METHOD> to_GLMB_POST_PROCESSING_METHOD(std::string type);

enum class TRACK_TO_TRACK_FUSION_STRATEGY
{
  CROSS_COVARIANCE_INTERSECTION,
  TAKE_BEST,
};
namespace impl
{

const std::map<TRACK_TO_TRACK_FUSION_STRATEGY, std::string> TRACK_TO_TRACK_FUSION_STRATEGY_2_STRING{
  { TRACK_TO_TRACK_FUSION_STRATEGY::CROSS_COVARIANCE_INTERSECTION, "CROSS_COVARIANCE_INTERSECTION" },
  { TRACK_TO_TRACK_FUSION_STRATEGY::TAKE_BEST, "TAKE_BEST" }
};
}
std::string to_string(TRACK_TO_TRACK_FUSION_STRATEGY type);
std::optional<TRACK_TO_TRACK_FUSION_STRATEGY> to_TRACK_TO_TRACK_FUSION_STRATEGY(std::string type);
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
}  // namespace ttb

template <>
struct std::formatter<ttb::STAGE>
{
  constexpr auto parse(std::format_parse_context& ctx)
  {
    return ctx.begin();
  }
  auto format(ttb::STAGE stage, std::format_context& ctx) const
  {
    return std::format_to(ctx.out(), "{}", ttb::to_string(stage));
  }
};

namespace ttb
{

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

std::vector const vru_classes = { CLASS::UNKNOWN,
                                  CLASS::PEDESTRIAN,
                                  CLASS::BICYCLE,
                                  CLASS::MOTORBIKE,
                                  CLASS::BIKE_UNION };

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
};
namespace impl
{
const std::map<TTT_FILTER_TYPE, std::string> TTT_FILTER_TYPE_2_STRING{
  { TTT_FILTER_TYPE::NO, "NO" },
  { TTT_FILTER_TYPE::TRANS, "TRANS" },
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
  GLMB_MS,     ///< multi-sensor GLMB update with suboptimal Gibbs sampling
               ///< B. -N. Vo, B. -T. Vo and M. Beard,
               ///< "Multi-Sensor Multi-Object Tracking With the Generalized Labeled Multi-Bernoulli Filter," in IEEE
               ///< Transactions on Signal Processing, vol. 67, no. 23, pp. 5952-5967, 1 Dec.1, 2019,
               ///< doi: 10.1109/TSP.2019.2946023.
  LMB_IC,      ///< implements the multi-sensor LMB update sequentiell (LMB2GLMB -> single sensor upd 1 -> GLMB2LMB ->
               ///< LMB2GLMB -> single sensor upd 2 -> GLMB2LMB)
  LMB_MS,      ///< uses the GLMB_MS update
  ID_TRACKER,  ///< perform the association on the object id only
  PMBM_IC
};
namespace impl
{
const std::map<FILTER_TYPE, std::string> FILTER_TYPE_2_STRING{
  { FILTER_TYPE::NO, "NO" },           { FILTER_TYPE::NN, "NN" },
  { FILTER_TYPE::GNN, "GNN" },         { FILTER_TYPE::GLMB_IC, "GLMB_IC" },
  { FILTER_TYPE::GLMB_MS, "GLMB_MS" }, { FILTER_TYPE::LMB_IC, "LMB_IC" },
  { FILTER_TYPE::LMB_MS, "LMB_MS" },   { FILTER_TYPE::ID_TRACKER, "ID_TRACKER" },
  { FILTER_TYPE::PMBM_IC, "PMBM_IC" }
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

  HULL_POS_X_MIN,  ///< the minimal POS_X value of a enclosing convex hull curve
  HULL_POS_Y_MIN,  ///< the minimal POS_Y value of a enclosing convex hull curve
  HULL_POS_X_MAX,  ///< the maximal POS_X value of a enclosing convex hull curve
  HULL_POS_Y_MAX,  ///< the maximal POS_Y value of a enclosing convex hull curve

  LENGTH,  ///< Length of object [m] // Todo: log space?
  WIDTH,   ///< Width of object [m] // Todo: log space?
  HEIGHT,  ///< Height of object [m] // Todo: log space?
  RADIUS,  ///< Radius of object [m]

  LENGTH_CHANGE,  ///< Change of the Length [m/s]
  WIDTH_CHANGE,   ///< Change of the Width [m/s]
  HEIGHT_CHANGE,  ///< Change of the Height [m/s]
  RADIUS_CHANGE   ///< Change of the Radius [m/s]
};
std::vector const ALL_COMPONENTS{ COMPONENT::POS_X,           COMPONENT::POS_Y,
                                  COMPONENT::POS_Z,           COMPONENT::VEL_X,
                                  COMPONENT::VEL_Y,           COMPONENT::VEL_Z,
                                  COMPONENT::VEL_ABS,         COMPONENT::ACC_X,
                                  COMPONENT::ACC_Y,           COMPONENT::ACC_Z,
                                  COMPONENT::ACC_ABS,         COMPONENT::JERK_X,
                                  COMPONENT::JERK_Y,          COMPONENT::JERK_Z,
                                  COMPONENT::JERK_ABS,        COMPONENT::ROT_Z,
                                  COMPONENT::VEL_ROT_Z,       COMPONENT::ACC_ROT_Z,
                                  COMPONENT::AZIMUTH,         COMPONENT::ELEVATION,
                                  COMPONENT::LENGTH,          COMPONENT::WIDTH,
                                  COMPONENT::HEIGHT,          COMPONENT::RADIUS,
                                  COMPONENT::LENGTH_CHANGE,   COMPONENT::WIDTH_CHANGE,
                                  COMPONENT::HEIGHT_CHANGE,   COMPONENT::RADIUS_CHANGE,
                                  COMPONENT::X_CC_LOWER_LEFT, COMPONENT::X_CC_UPPER_RIGHT,
                                  COMPONENT::Y_CC_LOWER_LEFT, COMPONENT::Y_CC_UPPER_RIGHT,
                                  COMPONENT::HULL_POS_X_MIN,  COMPONENT::HULL_POS_X_MAX,
                                  COMPONENT::HULL_POS_Y_MIN,  COMPONENT::HULL_POS_Y_MAX };
std::vector const REQUIRED_EGO_MOTION_COMPONENTS{
  COMPONENT::VEL_ABS,
  COMPONENT::VEL_ROT_Z,
};

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

  { COMPONENT::HULL_POS_X_MIN, "HULL_POS_X_MIN" },
  { COMPONENT::HULL_POS_X_MAX, "HULL_POS_X_MAX" },
  { COMPONENT::HULL_POS_Y_MIN, "HULL_POS_Y_MIN" },
  { COMPONENT::HULL_POS_Y_MAX, "HULL_POS_Y_MAX" },
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
  STATIC,
  ROI
};
namespace impl
{
const std::map<BIRTH_MODEL_TYPE, std::string> BIRTH_MODEL_TYPE_2_STRING{ { BIRTH_MODEL_TYPE::DYNAMIC, "DYNAMIC" },
                                                                         { BIRTH_MODEL_TYPE::STATIC, "STATIC" },
                                                                         { BIRTH_MODEL_TYPE::ROI, "ROI" } };
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
  ROI
};
namespace impl
{
const std::map<PERSISTENCE_MODEL_TYPE, std::string> PERSISTENCE_MODEL_TYPE_2_STRING{
  { PERSISTENCE_MODEL_TYPE::CONSTANT, "CONSTANT" },
  { PERSISTENCE_MODEL_TYPE::ROI, "ROI" },
};
}
std::string to_string(PERSISTENCE_MODEL_TYPE type);
std::optional<PERSISTENCE_MODEL_TYPE> to_PERSISTENCE_MODEL(std::string type);

// #####################################################################################################################
enum class DISTRIBUTION_TYPE
{
  GAUSSIAN,
  MIXTURE,
};
namespace impl
{
const std::map<DISTRIBUTION_TYPE, std::string> DISTRIBUTION_TYPE_2_STRING{ { DISTRIBUTION_TYPE::GAUSSIAN, "GAUSSIAN" },
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

enum class LMB_MULTI_SENSOR_UPDATE_METHOD
{
  GLMB,  /// do multi-sensor update (transform to GLMB, multi-sensor GLMB update, approx as LMB)
  FPM,   /// do the fpm update (single-sensor LMB update based on common prior, merge LMB posteriors)
  PU,    /// do the pu update (approximated solution for division of gm of robertson)
};

namespace impl
{
const std::map<LMB_MULTI_SENSOR_UPDATE_METHOD, std::string> LMB_MULTI_SENSOR_UPDATE_METHOD_2_STRING{
  { LMB_MULTI_SENSOR_UPDATE_METHOD::GLMB, "GLMB" },
  { LMB_MULTI_SENSOR_UPDATE_METHOD::FPM, "FPM" },
  { LMB_MULTI_SENSOR_UPDATE_METHOD::PU, "PU" },
};
}
std::string to_string(LMB_MULTI_SENSOR_UPDATE_METHOD type);
std::optional<LMB_MULTI_SENSOR_UPDATE_METHOD> to_LMB_MULTI_SENSOR_UPDATE_METHOD(std::string type);

enum class PM_MULTI_SENSOR_UPDATE_METHOD
{
  GENERALIZED_K_BEST,  /// do pm and fpm fusion using the GKBest algorithm (calculate the true best solutions)
  K_BEST,  /// do pm and fpm fusion using the KBest algorithm (calculate the approximate best solutions. K is calculated
           /// depending on prior hypotheses or mixture distribution weight)
  K_BEST_POISSON,  /// do pm and fpm fusion using the KBest algorithm (calculate the approximate best solutions. K is
                   /// calculated using a poisson distribution depending on prior hypotheses or mixture distribution
                   /// weight)
};

namespace impl
{
const std::map<PM_MULTI_SENSOR_UPDATE_METHOD, std::string> PM_MULTI_SENSOR_UPDATE_METHOD_2_STRING{
  { PM_MULTI_SENSOR_UPDATE_METHOD::GENERALIZED_K_BEST, "GENERALIZED_K_BEST" },
  { PM_MULTI_SENSOR_UPDATE_METHOD::K_BEST, "K_BEST" },
  { PM_MULTI_SENSOR_UPDATE_METHOD::K_BEST_POISSON, "K_BEST_POISSON" },
};
}
std::string to_string(PM_MULTI_SENSOR_UPDATE_METHOD type);
std::optional<PM_MULTI_SENSOR_UPDATE_METHOD> to_PM_MULTI_SENSOR_UPDATE_METHOD(std::string type);

enum class GLMB_MULTI_SENSOR_UPDATE_METHOD
{
  MS,  /// do multi-sensor update
  PM   /// do the pm update
};

namespace impl
{
const std::map<GLMB_MULTI_SENSOR_UPDATE_METHOD, std::string> GLMB_MULTI_SENSOR_UPDATE_METHOD_2_STRING{
  { GLMB_MULTI_SENSOR_UPDATE_METHOD::MS, "MS" },
  { GLMB_MULTI_SENSOR_UPDATE_METHOD::PM, "PM" },
};
}
std::string to_string(GLMB_MULTI_SENSOR_UPDATE_METHOD type);
std::optional<GLMB_MULTI_SENSOR_UPDATE_METHOD> to_GLMB_MULTI_SENSOR_UPDATE_METHOD(std::string type);

}  // namespace ttb

template <class T>
  requires requires(T const& t) {
    { ttb::to_string(t) } -> std::convertible_to<std::string>;
  }
struct std::formatter<T>  // NOLINT
{
  constexpr auto parse(std::format_parse_context& ctx)
  {
    return ctx.begin();
  }
  auto format(T const& t, std::format_context& ctx) const
  {
    return std::format_to(ctx.out(), "{}", ttb::to_string(t));
  }
};
