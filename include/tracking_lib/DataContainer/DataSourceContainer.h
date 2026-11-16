#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/States/State.h"
#include "tracking_lib/StateModels/PredictionInformation.h"
#include "tracking_lib/Measurements/SensorInformation.h"
#include "tracking_lib/Map/Map.h"

namespace ttb
{

struct GroundTruthMeasModelInfo
{
  SourceId id;
  double clutter_rate = std::numeric_limits<double>::quiet_NaN();
  double detection_prob = std::numeric_limits<double>::quiet_NaN();
  double meas_xy_dev = std::numeric_limits<double>::quiet_NaN();
};

struct GroundTruthStateModelInfo
{
  double noise_dev = std::numeric_limits<double>::quiet_NaN();
  double persistence_prob = std::numeric_limits<double>::quiet_NaN();
};
struct GroundTruthBirthModelInfo
{
  std::vector<std::map<COMPONENT, std::pair<double, double>>> poss_birth_state;
  double birth_probability;
};

struct GroundTruthScenarioInfo
{
  GroundTruthMeasModelInfo meas_model_info;
  GroundTruthStateModelInfo state_model_info;
  GroundTruthBirthModelInfo birth_model_info;
};

/// A DataSourceContainer is a collection of individual data (measurements/tracks) of the same data source of the same
/// time together with other information needed to use this information
template<class UDT>
class DataSourceContainer
{
public:
  [[nodiscard]] std::string toString() const;
  /// unique id connecting this data to a MeasurementModel
  SourceId _source_id;
  /// the individual measurements
  std::vector<State<UDT>> _measurements{};
  /// tracks
  std::vector<BernoulliState<UDT>> _tracks{};
  /// time of the measurements
  Time _time;
  /// other information about the data source, e.g., the source frame
  SensorInformation _sensorInfo{};
  /// data needed for the prediction step to this measurement data
  PredictionInformation _predictionInfo{};
  /// map
  std::optional<Map> _map{};
  /// unique id of this container
  DataSourceContainerId _id{ _idGenerator.getID() };
  std::optional<GroundTruthScenarioInfo> _ground_truth_info{};
  static IDGenerator<DataSourceContainerId> _idGenerator;
};

/// return the transformation from the usual tracking coords into sensor specific coords,
/// e.g., for cameras, this rotates the axis
[[nodiscard]] SE3Trafo default_tracking_cs_to_default_sensor_cs(SourceId const& id);

}  // namespace ttb