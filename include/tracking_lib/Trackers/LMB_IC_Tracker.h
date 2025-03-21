#pragma once

#include "tracking_lib/Trackers/BaseTracker.h"
// #####################################################################################################################
#include "tracking_lib/MultiObjectStateDistributions/LMBDistribution.h"
#include "tracking_lib/SelfAssessment/SelfAssessment.h"

namespace ttb
{

struct CycleData
{
  Duration _totDuration{ 0s };
  Duration _predicationDuration{ 0s };
  Duration _postProcessPredictionDuration{ 0s };
  Duration _staticBirthDuration{ 0s };
  Duration _calcInnovationDuration{ 0s };
  Duration _groupingDuration{ 0s };
  Duration _updateDuration{ 0s };
  Duration _dynmicBirthDuration{ 0s };
  Duration _postProcessUpdateDuration{ 0s };
  Duration _clutterRateDuration{ 0s };
  Duration _detectionProbDuration{ 0s };
  Duration _mergeDuration{ 0s };
};
/// This struct specify the data, that should be profiled on this level
struct LMB_IC_Tracker_ProfilerData
{
  Time _time;
  Duration _cycleDuration{ 0s };
  std::map<MeasModelId, CycleData> _measCycle{};
  std::size_t _numTracks{ 0 };
  std::size_t _numEstimatedTracks{ 0 };
  // ...
};
std::string to_string(LMB_IC_Tracker_ProfilerData const& data);
std::string to_stringStatistics(std::vector<LMB_IC_Tracker_ProfilerData> const& datas);

/// Labeled Multi Bernoulli Iterator Corrector filter
class LMB_IC_Tracker final : public BaseTracker
{
public:
  explicit LMB_IC_Tracker(TTBManager* manager);
  void cycle(Time time, std::vector<MeasurementContainer>&& measContainers) override;
  [[nodiscard]] std::vector<State> getEstimate() const override;
  void reset() override;
  [[nodiscard]] TTBManager* manager() const override;
  [[nodiscard]] Time time() const override;
  [[nodiscard]] FILTER_TYPE type() const override;
  /// add new tracks to the _dist_LMB
  void addBirthTracks(MeasurementContainer const& measurementContainer,
                      std::map<MeasurementId, Probability> const& rzMap);
  TTBManager* _manager;
  /// The current time of the Filter
  Time _time{ 0s };
  /// LMB MOT Distribution
  LMBDistribution _dist_LMB;
  /// Detection and Clutter Estimation
  sa::ParameterEstimation _parameter_estimation;
};

}  // namespace ttb