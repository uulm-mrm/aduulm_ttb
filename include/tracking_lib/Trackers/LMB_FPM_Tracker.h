#pragma once

#include "tracking_lib/Trackers/BaseTracker.h"
// #####################################################################################################################
#include "tracking_lib/MultiObjectStateDistributions/LMBDistribution.h"
#include "tracking_lib/MultiObjectStateDistributions/GLMBDistribution.h"
#include "tracking_lib//Graph/Graph.h"
#include "tracking_lib/TTTFilters/TTTUncorrelatedTracks.h"
#include "tracking_lib/SelfAssessment/SelfAssessment.h"

namespace ttb
{
struct MeasurementCycleData
{
  Duration _totDuration;
  Duration _groupingDuration;
  Duration _updateDuration;
  Duration _mergeDuration;
  Duration _dynamicBirthDuration;
};
/// This struct specify the data, that should be profiled on this level
struct LMB_FPM_Tracker_ProfilerData
{
  Time _time{ 0s };
  Duration _cycleDuration{ 0s };
  Duration _predictionDuration{ 0s };
  Duration _staticBirthDuration{ 0s };
  std::map<MeasModelId, Duration> _innovationDuration{};
  std::map<MeasModelId, MeasurementCycleData> _measCycle{};
  Duration _postProcessUpdateDuration{ 0s };
  Duration _fusionDurationTotal{ 0s };
  Duration _fusionDuration{ 0s };
  Duration _fusionDynamicBirthDuration{ 0s };
  std::size_t _numTracks{ 0 };
  std::size_t _numEstimatedTracks{ 0 };
  // ...
};

std::string to_string(LMB_FPM_Tracker_ProfilerData const& data);
std::string to_stringStatistics(std::vector<LMB_FPM_Tracker_ProfilerData> const& datas);

/// Labeled Multi Bernoulli (LMB) filter using Fast Product Multi-Sensor (FPM) LMB multi-sensor update or PM LMB
/// multi-sensor update.
class LMB_FPM_Tracker final : public BaseTracker
{
public:
  explicit LMB_FPM_Tracker(TTBManager* manager);

  void cycle(Time time, std::vector<MeasurementContainer>&& measContainerList) override;
  /// performs multi-sensor update described in
  /// C. Hermann, M. Herrmann, T. Griebel, M. Buchholz and K. Dietmayer, "The Fast Product Multi-Sensor Labeled
  /// Multi-Bernoulli Filter," 2023 26th International Conference on Information Fusion (FUSION), Charleston, SC, USA,
  /// 2023, pp. 1-8, doi: 10.23919/FUSION52260.2023.10224189. https://ieeexplore.ieee.org/document/10224189
  void fpm_update(std::vector<MeasurementContainer>&& measContainerList, LMB_FPM_Tracker_ProfilerData& profilerData);
  /// performs multi-sensor update described in
  /// M. Herrmann, T. Luchterhand, C. Hermann and M. Buchholz, "The Product Multi-Sensor Labeled Multi-Bernoulli
  /// Filter," 2023 26th International Conference on Information Fusion (FUSION), Charleston, SC, USA, 2023, pp. 1-8,
  /// doi: 10.23919/FUSION52260.2023.10224121. https://ieeexplore.ieee.org/document/10224121
  void pm_update(std::vector<MeasurementContainer>&& measContainerList, LMB_FPM_Tracker_ProfilerData& profilerData);

  [[nodiscard]] std::vector<State> getEstimate() const override;

  void reset() override;

  [[nodiscard]] TTBManager* manager() const override;

  [[nodiscard]] Time time() const override;

  [[nodiscard]] FILTER_TYPE type() const override;

  // #####################################################################################################################

  /// This function initializes birth tracks based on a static birth
  void addStaticBirthTracks(MeasurementContainer const& measurementContainer,
                            std::map<MeasurementId, double> const& rzMap);

  /// This function initializes birth tracks based on a dynamic birth model, given a measurementContainer and
  /// an appropriate rzMap
  [[nodiscard]] tttfusion::TracksSensor addDynamicBirthTracks(const LMBDistribution& updatedLmbDist,
                                                              MeasurementContainer const& measurementContainer,
                                                              std::map<MeasurementId, double> const& rzMap) const;

  /// fuses dynamic birth tracks of single-sensors
  [[nodiscard]] LMBDistribution multiSensorDynamicBirth(std::vector<tttfusion::TracksSensor>&& birth_lmbs,
                                                        Duration& dynamicBirthTime) const;

  /// needed for pruning of state distribution
  template <class Policy>
  std::size_t prune_if(Policy pred)
  {
    std::size_t const before = _dist_LMB._tracks.size();
    _dist_LMB._tracks.erase(std::remove_if(_dist_LMB._tracks.begin(), _dist_LMB._tracks.end(), pred),
                            _dist_LMB._tracks.end());
    return before - _dist_LMB._tracks.size();
  }

  /// postprocessing of state distribution (merge components, remove components with small weight, etc...)
  void postProcessStateDist();

  TTBManager* _manager;
  /// current time of the Filter
  Time _time{ 0s };
  /// LMB MOT Distribution
  LMBDistribution _dist_LMB;
  /// parameter estimation
  sa::ParameterEstimation _parameter_estimation;
};

}  // namespace ttb
