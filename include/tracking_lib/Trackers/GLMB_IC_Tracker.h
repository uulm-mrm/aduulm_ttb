#pragma once

#include "tracking_lib/Trackers/BaseTracker.h"
// #####################################################################################################################
#include "tracking_lib/MultiObjectStateDistributions/GLMBDistribution.h"
#include "tracking_lib/States/State.h"
#include "tracking_lib/Measurements/MeasurementContainer.h"
#include "tracking_lib/SelfAssessment/SelfAssessment.h"

namespace ttb
{

/// Generalized Labeled Multi Bernoulli Iterator Corrector filter
class GLMB_IC_Tracker final : public BaseTracker
{
public:
  explicit GLMB_IC_Tracker(TTBManager* manager);
  void cycle(Time time, std::vector<MeasurementContainer>&& measContainers) override;
  [[nodiscard]] std::vector<State> getEstimate() const override;
  void reset() override;
  [[nodiscard]] TTBManager* manager() const override;
  [[nodiscard]] Time time() const override;
  [[nodiscard]] FILTER_TYPE type() const override;
  TTBManager* _manager;
  /// The current time of the Filter
  Time _time{ 0s };
  /// the underlying distribution
  GLMBDistribution _distribution;
  /// Estimation for detection and clutter rate
  sa::ParameterEstimation _parameter_estimation;
};

}  // namespace ttb