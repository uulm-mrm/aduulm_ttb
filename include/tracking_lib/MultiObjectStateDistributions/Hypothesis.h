#pragma once

#include "tracking_lib//TTBTypes/TTBTypes.h"

namespace ttb
{

/// Represents one (1) GLMB Hypotheses
class Hypothesis
{
public:
  Hypothesis(std::vector<StateId> tracks, double weight_log, HypothesisId origin = NO_HYPOTHESIS_ID_HISTORY);
  /// string representation
  [[nodiscard]] std::string toString(std::string const& prefix = "") const;
  /// weight of the hypothesis
  [[nodiscard]] double getWeight() const;
  HypothesisId _id{ _idGenerator.getID() };
  /// The hypothesis id of which this hypothesis originated
  HypothesisId _origin_id = NO_HYPOTHESIS_ID_HISTORY;
  /// Weight of the Hypothesis (log-space)
  double _weightLog{ std::numeric_limits<double>::quiet_NaN() };
  /// List with tracks
  std::vector<StateId> _tracks;
  static IDGenerator<HypothesisId> _idGenerator;
};

}  // namespace ttb