#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"

namespace ttb::classification
{

/// Represent the classification of a measurement.
struct MeasClassification
{
  /// return the probability of typ if typ is in m_probs, or 1-sum(m_probs),
  [[nodiscard]] std::pair<double, bool> getProb(CLASS typ) const;
  /// number of classes in this measurement
  [[nodiscard]] std::size_t getSize() const;
  /// class with the highest probability
  [[nodiscard]] CLASS getEstimate() const;
  /// string representation
  [[nodiscard]] std::string toString() const;
  /// data
  std::map<CLASS, Probability> m_probs;
};

}  // namespace ttb::classification