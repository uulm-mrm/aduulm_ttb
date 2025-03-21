#include "tracking_lib/Classification/MeasClassification.h"
// ######################################################################################################################
#include <algorithm>
#include <numeric>

namespace ttb::classification
{

std::pair<double, bool> MeasClassification::getProb(CLASS typ) const
{
  if (auto const elem_it = std::ranges::find_if(m_probs, [&](auto other) { return typ == other.first; });
      elem_it != m_probs.end())
  {
    return { elem_it->second, true };
  }
  double const sum = std::accumulate(
      std::begin(m_probs), std::end(m_probs), 0.0, [](double old, auto const& entry) { return old + entry.second; });
  return { 1 - sum, false };
}

std::size_t MeasClassification::getSize() const
{
  return m_probs.size();
}

CLASS MeasClassification::getEstimate() const
{
  if (const auto max_it =
          std::ranges::max_element(m_probs, [](auto const& a, auto const& b) { return a.second < b.second; });
      max_it != m_probs.end())
  {
    return max_it->first;
  }
  return CLASS::NOT_CLASSIFIED;
}

std::string MeasClassification::toString() const
{
  std::string out;
  for (auto const& [clazz, prob] : m_probs)
  {
    out += "Class " + to_string(clazz) + ": " + std::to_string(prob) + "\n";
  }
  return out;
}

}  // namespace ttb::classification