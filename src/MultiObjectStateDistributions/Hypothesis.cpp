#include "tracking_lib/MultiObjectStateDistributions/Hypothesis.h"

namespace ttb
{

IDGenerator<HypothesisId> Hypothesis::_idGenerator{};

Hypothesis::Hypothesis(std::vector<StateId> tracks, double weight_log, HypothesisId origin)
  : _origin_id{ origin }, _weightLog{ weight_log }, _tracks{ std::move(tracks) }
{
  std::ranges::sort(_tracks);
}

std::string Hypothesis::toString(std::string const& prefix) const
{
  std::string out = prefix + "Hypothesis\n" + prefix + "|\tId " + std::to_string(_id.value_) + "|\tOriginId " +
                    std::to_string(_origin_id.value_) + "\n" + prefix +
                    "|\t#Tracks: " + std::to_string(_tracks.size()) + "\n" + prefix + "|\tWeight " +
                    std::to_string(getWeight()) + "\n" + prefix + "|\tTracks: ";
  for (StateId id : _tracks)
  {
    out += std::to_string(id.value_) + " ";
  }
  out += '\n';
  return out;
}

double Hypothesis::getWeight() const
{
  return std::exp(_weightLog);
}

}  // namespace ttb