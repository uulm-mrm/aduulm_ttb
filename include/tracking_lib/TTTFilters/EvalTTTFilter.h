#pragma once

#include "tracking_lib/TTTFilters/BaseTTTFilter.h"

namespace ttb
{

class TTBManager;

struct EvalData
{
  Time time;
  std::vector<std::tuple<Vector2, Vector2, double>> distances;  ///< all distances of tracks assigned to the reference
                                                                ///< tracks, normalized distances, v_abs difference
  std::size_t unassigned_reference_tracks;                      ///< number of not assigned reference tracks
  std::size_t unassigned_eval_tracks;                           ///< number of not assigned evaluation tracks
};

std::string to_string(EvalData const& data);
std::string to_stringStatistics(std::vector<EvalData> const& data);

class EvalTTTFilter : public BaseTTTFilter
{
public:
  explicit EvalTTTFilter(TTBManager* manager);

  void cycle(Time time, std::vector<StateContainer> trackContainers) override;

  [[nodiscard]] std::vector<State> getEstimate() const override;

  [[nodiscard]] TTBManager* manager() const override;

  [[nodiscard]] Time time() const override;

  [[nodiscard]] TTT_FILTER_TYPE type() const override;

  void reset() override;

  TTBManager* _manager;
  std::map<SourceId, StateContainer> _reference_tracks;
  std::map<std::pair<SourceId, SourceId>, std::vector<EvalData>> _eval;  ///< Evaluation of (SourceId,
                                                                         ///< EvaluationSourceId) -> Evaluation
  Time _time = Time{ 0s };
};

}  // namespace ttb