#pragma once

#include "tracking_lib/TTBTypes/Concepts.h"

namespace ttb
{

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

class EvalTTTFilter
{
public:
  explicit EvalTTTFilter(TTBManager* manager);

  void cycle(std::vector<DataSourceContainer> trackContainers);

  [[nodiscard]] std::vector<State> get_estimate() const;

  [[nodiscard]] TTBManager* manager() const;

  [[nodiscard]] Time time() const;

  [[nodiscard]] TTT_FILTER_TYPE type() const;

  void reset();

  TTBManager* _manager;
  std::map<SourceId, DataSourceContainer> _reference_tracks;
  std::map<std::pair<SourceId, SourceId>, std::vector<EvalData>> _eval;  ///< Evaluation of (SourceId,
                                                                         ///< EvaluationSourceId) -> Evaluation
  Time _time = Time{ 0s };
};

static_assert(CPTTracker<EvalTTTFilter>);

}  // namespace ttb