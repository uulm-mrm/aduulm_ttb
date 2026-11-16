#pragma once

#include "tracking_lib/DataContainer/DataSourceContainer.h"
#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/MultiObjectStateDistributions/Hypothesis.h"
#include "tracking_lib/StateModels/PredictionInformation.h"

namespace ttb
{

template <class UDT>
class MBMDistribution final
{
public:
  explicit MBMDistribution(TTBManager<UDT>* manager);

  void predict(Duration deltaT, PredictionInformation const& pred);

  void update(DataSourceContainer<UDT> const& measurements);

  struct TrimmedCostMatrixResult
  {
    Matrix<double> cost_matrix_trimmed;
    std::vector<std::pair<int, int>> fixed_assignments;
    double cost_fixed = 0.0;
    std::vector<int> original_row_indices;
    std::vector<int> original_col_indices;
  };
  [[nodiscard]] TrimmedCostMatrixResult
  buildCostMatrix(DataSourceContainer<UDT> const& measurements,
                  std::vector<StateId> const& tracks,
                  std::vector<StateId> const& new_tracks,
                  std::unordered_map<StateId, std::unordered_map<MeasurementId, UDT>> const& meas_log_likelihood,
                  std::unordered_map<StateId, State<UDT>> const& updated_bernoullis,
                  std::unordered_map<StateId, std::map<MeasurementId, StateId>> const& updated_tracks_map) const;

  [[nodiscard]] TrimmedCostMatrixResult trimCostMatrix(Matrix<double> const& cost_matrix) const;

  void add_hypothesis(Hypothesis<UDT> hyp,
                      std::unordered_map<HypothesisId, Hypothesis<UDT>>& hypotheses,
                      std::unordered_map<std::vector<StateId>, HypothesisId>& state_2_hypotheses);

  void normalizeGlobalHypotheses();
  [[nodiscard]] bool is_valid() const;

  void pruningGlobalHypotheses(double pruningThreshold);
  void cappingGlobalHypotheses(std::size_t maxNumber);
  void updateGlobalHypotheses(std::vector<StateId> pruned_ids);
  [[nodiscard]] std::vector<State<UDT>> recycleBernoulliStates();
  [[nodiscard]] std::vector<StateId> pruningBernoulliBasis(double pruningThreshold);
  [[nodiscard]] std::vector<StateId> removeEmptyStates();
  void cleanupBernoulliBasis();

  void projectToPMB();

  [[nodiscard]] std::vector<BernoulliState<UDT>> getEstimate() const;

  TTBManager<UDT>* _manager;

  std::unordered_map<StateId, State<UDT>> _bernoulli_basis;
  std::unordered_map<StateId, State<UDT>> _new_bernoullis;
  std::unordered_map<HypothesisId, Hypothesis<UDT>> _global_hypotheses;
  std::unordered_map<std::vector<StateId>, HypothesisId> _state_2_hypotheses;
};

}  // namespace ttb
