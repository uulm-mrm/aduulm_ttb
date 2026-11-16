#include "tracking_lib/MultiObjectStateDistributions/MBMDistribution.h"

#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/MultiObjectStateDistributions/Utils.h"
#include "tracking_lib/MultiObjectStateDistributions/Hypothesis.h"

#include <stack>
#include <tracking_lib/Misc/MurtyAlgorithm.h>
#include "tracking_lib/Misc/GibbsSampler.h"
#include "tracking_lib/Misc/HungarianMethod.h"

#include <boost/algorithm/string/join.hpp>

namespace ttb
{
constexpr const auto tracy_color = tracy::Color::DarkSalmon;

template class MBMDistribution<double>;
template class MBMDistribution<AD>;

template <class UDT>
MBMDistribution<UDT>::MBMDistribution(TTBManager<UDT>* manager) : _manager{ manager }
{
}

template <class UDT>
void MBMDistribution<UDT>::predict(Duration deltaT, PredictionInformation const& pred)
{
  assert(is_valid());
  LOG_DEB("MBMDistribution::predict with " << _bernoulli_basis.size() << " Bernoullis");
  {
    std::unordered_map<StateId, State<UDT>> predicted_tracks = utils::predict<UDT>(
        _bernoulli_basis, deltaT, pred._egoMotion, [](auto const& pair) -> State<UDT> const& { return pair.second; });
    std::unordered_map<HypothesisId, Hypothesis<UDT>> predicted_hyps;
    std::unordered_map<std::vector<StateId>, HypothesisId> predicted_state_2_hypotheses;

    for (auto const& [id, old_hyp] : _global_hypotheses)
    {
      Hypothesis predicted_hyp = old_hyp;
      predicted_hyp._tracks.clear();
      for (StateId old_id : old_hyp._tracks)
      {
        predicted_hyp._tracks.push_back(predicted_tracks.at(old_id)._id);
      }
      add_hypothesis(std::move(predicted_hyp), predicted_hyps, predicted_state_2_hypotheses);
    }
    _global_hypotheses = std::move(predicted_hyps);
    _state_2_hypotheses = std::move(predicted_state_2_hypotheses);
    _bernoulli_basis.clear();
    for (auto& [old_id, state] : predicted_tracks)
    {
      StateId const id = state._id;
      _manager->database().state_existences.emplace(
          id,
          _manager->get_state_existence_probability(old_id) *
              _manager->getPersistenceModel().getPersistenceProbability(state, deltaT, pred._roi));
      _bernoulli_basis.emplace(id, std::move(state));
    }
  }
  updateGlobalHypotheses(removeEmptyStates());
  cleanupBernoulliBasis();
  assert(is_valid());
}

template <class UDT>
void MBMDistribution<UDT>::update(DataSourceContainer<UDT> const& measurements)
{
  assert(is_valid());
  std::vector<std::tuple<StateDetection<UDT>,
                         std::vector<std::tuple<StateId, StateId, State<UDT>, UDT>>,
                         std::vector<MeasClutter<UDT>>>>
      bernoulli_updates =
          utils::innovate(_bernoulli_basis, measurements, [](auto const& id_state) { return id_state.second; });
  LOG_DEB("MBMDistribution::update");

  std::size_t maxNumUpdateHypotheses = _manager->params()->pmbm_distribution.mbm.max_number_hypotheses;

  std::unordered_map<StateId, State<UDT>> updated_bernoullis;  // Updated Bernoulli components for this cycle
  std::unordered_map<HypothesisId, Hypothesis<UDT>> updated_hypotheses;
  std::unordered_map<std::vector<StateId>, HypothesisId> updated_state_2_hypotheses;
  std::unordered_map<StateId, std::map<MeasurementId, StateId>> updated_tracks_map;  // which track and measurement
                                                                                     // created the updated track
  std::unordered_map<StateId, std::unordered_map<MeasurementId, UDT>> meas_log_likelihood;
  /*
   * Track Oriented local hypothesis update
   */

  // 1. UPDATE ALL (OLD) BERNOULLI COMPONENTS
  // iterate over all Bernoulli components
  for (auto& [state_detection, state_meas_updates, meas_infos] : bernoulli_updates)
  {
    LOG_DEB("_updated dists size: " << state_meas_updates.size());
    UDT const detectionProbability = state_detection.detection_probability;
    UDT const misdetectionProbability = 1 - detectionProbability;
    // iterate over Innovation components of one Bernoulli
    for (auto& [old_state_id, meas_id, updated_state, state_meas_log_likelihood] : state_meas_updates)
    {
      meas_log_likelihood[old_state_id][meas_id] = state_meas_log_likelihood;
      StateId const id = updated_state._id;
      // update map: (old) Bernoulli with ID *origin_id* is updated by the measurement with ID *meas_id* into the new
      // Bernoulli with ID *id*.
      updated_tracks_map[old_state_id].emplace(meas_id, id);
      LOG_DEB("Track with id " << old_state_id << " updated with Meas id: " << meas_id << " has new id: " << id);
      assert([&] {
        if (not _bernoulli_basis.contains(old_state_id))
        {
          for (auto const& [id, state] : _bernoulli_basis)
          {
            LOG_FATAL(state.toString());
          }
          return false;
        }
        return true;
      }());
      UDT const existenceProbability = _manager->get_state_existence_probability(_bernoulli_basis.at(old_state_id)._id);
      if (meas_id == NOT_DETECTED)
      {
        _manager->database().state_existences.emplace(
            updated_state._id,
            (existenceProbability * misdetectionProbability) /
                (1 - existenceProbability + existenceProbability * misdetectionProbability));
        updated_state._log_weight = std::log(1 - existenceProbability + existenceProbability * misdetectionProbability);
        assert([&] {  // NOLINT
          if (std::isnan(updated_state._log_weight))
          {
            LOG_FATAL("exProb " << existenceProbability);
            LOG_FATAL("misProb" << misdetectionProbability);
            LOG_FATAL("updated dist " << updated_state.toString());
            LOG_FATAL("origin " << _bernoulli_basis.at(old_state_id).toString());
            return false;
          }
          return true;
        }());
        assert(!std::isnan(updated_state._log_weight));
      }
      else  // if there is a measurement associated to the origin state
      {
        _manager->database().state_existences.emplace(updated_state._id, 1);
        updated_state._log_weight = log(existenceProbability) + log(detectionProbability) + state_meas_log_likelihood;
        assert(!std::isnan(updated_state._log_weight));
      }

      updated_bernoullis.emplace(id, std::move(updated_state));
    }
  }
  LOG_DEB("# of updated tracks moved to updated_bernoullis: " << updated_tracks_map.size());

  // 2. GETTING STATE IDS OF ALL NEW BERNOULLIS
  std::vector<StateId> new_tracks;
  for (StateId state_id : _new_bernoullis | std::views::keys)
  {
    new_tracks.emplace_back(state_id);
  }

  LOG_DEB("# of old global hypotheses: " << _global_hypotheses.size());
  // 3. GLOBAL HYPOTHESES MANAGEMENT
  if (_global_hypotheses.empty())
  {
    LOG_DEB("Generate initial and empty hypothesis");
    add_hypothesis(Hypothesis<UDT>(std::vector<StateId>{}, UDT{ 0 }), _global_hypotheses, _state_2_hypotheses);
  }
  LOG_INF("# of measurements: " << measurements._measurements.size());
  for (const auto& [hyp_id, hyp] : _global_hypotheses)
  {
    LOG_DEB("Update hypothesis: " << hyp.toString());

    // MeasurementContainer measurements_copy = measurements;
    // std::vector<StateId> old_tracks_copy = hyp._tracks;
    auto costMatrixStruct = buildCostMatrix(
        measurements, hyp._tracks, new_tracks, meas_log_likelihood, updated_bernoullis, updated_tracks_map);
    // Rows with no finite candidate make this parent hypothesis impossible.
    // Skip just this parent, other global hypotheses may still explain the scan.
    if (costMatrixStruct.fixed_assignments.size() + costMatrixStruct.original_row_indices.size() !=
        measurements._measurements.size())
    {
      LOG_WARN("Skipping globally infeasible PMBM parent hypothesis " << hyp_id);
      continue;
    }
    Matrix<double> costMatrix = costMatrixStruct.cost_matrix_trimmed;
    LOG_DEB("Cost matrix created: \n" << costMatrix);

    int col_size = 1;  // of assignment matrix later
    MatrixIndex calculated_assignments;
    Vector<double> costs;

    LOG_DEB("after declaration of calc_assig: " << calculated_assignments.rows() << ", "
                                                << calculated_assignments.cols());

    if (costMatrix.rows() > 0 && costMatrix.cols() > 0)
    {
      auto const numberOfKbest =
          static_cast<std::size_t>(std::ceil(static_cast<double>(maxNumUpdateHypotheses) * std::exp(hyp._weightLog)));
      LOG_DEB("There should be " << numberOfKbest << " new k-best hypotheses");

      std::size_t const k{ numberOfKbest };
      switch (_manager->params()->pmbm_distribution.mbm.assignment_method)
      {
        case GLMB_ASSIGNMENT_METHOD::MURTY:
        {
          auto [assignments, cost] = murty::getAssignments(costMatrix, k);
          calculated_assignments = assignments.cast<Index>();
          costs = cost;
          break;
        }
        case GLMB_ASSIGNMENT_METHOD::GIBBS_SAMPLING:
        {
          LOG_DEB("# hyp._tracks: " << hyp._tracks.size());
          LOG_DEB("# meas_data: " << measurements._measurements.size());
          LOG_DEB("# of new tracks: " << new_tracks.size());

          auto [assignment, cost] = hungarian::solve(costMatrix);

          std::tie(calculated_assignments, costs) =
              gibbs::sample_assignments(costMatrix, k, assignment, k * 5, k * 100);
          break;
        }
        default:
        {
          assert(false);
          std::unreachable();
        }
      }

      LOG_DEB("Calculated Assignments: \n" << calculated_assignments);
      LOG_DEB("Costs of assignments: \n" << costs);

      if (calculated_assignments.rows() == 0 || calculated_assignments.cols() == 0)
      {
        LOG_WARN("No feasible ranked assignment for PMBM parent hypothesis " << hyp_id);
        continue;
      }
      col_size = calculated_assignments.cols();
    }
    else
    {
      LOG_DEB("All tracks are misdetected, no need for assignment");
      // col_size = 1;
    }

    // Create assignments out of fixed and calculated assignments
    Eigen::MatrixXi assignments(measurements._measurements.size(), col_size);
    assignments.setConstant(-1);
    LOG_DEB("Size of assignments: " << assignments.rows() << ", " << assignments.cols());

    // adding fixes assignments to overall assignments matrix
    auto fixed_assignments = costMatrixStruct.fixed_assignments;
    for (auto [row, col] : fixed_assignments)
    {
      assignments.row(row).setConstant(col);
    }
    LOG_DEB("assignments after adding fixed: \n" << assignments);

    auto original_rows = costMatrixStruct.original_row_indices;
    auto original_cols = costMatrixStruct.original_col_indices;

    // adding calculated assignments to overall assignments matrix
    for (Index calcCol = 0; calcCol < calculated_assignments.cols(); calcCol++)
    {
      for (Index calcRow = 0; calcRow < calculated_assignments.rows(); calcRow++)
      {
        auto calculated_assignment = original_cols[calculated_assignments(calcRow, calcCol)];
        assignments(original_rows[calcRow], calcCol) = calculated_assignment;
      }
    }

    LOG_DEB("Final assignments:\n" << assignments);

    assert([&]() {  // NOLINT
      for (Eigen::Index row = 0; row < assignments.rows(); ++row)
      {
        for (Eigen::Index col = 0; col < assignments.cols(); ++col)
        {
          if (assignments(row, col) < 0)
          {
            LOG_FATAL("Invalid PMBM assignment: unassigned measurement row");
            return false;
          }
        }
      }
      return true;
    }());

    // Creating new global Hypotheses out of Assignments
    for (Index solutionIndex = 0; solutionIndex < assignments.cols(); solutionIndex++)
    {
      assert([&] {  // NOLINT
        std::unordered_set<Index> occupied_old_tracks;

        for (Index row = 0; row < assignments.rows(); ++row)
        {
          Index const track_index = assignments(row, solutionIndex);

          if (track_index < static_cast<Index>(hyp._tracks.size()) && !occupied_old_tracks.insert(track_index).second)
          {
            LOG_FATAL("PMBM assignment uses an old track more than once");
            return false;
          }
        }

        return true;
      }());

      std::vector<StateId> updatedTrackList;
      std::vector<StateId> updatedTracksOf;
      UDT new_log_weight = hyp._weightLog;

      for (auto [meas_index, measurement] : std::views::enumerate(measurements._measurements))
      {
        MeasurementId meas_id = measurement._id;
        std::size_t trackIndex = assignments(meas_index, solutionIndex);
        // LOG_DEB("hyp._tracks.size(): " << hyp._tracks.size());

        // if the measurement is assigned to a new_bernoulli
        if (trackIndex >= hyp._tracks.size())
        {
          std::size_t newBernoulliIndex = trackIndex - hyp._tracks.size();
          StateId assignedStateIndex = new_tracks.at(newBernoulliIndex);
          new_log_weight += _new_bernoullis.at(assignedStateIndex)._log_weight;
          updatedTrackList.emplace_back(assignedStateIndex);
          continue;
        }
        LOG_DEB("trackIndex: " << trackIndex << ", track_id: " << hyp._tracks.at(trackIndex)
                               << ", measurement: " << meas_index << ", meas_id: " << meas_id.value_);

        // if the measurement is assigned to an old bernoulli from hypothesis' tracks
        updatedTracksOf.emplace_back(hyp._tracks.at(trackIndex));
        StateId assignedStateIndex = updated_tracks_map.at(hyp._tracks.at(trackIndex)).at(meas_id);
        new_log_weight += updated_bernoullis.at(assignedStateIndex)._log_weight;
        updatedTrackList.push_back(assignedStateIndex);
      }
      // adding misdetected bernoullis from hypothesis' tracks
      for (auto& track : hyp._tracks)
      {
        if (!std::ranges::contains(updatedTracksOf, track))
        {
          StateId assignedStateIndex = updated_tracks_map.at(track).at(NOT_DETECTED);
          new_log_weight += updated_bernoullis.at(assignedStateIndex)._log_weight;
          updatedTrackList.push_back(assignedStateIndex);
        }
      }
      // adding the new hypothesis to new hypotheses
      add_hypothesis(Hypothesis(std::move(updatedTrackList), new_log_weight, hyp_id),
                     updated_hypotheses,
                     updated_state_2_hypotheses);
    }
  }

  assert([&] {  // NOLINT
    if (updated_hypotheses.empty())
    {
      LOG_FATAL("No feasible PMBM global hypothesis for this measurement scan");
      return false;
    }
    return true;
  }());

  _global_hypotheses = std::move(updated_hypotheses);
  _state_2_hypotheses = std::move(updated_state_2_hypotheses);
  _bernoulli_basis = std::move(updated_bernoullis);

  LOG_DEB("# of new global hypotheses: " << _global_hypotheses.size());
  normalizeGlobalHypotheses();
  assert(std::ranges::all_of(_bernoulli_basis | std::views::values, std::identity{}, &State<UDT>::isValid));

  double pruning_threshold_hypotheses = _manager->params()->pmbm_distribution.mbm.pruning_threshold_hypotheses;
  pruningGlobalHypotheses(pruning_threshold_hypotheses);
  LOG_DEB("# of new global hypotheses after pruning: " << _global_hypotheses.size());
  assert(std::ranges::all_of(_bernoulli_basis | std::views::values, std::identity{}, &State<UDT>::isValid));

  cappingGlobalHypotheses(maxNumUpdateHypotheses);
  // Keep reverse index consistent after hypothesis pruning and capping.
  updateGlobalHypotheses({});
  LOG_DEB("# of new global hypotheses after capping: " << _global_hypotheses.size());
  assert(std::ranges::all_of(_bernoulli_basis | std::views::values, std::identity{}, &State<UDT>::isValid));

  normalizeGlobalHypotheses();

  // Cleaning up
  _bernoulli_basis.merge(_new_bernoullis);
  assert(_new_bernoullis.empty());

  // Pruning Bernoullis
  LOG_DEB("# of Bernoullis before Pruning: " << _bernoulli_basis.size());
  std::vector<StateId> pruned_ids =
      pruningBernoulliBasis(_manager->params()->pmbm_distribution.mbm.pruning_threshold_bernoulli);
  updateGlobalHypotheses(pruned_ids);
  LOG_DEB("# of Bernoullis after Pruning: " << _bernoulli_basis.size());

  cleanupBernoulliBasis();
  LOG_DEB("# of cleaned up Bernoulli Ids after pruning: " << _bernoulli_basis.size());
  assert(is_valid());
}

template <class UDT>
std::vector<BernoulliState<UDT>> MBMDistribution<UDT>::getEstimate() const
{
  LOG_DEB("MBMDistribution::getEstimate");
  assert(is_valid());

  double thresholdGamma = _manager->params()->pmbm_distribution.estimation_threshold;
  std::vector<BernoulliState<UDT>> estimates;
  if (_global_hypotheses.empty())
  {
    return {};
  }
  auto max_it = std::ranges::max_element(
      _global_hypotheses, [](const auto& a, const auto& b) { return a.second._weightLog < b.second._weightLog; });
  LOG_INF("Hypothesis with max log_weight is: " << max_it->second.toString());
  for (StateId state_id : max_it->second._tracks)
  {
    LOG_DEB("Try estimate with ID: " << state_id.value_ << ", Label: " << _bernoulli_basis.at(state_id)._label.value_
                                     << ", EP: " << _manager->get_state_existence_probability(state_id));
    if (_manager->get_state_existence_probability(state_id) > thresholdGamma)
    {
      estimates.emplace_back(_bernoulli_basis.at(state_id), _manager->get_state_existence_probability(state_id));
      LOG_DEB("Estimate with ID: " << state_id.value_);
      LOG_DEB(_bernoulli_basis.at(state_id).toString());
    }
  }
  return estimates;
}

template <class UDT>
void MBMDistribution<UDT>::projectToPMB()
{
  assert(is_valid());
  LOG_DEB("Starting PMB projection");

  std::unordered_map<Label, std::map<StateId, UDT>> label_2_track;
  std::map<StateId, State<UDT>> lookUpState;

  // 1. Grouping Bernoullis and their weights per Label/Track
  for (const auto& hyp : _global_hypotheses | std::views::values)
  {
    UDT weight = std::exp(hyp._weightLog);

    for (auto const& state_id : hyp._tracks)
    {
      State<UDT> const& s = _bernoulli_basis.at(state_id);
      lookUpState.emplace(state_id, s);
      Label label = s._label;
      label_2_track[label][s._id] += weight;
    }
  }

  _bernoulli_basis.clear();

  // 2. For each Track: weighted fusion/merge
  for (auto& state_weight_list : label_2_track | std::views::values)
  {
    assert(!state_weight_list.empty());
    auto [first_state, first_weight] = *state_weight_list.begin();
    State only_state = lookUpState.at(first_state);
    if (state_weight_list.size() == 1)
    {
      if (_manager->get_state_existence_probability(only_state._id) == 0.0)
      {
        LOG_DEB("eB is 0 -> continue");
        continue;
      }
      LOG_DEB("only 1 track with label: " << only_state._label.value_
                                          << ", prior eB: " << _manager->get_state_existence_probability(only_state._id)
                                          << ", summed weight: " << first_weight);
      double const only_state_existenceProbability = std::clamp(
          to_val(_manager->get_state_existence_probability(only_state._id)) * to_val(first_weight), 0.0, 1.0);
      _manager->database().state_existences.emplace(only_state._id, only_state_existenceProbability);
      _bernoulli_basis.emplace(only_state._id, only_state);
    }
    else
    {
      double r_fused = 0.0;
      std::vector<std::pair<UDT, StateId>> for_merging;
      for (auto const& [state_id, weight] : state_weight_list)
      {
        r_fused += to_val(weight) * to_val(_manager->get_state_existence_probability(lookUpState.at(state_id)._id));
        _manager->database().states.insert(lookUpState.at(state_id));
        for_merging.emplace_back(weight, state_id);
      }
      State fusedState = merge(*_manager, for_merging);
      double const fusedState_existenceProbability = std::clamp(r_fused, 0.0, 1.0);
      if (not fusedState.isEmpty())
      {
        assert(fusedState.isValid());
        StateId const id = fusedState._id;
        _manager->database().state_existences.emplace(id, fusedState_existenceProbability);
        _bernoulli_basis.emplace(id, std::move(fusedState));
      }
    }
  }

  // 3. Create only one hypothesis
  std::vector<StateId> final_state_ids;
  for (const auto& state_id : _bernoulli_basis | std::views::keys)
  {
    final_state_ids.emplace_back(state_id);
  }
  _global_hypotheses.clear();
  _state_2_hypotheses.clear();

  add_hypothesis(Hypothesis<UDT>(std::move(final_state_ids), UDT{ 0.0 }), _global_hypotheses, _state_2_hypotheses);

  LOG_DEB("Finished PMB projection: new global hypothesis with " << _bernoulli_basis.size() << " tracks");
  assert(is_valid());
}

template <class UDT>
MBMDistribution<UDT>::TrimmedCostMatrixResult MBMDistribution<UDT>::buildCostMatrix(
    DataSourceContainer<UDT> const& measurements,
    std::vector<StateId> const& tracks,
    std::vector<StateId> const& new_tracks,
    std::unordered_map<StateId, std::unordered_map<MeasurementId, UDT>> const& meas_log_likelihood,
    std::unordered_map<StateId, State<UDT>> const& updated_bernoullis,
    std::unordered_map<StateId, std::map<MeasurementId, StateId>> const& updated_tracks_map) const
{
  const std::size_t num_measurements = measurements._measurements.size();
  const std::size_t num_locals = tracks.size();
  const std::size_t num_births = new_tracks.size();

  const std::size_t total_cols = num_locals + num_births;

  Matrix<double> cost_matrix = Matrix<double>::Constant(
      static_cast<Index>(num_measurements), static_cast<Index>(total_cols), std::numeric_limits<double>::infinity());

  for (const auto& [meas_idx, measurement] : std::views::enumerate(measurements._measurements))
  {
    MeasurementId meas_id = measurement._id;

    // existing Bernoulli tracks
    for (const auto& [trackIndex, state_id] : std::views::enumerate(tracks))
    {
      State<UDT> const& track = _bernoulli_basis.at(state_id);
      LOG_DEB("Track: \n" << track.toString());

      // log_likelihood_detection must be -inf because murty searches for maximum and matrix element gets inverted
      double log_likelihood_detection = -std::numeric_limits<double>::infinity();
      double log_likelihood_misdetection = std::numeric_limits<double>::quiet_NaN();

      if (auto const inno = meas_log_likelihood.at(track._id).find(meas_id);
          inno != meas_log_likelihood.at(track._id).end())
      {
        assert(std::isfinite(inno->second));
        // Association cost must use the same complete detection weight
        // that is used to construct the resulting global hypothesis.
        StateId const detected_id = updated_tracks_map.at(state_id).at(meas_id);
        log_likelihood_detection = to_val(updated_bernoullis.at(detected_id)._log_weight);
      }

      assert(not std::isnan(updated_bernoullis.at(updated_tracks_map.at(state_id).at(NOT_DETECTED))._log_weight));
      log_likelihood_misdetection =
          to_val(updated_bernoullis.at(updated_tracks_map.at(state_id).at(NOT_DETECTED))._log_weight);

      cost_matrix(meas_idx, trackIndex) = -(log_likelihood_detection - log_likelihood_misdetection);
    }

    // new birth tracks (PPP)
    for (const auto& [trackIndex, state_id] : std::views::enumerate(new_tracks))
    {
      double log_likelihood_new_detected = -std::numeric_limits<double>::infinity();
      assert(_new_bernoullis.at(state_id)._meta_data._meas_update_weight.contains(measurements._source_id));
      assert(_new_bernoullis.at(state_id)._meta_data._meas_update_weight.at(measurements._source_id).size() == 1);
      if (_new_bernoullis.at(state_id)._meta_data._meas_update_weight.at(measurements._source_id).front().first ==
          meas_id)
      {
        log_likelihood_new_detected = to_val(_new_bernoullis.at(state_id)._log_weight);
      }

      cost_matrix(meas_idx, trackIndex + static_cast<long>(num_locals)) = -log_likelihood_new_detected;
    }
  }

  LOG_DEB("Cost Matrix before trimming:\n" << cost_matrix);

  auto trim_result = trimCostMatrix(cost_matrix);

  LOG_DEB("Trimmed cost matrix:\n" << trim_result.cost_matrix_trimmed);

  return trim_result;
}

template <class UDT>
MBMDistribution<UDT>::TrimmedCostMatrixResult
MBMDistribution<UDT>::trimCostMatrix(Matrix<double> const& cost_matrix) const
{
  int const rows = static_cast<int>(cost_matrix.rows());
  int const cols = static_cast<int>(cost_matrix.cols());

  std::vector<std::pair<int, int>> fixed_assignments;
  std::vector<int> remaining_rows;
  std::unordered_set<int> occupied_cols;

  double cost_fixed = 0.0;

  auto infeasible = []() -> TrimmedCostMatrixResult {
    return { .cost_matrix_trimmed = Matrix<double>(0, 0),
             .fixed_assignments = {},
             .cost_fixed = 0.0,
             .original_row_indices = {},
             .original_col_indices = {} };
  };

  // Identify forced assignments and impossible rows
  for (int i = 0; i < rows; ++i)
  {
    int valid_count = 0;
    int only_col = -1;

    for (int j = 0; j < cols; ++j)
    {
      if (std::isfinite(cost_matrix(i, j)))
      {
        ++valid_count;
        only_col = j;
      }
    }

    if (valid_count == 0)
    {
      return infeasible();
    }

    if (valid_count == 1)
    {
      if (!occupied_cols.insert(only_col).second)
      {
        // Two measurements require the same track
        return infeasible();
      }

      fixed_assignments.emplace_back(i, only_col);
      cost_fixed += cost_matrix(i, only_col);
    }
    else
    {
      remaining_rows.push_back(i);
    }
  }

  // Remove all columns already occupied by fixed assignments
  std::vector<int> remaining_cols;

  for (int j = 0; j < cols; ++j)
  {
    if (occupied_cols.contains(j))
    {
      continue;
    }

    for (int i : remaining_rows)
    {
      if (std::isfinite(cost_matrix(i, j)))
      {
        remaining_cols.push_back(j);
        break;
      }
    }
  }

  // Every remaining row needs at least one available candidate
  for (int i : remaining_rows)
  {
    bool has_candidate = false;

    for (int j : remaining_cols)
    {
      if (std::isfinite(cost_matrix(i, j)))
      {
        has_candidate = true;
        break;
      }
    }

    if (!has_candidate)
    {
      return infeasible();
    }
  }

  if (remaining_cols.size() < remaining_rows.size())
  {
    return infeasible();
  }

  Matrix<double> trimmed(remaining_rows.size(), remaining_cols.size());

  for (int i = 0; i < remaining_rows.size(); ++i)
  {
    for (int j = 0; j < remaining_cols.size(); ++j)
    {
      trimmed(i, j) = cost_matrix(remaining_rows[i], remaining_cols[j]);
    }
  }

  return { .cost_matrix_trimmed = std::move(trimmed),
           .fixed_assignments = std::move(fixed_assignments),
           .cost_fixed = cost_fixed,
           .original_row_indices = std::move(remaining_rows),
           .original_col_indices = std::move(remaining_cols) };
}

template <class UDT>
void MBMDistribution<UDT>::add_hypothesis(Hypothesis<UDT> hyp,
                                          std::unordered_map<HypothesisId, Hypothesis<UDT>>& hypotheses,
                                          std::unordered_map<std::vector<StateId>, HypothesisId>& state_2_hypotheses)
{
  std::ranges::sort(hyp._tracks);
  if (auto it = state_2_hypotheses.find(hyp._tracks); it != state_2_hypotheses.end())
  {
    // Hypotheses with identical local states represent the same event.
    UDT const a = hypotheses.at(it->second)._weightLog;
    UDT const b = hyp._weightLog;
    UDT const neg_inf = UDT{ -std::numeric_limits<double>::infinity() };
    if (a == neg_inf)
    {
      hypotheses.at(it->second)._weightLog = b;
    }
    else if (b != neg_inf)
    {
      UDT const hi = std::max(a, b);
      hypotheses.at(it->second)._weightLog = hi + std::log1p(std::exp(std::min(a, b) - hi));
    }
    return;
  }
  std::vector<StateId> tracks = hyp._tracks;
  HypothesisId hyp_id = hyp._id;
  hypotheses.emplace(hyp_id, std::move(hyp));
  state_2_hypotheses.emplace(std::move(tracks), hyp_id);
}

template <class UDT>
bool MBMDistribution<UDT>::is_valid() const
{
  for (auto const& [state_id, state] : _bernoulli_basis)
  {
    if (state_id != state._id)
    {
      LOG_FATAL("inconsistent id of state " << state.toString());
      LOG_FATAL(std::stacktrace::current());
      return false;
    }
    if (not state.isValid())
    {
      LOG_FATAL("invalid state in bernoulli basis " << state.toString());
    }
  }
  for (auto const& [hyp_id, hyp] : _global_hypotheses)
  {
    if (hyp_id != hyp._id)
    {
      LOG_FATAL("inconsistent id of hyp " << hyp.toString());
      LOG_FATAL(std::stacktrace::current());
      return false;
    }
    for (StateId id : hyp._tracks)
    {
      if (not _bernoulli_basis.contains(id))
      {
        LOG_FATAL("Hyp contains track that is not in the bernoullis");
        LOG_FATAL(std::stacktrace::current());
        return false;
      }
    }
  }
  return true;
}

template <class UDT>
void MBMDistribution<UDT>::normalizeGlobalHypotheses()
{
  LOG_DEB("MBMDistribution::normalizeGlobalHypotheses");

  assert(not _global_hypotheses.empty());

  std::vector<UDT> log_weights;
  for (auto& hypothesis : _global_hypotheses | std::views::values)
  {
    log_weights.push_back(hypothesis._weightLog);
  }

  UDT max_log = 0.0;
  if (!log_weights.empty())
  {
    max_log = *std::ranges::max_element(log_weights);
  }
  else
  {
    LOG_FATAL("log_weights is empty! Cannot normalize.");
    assert(false);
    return;
  }

  UDT denom = 1.0;
  for (UDT lw : log_weights)
    if (lw != max_log)
      denom += exp(lw - max_log);

  for (auto& hypothesis : _global_hypotheses | std::views::values)
  {
    hypothesis._weightLog = hypothesis._weightLog - max_log - log(denom);
  }

  // Postcondition: Normalized weights must sum to one.
  assert([&] {  // NOLINT
    UDT sum = 0.0;
    for (auto const& hypothesis : _global_hypotheses | std::views::values)
    {
      sum += std::exp(hypothesis._weightLog);
    }
    return std::isfinite(sum) && std::abs(sum - 1.0) < 1e-6;
  }());

  LOG_DEB("Normalized " << _global_hypotheses.size() << " global hypothesis weights.");
}

template <class UDT>
std::vector<State<UDT>> MBMDistribution<UDT>::recycleBernoulliStates()
{
  double recycling_threshold = _manager->params()->pmbm_distribution.mbm.recycle_threshold_bernoulli;
  std::unordered_map<StateId, State<UDT>> recycled_states;

  for (auto it = _bernoulli_basis.begin(); it != _bernoulli_basis.end();)
  {
    if (_manager->get_state_existence_probability(it->second._id) <
        recycling_threshold)  // && it->second._meta_data._numUpdates > 0)
    {
      // it->second._recycled = 1;
      recycled_states.emplace(it->first, it->second);
      it = _bernoulli_basis.erase(it);  // erase returns next iterator
    }
    else
    {
      ++it;
    }
  }

  for (auto& [new_id, new_state] : recycled_states)
  {
    UDT new_weight = 0.0;
    for (auto& [state_ids, hypothesis_id] : _state_2_hypotheses)
    {
      if (std::ranges::contains(state_ids, new_id))
      {
        if (_global_hypotheses.contains(hypothesis_id))
        {
          new_weight += std::exp(_global_hypotheses.at(hypothesis_id)._weightLog) *
                        _manager->get_state_existence_probability(new_state._id);
        }
      }
    }

    new_state._log_weight = std::log(new_weight);
  }

  std::vector<State<UDT>> states_only;
  for (auto& state : recycled_states | std::views::values)
  {
    states_only.push_back(std::move(state));  // moved
  }
  return states_only;
}

template <class UDT>
std::vector<StateId> MBMDistribution<UDT>::pruningBernoulliBasis(double pruningThreshold)
{
  LOG_DEB("MBMDistribution::pruningBernoulliBasis");

  std::vector<StateId> pruned_ids;

  for (auto it = _bernoulli_basis.begin(); it != _bernoulli_basis.end();)
  {
    if (_manager->get_state_existence_probability(it->second._id) < pruningThreshold || it->second.isEmpty())
    {
      pruned_ids.push_back(it->first);
      it = _bernoulli_basis.erase(it);  // erase returns next iterator
    }
    else
    {
      ++it;
    }
  }

  return pruned_ids;
}

template <class UDT>
std::vector<StateId> MBMDistribution<UDT>::removeEmptyStates()
{
  LOG_DEB("MBMDistribution::removeEmptyStates");

  std::vector<StateId> removed_ids;

  for (auto it = _bernoulli_basis.begin(); it != _bernoulli_basis.end();)
  {
    if (it->second.isEmpty())
    {
      LOG_DEB("remove " << it->second.toString());
      assert(it->first == it->second._id);
      removed_ids.push_back(it->first);
      it = _bernoulli_basis.erase(it);  // erase returns next iterator
    }
    else
    {
      ++it;
    }
  }

  return removed_ids;
}

template <class UDT>
void MBMDistribution<UDT>::cleanupBernoulliBasis()
{
  LOG_DEB("MBMDistribution::cleanupBernoulliBasis");

  std::unordered_set<StateId> valid_ids;
  for (const auto& state_ids : _state_2_hypotheses | std::views::keys)
  {
    for (const auto& state_id : state_ids)
    {
      valid_ids.insert(state_id);
    }
  }

  LOG_DEB("# of valid_ids: " << valid_ids.size());

  for (auto it = _bernoulli_basis.begin(); it != _bernoulli_basis.end();)
  {
    if (!valid_ids.contains(it->first))
    {
      it = _bernoulli_basis.erase(it);
    }
    else
    {
      ++it;
    }
  }
}

template <class UDT>
void MBMDistribution<UDT>::updateGlobalHypotheses(std::vector<StateId> pruned_ids)
{
  LOG_DEB("MBMDistribution::updateGlobalHypotheses");
  std::unordered_map<HypothesisId, Hypothesis<UDT>> updated_hypotheses;
  std::unordered_map<std::vector<StateId>, HypothesisId> updated_index;
  for (Hypothesis<UDT>& hypothesis : _global_hypotheses | std::views::values)
  {
    Hypothesis candidate = hypothesis;
    std::erase_if(candidate._tracks, [&](StateId state_id) { return std::ranges::contains(pruned_ids, state_id); });
    // The empty object set is a valid global PMBM hypothesis.
    // Also reindex after pruning and merge identical resulting track lists.
    add_hypothesis(std::move(candidate), updated_hypotheses, updated_index);
  }
  _global_hypotheses = std::move(updated_hypotheses);
  _state_2_hypotheses = std::move(updated_index);
}

template <class UDT>
void MBMDistribution<UDT>::pruningGlobalHypotheses(const double pruningThreshold)
{
  LOG_DEB("MBMDistribution::pruningGlobalHypotheses");

  std::erase_if(_global_hypotheses, [=](const std::pair<HypothesisId, Hypothesis<UDT>>& gh) {
    return gh.second._weightLog < std::log(pruningThreshold);
  });
}

template <class UDT>
void MBMDistribution<UDT>::cappingGlobalHypotheses(std::size_t maxNumber)
{
  LOG_DEB("MBMDistribution::updateGlobalHypotheses");

  // Copy to vector for sorting
  std::vector<std::pair<HypothesisId, Hypothesis<UDT>>> top_n(_global_hypotheses.begin(), _global_hypotheses.end());

  if (maxNumber >= top_n.size())
  {
    maxNumber = top_n.size();
  }

  // Sort by _weightLog descending
  std::ranges::partial_sort(top_n,
                            top_n.begin() + static_cast<std::ptrdiff_t>(maxNumber),
                            [](const auto& a, const auto& b) { return a.second._weightLog > b.second._weightLog; });

  // Resize to keep only top N
  top_n.erase(top_n.begin() + maxNumber, top_n.end());

  // Overwrite original map with top N
  _global_hypotheses.clear();
  for (const auto& [id, hyp] : top_n)
  {
    _global_hypotheses.emplace(id, hyp);
  }

  LOG_DEB("Capped to " << _global_hypotheses.size() << " global hypotheses.");
}

}  // namespace ttb
