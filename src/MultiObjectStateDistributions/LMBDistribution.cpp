#include "tracking_lib/MultiObjectStateDistributions/LMBDistribution.h"
#include "tracking_lib/DataContainer/DataSourceContainer.h"
#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/DataContainer/Database.h"
#include "tracking_lib/Graph/Graph.h"
#include "tracking_lib/Misc/Numeric.h"
#include "tracking_lib/SelfAssessment/SelfAssessment.h"
#include "tracking_lib/MultiObjectStateDistributions/GLMBDistribution.h"
#include "tracking_lib/MultiObjectStateDistributions/Utils.h"
#include "tracking_lib/TTBTypes/Concepts.h"

#include <aduulm_logger/aduulm_logger.hpp>
#include <ranges>
#include <utility>
#include <immintrin.h>
#include <tracy/tracy/TracyC.h>

namespace ttb
{

template class LMBDistribution<double>;
template class LMBDistribution<AD>;

template <class UDT>
IDGenerator<MODistributionId> LMBDistribution<UDT>::_idGenerator{};

template <class UDT>
LMBDistribution<UDT>::LMBDistribution(TTBManager<UDT>* manager) : LMBDistribution(manager, std::vector<StateId>{})
{
}

template <class UDT>
LMBDistribution<UDT>::LMBDistribution(TTBManager<UDT>* manager, std::vector<State<UDT>> tracks)
  : _manager{ manager }
  , _previous_label_ex_probs{ std::chrono::milliseconds(
        _manager->params()->self_assessment.persistence_prob_estimation.window_length_ms) }
{
  std::vector<StateId> ids;
  for (State<UDT> const& track : tracks)
  {
    ids.push_back(track._id);
    _manager->database().states.insert(track);
  }
  _tracks = std::move(ids);
}

template <class UDT>
LMBDistribution<UDT>::LMBDistribution(TTBManager<UDT>* manager, std::vector<StateId> tracks)
  : _manager{ manager }
  , _previous_label_ex_probs{ std::chrono::milliseconds(
        _manager->params()->self_assessment.persistence_prob_estimation.window_length_ms) }
  , _tracks{ std::move(tracks) }
{
}

template <class UDT>
void LMBDistribution<UDT>::approx_from(GLMBDistribution<UDT> glmb)
{
  ZoneScopedN("LMBDistribution::approx_from");
  LOG_DEB("GLMB -> LMB Conversion");
  assert(glmb.is_valid());
  _tracks.clear();
  std::vector<StateId> merged_ids;
  for (auto const& [label, trackIds] : glmb.label_2_tracks())
  {
    assert(not trackIds.empty() and "Label has no corresponding Tracks");
    UDT const labelExProb{ [&] {
      UDT prob = glmb.label_2_ex_prob().at(label);
      LOG_DEB("label in GLMB2LMB conversion: " << label << " #trackIds: " << trackIds.size() << " exProb: " << prob);
      return prob;
    }() };
    if (labelExProb < TTB_EPS)
    {
      LOG_DEB("Ignore Label: " << label.value_ << " with existenceProb: " << labelExProb);
      continue;
    }
    std::vector<std::pair<UDT, StateId>> for_merging;

    std::vector<State<UDT>> other_states;
    for (StateId state_id : trackIds)
    {
      for_merging.emplace_back(glmb.track_2_ex_prob().at(state_id) / labelExProb, state_id);
    }
    LOG_DEB("#other states: " << other_states.size() << " of label " << label);
    State merged_track = ttb::merge(*_manager, for_merging);
    merged_track._type = [&] {
      STATE_TYPE type = _manager->get_state(for_merging.front().second)._type;
      return type;
    }();
    if (not merged_track.isEmpty())
    {
      merged_track.postProcess();
      assert(merged_track.isValid());
      StateId const merged_id = merged_track._id;
      _manager->database().states.insert(std::move(merged_track));
      _manager->database().state_existences.emplace(merged_id, labelExProb);
      merged_ids.push_back(merged_id);
    }
  }
  _tracks = std::move(merged_ids);
  LOG_DEB("LMB after conversion:\n" << to_string());
  assert(is_valid());
}

template <class UDT>
UDT LMBDistribution<UDT>::estimated_number_of_tracks() const
{
  return std::accumulate(_tracks.begin(), _tracks.end(), UDT{ 0.0 }, [&](UDT old, StateId const& id) {
    return old + _manager->get_state_existence_probability(id);
  });
}

template <class UDT>
Vector<UDT> LMBDistribution<UDT>::cardinality_distribution() const
{
  ZoneScopedN("LMBDistribution::cardinality_distribution");
  Vector<UDT> cardDist{ { 1 } };
  for (auto const& id : _tracks)
  {
    UDT const r = _manager->get_state_existence_probability(id);
    Vector<UDT> convKernel{ { 1 - r, r } };
    cardDist = numeric::fold<UDT>(cardDist, convKernel);
  }
  return cardDist;
}

template <class UDT>
std::vector<StateId> LMBDistribution<UDT>::all_tracks() const
{
  return _tracks;
}

template <class UDT>
std::vector<BernoulliState<UDT>> LMBDistribution<UDT>::get_estimate() const
{
  // hpx::tracy::region("LMBDistribution::getEstimate", 0, 0);
  std::vector<BernoulliState<UDT>> out;
  switch (_manager->params()->lmb_distribution.extraction.type)
  {
    case MO_DISTRIBUTION_EXTRACTION_TYPE::EXISTENCE_PROBABILITY:
    {
      for (StateId const& id : _tracks)
      {
        if (_manager->get_state_existence_probability(id) > _manager->params()->lmb_distribution.extraction.threshold)
        {
          out.emplace_back(_manager->get_state(id), _manager->get_state_existence_probability(id));
        }
      }
      return out;
    }
    case MO_DISTRIBUTION_EXTRACTION_TYPE::CARDINALITY:
    {
      Vector<UDT> cardDist = cardinality_distribution();
      Index const maxCard = std::distance(cardDist.begin(), std::ranges::max_element(cardDist));

      // sort trackMap by existence Prob in descending order
      std::vector<std::size_t> inds(_tracks.size());
      std::iota(inds.begin(), inds.end(), 0);
      std::ranges::sort(inds, [&](std::size_t a, std::size_t b) {
        return _manager->get_state_existence_probability(_tracks.at(a)) >
               _manager->get_state_existence_probability(_tracks.at(b));
      });

      for (Index i = 0; i < maxCard; ++i)
      {
        out.emplace_back(_manager->get_state(_tracks.at(i)), _manager->get_state_existence_probability(_tracks.at(i)));
      }
      return out;
    }
    case MO_DISTRIBUTION_EXTRACTION_TYPE::BEST_HYPOTHESIS:
    {
      LOG_FATAL("Best Hypothesis not available for LMB");
      return out;
    }
  }
  std::unreachable();
}

template <class UDT>
std::pair<UDT, std::size_t> LMBDistribution<UDT>::truncate(std::size_t maxNTracks)
{
  if (_tracks.size() > maxNTracks)
  {
    std::size_t const before = _tracks.size();
    std::ranges::sort(_tracks, [&](StateId const& a, StateId const& b) {
      return _manager->get_state_existence_probability(a) > _manager->get_state_existence_probability(b);
    });
    UDT weight_sum = 0;
    UDT pruned_weight = 0;
    for (auto [id, state] : std::views::enumerate(_tracks))
    {
      weight_sum += _manager->get_state_existence_probability(state);
      if (id >= maxNTracks)
      {
        pruned_weight += _manager->get_state_existence_probability(state);
      }
    }
    _tracks.erase(std::next(_tracks.begin(), maxNTracks), _tracks.end());
    return { pruned_weight / weight_sum, before - _tracks.size() };
  }
  return { UDT{ 0.0 }, 0 };
}

template <class UDT>
void LMBDistribution<UDT>::predict(Duration deltaT, PredictionInformation const& prediction_info)
{
  TracyFiberEnter("LMBDistribution::predict");
  TracyCZoneN(ctx, "LMBDistribution::predict", 1);
  LOG_DEB("Predict LMB Distribution with " + std::to_string(_tracks.size()) + " Tracks");
  if (deltaT < 1ms)
  {
    TracyFiberEnter("LMBDistribution::predict");
    TracyCZoneEnd(ctx);
    return;
  }
  _previous_label_ex_probs.insert(_manager->filter_time(), [&] {
    std::map<Label, UDT> ex_probs;
    for (StateId id : _tracks)
    {
      State<UDT> const& state = _manager->get_state(id);
      ex_probs.emplace(state._label, _manager->get_state_existence_probability(id));
    }
    return ex_probs;
  }());
  std::unordered_map<StateId, StateId> const predicted_ids =
      utils::predict_db(_manager, _tracks, deltaT, prediction_info._egoMotion);
  _tracks.clear();
  for (auto const& [old, predicted] : predicted_ids)
  {
    if (predicted != EMPTY_STATE)
    {
      State<UDT> const& old_state = _manager->get_state(old);
      UDT const survival_probability =
          _manager->getPersistenceModel().getPersistenceProbability(old_state, deltaT, prediction_info._roi);
      _manager->database().state_existences.emplace(
          predicted, _manager->get_state_existence_probability(old) * survival_probability);
      _tracks.push_back(predicted);
    }
  }
  post_process_prediction();
  TracyFiberEnter("LMBDistribution::predict");
  TracyCZoneEnd(ctx);
}

template <class UDT>
std::pair<std::vector<Matrix<UDT>>, Duration> LMBDistribution<UDT>::surviving_distribution()
{
  ZoneScopedN("LMBDistribution::surviving_distribution");
  // assert (Called after update)
  if (_previous_label_ex_probs.data_buffer.empty())
  {
    return { { Matrix<UDT>::Zero(1, 1) }, 1s };
  }
  std::vector<Matrix<UDT>> independent_data;
  for (auto const& [label, ex_prob_before] : _previous_label_ex_probs.data_buffer.front().data)
  {
    UDT const rb = ex_prob_before;
    auto const ait =
        std::ranges::find_if(_tracks, [&](StateId aid) { return _manager->get_state(aid)._label == label; });
    auto const [ra, PS] = [&] {
      UDT ps = std::pow(_manager->params()->persistence_model.constant.persistence_prob,
                        to_seconds(_manager->filter_time() - _previous_label_ex_probs.data_buffer.front().time));
      if (ait != _tracks.end())
      {
        return std::pair{ _manager->get_state_existence_probability(*ait), ps };
      }
      return std::pair{ UDT{ 0.0 }, ps };
    }();
    Matrix<UDT> const mult{ { std::log((1 - ra) * (1 - rb) / (1)), std::log((1 - ra) * rb / (1)) },
                            { std::log(ra), std::log(0.0) } };
    independent_data.push_back(std::move(mult));
  }
  return { std::move(independent_data), _manager->filter_time() - _previous_label_ex_probs.data_buffer.front().time };
}

template <class UDT>
void LMBDistribution<UDT>::update(DataSourceContainer<UDT> const& measurementContainer, bool do_innovate)
{
  TracyFiberEnter("LMBDistribution::update");
  TracyCZoneN(ctx, "LMBDistribution::update", 1);
  if (do_innovate)
  {
    innovate(std::span{ &measurementContainer, 1 });
  }
  clear_update_metadata();
  assert(is_valid());
  switch (_manager->params()->lmb_distribution.update_method)
  {
    case LMB_UPDATE_METHOD::GLMB:
    {
      single_group_glmb_update(measurementContainer);
      break;
    }
    case LMB_UPDATE_METHOD::LBP:
    {
      single_group_lbp_update(measurementContainer);
      break;
    }
      assert(false);
      std::unreachable();
  }
  post_process_update();
  assert(is_valid());
  TracyFiberEnter("LMBDistribution::update");
  TracyCZoneEnd(ctx);
}

template <class UDT>
void LMBDistribution<UDT>::update(std::vector<DataSourceContainer<UDT>> const& data_source_containers)
{
  TracyFiberEnter("LMBDistribution::update");
  TracyCZoneN(ctx, "LMBDistribution::update", 1);
  // hpx::tracy::region("LMBDistribution::ms_update", 0, 0);
  bool const disable =
      data_source_containers.size() > 1 and
      (_manager->params()->lmb_distribution.multi_sensor.method == LMB_MULTI_SENSOR_UPDATE_METHOD::FPM or
       (_manager->params()->lmb_distribution.multi_sensor.method == LMB_MULTI_SENSOR_UPDATE_METHOD::GLMB and
        _manager->params()->glmb_distribution.multi_sensor.method == GLMB_MULTI_SENSOR_UPDATE_METHOD::PM));
  if (disable)
  {
    disable_state_distribution_post_processing();
  }

  innovate(data_source_containers);

  clear_update_metadata();

  switch (_manager->params()->lmb_distribution.multi_sensor.method)
  {
    case LMB_MULTI_SENSOR_UPDATE_METHOD::GLMB:
    {
      single_group_glmb_update(data_source_containers);
      break;
    }
    case LMB_MULTI_SENSOR_UPDATE_METHOD::FPM:
    {
      single_group_fpm_update(data_source_containers);
      break;
    }
    case LMB_MULTI_SENSOR_UPDATE_METHOD::PU:
    {
      single_group_pu_update(data_source_containers);
      break;
    }
  }
  if (disable)
  {
    reset_state_distribution_post_processing();
  }
  post_process_update();
  assert(is_valid());
  TracyFiberEnter("LMBDistribution::update");
  TracyCZoneEnd(ctx);
}

template <class UDT>
void LMBDistribution<UDT>::single_group_glmb_update(std::vector<DataSourceContainer<UDT>> const& data_source_containers)
{
  for (StateId track : _tracks)
  {
    /// tracks are already predicted, use existence prob
    /// as survival prob for GLMB
    _manager->database().state_persistences.emplace(track, _manager->get_state_existence_probability(track));
  }
  GLMBDistribution glmbDist(_manager);
  glmbDist.add_tracks(_tracks);
  glmbDist.update(data_source_containers, true, false);
  LOG_DEB("GLMB Dist after update:\n" << glmbDist.to_string());
  for (DataSourceContainer<UDT> const& data_source_container : data_source_containers)
  {
    assert(not _meas_assignment_prob.contains(data_source_container._id));
    _meas_assignment_prob[data_source_container._id] = glmbDist.meas_assignment_prob(data_source_container);
  }
  approx_from(std::move(glmbDist));
  LOG_DEB("LMB Dist after update:\n" << to_string());
  assert(is_valid() && "after update");
}

template <class UDT>
void LMBDistribution<UDT>::single_group_fpm_update(std::vector<DataSourceContainer<UDT>> const& data_source_containers)
{
  std::vector<LMBDistribution<UDT>> posteriors;
  Params next_params = *_manager->params();
  next_params.lmb_distribution.post_process_update.enable =
      false;  // do not remove tracks after single sensor updates...
  _manager->update_params(next_params);
  for (DataSourceContainer<UDT> const& data_source_container : data_source_containers)
  {
    LMBDistribution lmb = *this;
    lmb.update(data_source_container, false);
    _meas_assignment_prob[data_source_container._id] = lmb.meas_assignment_prob(data_source_container);
    posteriors.push_back(std::move(lmb));
  }
  next_params.lmb_distribution.post_process_update.enable =
      _manager->original_params().lmb_distribution.post_process_update.enable;  // do not remove tracks after single
                                                                                // sensor updates...
  _manager->update_params(std::move(next_params));
  reset_state_distribution_post_processing();
  fpm_fusion(std::move(posteriors));
}

template <class UDT>
void LMBDistribution<UDT>::single_group_pu_update(std::vector<DataSourceContainer<UDT>> const& data_source_containers)
{
  std::vector<LMBDistribution> posteriors;
  Params next_params = *_manager->params();
  next_params.lmb_distribution.post_process_update.enable =
      false;  // do not remove tracks after single sensor updates...
  _manager->update_params(next_params);
  for (DataSourceContainer<UDT> const& data_source_container : data_source_containers)
  {
    LMBDistribution lmb = *this;
    lmb.update(data_source_container, false);
    _meas_assignment_prob[data_source_container._id] = lmb.meas_assignment_prob(data_source_container);
    posteriors.push_back(std::move(lmb));
  }
  next_params.lmb_distribution.post_process_update.enable =
      _manager->original_params().lmb_distribution.post_process_update.enable;  // do not remove tracks after single
  // sensor updates...
  _manager->update_params(std::move(next_params));
  reset_state_distribution_post_processing();
  pu_fusion(std::move(posteriors));
}

template <class UDT>
void LMBDistribution<UDT>::single_group_lbp_update(DataSourceContainer<UDT> const& measurementContainer)
{
  TracyFiberEnter("LMBDistribution::single_group_lbp_update");
  TracyCZoneN(ctx, "LMBDistribution::single_group_lbp_update", 1);
  LOG_DEB(std::format("LMBDistribution::single_group_lbp_update of {} tracks with {} measurements",
                      _tracks.size(),
                      measurementContainer._measurements.size()));
  Index const L = std::ssize(_tracks);
  if (L == 0)
  {
    _meas_assignment_prob = [&] {
      LOG_DEB("no tracks -> create rz map");
      std::map<DataSourceContainerId, std::map<MeasurementId, UDT>> rz;
      rz[measurementContainer._id];
      for (State<UDT> const& meas : measurementContainer._measurements)
      {
        rz.at(measurementContainer._id)[meas._id] = 0;
      }
      return rz;
    }();
    assert(is_valid());
    LOG_DEB("LMBDistribution::single_group_lbp_update");
    TracyFiberEnter("LMBDistribution::single_group_lbp_update");
    TracyCZoneEnd(ctx);
    return;
  }
  auto const M = static_cast<Index>(measurementContainer._measurements.size());
  Index const not_detected = M;
  Index const not_existent = M + 1;
  Matrix<UDT> XI(L, M + 2);
  Matrix<UDT> NU = Matrix<UDT>::Ones(M + 2, L);
  Matrix<UDT> BETA = Matrix<UDT>::Zero(L, M + 2);
  std::vector<std::size_t> estimated_tracks_ind;
  for (auto const& [l, state_id] : std::views::enumerate(_tracks) | std::views::as_const)
  {
    State<UDT> const& state = _manager->get_state(state_id);
    UDT const state_existence = _manager->get_state_existence_probability(state_id);
    UDT const detection_prob = [&] {
      auto const it = _manager->database().state_detection_probabilities.template get<StateIdContainerId>().find(
          std::tuple{ state_id, measurementContainer._id });
      assert(it != _manager->database().state_detection_probabilities.template get<StateIdContainerId>().end());
      return it->detection_probability;
    }();
    for (auto const& [m, meas] : std::views::enumerate(measurementContainer._measurements))
    {
      if (_manager->database().state_updates.contains(std::tuple{ state._id, measurementContainer._id, meas._id }))
      {
        UDT const log_likelihood = [&] {
          auto const it = _manager->database().state_meas_log_likelihood.find(std::tuple{ state_id, meas._id });
          assert(it != _manager->database().state_meas_log_likelihood.end());
          return it->meas_log_likelihood;
        }();
        UDT const clutter_intensity = [&] {
          auto const it = _manager->database().meas_clutter_intensity.find(meas._id);
          assert(it != _manager->database().meas_clutter_intensity.end());
          return it->clutter_intensity;
        }();
        BETA(l, m) = state_existence * std::exp(log_likelihood) * detection_prob / clutter_intensity;
      }
    }
    BETA(l, not_detected) = state_existence * (1 - detection_prob);
    BETA(l, not_existent) = 1 - state_existence;
  }
  bool converged = false;
  UDT convergence_crit = UDT{ std::numeric_limits<double>::infinity() };
  for (std::size_t iteration = 0;
       not converged and iteration < _manager->params()->lmb_distribution.loopy_belief_propagation.max_iterations;
       ++iteration)
  {
    Matrix<UDT> const& BETA_RED = BETA(Eigen::all, Eigen::seq(0, M - 1));
    Matrix<UDT> const& NU_RED = NU(Eigen::seq(0, M - 1), Eigen::all);
    Vector<UDT> DENUM =
        (BETA_RED * NU_RED).diagonal() + BETA(Eigen::all, not_existent) + BETA(Eigen::all, not_detected);
    XI(Eigen::all, Eigen::seq(0, M - 1)) =
        BETA_RED.array() / (DENUM.replicate(1, M).array() - BETA_RED.array() * NU_RED.transpose().array());
    XI(Eigen::all, { not_detected, not_existent }) =
        BETA(Eigen::all, { not_detected, not_existent }).array() / DENUM.replicate(1, 2).array();
    Matrix<UDT> NU_Update = (1 / (1 + (XI.colwise().sum().replicate(L, 1) - XI).array())).transpose();
    convergence_crit = (NU - NU_Update).array().abs().sum() / NU.array().abs().sum();
    assert([&] {  // NOLINT
      if (not std::isfinite(convergence_crit))
      {
        LOG_FATAL("NU " << NU);
        LOG_FATAL("NU_UPDATE" << NU_Update);
        LOG_FATAL("Criteria " << convergence_crit);
        LOG_FATAL("Beta" << BETA);
        return false;
      }
      return true;
    }());
    if (convergence_crit < _manager->params()->lmb_distribution.loopy_belief_propagation.tol)
    {
      converged = true;
      TracyPlot("#LBP Iterations", static_cast<int64_t>(iteration));
    }
    NU = std::move(NU_Update);
  }
  if (not converged)
  {
    TracyPlot("#LBP Iterations",
              static_cast<int64_t>(_manager->params()->lmb_distribution.loopy_belief_propagation.max_iterations));
    LOG_WARN_THROTTLE(10,
                      "LBP not converged after " +
                          std::to_string(_manager->params()->lmb_distribution.loopy_belief_propagation.max_iterations) +
                          " Convergence criteria: " + std::to_string(convergence_crit) + " <! " +
                          std::to_string(_manager->params()->lmb_distribution.loopy_belief_propagation.tol) +
                          " not satisfied");
  }
  Matrix<UDT> P(L, M + 2);
  P(Eigen::all, Eigen::seq(0, M - 1)) =
      BETA(Eigen::all, Eigen::seq(0, M - 1)).array() * NU(Eigen::seq(0, M - 1), Eigen::all).transpose().array();
  P(Eigen::all, { not_detected, not_existent }) = BETA(Eigen::all, { not_detected, not_existent });
  P(Eigen::all, Eigen::all).array() /= P.rowwise().sum().replicate(1, M + 2).array();
  std::map<MeasurementId, UDT> assignment_any_track;
  for (auto const& [m, meas] : std::views::enumerate(measurementContainer._measurements))
  {
    assignment_any_track[meas._id] = 1.0 - 1.0 / (1 + XI.col(m).sum());
  }
  LOG_DEB("LBP done");
  // update state
  std::vector<hpx::future<State<UDT>>> merged_states;
  std::vector<UDT> merged_states_existence_probs;
  for (auto const& [l, state_id] : std::views::enumerate(_tracks))
  {
    std::vector<std::pair<UDT, StateId>> for_merging;
    for (auto const& [m, meas] : std::views::enumerate(measurementContainer._measurements))
    {
      if (_manager->database().state_updates.contains(std::tuple{ state_id, measurementContainer._id, meas._id }))
      {
        for_merging.emplace_back(P(l, m),
                                 _manager->database()
                                     .state_updates.find(std::tuple{ state_id, measurementContainer._id, meas._id })
                                     ->updated_state_id);
      }
    }
    for_merging.emplace_back(P(l, not_detected),
                             _manager->database()
                                 .state_updates.find(std::tuple{ state_id, measurementContainer._id, NOT_DETECTED })
                                 ->updated_state_id);

    UDT const merged_state_existence = [&] {
      UDT ex = std::accumulate(for_merging.begin(), for_merging.end(), UDT{ 0.0 }, [&](UDT first, auto const& val) {
        return first + val.first;
      });
      if (ex > 1 or ex < 0)
      {
        LOG_WARN_THROTTLE(1,
                          "merged state existence prob not in [0, 1]: " << ex << " in update of sensor: "
                                                                        << measurementContainer._source_id.value_);
        ex = std::clamp(ex, UDT{ 0.0 }, UDT{ 1.0 });
      }
      return ex;
    }();
    merged_states_existence_probs.push_back(merged_state_existence);
    merged_states.emplace_back(hpx::async([&, for_merging] { return ttb::merge(*_manager, for_merging); }));
  }
  std::vector<StateId> updated_states;
  for (auto const& [state_fut, ex_prob] : std::views::zip(merged_states, merged_states_existence_probs))
  {
    State merged_state = state_fut.get();
    if (not merged_state.isEmpty())
    {
      assert(merged_state.isValid());
      StateId const id = merged_state._id;
      _manager->database().states.insert(std::move(merged_state));
      _manager->database().state_existences.emplace(id, ex_prob);
      updated_states.push_back(id);
    }
  }
  _tracks = std::move(updated_states);
  _meas_assignment_prob[measurementContainer._id] = std::move(assignment_any_track);
  assert(is_valid());
  LOG_DEB("LMBDistribution::single_group_lbp_update");
  TracyFiberEnter("LMBDistribution::single_group_lbp_update");
  TracyCZoneEnd(ctx);
}

template <class UDT>
void LMBDistribution<UDT>::single_group_glmb_update(DataSourceContainer<UDT> const& measurementContainer)
{
  // hpx::tracy::region("LMBDistribution::single_group_glmb_update", 0, 0);
  LOG_DEB("Update LMB Distribution: " << to_string());
  for (StateId track : _tracks)
  {
    /// tracks are already predicted, use existence prob
    /// as survival prob for GLMB
    _manager->database().state_persistences.emplace(track, _manager->get_state_existence_probability(track));
  }
  GLMBDistribution glmbDist(_manager);
  glmbDist.add_tracks(_tracks);
  glmbDist.update(measurementContainer, false);
  _meas_assignment_prob[measurementContainer._id] = glmbDist.meas_assignment_prob(measurementContainer);
  LOG_DEB("GLMB Dist after update:\n" << glmbDist.to_string());
  approx_from(std::move(glmbDist));
  LOG_DEB("LMB Dist after update:\n" << to_string());
  assert(is_valid() && "after update");
}

template <class UDT>
void LMBDistribution<UDT>::fpm_fusion(std::vector<LMBDistribution<UDT>> posterior_lmbs)
{
  ZoneScopedN("LMBDistribution::fpm_fusion");
  LOG_DEB("FPM fusion with " << posterior_lmbs.size() << " single-sensor updated lmb densities.");
  if (posterior_lmbs.size() == 1)
  {
    LOG_DEB("single sensor case");
    *this = std::move(posterior_lmbs.at(0));
    return;
  }

  // this state created the following posteriors
  std::map<StateId, std::vector<StateId>> prior_2_posteriors;
  for (LMBDistribution& posterior_lmb : posterior_lmbs)
  {
    for (StateId posterior : posterior_lmb._tracks)
    {
      prior_2_posteriors[_manager->get_state(posterior)._prior_id].push_back(posterior);
    }
  }
  std::vector<StateId> updated_tracks;
  for (auto& [prior, posteriors] : prior_2_posteriors)
  {
    if (posteriors.size() != posterior_lmbs.size())
    {
      continue;
    }
    auto const prior_it = std::ranges::find(_tracks, prior);
    assert(prior_it != _tracks.end());
    std::vector<State<UDT>> copies;
    for (StateId id : posteriors)
    {
      copies.push_back(_manager->get_state(id));
    }
    if (std::optional<std::pair<State<UDT>, UDT>> fused =
            fuse_tracks_fpm(_manager->get_state(*prior_it), std::move(copies));
        fused.has_value())
    {
      StateId const id = fused.value().first._id;
      _manager->database().states.insert(std::move(fused.value().first));
      _manager->database().state_existences.emplace(id, fused.value().second);
      updated_tracks.push_back(id);
    }
  }
  _tracks = std::move(updated_tracks);
  assert(is_valid() && "Fused dist after fpm fusion is not valid!");
}

template <class UDT>
std::optional<std::pair<State<UDT>, UDT>>
LMBDistribution<UDT>::fuse_tracks_fpm(State<UDT> priorTrack, std::vector<State<UDT>> posteriors) const
{
  ZoneScopedN("LMBDistribution::fuse_tracks_fpm");
  LOG_DEB("Fuse Tracks with label " << priorTrack._label);
  assert([&] {
    if (not priorTrack.isValid())
    {
      LOG_FATAL("pior track s is invalid");
      return false;
    }
    for (State<UDT> const& post : posteriors)
    {
      if (not post.isValid())
      {
        LOG_FATAL("post is invalid:");
        return false;
      }
    }
    return true;
  }());
  UDT const r_prior = _manager->get_state_existence_probability(priorTrack._id);
  auto const [r_posterior_product, r_inverse_posterior_product] = [&] {
    UDT r_prod = 1;
    UDT r_inv_prod = 1;
    for (State<UDT> const& state : posteriors)
    {
      r_prod *= _manager->get_state_existence_probability(state._id);
      r_inv_prod *= (1 - _manager->get_state_existence_probability(state._id));
    }
    return std::pair{ r_prod, r_inv_prod };
  }();

  // Fusion
  std::optional<State<UDT>> fusedDist = fuse_posteriors(
      priorTrack,
      posteriors,
      _manager->params()->lmb_distribution.multi_sensor.fpm.num_components,
      _manager->params()->lmb_distribution.multi_sensor.fpm.method,
      _manager->params()->lmb_distribution.multi_sensor.fpm.poisson_method.equal_allocation_share_ratio);
  if (not fusedDist.has_value())
  {
    LOG_WARN("Fused mixture is empty due to pruning in posteriors -> Track is removed. It does not matter if it has a "
             "high existence probability. Result could be worse because of this...");
    // This case happens when not all posteriors have a value for one prior component! (Fused mixture is empty due to
    // pruning in local updates) This happens in very complex scenarios like bearing only! Todo: Make sure there is at
    // least one common component! (Must be probably fixed in innovation part...) this fused component does not exist
    // which is worse....
    return std::nullopt;
  }
  assert(fusedDist.value()._misc.contains("dist_weight_sum"));
  UDT const fused_existence = [&] {
    UDT exp = 1.0 - posteriors.size();
    UDT r_prior_product = std::pow(r_prior, exp);
    UDT r_prior_inverse_product = std::pow(1.0 - r_prior, exp);
    UDT r_fused_num =
        std::any_cast<UDT>(fusedDist.value()._misc.at("dist_weight_sum")) * r_prior_product * r_posterior_product;
    UDT r_fused_den = r_prior_inverse_product * r_inverse_posterior_product + r_fused_num;
    assert(std::isfinite(r_fused_num / r_fused_den));
    return r_fused_num / r_fused_den;
  }();
  assert([&] {
    if (not fusedDist.value().isValid())
    {
      LOG_FATAL("fused track is not valid");
      LOG_FATAL(fusedDist.value().toString());
      LOG_FATAL("prior");
      LOG_FATAL(priorTrack.toString());
      LOG_FATAL("posteriors");
      for (State<UDT> const& post : posteriors)
      {
        LOG_FATAL(post.toString());
      }
      return false;
    }
    return true;
  }());
  return std::pair{ fusedDist.value(), fused_existence };
}

template <class UDT>
void LMBDistribution<UDT>::pu_fusion(std::vector<LMBDistribution<UDT>> posterior_lmbs)
{
  ZoneScopedN("LMBDistribution::fpm_fusion");
  LOG_DEB("FPM fusion with " << posterior_lmbs.size() << " single-sensor updated lmb densities.");
  if (posterior_lmbs.size() == 1)
  {
    LOG_DEB("single sensor case");
    *this = std::move(posterior_lmbs.at(0));
    return;
  }

  // this state created the following posteriors
  std::map<StateId, std::vector<StateId>> prior_2_posteriors;
  for (LMBDistribution& posterior_lmb : posterior_lmbs)
  {
    for (StateId posterior : posterior_lmb._tracks)
    {
      prior_2_posteriors[_manager->get_state(posterior)._prior_id].push_back(posterior);
    }
  }
  std::vector<StateId> updated_tracks;
  for (auto& [prior, posteriors] : prior_2_posteriors)
  {
    if (posteriors.size() != posterior_lmbs.size())
    {
      continue;
    }
    auto const prior_it = std::ranges::find(_tracks, prior);
    assert(prior_it != _tracks.end());
    std::vector<State<UDT>> copies;
    for (StateId id : posteriors)
    {
      copies.push_back(_manager->get_state(id));
    }
    if (std::optional<std::pair<State<UDT>, UDT>> fused =
            fuse_tracks_pu(_manager->get_state(*prior_it), std::move(copies));
        fused.has_value())
    {
      StateId const id = fused.value().first._id;
      _manager->database().states.insert(std::move(fused.value().first));
      _manager->database().state_existences.emplace(id, fused.value().second);
      updated_tracks.push_back(id);
    }
  }
  _tracks = std::move(updated_tracks);
  assert(is_valid() && "Fused dist after fpm fusion is not valid!");
}

template <class UDT>
std::optional<std::pair<State<UDT>, UDT>> LMBDistribution<UDT>::fuse_tracks_pu(State<UDT> priorTrack,
                                                                               std::vector<State<UDT>> posteriors) const
{
  ZoneScopedN("LMBDistribution::fuse_tracks_fpm");
  LOG_DEB("Fuse Tracks with label " << priorTrack._label);
  assert([&] {
    if (not priorTrack.isValid())
    {
      LOG_FATAL("pior track s is invalid");
      return false;
    }
    for (State<UDT> const& post : posteriors)
    {
      if (not post.isValid())
      {
        LOG_FATAL("post is invalid:");
        return false;
      }
    }
    return true;
  }());
  UDT const r_prior = _manager->get_state_existence_probability(priorTrack._id);
  auto const [r_posterior_product, r_inverse_posterior_product] = [&] {
    UDT r_prod = 1;
    UDT r_inv_prod = 1;
    for (State<UDT> const& state : posteriors)
    {
      r_prod *= _manager->get_state_existence_probability(state._id);
      r_inv_prod *= (1 - _manager->get_state_existence_probability(state._id));
    }
    return std::pair{ r_prod, r_inv_prod };
  }();

  // approximation of posteriors to one gaussian component in order to calculate division of states
  // Fusion
  std::optional<State<UDT>> fusedDist =
      fuse_posteriors_pu(priorTrack,
                         posteriors,
                         _manager->params()->lmb_distribution.multi_sensor.pu.use_best_component,
                         _manager->params()->lmb_distribution.multi_sensor.pu.num_components);
  if (not fusedDist.has_value())
  {
    LOG_WARN("Fused mixture is empty -> Track is removed. It does not matter if it has a "
             "high existence probability. Result could be worse because of this... Case should not happen! Check!!!!");
    return std::nullopt;
  }
  assert(fusedDist.value()._misc.contains("eta"));
  UDT const fused_existence = [&] {
    UDT exp = 1.0 - posteriors.size();
    UDT r_prior_product = std::pow(r_prior, exp);
    UDT r_prior_inverse_product = std::pow(1.0 - r_prior, exp);
    UDT r_fused_num = std::any_cast<UDT>(fusedDist.value()._misc.at("eta")) * r_prior_product * r_posterior_product;
    UDT r_fused_den = r_prior_inverse_product * r_inverse_posterior_product + r_fused_num;
    assert(std::isfinite(r_fused_num / r_fused_den));
    return r_fused_num / r_fused_den;
  }();
  assert([&] {
    if (not fusedDist.value().isValid())
    {
      LOG_FATAL("fused track is not valid");
      LOG_FATAL(fusedDist.value().toString());
      LOG_FATAL("prior");
      LOG_FATAL(priorTrack.toString());
      LOG_FATAL("posteriors");
      for (State<UDT> const& post : posteriors)
      {
        LOG_FATAL(post.toString());
      }
      return false;
    }
    return true;
  }());
  return std::pair{ fusedDist.value(), fused_existence };
}

template <class UDT>
void LMBDistribution<UDT>::add_tracks(std::vector<BernoulliState<UDT>> tracks)
{
  for (BernoulliState<UDT>& birth_track : tracks)
  {
    assert([&] {  // NOLINT
      if (auto const it = std::ranges::find_if(
              _tracks,
              [&](StateId existingid) { return _manager->get_state(existingid)._label == birth_track.state._label; });
          it != _tracks.end())
      {
        LOG_FATAL("Track already exists in current distribution!");
        return false;
      }

      if (not birth_track.state.isValid())
      {
        LOG_FATAL("requested to add invalid track to LMB");
        LOG_FATAL(birth_track.state.toString());
        LOG_FATAL(std::stacktrace::current());
        return false;
      }
      return true;
    }());
    StateId const id = birth_track.state._id;
    _manager->database().states.insert(std::move(birth_track.state));
    _manager->database().state_existences.emplace(id, birth_track.existence_probability);
    _tracks.push_back(id);
  }
}

template <class UDT>
void LMBDistribution<UDT>::add_tracks(std::vector<StateId> const& tracks)
{
  for (StateId id : tracks)
  {
    _tracks.push_back(id);
  }
}

template <class UDT>
void LMBDistribution<UDT>::merge(LMBDistribution<UDT> otherDist)
{
  LOG_DEB("LMBDistribution::merge");
  ZoneScopedN("LMBDistribution::merge");
  add_tracks(otherDist._tracks);
  for (auto& [source_id, meas_probs] : _meas_assignment_prob)
  {
    assert([&] {
      if (not otherDist._meas_assignment_prob.contains(source_id))
      {
        LOG_FATAL("the other lmb distribution does not contain the meas assignment matrix for source "
                  << source_id.value_);
        LOG_FATAL("it contains:");
        for (auto const& [source, _] : otherDist._meas_assignment_prob)
        {
          LOG_FATAL("Source : " << source);
        }
        LOG_FATAL(otherDist.to_string());
        LOG_FATAL("this lmb contains: ");
        for (auto const& [source, matrix] : _meas_assignment_prob)
        {
          LOG_FATAL("Source : " << source);
        }
        return false;
      }
      return true;
    }());
    meas_probs.merge(std::move(otherDist._meas_assignment_prob.at(source_id)));
  }
  LOG_DEB("LMBDistribution::merge");
}

template <class UDT>
std::string LMBDistribution<UDT>::to_string(std::string const& prefix) const
{
  std::string out = prefix + "LMB Distribution\n";
  out += prefix + "|\tId: " + std::to_string(_id.value_) + "\n";
  for (StateId id : _tracks)
  {
    out += _manager->get_state(id).toString(prefix + "|\t");
  }
  return out;
}

template <class UDT>
void LMBDistribution<UDT>::disable_state_distribution_post_processing()
{
  Params next_params = *_manager->params();
  for (auto const& [model_id, _] : _manager->getStateModelMap())
  {
    next_params.state_model_params(model_id).distribution.post_process.enable = false;
  }
  _manager->update_params(std::move(next_params));
}

template <class UDT>
void LMBDistribution<UDT>::reset_state_distribution_post_processing()
{
  Params next_params = *_manager->params();
  for (auto const& [model_id, _] : _manager->getStateModelMap())
  {
    next_params.state_model_params(model_id).distribution.post_process.enable =
        _manager->original_params().state_model_params(model_id).distribution.post_process.enable;
  }
  _manager->update_params(std::move(next_params));
}

template <class UDT>
void LMBDistribution<UDT>::innovate(std::span<DataSourceContainer<UDT> const> data_source_containers)
{
  for (DataSourceContainer<UDT> const& data_source_container : data_source_containers)
  {
    utils::innovate_db(_manager, _tracks, data_source_container);
  }
}

template <class UDT>
void LMBDistribution<UDT>::clear_update_metadata()
{
  _meas_assignment_prob.clear();
  for (StateId id : _tracks)
  {
    auto const it = _manager->database().states.find(id);
    State state = *it;
    state._meta_data._last_meas_log_likelihood.clear();
    state._meta_data._meas_update_weight.clear();
    _manager->database().states.replace(it, std::move(state));
  }
}

template <class UDT>
bool LMBDistribution<UDT>::is_valid() const
{
  return std::ranges::all_of(_tracks, [&](StateId id) {
    State<UDT> const& track = _manager->get_state(id);
    if (not track.isValid())
    {
      LOG_FATAL("Track " + track.toString() + "not valid");
      return false;
    }
    return true;
  });
}

template <class UDT>
void LMBDistribution<UDT>::post_process_prediction()
{
  LOG_DEB("LMBDistribution::post_process_prediction");
  ZoneScopedN("LMBDistribution::post_process_prediction");
  prune_if([&](StateId id) {
    State<UDT> const& track = _manager->get_state(id);
    return track.isEmpty();
  });
  if (_manager->params()->lmb_distribution.post_process_prediction.enable)
  {
    prune_if([&](StateId id) {
      return _manager->get_state_existence_probability(id) <
             _manager->params()->lmb_distribution.post_process_prediction.pruning_threshold;
    });
    prune_if([&](StateId id) {
      State<UDT> const& track = _manager->get_state(id);
      return track._meta_data._durationSinceLastAssociation >
             std::chrono::milliseconds(
                 _manager->params()->lmb_distribution.post_process_prediction.max_last_assoc_duration_ms);
    });
    truncate(_manager->params()->lmb_distribution.post_process_prediction.max_tracks);
  }
  assert([&] {  // NOLINT
    if (not is_valid())
    {
      LOG_FATAL(to_string());
      return false;
    }
    return true;
  }());
}

template <class UDT>
template <class Policy>
std::pair<UDT, std::size_t> LMBDistribution<UDT>::prune_if(Policy pred)
{
  std::vector<StateId> keep;
  std::size_t num_pruned = 0;
  UDT weight_pruned = 0;
  UDT weight_sum = 0;
  for (StateId id : _tracks)
  {
    UDT const ex = _manager->get_state_existence_probability(id);
    weight_sum += ex;
    if (pred(id))
    {
      ++num_pruned;
      weight_pruned += ex;
    }
    else
    {
      keep.push_back(id);
    }
  }
  _tracks = std::move(keep);
  return { weight_pruned / weight_sum, num_pruned };
}

template <class UDT>
void LMBDistribution<UDT>::post_process_update()
{
  LOG_DEB("LMBDistribution::post_process_update");
  ZoneScopedN("LMBDistribution::post_process_update");
  prune_if([&](StateId const& id) {
    State<UDT> const& track = _manager->get_state(id);
    return track.isEmpty();
  });
  if (_manager->params()->lmb_distribution.post_process_update.enable)
  {
    prune_if([&](StateId id) {
      return _manager->get_state_existence_probability(id) <
             _manager->params()->lmb_distribution.post_process_update.pruning_threshold;
    });
    truncate(_manager->params()->lmb_distribution.post_process_update.max_tracks);
  }
  assert([&] {  // NOLINT
    if (not is_valid())
    {
      LOG_FATAL(to_string());
      return false;
    }
    return true;
  }());
  LOG_DEB("postProcessUpdate done");
}

template <class UDT>
Vector<UDT> LMBDistribution<UDT>::clutter_distribution(DataSourceContainer<UDT> const& data_source_container) const
{
  std::vector<Matrix<UDT>> const detection_dists = detection_distribution(data_source_container);
  Matrix<UDT> detection_dist{ { 1 } };
  for (Matrix<UDT> const& D : detection_dists)
  {
    detection_dist = numeric::fold<UDT>(detection_dist, D);
  }

  Index const num_meas = data_source_container._measurements.size();
  assert(std::abs(detection_dist.sum() - 1) < 1e-5);
  Vector<UDT> clutter_dist = Vector<UDT>::Zero(num_meas + 1);
  for (Index detected = 0; detected < detection_dist.rows(); ++detected)
  {
    for (Index missed = 0; missed < detection_dist.cols(); ++missed)
    {
      clutter_dist(std::max(Index{ 0 }, num_meas - detected)) += detection_dist(detected, missed);
    }
  }
  clutter_dist /= clutter_dist.sum();
  assert([&] {  // NOLINT
    if (not std::ranges::all_of(clutter_dist.reshaped(), [](UDT val) { return std::isfinite(val); }))
    {
      LOG_FATAL("Invalid clutter dist: " << clutter_dist << " with sum: " << clutter_dist.sum()
                                         << " detection dist sum: " << detection_dist.sum());
      return false;
    }
    return true;
  }());
  return clutter_dist;
}

template <class UDT>
std::vector<Matrix<UDT>>
LMBDistribution<UDT>::detection_distribution(DataSourceContainer<UDT> const& data_source_container) const
{
  std::vector<Matrix<UDT>> detection_dists;
  for (StateId id : _tracks)
  {
    State<UDT> const& track = _manager->get_state(id);
    assert([&] {
      if (not track._meta_data._detectable.contains(data_source_container._source_id))
      {
        LOG_FATAL(track.toString());
        return false;
      }
      return true;
    }());
    if (not track._meta_data._detectable.at(data_source_container._source_id) or
        track._meta_data._numUpdates <= _manager->params()->meas_models.gaussian_models.size() + 1)
    {
      continue;
    }
    assert([&] {  // NOLINT
      if (not track._meta_data._meas_update_weight.contains(data_source_container._source_id) or
          not _manager->database().state_existences.contains(track._id))
      {
        LOG_FATAL("track not ok: " << track.toString());
        return false;
      }
      return true;
    }());
    UDT const weight_mis_detection = [&] {
      if (auto const it = std::ranges::find(track._meta_data._meas_update_weight.at(data_source_container._source_id),
                                            NOT_DETECTED,
                                            &std::pair<MeasurementId, UDT>::first);
          it != track._meta_data._meas_update_weight.at(data_source_container._source_id).end())
      {
        return it->second;
      }
      return UDT{ 0.0 };
    }();

    UDT const track_ex_prob = _manager->get_state_existence_probability(track._id);

    UDT const ed = track_ex_prob * (1 - weight_mis_detection);  ///< track exists + detected
    UDT const em = track_ex_prob * weight_mis_detection;        ///< track exists + missed
    UDT const n = 1 - track_ex_prob;                            ///< track does not exists
    detection_dists.push_back(Matrix<UDT>{ { n, em }, { ed, 0 } });
  }
  return detection_dists;
}

template <class UDT>
UDT LMBDistribution<UDT>::bayes_factor_statistics_measurement_likelihood(
    DataSourceContainer<UDT> const& data_source_container) const
{
  if (not data_source_container._sensorInfo._sensor_fov.has_value())
  {
    LOG_FATAL("Can not compute Self Assessment Statistics if no fov is defined");
    LOG_FATAL(std::stacktrace::current());
    std::terminate();
  }
  double const fov_area =
      data_source_container._sensorInfo._sensor_fov.value().area(Components({ COMPONENT::POS_X, COMPONENT::POS_Y }));

  double const clutter_rate = [&] {
    if (_manager->params()->meas_model_params(data_source_container._source_id).clutter.rate.has_value())
    {
      return _manager->params()->meas_model_params(data_source_container._source_id).clutter.rate.value();
    }
    return fov_area * _manager->params()->meas_model_params(data_source_container._source_id).clutter.intensity.value();
  }();

  UDT meas_log_likelihood = -clutter_rate;
  for (State<UDT> const& meas : data_source_container._measurements)
  {
    if (not _manager->database().meas_clutter_intensity.contains(meas._id))
    {
      // todo: refactor compute innovation: problem if no tracksare there, no clutter for measurment is computed
      LOG_INF(std::format("The clutter intensity was never computed for meas {}", meas._id));
      _manager->database().meas_clutter_intensity.insert(
          { meas._id,
            _manager->getMeasModel(data_source_container._source_id)
                .calculate_clutter_intensity(meas, data_source_container._sensorInfo) });
    }
    meas_log_likelihood += std::log(_manager->database().meas_clutter_intensity.find(meas._id)->clutter_intensity);
  }
  std::map<StateId, UDT> betas;
  for (StateId track : _tracks)
  {
    State<UDT> const& state = _manager->get_state(track);
    UDT const r = _manager->get_state_existence_probability(track);
    UDT const& r_ = _manager->get_state_existence_probability(state._prior_id);
    assert(_manager->database().state_detection_probabilities.contains(
        std::tuple{ state._prior_id, data_source_container._id }));
    UDT const pd = _manager->database()
                       .state_detection_probabilities.find(std::tuple{ state._prior_id, data_source_container._id })
                       ->detection_probability;
    meas_log_likelihood += (1 - r) * std::log(1 - r_) + r * std::log(r_);
    meas_log_likelihood += -(1 - r) * log(1 - r);
    assert(state._meta_data._meas_update_weight.contains(data_source_container._source_id));
    for (auto const& [meas, weight] : state._meta_data._meas_update_weight.at(data_source_container._source_id))
    {
      UDT const alpha = r * weight;
      if (meas == NOT_DETECTED)
      {
        meas_log_likelihood += alpha * std::log(1 - pd);
        meas_log_likelihood += -alpha * log(alpha);
      }
      else
      {
        meas_log_likelihood += (1 - alpha) * log(1 - alpha);
        meas_log_likelihood += -alpha * log(alpha);
        betas[meas] += alpha;
        assert(_manager->database().state_meas_log_likelihood.contains(std::tuple{ state._prior_id, meas }));
        UDT const l = _manager->database()
                          .state_meas_log_likelihood.find(std::tuple{ state._prior_id, meas })
                          ->meas_log_likelihood;
        meas_log_likelihood +=
            alpha *
            (std::log(pd) + l - std::log(_manager->database().meas_clutter_intensity.find(meas)->clutter_intensity));
      }
    }
  }
  for (const auto& weight : betas | std::views::values)
  {
    meas_log_likelihood += -(1 - weight) * log(1 - weight);
  }
  return meas_log_likelihood;
}

template <class UDT>
std::map<std::pair<UDT, UDT>, UDT>
LMBDistribution<UDT>::bayes_factor_statistics_measurement_likelihood_without_detection(
    DataSourceContainer<UDT> const& data_source_container) const
{
  if (not data_source_container._sensorInfo._sensor_fov.has_value())
  {
    LOG_FATAL("Can not compute Self Assessment Statistics if no fov is defined");
    LOG_FATAL(std::stacktrace::current());
    std::terminate();
  }
  double const fov_area =
      data_source_container._sensorInfo._sensor_fov.value().area(Components({ COMPONENT::POS_X, COMPONENT::POS_Y }));

  double const clutter_rate = [&] {
    if (_manager->params()->meas_model_params(data_source_container._source_id).clutter.rate.has_value())
    {
      return _manager->params()->meas_model_params(data_source_container._source_id).clutter.rate.value();
    }
    return fov_area * _manager->params()->meas_model_params(data_source_container._source_id).clutter.intensity.value();
  }();

  UDT meas_log_likelihood = -clutter_rate;
  for (State<UDT> const& meas : data_source_container._measurements)
  {
    assert(_manager->database().meas_clutter_intensity.contains(meas._id));
    meas_log_likelihood += std::log(_manager->database().meas_clutter_intensity.find(meas._id)->clutter_intensity);
  }
  UDT num_detected = 0;
  UDT num_missed = 0;
  std::map<StateId, UDT> betas;
  for (StateId track : _tracks)
  {
    State<UDT> const& state = _manager->get_state(track);
    UDT const r = _manager->get_state_existence_probability(track);
    UDT const& r_ = _manager->get_state_existence_probability(state._prior_id);
    assert(_manager->database().state_detection_probabilities.contains(
        std::tuple{ state._prior_id, data_source_container._id }));
    meas_log_likelihood += (1 - r) * std::log(1 - r_) + r * std::log(r_);
    meas_log_likelihood += -(1 - r) * log(1 - r);
    assert(state._meta_data._meas_update_weight.contains(data_source_container._source_id));
    for (auto const& [meas, weight] : state._meta_data._meas_update_weight.at(data_source_container._source_id))
    {
      UDT const alpha = r * weight;
      if (meas == NOT_DETECTED)
      {
        num_missed += alpha;
        meas_log_likelihood += -alpha * log(alpha);
      }
      else
      {
        meas_log_likelihood += (1 - alpha) * log(1 - alpha);
        meas_log_likelihood += -alpha * log(alpha);
        betas[meas] += alpha;
        num_detected += alpha;
        assert(_manager->database().state_meas_log_likelihood.contains(std::tuple{ state._prior_id, meas }));
        UDT const l = _manager->database()
                          .state_meas_log_likelihood.find(std::tuple{ state._prior_id, meas })
                          ->meas_log_likelihood;
        meas_log_likelihood +=
            alpha * (l - std::log(_manager->database().meas_clutter_intensity.find(meas)->clutter_intensity));
      }
    }
  }
  for (const auto& weight : betas | std::views::values)
  {
    meas_log_likelihood += -(1 - weight) * log(1 - weight);
  }
  return { { { num_detected, num_missed }, meas_log_likelihood } };
}

template <class UDT>
std::map<UDT, UDT> LMBDistribution<UDT>::bayes_factor_statistics_measurement_likelihood_without_clutter(
    DataSourceContainer<UDT> const& data_source_container) const
{
  if (not data_source_container._sensorInfo._sensor_fov.has_value())
  {
    LOG_FATAL("Can not compute Self Assessment Statistics if no fov is defined");
    LOG_FATAL(std::stacktrace::current());
    std::terminate();
  }
  UDT meas_log_likelihood = 0;
  UDT num_detected = 0;
  std::map<StateId, UDT> betas;
  for (StateId track : _tracks)
  {
    State<UDT> const& state = _manager->get_state(track);
    UDT const r = _manager->get_state_existence_probability(track);
    UDT const& r_ = _manager->get_state_existence_probability(state._prior_id);
    assert(_manager->database().state_detection_probabilities.contains(
        std::tuple{ state._prior_id, data_source_container._id }));
    UDT const pd = _manager->database()
                       .state_detection_probabilities.find(std::tuple{ state._prior_id, data_source_container._id })
                       ->detection_probability;
    meas_log_likelihood += (1 - r) * std::log(1 - r_) + r * std::log(r_);
    meas_log_likelihood += -(1 - r) * log(1 - r);
    assert(state._meta_data._meas_update_weight.contains(data_source_container._source_id));
    for (auto const& [meas, weight] : state._meta_data._meas_update_weight.at(data_source_container._source_id))
    {
      UDT const alpha = r * weight;
      if (meas == NOT_DETECTED)
      {
        meas_log_likelihood += alpha * std::log(1 - pd);
        meas_log_likelihood += -alpha * log(alpha);
      }
      else
      {
        num_detected += alpha;
        betas[meas] += alpha;
        meas_log_likelihood += (1 - alpha) * log(1 - alpha);
        meas_log_likelihood += -alpha * log(alpha);
        assert(_manager->database().state_meas_log_likelihood.contains(std::tuple{ state._prior_id, meas }));
        UDT const l = _manager->database()
                          .state_meas_log_likelihood.find(std::tuple{ state._prior_id, meas })
                          ->meas_log_likelihood;
        meas_log_likelihood += alpha * (std::log(pd) + l);
      }
    }
  }
  for (const auto& weight : betas | std::views::values)
  {
    meas_log_likelihood += -(1 - weight) * log(1 - weight);
  }
  return { { num_detected, meas_log_likelihood } };
}

template <class UDT>
std::map<MeasurementId, UDT> const&
LMBDistribution<UDT>::meas_assignment_prob(DataSourceContainer<UDT> const& data_source_container) const
{
  LOG_DEB("LMBDistribution::meas_assignment_prob");

  assert([&] {  // NOLINT
    if (not _meas_assignment_prob.contains(data_source_container._id))
    {
      LOG_FATAL(data_source_container.toString());
      LOG_FATAL(std::stacktrace::current());
      for (auto const& [id, data] : _meas_assignment_prob)
      {
        LOG_FATAL(id);
      }
      return false;
    }
    return true;
  }());
  return _meas_assignment_prob.at(data_source_container._id);
}

}  // namespace ttb
