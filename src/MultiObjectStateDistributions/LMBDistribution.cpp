#include "tracking_lib/MultiObjectStateDistributions/LMBDistribution.h"
// #####################################################################################################################
#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/Measurements/MeasurementContainer.h"
#include "tracking_lib/States/Innovation.h"
#include "tracking_lib/StateModels/BaseStateModel.h"
#include "tracking_lib/Graph/Graph.h"
#include "tracking_lib/Misc/Numeric.h"
#include "tracking_lib/SelfAssessment/SelfAssessment.h"
#include "tracking_lib/Misc/Grouping.h"
#include "tracking_lib/MultiObjectStateDistributions/Utils.h"
#include "tracking_lib/Misc/AngleNormalization.h"
#include "tracking_lib/Misc/ProportionalAllocation.h"

#include <execution>
#include <utility>

#include <tracy/tracy/Tracy.hpp>

namespace ttb
{
constexpr const auto tracy_color = tracy::Color::LightSalmon;

IDGenerator<MODistributionId> LMBDistribution::_idGenerator{};

std::string to_string(GLMB2LMBConversionProfilerData const& data)
{
  std::string out = "GLMB2LMB Conversion\n";
  out += "\tDuration: " + std::to_string(to_milliseconds(data._duration)) + "ms\n";
  out += "\t#GLMB Hypotheses: " + std::to_string(data._numGlmbHypotheses) + "\n";
  out += "\t#Tracks: " + std::to_string(data._numTracks) + "\n";
  return out;
}
std::string to_stringStatistics(std::vector<GLMB2LMBConversionProfilerData> const& datas)
{
  std::string out = "GLMB2LMB Conversion Statistics\n";
  Duration meanDuration(0);
  double meanHyp = 0;
  double meanTracks = 0;
  for (auto const& data : datas)
  {
    meanDuration += data._duration / datas.size();
    meanHyp += static_cast<double>(data._numGlmbHypotheses) / datas.size();
    meanTracks += static_cast<double>(data._numTracks) / datas.size();
  }
  out += "\tMean Duration: " + std::to_string(to_milliseconds(meanDuration)) + "ms\n";
  out += "\tMean #GLMB Hypotheses: " + std::to_string(meanHyp) + "\n";
  out += "\tMean #Tracks: " + std::to_string(meanTracks) + "\n";
  return out;
}

std::string to_string(LMBCalcInnovationProfilerData const& data)
{
  std::string out = "LMB Innovation\n";
  out += "\tDuration: " + std::to_string(to_milliseconds(data._duration)) + "ms\n";
  out += "\t#Tracks: " + std::to_string(data._numTracks) + "\n";
  out += "\t#Updated Tracks: " + std::to_string(data._numUpdatedTracks) + "\n";
  out += "\t#Measurements: " + std::to_string(data._numMeasurements) + "\n";
  out += "\tMeasModel ID: " + data._id.value_ + "\n";
  return out;
}
std::string to_stringStatistics(std::vector<LMBCalcInnovationProfilerData> const& datas)
{
  std::string out = "LMB Innovation Statistics\n";
  Duration meanDuration(0);
  double meanTracks = 0;
  double meanUpdatedTracks = 0;
  double meanMeasurements = 0;
  for (auto const& data : datas)
  {
    meanDuration += data._duration / datas.size();
    meanTracks += static_cast<double>(data._numTracks) / datas.size();
    meanUpdatedTracks += static_cast<double>(data._numUpdatedTracks) / datas.size();
    meanMeasurements += static_cast<double>(data._numMeasurements) / datas.size();
  }
  out += "\tMean Duration: " + std::to_string(to_milliseconds(meanDuration)) + "ms\n";
  out += "\tMean #Tracks: " + std::to_string(meanTracks) + "\n";
  out += "\tMean #Updated Tracks: " + std::to_string(meanUpdatedTracks) + "\n";
  out += "\tMean #Measurements: " + std::to_string(meanMeasurements) + "\n";
  return out;
}

std::string to_string(LMBPredictProfilerData const& data)
{
  std::string out = "LMB Prediction\n";
  out += "\tDuration: " + std::to_string(to_milliseconds(data._duration)) + "ms\n";
  out += "\t#Tracks: " + std::to_string(data._numTracks) + "\n";
  return out;
}
std::string to_stringStatistics(std::vector<LMBPredictProfilerData> const& datas)
{
  std::string out = "LMB Prediction Statistics\n";
  Duration meanDuration(0);
  double meanTracks = 0;
  for (auto const& data : datas)
  {
    meanDuration += data._duration / datas.size();
    meanTracks += static_cast<double>(data._numTracks) / datas.size();
  }
  out += "\tMean Duration: " + std::to_string(to_milliseconds(meanDuration)) + "ms\n";
  out += "\tMean #Tracks: " + std::to_string(meanTracks) + "\n";
  return out;
}

std::string to_string(LMBGroupingProfilerData const& data)
{
  std::string out = "LMB Grouping\n";
  out += "\tDuration: " + std::to_string(to_milliseconds(data._duration)) + "ms\n";
  out += "\t#Tracks: " + std::to_string(data._numTracks) + "\n";
  out += "\t#Measurements: " + std::to_string(data._numMesurements) + "\n";
  out += "\tGroups\n";
  for (auto const& [numTracks, numMeas] : data._groups_num_tracks_num_meas)
  {
    out += "\t\t#Tracks: " + std::to_string(numTracks) + "\t#Mesurements: " + std::to_string(numMeas) + "\n";
  }
  out += "\t#NonAssocMeasurements: " + std::to_string(data._numNonAssocMeasurements);
  return out;
}
std::string to_stringStatistics(std::vector<LMBGroupingProfilerData> const& datas)
{
  std::string out = "LMB Grouping Statistics\n";
  double meanDur = 0;
  double meanTracks = 0;
  double meanMeasurements = 0;
  double meanGroups = 0;
  double meanNonAssoc = 0;
  std::pair<std::size_t, std::size_t> biggestGroup{ 0, 0 };
  for (auto const& data : datas)
  {
    meanDur += to_milliseconds(data._duration) / datas.size();
    meanTracks += static_cast<double>(data._numTracks) / datas.size();
    meanMeasurements += static_cast<double>(data._numMesurements) / datas.size();
    meanNonAssoc += static_cast<double>(data._numNonAssocMeasurements) / datas.size();
    auto maxIt = std::ranges::max_element(data._groups_num_tracks_num_meas, [](auto const& a, auto const& b) {
      return a.first + a.second < b.first + b.second;
    });
    if (maxIt != data._groups_num_tracks_num_meas.end())
    {
      if (maxIt->first + maxIt->second > biggestGroup.first + biggestGroup.second)
      {
        biggestGroup = { maxIt->first, maxIt->second };
      }
    }
    meanGroups += static_cast<double>(data._groups_num_tracks_num_meas.size()) / datas.size();
  }
  out += "\tMean Duration: " + std::to_string(meanDur) + "ms\n";
  out += "\tMean #Tracks: " + std::to_string(meanTracks) + "\n";
  out += "\tMean #Measurements: " + std::to_string(meanMeasurements) + "\n";
  out += "\tMean #Groups: " + std::to_string(meanGroups) + "\n";
  out += "\tBiggest Group #Track: " + std::to_string(biggestGroup.second) +
         " #Measurements: " + std::to_string(biggestGroup.second) + "\n";
  out += "\tMean #NonAssocMeasurements: " + std::to_string(meanNonAssoc) + "\n";
  return out;
}

LMBDistribution::LMBDistribution(TTBManager* manager) : LMBDistribution(manager, {})
{
}

LMBDistribution::LMBDistribution(TTBManager* manager, std::vector<State> tracks)
  : _manager{ manager }, _tracks{ std::move(tracks) }
{
}

void LMBDistribution::convertGLMB2LMB(GLMBDistribution glmb)
{
  ZoneScopedNC("LMBDistribution::convertGLMB", tracy_color);
  LOG_DEB("GLMB -> LMB Conversion");
  if (glmb._tracks.empty() and glmb._hypotheses.empty())
  {
    LOG_DEB("Updated Tracks of GLMB are empty");
  }
  LOG_DEB("Convert " << glmb._hypotheses.size() << " GLMB Hypotheses into " << glmb.label2Tracks().size()
                     << " LMB Tracks");
  _tracks.clear();
  for (auto const& [label, trackIds] : glmb.label2Tracks())
  {
    assert(not trackIds.empty() and "Label has no corresponding Tracks");
    Probability const labelExProb{ [&] {
      double prob = glmb.label2ExProb().at(label);
      LOG_DEB("label in GLMB2LMB conversion: " << label << " #trackIds: " << trackIds.size() << " exProb: " << prob);
      return prob;
    }() };
    if (labelExProb < TTB_EPS)
    {
      LOG_DEB("Ignore Label: " << label.value_ << " with existenceProb: " << labelExProb);
      continue;
    }
    State merged_track = std::move(glmb._tracks.at(trackIds.at(0)));
    assert(merged_track.isValid());
    Probability trackExProb = glmb.track2ExProb().at(trackIds.at(0));
    LOG_DEB("mult. state weight of label " << label << " with weight of exProb: " << trackExProb);
    merged_track.multiplyWeights(trackExProb / labelExProb);
    if (merged_track._meta_data._lastAssociatedMeasurement == NOT_DETECTED)
    {
      merged_track._weight_mis_detection = trackExProb / labelExProb;
    }
    std::vector<State> other_states;
    for (std::size_t i = 1; i < trackIds.size(); ++i)
    {
      trackExProb = glmb.track2ExProb().at(trackIds.at(i));
      LOG_DEB("mult. state weight of label " << label << "  with weight of exProb: " << trackExProb);

      State nextTrack = std::move(glmb._tracks.at(trackIds.at(i)));
      assert(nextTrack.isValid());
      nextTrack.multiplyWeights(trackExProb / labelExProb);
      if (nextTrack._meta_data._lastAssociatedMeasurement == NOT_DETECTED)
      {
        merged_track._weight_mis_detection = trackExProb / labelExProb;
      }
      LOG_DEB("nextTrack with label: " << label << nextTrack.toString());
      other_states.push_back(std::move(nextTrack));
    }
    LOG_DEB("#other states: " << other_states.size() << " of label " << label);
    merged_track.merge(std::move(other_states));
    merged_track._existenceProbability = labelExProb;
    if (not merged_track.isEmpty())
    {
      assert(merged_track.isValid());
      _tracks.push_back(std::move(merged_track));
    }
  }
  LOG_DEB("LMB after conversion:\n" << toString());
  assert(isValid());
}

double LMBDistribution::estimatedNumberOfTracks() const
{
  return std::accumulate(_tracks.begin(), _tracks.end(), 0.0, [](double old, State const& track) {
    return old + track._existenceProbability;
  });
}

Vector LMBDistribution::cardinalityDistribution() const
{
  Vector cardDist{ { 1 } };
  for (auto const& track : _tracks)
  {
    double r = track._existenceProbability;
    Vector convKernel{ { 1 - r, r } };
    cardDist = numeric::fold(cardDist, convKernel);
  }
  return cardDist;
}

std::vector<State> LMBDistribution::getEstimate() const
{
  std::vector<State> out;
  switch (_manager->params().lmb_distribution.extraction.type)
  {
    case MO_DISTRIBUTION_EXTRACTION_TYPE::EXISTENCE_PROBABILITY:
    {
      for (State const& track : _tracks)
      {
        if (track._existenceProbability > _manager->params().lmb_distribution.extraction.threshold)
        {
          out.push_back(track);
        }
      }
      return out;
    }
    case MO_DISTRIBUTION_EXTRACTION_TYPE::CARDINALITY:
    {
      Vector cardDist = cardinalityDistribution();
      Index const maxCard = std::distance(cardDist.begin(), std::ranges::max_element(cardDist));

      // sort trackMap by existence Prob in descending order
      std::vector<std::size_t> inds(_tracks.size());
      std::iota(inds.begin(), inds.end(), 0);
      std::ranges::sort(inds, [&](std::size_t a, std::size_t b) {
        return _tracks.at(a)._existenceProbability > _tracks.at(b)._existenceProbability;
      });

      for (Index i = 0; i < maxCard; ++i)
      {
        out.push_back(_tracks.at(i));
      }
      return out;
    }
    case MO_DISTRIBUTION_EXTRACTION_TYPE::BEST_HYPOTHESIS:
    {
      LOG_FATAL("Best Hypothesis not available for LMB");
      return out;
    }
  }
  LOG_FATAL("Something serious wrong going on");
  return {};
}

std::size_t LMBDistribution::truncate(std::size_t maxNTracks)
{
  if (_tracks.size() > maxNTracks)
  {
    std::size_t const before = _tracks.size();
    std::ranges::sort(_tracks,
                      [](State const& a, State const& b) { return a._existenceProbability < b._existenceProbability; });
    _tracks.erase(std::next(_tracks.begin(), maxNTracks), _tracks.end());
    return before - _tracks.size();
  }
  return 0;
}

void LMBDistribution::resetPriorId()
{
  std::ranges::for_each(_tracks, [&](State& track) { track.resetPriorId(); });
}

void LMBDistribution::predict(Duration deltaT, EgoMotionDistribution const& egoDist)
{
  ZoneScopedNC("LMBDistribution::predict", tracy_color);
  LOG_DEB("Predict LMB Distribution with " + std::to_string(_tracks.size()) + " Tracks");
  utils::predict(_manager, _tracks, deltaT, egoDist, {}, [&](State& state) {
    double const survival_probability = _manager->getPersistenceModel().getPersistenceProbability(state, deltaT);
    state._existenceProbability *= survival_probability;
  });
  postProcessPrediction();
}

void LMBDistribution::update(Groups groups)
{
  ZoneScopedNC("LMBDistribution::group_update", tracy_color);
  assert(not groups.empty() and "invalid group size");
  if (_manager->params().thread_pool_size > 0 and
      _manager->params().lmb_distribution.calculate_single_sensor_group_updates_parallel)
  {
    LOG_DEB("parallel group update");
    _manager->thread_pool().detach_loop(std::size_t{ 0 }, groups.size(), [&](std::size_t i) {
      groups.at(i).first.single_group_update(groups.at(i).second);
    });
    _manager->thread_pool().wait();
  }
  else
  {
    LOG_DEB("sequential group update");
    for (std::size_t i = 0; i != groups.size(); i++)
    {
      groups.at(i).first.single_group_update(groups.at(i).second);
    }
  }
  for (std::size_t i = 1; i < groups.size(); ++i)
  {
    groups.at(0).first.merge(std::move(groups.at(i).first));
  }
  *this = std::move(groups.at(0).first);
}

void LMBDistribution::update(MeasurementContainer const& measurementContainer)
{
  ZoneScopedNC("LMBDistribution::update", tracy_color);
  if (_manager->params().lmb_distribution.use_grouping)
  {
    LOG_DEB("use grouping");
    std::vector<grouping::Group> groups = grouping::group(_tracks, measurementContainer);
    Groups lmb_groups;
    for (auto& [tracks, measurement_container] : groups)
    {
      lmb_groups.emplace_back(LMBDistribution(_manager, std::move(tracks)), std::move(measurement_container));
    }
    update(std::move(lmb_groups));
    postProcessUpdate();
  }
  else
  {
    LOG_DEB("no grouping");
    single_group_update(measurementContainer);
    postProcessUpdate();
  }
  if (_manager->params().show_gui)
  {
    double assoc = 0;
    for (auto const& [meas_id, assoc_prob] : meas_assignment_prob())
    {
      assoc += assoc_prob;
    }
    {
      std::lock_guard lock(_manager->vizu().add_data_mutex);
      _manager->vizu()._meas_model_data[measurementContainer._id]._num_assoc_measurements.emplace_back(
          measurementContainer._time, assoc);
    }
    auto tracks = getEstimate();
    std::map<COMPONENT, std::pair<double, std::size_t>> nis_values;
    for (State const& track : tracks)
    {
      double best_weight = 0;
      std::map<COMPONENT, double> track_nis;
      for (auto const& [model_nis, data] : track._nis)
      {
        for (auto const& [weight, values] : data)
        {
          if (weight > best_weight)
          {
            best_weight = weight;
            track_nis = values;
          }
        }
      }
      for (auto const& [comp, nis] : track_nis)
      {
        nis_values[comp] = { nis_values[comp].first + nis, nis_values[comp].second + 1 };
      }
    }
    std::lock_guard lock(_manager->vizu().add_data_mutex);
    for (auto const& [comp, nis_dof] : nis_values)
    {
      _manager->vizu()._meas_model_data[measurementContainer._id]._nis_dof[comp].emplace_back(
          measurementContainer._time, nis_dof.first, nis_dof.second);
    }
  }
}

void LMBDistribution::single_group_update(MeasurementContainer const& measurementContainer)
{
  ZoneScopedNC("LMBDistribution::single_group_lbp_update", tracy_color);
  switch (_manager->params().lmb_distribution.update_method)
  {
    case LMB_UPDATE_METHOD::GLMB:
      single_group_glmb_update(measurementContainer);
      return;
    case LMB_UPDATE_METHOD::LBP:
      single_group_lbp_update(measurementContainer);
      return;
  }
  DEBUG_ASSERT_MARK_UNREACHABLE;
}

void LMBDistribution::single_group_lbp_update(MeasurementContainer const& measurementContainer)
{
  ZoneScopedNC("LMBDistribution::single_group_lbp_update", tracy_color);
  std::size_t const L = _tracks.size();
  if (L == 0)
  {
    _meas_assignment_prob = [&] {
      std::map<MeasurementId, Probability> rz;
      for (Measurement const& meas : measurementContainer._data)
      {
        rz[meas._id] = 0;
      }
      return rz;
    }();
    sa::ClutterDetectionDistribution stats = sa::compute_dists(_tracks, measurementContainer._data.size());
    _clutter_dist = std::move(stats.clutter_dist);
    _detection_dist = std::move(stats.detection_dist);
    return;
  }
  auto const M = static_cast<Index>(measurementContainer._data.size());
  Index const not_detected = M;
  Index const not_existent = M + 1;
  Matrix XI(L, M + 2);
  Matrix NU = Matrix::Ones(M + 2, L);
  Matrix BETA = Matrix::Zero(L, M + 2);
  std::vector<std::size_t> estimated_tracks_ind;
  for (auto const& [l, state] : std::views::enumerate(_tracks))
  {
    Innovation const& innovation = state._innovation.at(measurementContainer._id);
    for (auto const& [m, meas] : std::views::enumerate(measurementContainer._data))
    {
      if (innovation._updates.contains(meas._id))
      {
        BETA(l, m) = state._existenceProbability * std::exp(innovation._updates.at(meas._id).log_likelihood) *
                     innovation._detectionProbability / innovation._updates.at(meas._id).clutter_intensity;
      }
    }
    BETA(l, not_detected) = state._existenceProbability * (1 - innovation._detectionProbability);
    BETA(l, not_existent) = 1 - state._existenceProbability;
  }
  bool converged = false;
  double convergence_crit = std::numeric_limits<double>::infinity();
  for (std::size_t iteration = 0;
       not converged and iteration < _manager->params().lmb_distribution.loopy_belief_propagation.max_iterations;
       ++iteration)
  {
    Matrix const& BETA_RED = BETA(Eigen::all, Eigen::seq(0, M - 1));
    Matrix const& NU_RED = NU(Eigen::seq(0, M - 1), Eigen::all);
    Vector DENUM = (BETA_RED * NU_RED).diagonal() + BETA(Eigen::all, not_existent) + BETA(Eigen::all, not_detected);
    XI(Eigen::all, Eigen::seq(0, M - 1)) =
        BETA_RED.array() / (DENUM.replicate(1, M).array() - BETA_RED.array() * NU_RED.transpose().array());
    XI(Eigen::all, { not_detected, not_existent }) =
        BETA(Eigen::all, { not_detected, not_existent }).array() / DENUM.replicate(1, 2).array();
    Matrix NU_Update = (1 / (1 + (XI.colwise().sum().replicate(L, 1) - XI).array())).transpose();
    convergence_crit = (NU - NU_Update).array().abs().sum() / NU.array().abs().sum();
    assert([&] {  // NOLINT
      if (not std::isfinite(convergence_crit))
      {
        LOG_FATAL("NU " << NU);
        LOG_FATAL("NU_UPDATE" << NU_Update);
        LOG_FATAL("Criteria " << convergence_crit);
        return false;
      }
      return true;
    }());
    if (convergence_crit < _manager->params().lmb_distribution.loopy_belief_propagation.tol)
    {
      converged = true;
      TracyPlot("#LBP Iterations", static_cast<int64_t>(iteration));
    }
    NU = std::move(NU_Update);
  }
  if (not converged)
  {
    TracyPlot("#LBP Iterations",
              static_cast<int64_t>(_manager->params().lmb_distribution.loopy_belief_propagation.max_iterations));
    LOG_WARN_THROTTLE(10,
                      "LBP not converged after " +
                          std::to_string(_manager->params().lmb_distribution.loopy_belief_propagation.max_iterations) +
                          " Convergence criteria: " + std::to_string(convergence_crit) + " <! " +
                          std::to_string(_manager->params().lmb_distribution.loopy_belief_propagation.tol) +
                          " not satisfied");
  }
  Matrix P(L, M + 2);
  P(Eigen::all, Eigen::seq(0, M - 1)) =
      BETA(Eigen::all, Eigen::seq(0, M - 1)).array() * NU(Eigen::seq(0, M - 1), Eigen::all).transpose().array();
  P(Eigen::all, { not_detected, not_existent }) = BETA(Eigen::all, { not_detected, not_existent });
  P(Eigen::all, Eigen::all).array() /= P.rowwise().sum().replicate(1, M + 2).array();
  std::map<MeasurementId, double> assignment_any_track;
  for (auto const& [m, meas] : std::views::enumerate(measurementContainer._data))
  {
    assignment_any_track[meas._id] = 1.0 - 1.0 / (1 + XI.col(m).sum());
  }
  // update state
  std::vector<State> updated_states;
  for (auto const& [l, state] : std::views::enumerate(_tracks))
  {
    assert(not state._innovation.at(measurementContainer._id)._updates.empty());
    auto& [meas_id, update] = *state._innovation.at(measurementContainer._id)._updates.begin();
    State mergedState = std::move(update.updated_dist);
    Index const m = std::distance(measurementContainer._data.begin(),
                                  std::ranges::find_if(measurementContainer._data, [meas_id](Measurement const& meas) {
                                    return meas._id == meas_id;
                                  }));
    double const weight = P(l, m);
    assert(weight >= 0 and weight <= 1);
    mergedState._existenceProbability = weight;
    mergedState.multiplyWeights(weight);
    mergedState.multiplyNisWeights(weight);

    if (meas_id == NOT_DETECTED)
    {
      mergedState._weight_mis_detection = weight;
    }
    for (auto& [next_meas_id, next_update] : state._innovation.at(measurementContainer._id)._updates)
    {
      if (meas_id == next_meas_id)
      {
        continue;
      }
      Index const next_m =
          std::distance(measurementContainer._data.begin(),
                        std::ranges::find_if(measurementContainer._data, [next_meas_id](Measurement const& meas) {
                          return meas._id == next_meas_id;
                        }));
      double const next_weight = P(l, next_m);
      assert(next_weight >= 0 and next_weight <= 1);
      next_update.updated_dist.multiplyWeights(next_weight);
      next_update.updated_dist.multiplyNisWeights(next_weight);
      mergedState.merge(std::move(next_update.updated_dist));

      mergedState._existenceProbability += next_weight;
      if (next_meas_id == NOT_DETECTED)
      {
        mergedState._weight_mis_detection = next_weight;
      }
    }
    if (mergedState._existenceProbability > 1 + 1e-5)
    {
      LOG_WARN("created existence prob > 1 + 1e-5. Clamp to 1.");
      mergedState._existenceProbability = 1;
    }
    else if (mergedState._existenceProbability > 1)
    {
      mergedState._existenceProbability = 1;
    }
    assert(mergedState._existenceProbability >= 0 and mergedState._existenceProbability <= 1);
    mergedState.multiplyWeights(1 / mergedState.sumWeights());
    updated_states.push_back(std::move(mergedState));
  }
  _tracks = std::move(updated_states);
  _meas_assignment_prob = std::move(assignment_any_track);
  sa::ClutterDetectionDistribution stats = sa::compute_dists(_tracks, measurementContainer._data.size());
  _clutter_dist = std::move(stats.clutter_dist);
  _detection_dist = std::move(stats.detection_dist);
}

void LMBDistribution::single_group_glmb_update(MeasurementContainer const& measurementContainer)
{
  ZoneScopedNC("LMBDistribution::single_group_glmb_update", tracy_color);
  LOG_DEB("Update LMB Distribution: " << toString());
  GLMBDistribution glmbDist(_manager, std::move(_tracks));
  glmbDist.update(measurementContainer);
  _meas_assignment_prob = glmbDist.probOfAssigment(measurementContainer);
  _clutter_dist = glmbDist.clutter_distribution(static_cast<Index>(measurementContainer._data.size()));
  _detection_dist = glmbDist.detection_distribution();
  LOG_DEB("GLMB Dist after update:\n" << glmbDist.toString());
  convertGLMB2LMB(std::move(glmbDist));
  LOG_DEB("LMB Dist after update:\n" << toString());
  assert(isValid() && "after update");
}

void LMBDistribution::fpm_fusion(std::vector<LMBDistribution>&& updated_lmbs, bool isDynamicBirth)
{
  ZoneScopedN("LMBDistribution::fpm_fusion");
  LOG_DEB("FPM fusion with " << updated_lmbs.size()
                             << " single-sensor updated lmb densities. Is is dynamic birth?: " << isDynamicBirth);
  if (_manager->params().filter.lmb_fpm.sensor_number != updated_lmbs.size() && !isDynamicBirth)
  {
    LOG_ERR("number of updated lmb densities does not fit to number of sensors");
  }
  LMBDistribution fusedLMB(_manager);
  if (updated_lmbs.size() == 1)
  {
    ZoneScopedN("LMB_FPM_Tracker::fusion_single_sensor_case");
    // no fusion is needed since only one sensor update is given
    _tracks.clear();
    merge(updated_lmbs.at(0));
    return;
  }

  // FPM-LMB fusion of lmb dists
  std::vector<Label> prior_labels;

  // sort prior and local posterior tracks depending on their label
  std::map<Label, std::vector<State>> label2UpdatedTracksMap;
  std::map<Label, std::vector<State>> label2PriorTracksMap;
  std::stringstream priorLabelsStr;
  {
    ZoneScopedN("LMB_FPM_Tracker::fpmFusion_fillMaps");
    for (const auto& track_prior : _tracks)
    {
      auto it = label2PriorTracksMap.find(track_prior._label);
      if (it != label2PriorTracksMap.end())
      {
        std::stringstream msg;
        msg << "Prior lmb density contains the same label twice...";
        throw std::runtime_error(msg.str());
      }
      priorLabelsStr << track_prior._label << " ";
      prior_labels.push_back(track_prior._label);
      label2PriorTracksMap[track_prior._label].push_back(track_prior);
    }
    LOG_DEB("Prior Labels: " << priorLabelsStr.str());

    bool emptySensorUpdate = false;
    bool emptyTrackUpdate = false;
    std::vector<Label> emptyTrackLabels;
    std::vector<Label> addedLabels;
    for (const auto& posterior_lmb : updated_lmbs)
    {
      LOG_DEB("next sensor!");

      // one sensor has an empty distribution -> fusion would throw away all tracks
      if (posterior_lmb._tracks.empty())
      {
        emptySensorUpdate = true;
        break;
      }
      bool foundAtrack = false;
      for (auto& label_prior : prior_labels)
      {
        auto trackIt = std::find_if(posterior_lmb._tracks.begin(),
                                    posterior_lmb._tracks.end(),
                                    [&label_prior](const State& obj) { return obj._label == label_prior; });
        if (trackIt != posterior_lmb._tracks.end())
        {
          // add posterior track to track list of label
          LOG_DEB("Add label " << label_prior << " to map ");
          addedLabels.push_back(label_prior);
          label2UpdatedTracksMap[label_prior].push_back(std::move(*trackIt));
          foundAtrack = true;
        }
        else
        {
          // track doesn't exist anymore in this distribution -> fusion will throw away this track
          auto itLabel = std::find(emptyTrackLabels.begin(), emptyTrackLabels.end(), label_prior);
          if (itLabel == emptyTrackLabels.end())
          {
            // this is the first distribution, where this track is vanished -> add label to list
            LOG_INF("Remove track with label " << label_prior << " since it vanished in one ore more posterior dists!");
            emptyTrackLabels.push_back(label_prior);
          }
          emptyTrackUpdate = true;
        }
      }
      if (!foundAtrack)
      {
        // This case should never happen!!
        LOG_ERR(" No Track found for fusion of this sensor");
        emptySensorUpdate = true;
        std::stringstream msg;
        msg << "LMB dist of sensor has labels which are not in lmb prior dist";
        throw std::runtime_error(msg.str());
      }
    }

    // remove tracks, which can not exist anymore after fusion
    if (emptyTrackUpdate)
    {
      assert(emptyTrackLabels.size() > 0 && "no labels to delete!");
      if (emptyTrackLabels.empty())
      {
        throw std::runtime_error("No Labels to delete. Something is weird!");
      }
      // delete track from label2localTracksMap
      for (auto& emptyLabel : emptyTrackLabels)
      {
        if (std::find(addedLabels.begin(), addedLabels.end(), emptyLabel) != addedLabels.end())
        {
          label2UpdatedTracksMap.erase(emptyLabel);
        }
      }
    }

    if (emptySensorUpdate || label2UpdatedTracksMap.empty())
    {
      if (!isDynamicBirth)  // in dynamic birth his behaviour is ok!
      {
        LOG_WARN("Received at least one empty distribution or each track is vanished in at least one posterior "
                 "distribution => no fusion is performed");
        if (emptySensorUpdate)
        {
          LOG_WARN("Empty sensor update (is ok in first cycle of fpm-lmb and if you use a dynamic birth)");
        }
      }
      // return empty LMB distribution
      _tracks.clear();
      return;
    }
  }

  fuse_distributions_fpm(std::move(label2PriorTracksMap), std::move(label2UpdatedTracksMap));
  LOG_DEB("Fused dist after fpm fusion: " << fusedLMB.toString());
  assert(isValid() && "Fused dist after fpm fusion is not valid!");
}

void LMBDistribution::fuse_distributions_fpm(std::map<Label, std::vector<State>>&& label2PriorTracksMap,
                                             std::map<Label, std::vector<State>>&& label2UpdatedTracksMap)
{
  ZoneScopedN("LMB_FPM_Tracker::fuseLMBDistributions");
  _tracks.clear();
  std::mutex fused_lmbs_mutex;
  if (_manager->params().thread_pool_size > 0 && _manager->params().filter.lmb_fpm.calculate_fpm_fusion_tracks_parallel)
  {
    auto kv = std::views::keys(label2UpdatedTracksMap);
    std::vector<Label> labels{ kv.begin(), kv.end() };

    _manager->thread_pool().detach_loop(std::size_t{ 0 }, labels.size(), [&](std::size_t i) {
      LOG_DEB("Fusion of tracks with label " << labels.at(i));
      if (label2PriorTracksMap[labels.at(i)].size() > 1 || label2PriorTracksMap[labels.at(i)].empty())
      {
        throw std::runtime_error("more than one possible prior tracks or no existing prior track for this label!");
      }
      State prior_track = std::move(*label2PriorTracksMap[labels.at(i)].begin());
      Time fused_time = label2UpdatedTracksMap[labels.at(i)].begin()->_time;
      State fused_track = fuse_tracks_fpm(std::move(prior_track),
                                          label2UpdatedTracksMap[labels.at(i)].begin(),
                                          label2UpdatedTracksMap[labels.at(i)].end());
      fused_track._time = fused_time;
      if (fused_track._existenceProbability != 0.0)
      {
        std::unique_lock lock(fused_lmbs_mutex);
        addTrack(std::move(fused_track));
      }
      else
      {
        LOG_DEB("Track with label " << labels.at(i) << " vanished since no fusion result is available!");
      }
    });
    _manager->thread_pool().wait();
  }
  else
  {
    for (const auto& [label, trackList] : label2UpdatedTracksMap)
    {
      // fusion
      LOG_DEB("Fusion of tracks with label " << label);
      if (label2PriorTracksMap[label].size() > 1 || label2PriorTracksMap[label].empty())
      {
        throw std::runtime_error("more than one possible prior tracks or no existing prior track for this label!");
      }
      State prior_track = std::move(*label2PriorTracksMap[label].begin());
      Time fused_time = trackList.begin()->_time;
      State fused_track = fuse_tracks_fpm(std::move(prior_track), trackList.begin(), trackList.end());
      fused_track._time = fused_time;
      if (fused_track._existenceProbability > 0.0)
      {
        addTrack(std::move(fused_track));
      }
      else
      {
        LOG_DEB("Track with label " << label << " vanished since no fusion result is available!");
      }
    }
  }
}

template <typename TrackIt>
State LMBDistribution::fuse_tracks_fpm(State&& priorTrack, TrackIt localTracksBegin, TrackIt localTracksEnd) const
{
  ZoneScopedN("LMB_FPM_Tracker::fuseTracks_fpm");
  LOG_DEB("Fuse Tracks with label " << priorTrack._label);
  if (localTracksBegin == localTracksEnd)
  {
    // no fusion is needed since no posterior distribution is given
    LOG_ERR("Should not end up here!");
    throw std::runtime_error("No posterior updates are given! How can you end up here?!? Should be catched before");
  }
  // calculate existence probability products
  double r_fused = 0.;  // is calculated later
  double r_prior = priorTrack._existenceProbability;
  LOG_DEB("r_prior: " << r_prior);
  double r_posterior_product = 1.;
  double r_inverse_posterior_product = 1.;
  calculate_product_r_factors(localTracksBegin, localTracksEnd, r_posterior_product, r_inverse_posterior_product);

  // Fusion
  const Label priorLabel = priorTrack._label;
  State fusedDist = _manager->createState();
  std::size_t numSensors = 0;
  double weightSumMMF = 0.0;  // check calculation in case of MMF!
  bool fusedResultIsEmpty = true;
  auto const& stateModels = _manager->getStateModelMap();

  //  LOG_INF("Prior Track: " << priorTrack.toString());
  //  for (TrackIt track = localTracksBegin; track != localTracksEnd; ++track)
  //  {
  //    LOG_INF("posterior Track: " << track->toString());
  //  }

  for (auto const& [model_id, mixture] : priorTrack._state_dist)
  {
    BaseStateModel const& stateModel = *stateModels.at(model_id);

    const std::vector<BaseDistribution*>& priorMixtureComponents = mixture->dists();
    if (priorMixtureComponents.empty())
    {
      LOG_INF("Skipping empty mixture in track fusion");
      continue;
    }

    if (_manager->params().filter.lmb_fpm.calculate_true_k_best)
    {
      // calculate the totally k best solutions using the generalizedKBest algorithm
      LOG_INF("GeneralizedKBest");
      FPMFusionInfos fusionInfos =
          create_graph_fpm(localTracksBegin, localTracksEnd, priorMixtureComponents, model_id, numSensors);

      //    size_t counter_priorComps = fusionInfos._usedPriorComps.size();

      if (fusionInfos._startNode == fusionInfos._endNode)
      {
        LOG_WARN("No Fusion result for model_id " << model_id
                                                  << " since no prior mixture component is given by all sensors!");
        continue;
      }
      // GeneralizedKBestSelection algorithm
      LOG_INF("start with generalizedKBestSelection algorithm, pmNumBestComponents_k: "
              << _manager->params().filter.lmb_fpm.pmNumBestComponents_k);
      std::vector<std::pair<std::vector<Edge_FPM>, double>> solutions = fusionInfos._graph.k_shortest_paths(
          fusionInfos._startNode, fusionInfos._endNode, _manager->params().filter.lmb_fpm.pmNumBestComponents_k);

      // calculate fused result of each selection
      if (solutions.empty())
      {
        // There is no fused result for this label, but there should be one!
        LOG_ERR("Something is weird with the generalizedKbestSelectionAlgorithm!");  // This should never happen
        throw std::runtime_error("Something is weird with the generalizedKBestAlgorithm!");
        continue;
      }
      // calculate for every prior component fused gm components
      std::vector<std::unique_ptr<BaseDistribution>> fusedComponents =
          fuse_mixture_components_generalized_k_best_fpm(fusionInfos, solutions, numSensors, weightSumMMF, stateModel);
      fusedDist._state_dist.at(stateModel.id())->merge(std::move(fusedComponents));
      fusedResultIsEmpty = false;
    }
    else
    {
      LOG_INF("KBest");
      // calculate k depending on the prior mixture component weight and only use the kBestSelection algorithm for
      // each prior component separately calculate the actual k best solutions using the generalizedKBestalgorithm
      std::vector<FPMFusionInfos> fusionInfosVec =
          create_graphs_fpm(localTracksBegin, localTracksEnd, priorMixtureComponents, model_id, numSensors);

      if (fusionInfosVec.empty())
      {
        LOG_WARN("No Fusion result for model_id " << model_id
                                                  << " since no prior mixture component is given by all sensors!");
        continue;
      }

      // K BestSelection algorithm
      LOG_INF("start with kBestSelection algorithm, pmNumBestComponents_k: "
              << _manager->params().filter.lmb_fpm.pmNumBestComponents_k);
      std::vector<std::pair<FPMFusionPart, std::vector<std::pair<std::vector<Edge_FPM>, double>>>> solutionsVec;
      for (auto const& fusionInfos : fusionInfosVec)
      {
        if (fusionInfos._k == 0)
        {
          continue;
        }
        std::vector<std::pair<std::vector<Edge_FPM>, double>> solutions =
            fusionInfos._graph.k_shortest_paths(fusionInfos._startNode, fusionInfos._endNode, fusionInfos._k);

        if (solutions.empty())
        {
          // There is no fused result for this label!
          LOG_ERR(
              "Something is weird with the kBestSelectionAlgorithm! There should be at least one solution!");  // This
                                                                                                               // should
                                                                                                               // never
                                                                                                               // happen
          throw std::runtime_error("Something is weird with the kBestSelectionAlgorithm! There should be at least "
                                   "one solution!");
        }
        solutionsVec.emplace_back(std::move(fusionInfos._partInfos), std::move(solutions));
      }

      // calculate for every prior component fused gm components
      std::vector<std::unique_ptr<BaseDistribution>> fusedComponents =
          fuse_mixture_components_k_best_fpm(solutionsVec, numSensors, weightSumMMF, stateModel);
      fusedDist._state_dist.at(stateModel.id())->merge(std::move(fusedComponents));
      fusedResultIsEmpty = false;
    }
  }

  // calculate fused r, for the formula derivation check the FPM-LMB Paper!
  // (https://ieeexplore.ieee.org/document/10224189) ToDo(hermann): In case of multi model filter (multiple state
  // models) the calculation of the fused existence probability is wrong! (weightSumMMF!!!)
  if (fusedResultIsEmpty)  // This check is not necessary, since in case of an empty fusion result, weightSumMMF
                           // should be empty and then r_fused = 0. It is here for safety reasons
  {
    LOG_WARN("Fused result is empty!");
    if (weightSumMMF > 0)
    {
      throw std::runtime_error("Weight sum of gm components should be empty in case of an empty fusion result!");
    }
    r_fused = 0.0;
    LOG_WARN("r_fused = " << r_fused);
    fusedDist._existenceProbability = r_fused;  // todo(hermann): is this necessary?
    return fusedDist;
  }

  // calculate fused existence probability
  r_fused = calculate_fused_existence_probability(
      r_prior, r_posterior_product, r_inverse_posterior_product, weightSumMMF, numSensors);
  LOG_DEB("r_fused = " << r_fused);
  fusedDist._existenceProbability = r_fused;
  fusedDist._label = priorLabel;
  return fusedDist;
}

template <typename TrackIt>
void LMBDistribution::calculate_product_r_factors(TrackIt localTracksBegin,
                                                  TrackIt localTracksEnd,
                                                  double& r_posterior_product,
                                                  double& r_inverse_posterior_product) const
{
  for (TrackIt localTrack = localTracksBegin; localTrack != localTracksEnd; ++localTrack)
  {
    double r_posterior = localTrack->_existenceProbability;
    LOG_DEB("r_posterior: " << r_posterior);
    if (r_posterior >= 1.0)
    {
      r_posterior = std::min(1.0 - TTB_EPS, r_posterior);
    }
    r_posterior_product *= r_posterior;
    r_inverse_posterior_product *= (1 - r_posterior);
  }
}

template <typename TrackIt>
LMBDistribution::FPMFusionInfos
LMBDistribution::create_graph_fpm(TrackIt localTracksBegin,
                                  TrackIt localTracksEnd,
                                  const std::vector<BaseDistribution*>& priorMixtureComponents,
                                  const StateModelId model_id,
                                  std::size_t& numSensors) const
{
  ZoneScopedN("LMB_FPM_Tracker::createGraph");
  if constexpr (std::is_base_of_v<std::random_access_iterator_tag,
                                  typename std::iterator_traits<TrackIt>::iterator_category>)
  {
    numSensors = localTracksEnd - localTracksBegin;
  }
  std::map<DistributionId, BaseDistribution*> usedPriorComps;
  std::vector<std::map<DistributionId, const BaseDistribution*>> edgeId2localComp(numSensors);
  std::vector<std::size_t> nodes;
  std::vector<graph::Edge<Edge_FPM, std::size_t, double>> edges;
  std::size_t priorCompsNumber = priorMixtureComponents.size();
  LOG_INF("number of prior components: " << priorCompsNumber << " numSensors: " << numSensors);

  std::size_t priorCompCounter = 1;
  std::size_t highestPossibleVal = priorCompsNumber * (numSensors + 1) + 1;
  std::size_t startNode = 0;
  nodes.push_back(startNode);
  std::size_t endNode = highestPossibleVal;
  for (const auto& prior_component : priorMixtureComponents)
  {
    DistributionId prior_id = prior_component->id();

    bool isempty = false;
    std::size_t sensorCounter = 1;
    std::vector<std::size_t> nodes_tmp;
    std::vector<graph::Edge<Edge_FPM, std::size_t, double>> edges_comp;
    std::size_t sensorId = 0;
    for (TrackIt track = localTracksBegin; track != localTracksEnd; ++track, sensorId++)
    {
      const std::map<StateModelId, std::unique_ptr<BaseDistribution>>& locDist = track->_state_dist;
      std::vector<graph::Edge<Edge_FPM, std::size_t, double>> edges_sensor;

      size_t node1 = (priorCompCounter - 1) * (numSensors + 1) + sensorCounter;
      sensorCounter++;
      size_t node2 = (priorCompCounter - 1) * (numSensors + 1) + sensorCounter;
      if (node2 == priorCompCounter * (numSensors + 1))
      {
        // last node of this prior component id
        node2 = endNode;
      }
      for (const auto& [loc_model_id, locMixture] : locDist)
      {
        if (loc_model_id != model_id)
        {
          // gaussian components must have the same state model id!
          continue;
        }
        const auto& localMixtureComponents = locMixture->dists();
        for (const auto& loc_comp : localMixtureComponents)
        {
          if ((loc_comp->priorId() == prior_id) ||
              ((loc_comp->priorId() == NO_DISTRIBUTION_ID_HISTORY) &&
               (loc_comp->id() == prior_id)))  // prior_id==loc_comp->priorId() || missdetected component(corresponds
                                               // to priorId=0) with id==prior_id
          {
            //           LOG_INF("found fitting component! " << loc_comp->toString());
            LOG_INF("Create new edge with (dist_id, sensorId)=("
                    << loc_comp->id() << ", " << sensorId << "), startNode: " << node1 << " end node: " << node2);
            edges_sensor.emplace_back(std::make_pair(sensorId, loc_comp->id()), node1, node2, loc_comp->sumWeights());
            edgeId2localComp[sensorId].emplace(loc_comp->id(), loc_comp);
          }
        }
      }

      if (edges_sensor.empty())
      {
        // one sensor has no MixtureComponents!
        LOG_INF("Sensor does not deliver results for component id " << prior_id);
        isempty = true;
        break;
      }
      nodes_tmp.push_back(node1);
      nodes_tmp.push_back(node2);
      edges_comp.insert(
          edges_comp.end(), std::make_move_iterator(edges_sensor.begin()), std::make_move_iterator(edges_sensor.end()));
    }
    if (!isempty)
    {
      LOG_INF("insert nodes and edges for this prior id: " << prior_id);
      nodes.insert(nodes.end(), std::make_move_iterator(nodes_tmp.begin()), std::make_move_iterator(nodes_tmp.end()));
      edges.insert(edges.end(), std::make_move_iterator(edges_comp.begin()), std::make_move_iterator(edges_comp.end()));
      double factor = 1 / std::pow(prior_component->sumWeights(), numSensors - 1);
      // Add Dummy Edge containing weight of 1/(V-1)*priorCompWeight
      std::size_t first_node_id = (priorCompCounter - 1) * (numSensors + 1) + 1;
      edges.emplace_back(std::make_pair(numSensors, prior_id), startNode, first_node_id, factor);
      LOG_INF("Create new dummy edge (dist_id, sensorId)=("
              << prior_id << ", " << numSensors << ") startNode: " << startNode << " end node: " << first_node_id);
      usedPriorComps.emplace(prior_id, prior_component);
    }
    priorCompCounter++;
  }
  // remove duplicated nodes
  sort(nodes.begin(), nodes.end());
  nodes.erase(unique(nodes.begin(), nodes.end()), nodes.end());

  LOG_INF("startNode: " << startNode << ", endNode: " << endNode);
  if (edges.empty())
  {
    // Empty graph! all prior components are not present in all posterior tracks!
    endNode = startNode;
  }
  graph::DiGraph<std::size_t, Edge_FPM, double> graph = graph::DiGraph(std::move(nodes), std::move(edges));

  FPMFusionPart partInfos;
  partInfos._usedPriorComps = std::move(usedPriorComps);
  partInfos._edgeId2localComp = std::move(edgeId2localComp);
  return { ._graph = std::move(graph),
           ._partInfos = std::move(partInfos),
           ._startNode = startNode,
           ._endNode = endNode,
           ._k = _manager->params().filter.lmb_fpm.pmNumBestComponents_k };
}

std::size_t LMBDistribution::scaled_by_prior_weight(const double weight) const
{
  return static_cast<std::size_t>(weight *
                                  static_cast<double>(_manager->params().filter.lmb_fpm.pmNumBestComponents_k));
}

std::size_t LMBDistribution::scaled_by_prior_weight_poisson(std::map<DistributionId, Index>& componentID2Alloc,
                                                            DistributionId prior_id) const
{
  ZoneScopedN("LMB_FPM_Tracker::scaledByPriorWeightPoisson");
  Vector scaling_weights{ { 1 } };
  if (!componentID2Alloc.empty())
  {
    Indices slots = propAllocation(componentID2Alloc.at(prior_id),
                                   scaling_weights,
                                   _manager->params().filter.lmb_fpm.equal_allocation_share_ratio);
    return slots(0);
  }
  return 0;
}

std::size_t LMBDistribution::get_k(const double weight,
                                   std::map<DistributionId, Index>& componentID2Alloc,
                                   DistributionId prior_id) const
{
  if (_manager->params().filter.lmb_fpm.calculate_poisson_k_best)
  {
    return scaled_by_prior_weight_poisson(componentID2Alloc, prior_id);
  }
  else
  {
    return scaled_by_prior_weight(weight);
  }
}

template <typename TrackIt>
std::vector<LMBDistribution::FPMFusionInfos>
LMBDistribution::create_graphs_fpm(TrackIt localTracksBegin,
                                   TrackIt localTracksEnd,
                                   const std::vector<BaseDistribution*>& priorMixtureComponents,
                                   const StateModelId model_id,
                                   std::size_t& numSensors) const
{
  ZoneScopedN("LMB_FPM_Tracker::createGraphs");
  std::vector<LMBDistribution::FPMFusionInfos> graphs;
  if constexpr (std::is_base_of_v<std::random_access_iterator_tag,
                                  typename std::iterator_traits<TrackIt>::iterator_category>)
  {
    numSensors = localTracksEnd - localTracksBegin;
  }

  std::map<DistributionId, Index> componentID2Alloc;
  std::vector<DistributionId> component_ids;

  if (_manager->params().filter.lmb_fpm.calculate_poisson_k_best)
  {
    std::size_t num_prior_comps = priorMixtureComponents.size();
    Eigen::ArrayXf weights(num_prior_comps);
    component_ids.resize(num_prior_comps);
    std::size_t counter = 0;
    for (const auto& prior_component : priorMixtureComponents)
    {
      component_ids.at(counter) = prior_component->id();
      weights(counter) = prior_component->sumWeights();
      counter++;
    }

    if (num_prior_comps != 1)
    {
      Eigen::ArrayXf linspace = Eigen::ArrayXf::LinSpaced(num_prior_comps, 0, num_prior_comps - 1);
      Eigen::ArrayXf prod = weights * linspace;
      double lambda = prod.sum();
      Vector poiss_probs = Vector::Zero(num_prior_comps);
      poiss_probs(0) = std::exp(-lambda);
      for (std::size_t n = 1; n <= num_prior_comps - 1; ++n)
      {
        poiss_probs(n) = lambda / (n)*poiss_probs(n - 1);
      }
      Indices alloc = propAllocation(_manager->params().filter.lmb_fpm.pmNumBestComponents_k,
                                     poiss_probs,
                                     _manager->params().filter.lmb_fpm.equal_allocation_share_ratio);
      const auto alloc_slots = alloc.sum();
      assert(alloc_slots <= _manager->params().filter.lmb_fpm.pmNumBestComponents_k && "too much slots are allocated");
      if (static_cast<double>(alloc_slots) < _manager->params().filter.lmb_fpm.pmNumBestComponents_k * 0.85)
      {
        LOG_WARN("Number of allocated slots is very small (" << alloc_slots << " / "
                                                             << _manager->params().filter.lmb_fpm.pmNumBestComponents_k
                                                             << " are allocated).");
      }
      std::size_t counter2 = 0;
      for (auto const& val : alloc)
      {
        componentID2Alloc.emplace(component_ids.at(counter2), val);
        counter2++;
      }
    }
    else
    {
      componentID2Alloc.emplace(component_ids.at(0), _manager->params().filter.lmb_fpm.pmNumBestComponents_k);
    }
  }
  for (const auto& prior_component : priorMixtureComponents)
  {
    DistributionId prior_id = prior_component->id();
    bool isempty = false;

    std::map<DistributionId, BaseDistribution*> usedPriorComps;
    std::vector<std::map<DistributionId, const BaseDistribution*>> edgeId2localComp(numSensors);
    std::vector<std::size_t> nodes;
    std::vector<graph::Edge<Edge_FPM, std::size_t, double>> edges;

    std::size_t startNode = 0;
    std::size_t endNode = 0;
    nodes.push_back(startNode);

    std::size_t node_ctr = 0;

    std::vector<std::size_t> nodes_tmp;
    std::vector<graph::Edge<Edge_FPM, std::size_t, double>> edges_comp;
    std::size_t sensorId = 0;
    for (TrackIt track = localTracksBegin; track != localTracksEnd; ++track)
    {
      const std::map<StateModelId, std::unique_ptr<BaseDistribution>>& locDist = track->_state_dist;
      std::vector<graph::Edge<Edge_FPM, std::size_t, double>> edges_sensor;

      std::size_t node1 = node_ctr;
      node_ctr++;
      std::size_t node2 = node_ctr;

      for (const auto& [loc_model_id, locMixture] : locDist)
      {
        if (loc_model_id != model_id)
        {
          // gaussian components must have the same state model id!
          continue;
        }
        const auto& localMixtureComponents = locMixture->dists();
        LOG_INF("Mixture size: " << localMixtureComponents.size());
        for (const auto& loc_comp : localMixtureComponents)
        {
          if ((loc_comp->priorId() == prior_id) ||
              ((loc_comp->priorId() == NO_DISTRIBUTION_ID_HISTORY) &&
               (loc_comp->id() == prior_id)))  // prior_id==loc_comp->priorId()
                                               //  || misdetected component (corresponds
                                               //  to priorId=0) with id==prior_id
          {
            LOG_INF("Create new edge with distId " << loc_comp->id() << ", sensorId " << sensorId
                                                   << " startNode: " << node1 << " end node: " << node2);
            edges_sensor.emplace_back(std::make_pair(sensorId, loc_comp->id()), node1, node2, loc_comp->sumWeights());
            edgeId2localComp[sensorId].emplace(loc_comp->id(), loc_comp);
          }
        }
      }
      sensorId++;
      if (edges_sensor.empty())
      {
        // one sensor has no MixtureComponents with this prior component id!
        LOG_INF("Sensor does not deliver results for component id " << prior_id);
        isempty = true;
        break;
      }
      LOG_INF("Add Node with id: " << node1);
      LOG_INF("add Node with id: " << node2);
      nodes_tmp.push_back(node1);
      nodes_tmp.push_back(node2);
      endNode = node2;
      edges_comp.insert(
          edges_comp.end(), std::make_move_iterator(edges_sensor.begin()), std::make_move_iterator(edges_sensor.end()));
    }
    if (!isempty)
    {
      LOG_INF("insert nodes and edges for this prior id: " << prior_id);
      nodes.insert(nodes.end(), std::make_move_iterator(nodes_tmp.begin()), std::make_move_iterator(nodes_tmp.end()));
      edges.insert(edges.end(), std::make_move_iterator(edges_comp.begin()), std::make_move_iterator(edges_comp.end()));
      usedPriorComps.emplace(prior_id, prior_component);

      // remove duplicated nodes
      sort(nodes.begin(), nodes.end());
      nodes.erase(unique(nodes.begin(), nodes.end()), nodes.end());

      LOG_DEB("extern startNode: " << startNode << " extern endNode: " << endNode);
      graph::DiGraph<std::size_t, Edge_FPM, double> graph = graph::DiGraph(std::move(nodes), std::move(edges));
      // calculate k dependent on which mode
      std::size_t k = get_k(prior_component->sumWeights(), componentID2Alloc, prior_id);
      LOG_INF("Calculate the k best solutions with k = " << k);
      LOG_INF("start node: " << startNode << " endNode: " << endNode);
      FPMFusionPart partInfos;
      partInfos._usedPriorComps = std::move(usedPriorComps);
      partInfos._edgeId2localComp = std::move(edgeId2localComp);
      graphs.emplace_back(std::move(graph), std::move(partInfos), startNode, endNode, k);
    }
  }
  return graphs;
}

std::vector<std::unique_ptr<BaseDistribution>> LMBDistribution::fuse_mixture_components_generalized_k_best_fpm(
    const FPMFusionInfos& fusionInfos,
    std::vector<std::pair<std::vector<Edge_FPM>, double>>& solutions,
    const std::size_t numSensors,
    double& weightSumMMF,
    const BaseStateModel& stateModel) const
{
  std::vector<std::unique_ptr<BaseDistribution>> fusedComponents;

  double weightSum = 0;
  for (const auto& [solution, fusedWeightGraph] : solutions)
  {
    LOG_INF("new solution with weight " << fusedWeightGraph);
    DistributionId priorId;
    Matrix priorInfoMat;
    Vector priorInfoVec;
    double priorWeight = 0.0;
    Matrix infoMatSum;
    Vector infoVecSum;
    double weightProduct = 1.0;
    bool hasPriorComponent = false;
    for (const auto& edgeId : solution)
    {
      LOG_INF("edgeId: (" << edgeId.second << ", " << edgeId.first << ")");
      if (fusionInfos._partInfos._usedPriorComps.contains(edgeId.second) && edgeId.first == numSensors)
      {
        auto* const prior_component = fusionInfos._partInfos._usedPriorComps.at(edgeId.second);
        const auto* priorDist = impl::smartPtrCast<GaussianDistribution>(prior_component,
                                                                         "prior dist has "
                                                                         "unsupported Gaussian "
                                                                         "distribution type "
                                                                         "(fuseTracks_fpm)");
        priorInfoMat = priorDist->getInfoMat();
        priorInfoVec = priorDist->getInfoVec();
        if (stateModel.state_comps().indexOf(COMPONENT::ROT_Z).has_value())
        {
          angles::normalizeAngle(priorInfoVec(stateModel.state_comps().indexOf(COMPONENT::ROT_Z).value()));
        }
        priorWeight = priorDist->sumWeights();
        priorId = edgeId.second;
        hasPriorComponent = true;
        LOG_INF("is prior component!");
        continue;
      }
      // local component
      const auto* localDist = impl::smartPtrCast<GaussianDistribution>(
          fusionInfos._partInfos._edgeId2localComp[edgeId.first].at(edgeId.second),
          "local dist has unsupported Gaussian distribution type "
          "(fuseTracks_fpm)");
      const Matrix infoMat = localDist->getInfoMat();
      Vector infoVec = localDist->getInfoVec();
      if (stateModel.state_comps().indexOf(COMPONENT::ROT_Z).has_value())
      {
        angles::normalizeAngle(infoVec(stateModel.state_comps().indexOf(COMPONENT::ROT_Z).value()));
      }
      if (infoMatSum.size() == 0)
      {
        infoMatSum = infoMat;
      }
      else
      {
        infoMatSum = infoMatSum + infoMat;
      }

      if (infoVecSum.size() == 0)
      {
        infoVecSum = infoVec;
      }
      else
      {
        infoVecSum = infoVecSum + infoVec;
      }
      weightProduct = weightProduct * localDist->sumWeights();
    }
    if (!hasPriorComponent)
    {
      LOG_FATAL("Has no prior component in solution of generalizedKBest algorithm!!!! -> BUG");
      throw std::runtime_error("Has no prior component in solution of generalizedKBest algorithm!!!! -> BUG");
    }
    double normConst = 1.0;
    Matrix diff = infoMatSum - static_cast<double>(numSensors - 1) * priorInfoMat;
    //        LOG_ERR("Check Semidefiniteness! FPM_fusion before .i(): " << checkSemidefiniteness(diff));
    Matrix fusedCov = diff.inverse();
    //        if(!checkSemidefiniteness(fusedCov))
    //        {
    //          LOG_ERR("not semidefinit!");
    //          // Debug print selections of this gm id
    //          for(const auto& comp:selection)
    //          {
    //            LOG_WARN("comp id: " << comp->id << " weight: " << comp->weight);
    //            comp->distribution->print();
    //          }
    //          throw std::runtime_error("Not semidefinit!");
    //        }
    Vector fusedMean = fusedCov * (infoVecSum - static_cast<double>(numSensors - 1) * priorInfoVec);
    if (stateModel.state_comps().indexOf(COMPONENT::ROT_Z).has_value())
    {
      angles::normalizeAngle(fusedMean(stateModel.state_comps().indexOf(COMPONENT::ROT_Z).value()));
    }
    // Fused Weights can be calculated using the BPCR formula. See derivation of "The Fast Product Multi-Sensor Labeled
    // Multi-Bernoulli Filter"
    double fusedWeight =
        normConst * weightProduct /
        (std::pow(priorWeight, static_cast<double>(numSensors - 1)));  // ToDo(hermann): Check if logarithm
                                                                       // implementation for weights is necessary
    if (fabs(fusedWeight - fusedWeightGraph) > 1e-6)
    {
      LOG_ERR("in fpm-lmb calculated fused weight is "
              << fusedWeight << " in generalizedKBest algorithm calculated fusedWeight is " << fusedWeightGraph);
      //        throw std::runtime_error("Different fused weight results!");
    }
    weightSum += fusedWeight;
    auto fused_dist = std::make_unique<GaussianDistribution>(
        std::move(fusedMean), std::move(fusedCov));  // todo(hermann): add version for use with reference points!
    fused_dist->multiplyWeights(fusedWeight);
    fusedComponents.push_back(std::move(fused_dist));
  }
  // normalize component weights
  LOG_DEB("Normalize weights with weightSum: " << weightSum);
  if (weightSum > 0)
  {
    for (auto& comp : fusedComponents)
    {
      comp->multiplyWeights(1 / weightSum);
      LOG_DEB("comp id: " << comp->id() << " comp weight: " << comp->sumWeights());
    }
  }
  weightSumMMF = weightSum;
  return fusedComponents;
}

std::vector<std::unique_ptr<BaseDistribution>> LMBDistribution::fuse_mixture_components_k_best_fpm(
    const std::vector<std::pair<FPMFusionPart, std::vector<std::pair<std::vector<Edge_FPM>, double>>>>& solutionsVec,
    const std::size_t numSensors,
    double& weightSumMMF,
    const BaseStateModel& stateModel) const
{
  std::vector<std::unique_ptr<BaseDistribution>> fusedComponents;
  LOG_INF("fuse_mixture_components_k_best_fpm: ");
  double weightSum = 0;
  for (const auto& solutions : solutionsVec)
  {
    if (solutions.first._usedPriorComps.size() > 1 || solutions.first._usedPriorComps.empty())
    {
      throw std::runtime_error("more than one prior component or no prior component given! Can not perform fusion!");
    }
    auto* const prior_component = solutions.first._usedPriorComps.begin()->second;
    const auto* priorDist = impl::smartPtrCast<GaussianDistribution>(prior_component,
                                                                     "prior dist has unsupported "
                                                                     "Gaussian distribution type "
                                                                     "(fuse_tracks_fpm)");
    Matrix priorInfoMat = priorDist->getInfoMat();
    Vector priorInfoVec = priorDist->getInfoVec();
    if (stateModel.state_comps().indexOf(COMPONENT::ROT_Z).has_value())
    {
      angles::normalizeAngle(priorInfoVec(stateModel.state_comps().indexOf(COMPONENT::ROT_Z).value()));
    }
    double priorWeight = priorDist->sumWeights();
    for (const auto& solution : solutions.second)
    {
      Matrix infoMatSum;
      Vector infoVecSum;
      double weightProduct = 1.0;
      LOG_INF("next solution with weight: " << solution.second);
      for (const auto& edgeId : solution.first)
      {
        LOG_INF("edge with distId " << edgeId.second << " and sensorId: " << edgeId.first);
        // local component
        const auto* localDist =
            impl::smartPtrCast<GaussianDistribution>(solutions.first._edgeId2localComp[edgeId.first].at(edgeId.second),
                                                     "local dist has unsupported Gaussian distribution type "
                                                     "(fuse_tracks_fpm)");
        const Matrix infoMat = localDist->getInfoMat();
        Vector infoVec = localDist->getInfoVec();
        if (stateModel.state_comps().indexOf(COMPONENT::ROT_Z).has_value())
        {
          angles::normalizeAngle(infoVec(stateModel.state_comps().indexOf(COMPONENT::ROT_Z).value()));
        }

        if (infoMatSum.size() == 0)
        {
          infoMatSum = infoMat;
        }
        else
        {
          infoMatSum = infoMatSum + infoMat;
        }

        if (infoVecSum.size() == 0)
        {
          infoVecSum = infoVec;
        }
        else
        {
          infoVecSum = infoVecSum + infoVec;
        }
        weightProduct = weightProduct * localDist->sumWeights();
      }

      double normConst = 1.0;
      Matrix diff = infoMatSum - static_cast<double>(numSensors - 1) * priorInfoMat;
      //        LOG_ERR("Check Semidefiniteness! FPM_fusion before .i(): " << checkSemidefiniteness(diff));
      Matrix fusedCov = diff.inverse();
      //        if(!checkSemidefiniteness(fusedCov))
      //        {
      //          LOG_ERR("not semidefinit!");
      //          // Debug print selections of this gm id
      //          for(const auto& comp:selection)
      //          {
      //            LOG_WARN("comp id: " << comp->id << " weight: " << comp->weight);
      //            comp->distribution->print();
      //          }
      //          throw std::runtime_error("Not semidefinit!");
      //        }
      Vector fusedMean = fusedCov * (infoVecSum - static_cast<double>(numSensors - 1) * priorInfoVec);
      if (stateModel.state_comps().indexOf(COMPONENT::ROT_Z).has_value())
      {
        angles::normalizeAngle(fusedMean(stateModel.state_comps().indexOf(COMPONENT::ROT_Z).value()));
      }
      // Fused Weights can be calculated using the BPCR formula. See derivation of "The Fast Product Multi-Sensor
      // Labeled Multi-Bernoulli Filter"
      double fusedWeight = normConst * weightProduct / (std::pow(priorWeight, static_cast<double>(numSensors - 1)));
      weightSum += fusedWeight;
      auto fused_dist = std::make_unique<GaussianDistribution>(std::move(fusedMean), std::move(fusedCov));
      fused_dist->multiplyWeights(fusedWeight);
      fusedComponents.push_back(std::move(fused_dist));
    }
  }
  // normalize component weights
  LOG_DEB("Normalize weights with weightSum: " << weightSum);
  if (weightSum > 0)
  {
    for (auto& comp : fusedComponents)
    {
      comp->multiplyWeights(1 / weightSum);
      LOG_DEB("comp id: " << comp->id() << " comp weight: " << comp->sumWeights());
    }
  }
  weightSumMMF = weightSum;
  return fusedComponents;
}

double LMBDistribution::calculate_fused_existence_probability(const double r_prior,
                                                              const double r_posterior_product,
                                                              const double r_inverse_posterior_product,
                                                              const double weightSumMMF,
                                                              const std::size_t numSensors) const
{
  double exp = 1.0 - numSensors;
  double r_prior_product = std::pow(r_prior, exp);
  double r_prior_inverse_product = std::pow(1.0 - r_prior, exp);
  double r_fused_num = weightSumMMF * r_prior_product * r_posterior_product;
  double r_fused_den = r_prior_inverse_product * r_inverse_posterior_product + r_fused_num;
  return r_fused_num / r_fused_den;
}

void LMBDistribution::calcInnovation(MeasurementContainer const& Z)
{
  ZoneScopedNC("LMBDistribution::innovation", tracy_color);
  LOG_DEB("Innovation LMB Distribution with " + std::to_string(_tracks.size()) + " Tracks and " +
          std::to_string(Z._data.size()) + " Measurements");
  utils::innovate(_manager, _tracks, Z);
  LOG_DEB("After calcInnovation:\n" + toString());
}

void LMBDistribution::addTrack(State track)
{
  if (auto const it = std::ranges::find_if(
          _tracks, [&](State const& existingTrack) { return existingTrack._label == track._label; });
      it != _tracks.end())
  {
    LOG_FATAL("Track already exists in current distribution! Overwriting track!");
    throw std::runtime_error("Track already exists in current distribution! RUN");
  }
  track._survival_probability = std::numeric_limits<double>::quiet_NaN();
  _tracks.push_back(std::move(track));
}

bool LMBDistribution::eraseTrack(Label label)
{
  bool labelFound = false;
  if (const auto it = std::ranges::find_if(_tracks, [&](State const& track) { return track._label == label; });
      it != _tracks.end())
  {
    labelFound = true;
    _tracks.erase(it);
  }
  return labelFound;
}

void LMBDistribution::merge(LMBDistribution otherDist)
{
  ZoneScopedNC("LMBDistribution::merge", tracy_color);
  std::ranges::for_each(otherDist._tracks, [&](State& track) { addTrack(std::move(track)); });
  _meas_assignment_prob.merge(std::move(otherDist._meas_assignment_prob));
  _detection_dist = sa::merge_detection_dist(_detection_dist, otherDist._detection_dist);
  _clutter_dist = sa::merge_clutter_dist(_clutter_dist, otherDist._clutter_dist);
}

std::string LMBDistribution::toString(std::string const& prefix) const
{
  std::string out = prefix + "LMB Distribution\n";
  out += prefix + "|\tId: " + std::to_string(_id.value_) + "\n";
  for (auto const& track : _tracks)
  {
    out += track.toString(prefix + "|\t");
  }
  return out;
}

bool LMBDistribution::isValid() const
{
  return std::ranges::all_of(_tracks, [&](State const& track) {
    if (not track.isValid())
    {
      LOG_FATAL("Track " + track.toString() + "not valid");
      return false;
    }
    return true;
  });
}

void LMBDistribution::postProcessPrediction()
{
  ZoneScopedNC("LMBDistribution::postProcessPrediction", tracy_color);
  prune_if([](State const& state) { return state.isEmpty(); });
  if (_manager->params().lmb_distribution.post_process_prediction.enable)
  {
    prune_if([&](State const& track) {
      return track._existenceProbability <
             _manager->params().lmb_distribution.post_process_prediction.pruning_threshold;
    });
    prune_if([&](State const& track) {
      return track._meta_data._durationSinceLastAssociation >
             std::chrono::milliseconds(
                 _manager->params().lmb_distribution.post_process_prediction.max_last_assoc_duration_ms);
    });
    truncate(_manager->params().lmb_distribution.post_process_prediction.max_tracks);
  }
  assert([&] {  // NOLINT
    if (not isValid())
    {
      LOG_FATAL(toString());
      return false;
    }
    return true;
  }());
}

void LMBDistribution::postProcessUpdate()
{
  LOG_DEB("post process update");
  ZoneScopedNC("LMBDistribution::postProcessUpdate", tracy_color);
  prune_if([](State const& state) { return state.isEmpty(); });
  if (_manager->params().lmb_distribution.post_process_update.enable)
  {
    prune_if([&](State const& track) {
      return track._existenceProbability < _manager->params().lmb_distribution.post_process_update.pruning_threshold;
    });
    truncate(_manager->params().lmb_distribution.post_process_update.max_tracks);
  }
  assert([&] {  // NOLINT
    if (not isValid())
    {
      LOG_FATAL(toString());
      return false;
    }
    return true;
  }());
  LOG_DEB("postProcessUpdate done");
}

Vector const& LMBDistribution::clutter_distribution() const
{
  return _clutter_dist;
}

Matrix const& LMBDistribution::detection_distribution() const
{
  return _detection_dist;
}

std::map<MeasurementId, Probability> const& LMBDistribution::meas_assignment_prob() const
{
  return _meas_assignment_prob;
}

}  // namespace ttb
