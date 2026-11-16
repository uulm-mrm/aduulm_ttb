#include "tracking_lib/Trackers/GNN_Tracker.h"
// #####################################################################################################################
#include "tracking_lib/States/State.h"
#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/MultiObjectStateDistributions/LMBDistribution.h"
#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/Misc/MurtyAlgorithm.h"
#include "tracking_lib/Misc/Grouping.h"
#include "tracking_lib/MultiObjectStateDistributions/Utils.h"

namespace ttb
{

template class GNN_Tracker<double>;
template class GNN_Tracker<AD>;

template <class UDT>
GNN_Tracker<UDT>::GNN_Tracker(TTBManager<UDT>* manager) : _manager{ manager }
{
}

template <class UDT>
FILTER_TYPE GNN_Tracker<UDT>::type() const
{
  return FILTER_TYPE::GNN;
}

template <class UDT>
std::vector<StateId> GNN_Tracker<UDT>::all_tracks() const
{
  return {};
}

template <class UDT>
TTBManager<UDT>* GNN_Tracker<UDT>::manager() const
{
  return _manager;
}

template <class UDT>
Time GNN_Tracker<UDT>::time() const
{
  return _time;
}

template <class UDT>
bool GNN_Tracker<UDT>::hasValidTracks() const
{
  return std::ranges::all_of(_gnn_tracks, [&](State<UDT> const& track) {
    if (not track.isValid())
    {
      LOG_FATAL("Track " + track.toString() + "not valid");
      return false;
    }
    return true;
  });
}

template <class UDT>
std::string GNN_Tracker<UDT>::toString(std::string const& prefix) const
{
  std::string out = prefix + "GNN distribution\n";
  for (auto const& track : _gnn_tracks)
  {
    out += track.toString(prefix + "|\t");
  }
  return out;
}

constexpr const auto tracy_color1 = tracy::Color::LightSkyBlue;
constexpr const auto tracy_color2 = tracy::Color::LightCyan;
constexpr const auto tracy_color3 = tracy::Color::LightCyan;
constexpr const auto tracy_color4 = tracy::Color::Orange;
constexpr const auto tracy_color5 = tracy::Color::CornflowerBlue;

template <class UDT>
void GNN_Tracker<UDT>::cycle(std::vector<DataSourceContainer<UDT>> measContainerList)
{
  assert(hasValidTracks() and "Invalid GNN distribution");
  LOG_DEB("GNN cycle with " << measContainerList.size() << " measurement containers");
  LOG_INF("GNN-Tracker is about to update " + std::to_string(_gnn_tracks.size()) + " tracks with "
          << std::to_string(measContainerList.size()) + " measurement containers at time " << _time);

  if (measContainerList.empty())
  {
    LOG_DEB("Measurement containers are empty - just perform the prediction");
    return;
  }

  std::size_t num_sensors = measContainerList.size();

  for (auto const& measContainer : measContainerList)
  {
    LOG_DEB("Measurement container's time: " << measContainer._time);
    LOG_DEB("Tracker's time : " << _time);
    Duration deltaT = measContainer._time - _time;
    if (deltaT < 0ms)
    {
      LOG_FATAL("Receiving measurement container from the ancient past .... - MUST NOT HAPPEN");
      LOG_FATAL("Measurement container with ID " << measContainer._source_id.value_);
      throw std::runtime_error("Receiving measurement container from the ancient past .... - MUST NOT HAPPEN");
    }
    LOG_DEB("Predict GNN distribution with time duration " + std::to_string(to_seconds(deltaT)));
    performPrediction(deltaT, measContainer._predictionInfo);
    _time = measContainer._time;

    if (_manager->getBirthModel().type() == BIRTH_MODEL_TYPE::STATIC)
    {
      LOG_DEB("Add static birth tracks");
      addBirthTracks(measContainer, {});
    }

    LOG_DEB("Calculate innovation");
    performInnovation(measContainer);
    LOG_DEB("Perform update step");
    auto assignment_prob = performUpdate(measContainer, num_sensors);

    if (_manager->getBirthModel().type() == BIRTH_MODEL_TYPE::DYNAMIC)
    {
      LOG_DEB("Add dynamic birth tracks");
      LOG_DEB("Measurement container data size: " << measContainer._measurements.size());
      LOG_DEB("Assignment probabilities size: " << assignment_prob.size());
      LOG_DEB("number of tracks: " << _gnn_tracks.size());
      addBirthTracks(measContainer, assignment_prob);
    }

    LOG_DEB("Updated " + std::to_string(_gnn_tracks.size()) + " tracks");
    if (_manager->params()->filter.gnn.trackManagementParams.do_post_process_after_update)  // post_process
    {
      LOG_DEB("Start post processing update");
      postProcessUpdate(num_sensors);
      LOG_DEB("Done post processing update");
    }
    LOG_DEB(std::to_string(_gnn_tracks.size()) + " tracks left (after post processing after update)");
  }
  LOG_DEB("GNN cycle is done");
  LOG_DEB("");
}

template <class UDT>
void GNN_Tracker<UDT>::performPrediction(Duration deltaT, PredictionInformation const& prediction_info)
{
  LOG_DEB("Predict GNN distribution with " + std::to_string(_gnn_tracks.size()) + " tracks");
  for (State<UDT>& state : _gnn_tracks)
  {
    state.predict(deltaT, prediction_info._egoMotion);
  }
}

template <class UDT>
void GNN_Tracker<UDT>::performInnovation(DataSourceContainer<UDT> const& Z)
{
  LOG_DEB("Perform the innovation of the GNN distribution with " + std::to_string(_gnn_tracks.size()) + " tracks and " +
          std::to_string(Z._measurements.size()) + " measurements");
  for (State<UDT> const& state : _gnn_tracks)
  {
    auto t = state.innovate(Z);
  }
  LOG_DEB("After calcInnovation:\n" + toString());
}

template <class UDT>
std::map<MeasurementId, UDT> GNN_Tracker<UDT>::performUpdate(DataSourceContainer<UDT> const& measurementContainer,
                                                             std::size_t num_sensors)
{
  LOG_DEB("Calculate groups");
  std::vector<State<UDT>> updated_tracks;
  std::vector<MeasurementId> used_to_update;
  std::map<MeasurementId, UDT> assignment_prob;
  std::map<Label, MeasurementId> meas_to_track_assignments;
  std::vector<grouping::Group<UDT>> gnn_groups;
  if (_manager->params()->filter.gnn.use_grouping)
  {
    LOG_DEB("Use grouping");
    std::vector<grouping::Group<UDT>> groups = grouping::group(_gnn_tracks, measurementContainer);
    for (auto& [group_tracks, group_measurement_container] : groups)
    {
      gnn_groups.emplace_back(std::move(group_tracks), std::move(group_measurement_container));
    }
  }
  else
  {
    gnn_groups.emplace_back(std::move(_gnn_tracks), measurementContainer);
  }

  for (auto const& gnn_group : gnn_groups)
  {
    for (auto const& meas : gnn_group.measurement_container._measurements)
    {
      assignment_prob.emplace(meas._id, 0);
    }
  }
  LOG_DEB("assignment probabilities size: " << assignment_prob.size());

  int num_misses = 0;
  int num_hits = 0;
  for (std::size_t i = 0; i != gnn_groups.size(); i++)
  {
    auto& gnn_group = gnn_groups.at(i);
    LOG_DEB("Starting association in cluster " << i);
    auto& group_tracks = gnn_group.tracks;
    LOG_DEB("This cluster " << i << " contains " << group_tracks.size() << " tracks");
    auto& group_measurements = gnn_group.measurement_container;
    LOG_DEB("The number of measurements in this cluster is " << group_measurements._measurements.size());
    LOG_DEB("The number of tracks in this cluster is " << group_tracks.size());
    auto costs = buildCostMatrix(group_tracks, group_measurements);
    Eigen::MatrixXi assignments;
    Vector<double> resultingCosts;
    if (costs.size() == 0)
    {
      LOG_DEB("Cost matrix is empty");
    }
    else
    {
      LOG_DEB("The cost matrix looks like this: " << costs);
      auto result = murty::getAssignments(costs, 1);
      assignments = result.first;
      resultingCosts = result.second;
      LOG_DEB("The size of the solution of the assignment-problem is: " << assignments.size());
      LOG_DEB("assignments: " << assignments);
      LOG_DEB("The solution of the assignment-problem is: " << assignments.transpose());
    }

    // check assignment results
    if (assignments.size() == 0)
    {
      LOG_DEB("There is no assignment - murty result is empty!");
      LOG_DEB("cost matrix: " << costs);
      LOG_DEB("assignments: " << assignments.transpose());
    }
    else if (assignments.size() != group_measurements._measurements.size())
    {
      LOG_DEB("assignments.size() =" << assignments.size());
      LOG_DEB("measurements._measurements.size() =" << group_measurements._measurements.size());
      LOG_FATAL("Solution size is not measurements size.");
    }
    else
    {
      for (auto [measIdx, meas] : std::views::enumerate(group_measurements._measurements))
      {
        LOG_DEB("measurement index: " << measIdx);
        LOG_DEB("assignment: " << assignments(measIdx, 0));
        LOG_DEB("cost of assignment: " << costs(measIdx, assignments(measIdx, 0)));
        if (assignments(measIdx, 0) >= static_cast<int>(group_tracks.size()))
        {
          LOG_DEB("Measurement at index " << measIdx << " with ID " << meas._id << " is NOT associated with a track");
          assignment_prob[meas._id] = 0;
          LOG_DEB("assignment probabilities size: " << assignment_prob.size());
          continue;
        }
        auto const trackIndex = static_cast<std::size_t>(assignments(measIdx, 0));
        LOG_DEB("Update Track " << group_tracks.at(trackIndex)._label << " with Measurement " << meas._id);
        assignment_prob[meas._id] = 1.0;
        used_to_update.push_back(meas._id);
        meas_to_track_assignments.insert({ group_tracks.at(trackIndex)._label, meas._id });
        LOG_DEB("" << group_tracks.at(trackIndex).toString());
        State new_track = _manager->get_state(
            _manager->database()
                .state_updates.find(std::tuple{ group_tracks.at(trackIndex)._id, measurementContainer._id, meas._id })
                ->updated_state_id);
        updated_tracks.push_back(std::move(new_track));
        LOG_DEB("Updated Track " << group_tracks.at(trackIndex)._label << " has score "
                                 << updated_tracks.at(updated_tracks.size() - 1)._score);
        num_hits++;
      }
    }

    for (auto& track : group_tracks)
    {
      if (meas_to_track_assignments.find(track._label) == meas_to_track_assignments.end())
      {
        LOG_DEB("Score for unassociated Track " << track._label << " was " << track._score << " and now is");
        State new_track = _manager->get_state(
            _manager->database()
                .state_updates.find(std::tuple{ track._id, measurementContainer._id, NOT_DETECTED })
                ->updated_state_id);
        updated_tracks.emplace_back(std::move(new_track));
        LOG_DEB("Add score " << updated_tracks.at(updated_tracks.size() - 1)._score);
        meas_to_track_assignments.insert({ track._label, NOT_DETECTED });
        num_misses++;
      }
    }
    LOG_DEB("# tracks: " << updated_tracks.size() << " # misses: " << num_misses << " # hits: " << num_hits);
  }
  _gnn_tracks = std::move(updated_tracks);
  return assignment_prob;
}

template <class UDT>
Matrix<double> GNN_Tracker<UDT>::buildCostMatrix(std::vector<State<UDT>> const& tracks,
                                                 DataSourceContainer<UDT> const& splitMeasurements)
{
  LOG_DEB("Start building cost matrix");
  // cost matrix with size: # measurements x (# measurements + # tracks)
  // rows are the measurements, cols are the measurements + tracks
  LOG_DEB("tracks: " << tracks.size() << " measurements: " << splitMeasurements._measurements.size());
  Matrix<double> negative_log_cost_matrix = Matrix<double>::Constant(
      splitMeasurements._measurements.size(), splitMeasurements._measurements.size() + tracks.size(), INFINITY);
  if (negative_log_cost_matrix.size() == 0)
  {
    LOG_DEB("Returning empty cost matrix");
    return negative_log_cost_matrix;
  }
  double max_cost = INFINITY * -1;
  for (auto const& [measIndex, meas] : std::views::enumerate(splitMeasurements._measurements))
  {
    // Iterate over all measurements in partition
    for (auto const& [trackIndex, track] : std::views::enumerate(tracks))
    {
      Probability<double> const PD = [&] {
        auto const& rlock = _manager->database().state_detection_probabilities;
        auto const it = rlock.template get<StateIdContainerId>().find(std::tuple{ track._id, splitMeasurements._id });
        assert(it != rlock.template get<StateIdContainerId>().end());
        return to_val(it->detection_probability);
      }();
      LOG_DEB("PD: " << PD);
      double logPD = std::log(PD);
      LOG_DEB("logPD: " << logPD);
      if (_manager->database().state_updates.contains(std::tuple{ track._id, splitMeasurements._id, meas._id }))
      {
        double const log_likelihood = [&] {
          auto const& rlock = _manager->database().state_meas_log_likelihood;
          auto const it = rlock.find(std::tuple{ track._id, meas._id });
          assert(it != rlock.end());
          return to_val(it->meas_log_likelihood);
        }();
        double cost_assignment = -(logPD + log_likelihood);
        negative_log_cost_matrix(measIndex, trackIndex) = cost_assignment;
        if (cost_assignment > max_cost)
        {
          max_cost = cost_assignment;
        }
        double const clutter_intensity = [&] {
          auto const& rlock = _manager->database().meas_clutter_intensity;
          auto const it = rlock.find(meas._id);
          assert(it != rlock.end());
          return to_val(it->clutter_intensity);
        }();
        double external_sources_density = clutter_intensity + birth_track_density;
        LOG_DEB("clutter intensity " << clutter_intensity);
        LOG_DEB("birth_track_density " << birth_track_density);
        LOG_DEB("external sources density " << external_sources_density);
        LOG_DEB("log external sources density " << log(external_sources_density));

        negative_log_cost_matrix(measIndex, measIndex + tracks.size()) =
            -(log(1 - PD) + log(external_sources_density));  // source
        // https://users.metu.edu.tr/umut/ee793/files/METULecture5.pdf
      }
    }
  }
  LOG_DEB("max cost value : " << max_cost);
  if (!_manager->params()->filter.gnn.costMatrixParams.use_external_sources_density)
  {
    for (auto const& [measIndex, meas] : std::views::enumerate(splitMeasurements._measurements))
    {
      negative_log_cost_matrix(measIndex, tracks.size() + measIndex) = max_cost + 1;
    }
  }
  return negative_log_cost_matrix;
}

template <class UDT>
std::vector<BernoulliState<UDT>> GNN_Tracker<UDT>::get_estimate() const
{
  std::vector<BernoulliState<UDT>> estimate;
  LOG_DEB("num tracks (GNN) before getEstimate(): " << _gnn_tracks.size());
  if (_manager->params()->filter.gnn.trackManagementParams.output_only_confirmed)
  {
    for (auto& track : _gnn_tracks)
    {
      if (track._stage == STAGE::CONFIRMED)
      {
        estimate.emplace_back(track, 1);
      }
    }
  }
  else
  {
    estimate = _gnn_tracks |
               std::views::transform([](State<UDT> const& state) { return BernoulliState<UDT>{ state, UDT{ 1 } }; }) |
               std::ranges::to<std::vector<BernoulliState<UDT>>>();
  }
  LOG_DEB("num tracks (GNN) after getEstimate(): " << estimate.size());
  return estimate;
}

template <class UDT>
void GNN_Tracker<UDT>::reset()
{
  LOG_DEB("Reset GNN Filter");
  _gnn_tracks = std::vector<State<UDT>>();
  _time = Time{ 0s };
}

template <class UDT>
void GNN_Tracker<UDT>::addBirthTracks(DataSourceContainer<UDT> const& measurementContainer,
                                      std::map<MeasurementId, UDT> const& rzMap)
{
  LOG_DEB("Start adding birth tracks");
  LOG_DEB("Type (birth model): " << to_string(_manager->getBirthModel().type()));
  BirthModel<UDT>& birthModel = _manager->getBirthModel();
  BirthTracks<UDT> birth_tracks = birthModel.getBirthTracks(measurementContainer, rzMap, _gnn_tracks);
  for (auto const& [track, meas] : std::views::zip(birth_tracks.birth_tracks, birth_tracks.birth_measurements))
  {
    assert(meas.has_value());
    LOG_DEB("Start tracks are born");
    if (!_manager->params()
             ->filter.gnn.trackManagementParams.use_existence_prob_from_birth_model_for_score_initialisation and
        _manager->getBirthModel().type() == BIRTH_MODEL_TYPE::DYNAMIC and
        _manager->database().state_updates.contains(
            std::tuple{ track.state._id, measurementContainer._id, meas.value()._id }))
    {
      LOG_DEB("Start calculating track score for dynamic birth tracks");
      Probability<double> const PD = [&] {
        auto const& rlock = _manager->database().state_detection_probabilities;
        auto const it =
            rlock.template get<StateIdContainerId>().find(std::tuple{ track.state._id, measurementContainer._id });
        assert(it != rlock.template get<StateIdContainerId>().end());
        return to_val(it->detection_probability);
      }();
      auto measurements = measurementContainer._measurements;
      const auto measIt = std::ranges::find_if(measurementContainer._measurements, [&](const State<UDT>& measurement) {
        assert(track.state._meta_data._meas_update_weight.size() == 1);
        assert(track.state._meta_data._meas_update_weight.begin()->second.size() == 1);
        return measurement._id == track.state._meta_data._meas_update_weight.begin()->second.front().first;
      });
      double const clutter_intensity = [&] {
        auto const& rlock = _manager->database().meas_clutter_intensity;
        auto const it = rlock.find(measIt->_id);
        assert(it != rlock.end());
        return to_val(it->clutter_intensity);
      }();
      double false_target_density = clutter_intensity;
      if (_manager->params()->filter.gnn.costMatrixParams.calculate_birth_track_density)
      {
        LOG_DEB("calculate birth_track_density");
        double fov_size = measurementContainer._sensorInfo._sensor_fov.value().area(
            _manager->state_comps(measIt->_state_dist.begin()->first));
        double mean_num_birth = _manager->params()->birth_model->dynamic_model.mean_num_birth;
        LOG_DEB("fov_size " << fov_size);
        birth_track_density = mean_num_birth / fov_size;
      }
      else
      {
        LOG_DEB("use default (given in configs) birth_track_density");
        birth_track_density = _manager->params()->filter.gnn.costMatrixParams.birth_track_density;
      }
      LOG_DEB("birth_track_density: " << birth_track_density);
      track.state._score = log(birth_track_density) - log(false_target_density) + log(PD);
    }
    _gnn_tracks.push_back(std::move(track.state));
    LOG_DEB("Initial track score of Track " << track.state._label << ": " << track.state._score);
  }
}

template <class UDT>
void GNN_Tracker<UDT>::postProcessUpdate(std::size_t num_sensors)
{
  int tracks_before = _gnn_tracks.size();
  int tentative_tracks = 0;
  int preliminary_tracks = 0;
  int confirmed_tracks = 0;
  int dead_tracks = 0;

  auto to_remove = [&](State<UDT>& track) {
    track.performStageUpdate(num_sensors);
    if (track._stage == STAGE::TENTATIVE)
    {
      tentative_tracks++;
    }
    else if (track._stage == STAGE::PRELIMINARY)
    {
      preliminary_tracks++;
    }
    else if (track._stage == STAGE::CONFIRMED)
    {
      confirmed_tracks++;
    }
    else if (track._stage == STAGE::DEAD)
    {
      dead_tracks++;
    }
    else
    {
      LOG_FATAL("Invalid track state stage (tentative, preliminary, confirmed) for GNN-Tracker.");
    }
    return track._stage == STAGE::DEAD;
  };
  erase_if(_gnn_tracks, to_remove);

  int total_tracks = _gnn_tracks.size();
  int pruned = tracks_before - _gnn_tracks.size();
  LOG_DEB("Tentative: " << tentative_tracks << " Preliminary: " << preliminary_tracks << " Confirmed: "
                        << confirmed_tracks << " Dead: " << dead_tracks << " Total: " << total_tracks);
  LOG_DEB("Pruned " << pruned << " tracks out of " << tracks_before);
}

}  // namespace ttb
