#include "tracking_lib/Trackers/GNN_Tracker.h"
// #####################################################################################################################
#include "tracking_lib/States/State.h"
#include "tracking_lib/States/Innovation.h"
#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/Measurements/MeasurementContainer.h"
#include "tracking_lib/MultiObjectStateDistributions/LMBDistribution.h"
#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/Misc/MurtyAlgorithm.h"
#include "tracking_lib/Misc/Grouping.h"
#include "tracking_lib/MultiObjectStateDistributions/Utils.h"

#include <tracy/tracy/Tracy.hpp>

namespace ttb
{
GNN_Tracker::GNN_Tracker(TTBManager* manager) : _manager{ manager }
{
}

FILTER_TYPE GNN_Tracker::type() const
{
  return FILTER_TYPE::GNN;
}

TTBManager* GNN_Tracker::manager() const
{
  return _manager;
}

Time GNN_Tracker::time() const
{
  return _time;
}

bool GNN_Tracker::hasValidTracks() const
{
  return std::ranges::all_of(_gnn_tracks, [&](State const& track) {
    if (not track.isValid())
    {
      LOG_FATAL("Track " + track.toString() + "not valid");
      return false;
    }
    return true;
  });
}

std::string GNN_Tracker::toString(std::string const& prefix) const
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

void GNN_Tracker::cycle(Time time, std::vector<MeasurementContainer>&& measContainerList)
{
  ZoneScopedNC("GNNTracker::cycle", tracy_color1);
  assert(hasValidTracks() and "Invalid GNN distribution");
  LOG_DEB("GNN cycle with " << measContainerList.size() << " measurement containers");
  LOG_INF("GNN-Tracker is about to update " + std::to_string(_gnn_tracks.size()) + " tracks with "
          << std::to_string(measContainerList.size()) + " measurement containers at time " << _time);

  if (measContainerList.empty())
  {
    LOG_DEB("Measurement containers are empty - just perform the prediction");
    performPrediction(time - _time, EgoMotionDistribution::zero());
    _time = time;
  }

  std::size_t num_sensors = measContainerList.size();

  for (auto const& measContainer : measContainerList)
  {
    ZoneScopedNC("GNNTracker::cycleMeasurementContainers", tracy_color2);
    LOG_DEB("Measurement container's time: " << measContainer._time);
    LOG_DEB("Tracker's time : " << _time);
    Duration deltaT = measContainer._time - _time;
    if (deltaT < 0ms)
    {
      LOG_FATAL("Receiving measurement container from the ancient past .... - MUST NOT HAPPEN");
      LOG_FATAL("Tracker's time: " + to_string(_time));
      LOG_FATAL("Measurement container's time: " + to_string(measContainer._time));
      LOG_FATAL("Measurement container with ID " << measContainer._id.value_);
      throw std::runtime_error("Receiving measurement container from the ancient past .... - MUST NOT HAPPEN");
    }
    LOG_DEB("Predict GNN distribution with time duration " + std::to_string(to_seconds(deltaT)));
    performPrediction(deltaT, measContainer._egoMotion);
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
      LOG_DEB("Measurement container data size: " << measContainer._data.size());
      LOG_DEB("Assignment probabilities size: " << assignment_prob.size());
      LOG_DEB("number of tracks: " << _gnn_tracks.size());
      addBirthTracks(measContainer, assignment_prob);
    }

    LOG_DEB("Updated " + std::to_string(_gnn_tracks.size()) + " tracks");
    if (_manager->params().filter.gnn.trackManagementParams.do_post_process_after_update)  // post_process
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

void GNN_Tracker::performPrediction(Duration deltaT, EgoMotionDistribution const& egoDist)
{
  LOG_DEB("Predict GNN distribution with " + std::to_string(_gnn_tracks.size()) + " tracks");
  utils::predict(_manager, _gnn_tracks, deltaT, egoDist, {}, [&](State& state) {});
}

void GNN_Tracker::performInnovation(MeasurementContainer const& Z)
{
  LOG_DEB("Perform the innovation of the GNN distribution with " + std::to_string(_gnn_tracks.size()) + " tracks and " +
          std::to_string(Z._data.size()) + " measurements");
  utils::innovate(_manager, _gnn_tracks, Z);
  LOG_DEB("After calcInnovation:\n" + toString());
}

std::map<MeasurementId, double> GNN_Tracker::performUpdate(MeasurementContainer const& measurementContainer,
                                                           std::size_t num_sensors)
{
  LOG_DEB("Calculate groups");
  std::vector<State> updated_tracks;
  std::vector<MeasurementId> used_to_update;
  std::map<MeasurementId, double> assignment_prob;
  std::map<Label, MeasurementId> meas_to_track_assignments;
  std::vector<grouping::Group> gnn_groups;
  if (_manager->params().filter.gnn.use_grouping)
  {
    LOG_DEB("Use grouping");
    std::vector<grouping::Group> groups = grouping::group(_gnn_tracks, measurementContainer);
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
    for (auto const& meas : gnn_group.measurement_container._data)
    {
      assignment_prob.emplace(meas._id, 0);
    }
  }
  LOG_DEB("assignment probabilities size: " << assignment_prob.size());

  int num_misses = 0;
  int num_hits = 0;
  for (std::size_t i = 0; i != gnn_groups.size(); i++)
  {
    ZoneScopedNC("GNNTracker::ForLoopGroups", tracy_color3);
    auto& gnn_group = gnn_groups.at(i);
    LOG_DEB("Starting association in cluster " << i);
    auto& group_tracks = gnn_group.tracks;
    LOG_DEB("This cluster " << i << " contains " << group_tracks.size() << " tracks");
    auto& group_measurements = gnn_group.measurement_container;
    LOG_DEB("The number of measurements in this cluster is " << group_measurements._data.size());
    LOG_DEB("The number of tracks in this cluster is " << group_tracks.size());
    auto costs = buildCostMatrix(group_tracks, group_measurements, measurementContainer._id);
    Eigen::MatrixXi assignments;
    Vector resultingCosts;
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
    else if (assignments.size() != group_measurements._data.size())
    {
      LOG_DEB("assignments.size() =" << assignments.size());
      LOG_DEB("measurements._measurements.size() =" << group_measurements._data.size());
      LOG_FATAL("Solution size is not measurements size.");
    }
    else
    {
      for (auto [measIdx, meas] : std::views::enumerate(group_measurements._data))
      {
        ZoneScopedNC("GNNTracker::ForLoopMeas", tracy_color4);
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
        State new_track =
            group_tracks.at(trackIndex)._innovation.at(measurementContainer._id)._updates.at(meas._id).updated_dist;
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
        LOG_DEB("Score for unassociated Track "
                << track._label << " was " << track._score << " and now is "
                << track._innovation.at(measurementContainer._id)._updates.at(MeasurementId{ 0 }).updated_dist._score);
        State new_track = track._innovation.at(measurementContainer._id)._updates.at(MeasurementId{ 0 }).updated_dist;
        updated_tracks.emplace_back(std::move(new_track));
        LOG_DEB("Add score " << updated_tracks.at(updated_tracks.size() - 1)._score);
        meas_to_track_assignments.insert({ track._label, MeasurementId{ 0 } });
        num_misses++;
      }
    }
    LOG_DEB("# tracks: " << updated_tracks.size() << " # misses: " << num_misses << " # hits: " << num_hits);
  }
  _gnn_tracks = std::move(updated_tracks);
  return assignment_prob;
}

Matrix GNN_Tracker::buildCostMatrix(std::vector<State> const& tracks,
                                    MeasurementContainer const& splitMeasurements,
                                    const MeasModelId& measContainerID)
{
  ZoneScopedNC("GNNTracker::buildCostMatrix", tracy_color5);
  LOG_DEB("Start building cost matrix");
  // cost matrix with size: # measurements x (# measurements + # tracks)
  // rows are the measurements, cols are the measurements + tracks
  LOG_DEB("tracks: " << tracks.size() << " measurements: " << splitMeasurements._data.size());
  Matrix negative_log_cost_matrix =
      Matrix::Constant(splitMeasurements._data.size(), splitMeasurements._data.size() + tracks.size(), INFINITY);
  if (negative_log_cost_matrix.size() == 0)
  {
    LOG_DEB("Returning empty cost matrix");
    return negative_log_cost_matrix;
  }
  double max_cost = INFINITY * -1;
  for (auto const& [measIndex, meas] : std::views::enumerate(splitMeasurements._data))
  {
    // Iterate over all measurements in partition
    for (auto const& [trackIndex, track] : std::views::enumerate(tracks))
    {
      Innovation const& innoMap = track._innovation.at(measContainerID);
      double PD = innoMap._detectionProbability;
      LOG_DEB("PD: " << PD);
      double logPD = std::log(PD);
      LOG_DEB("logPD: " << logPD);
      auto const& inno = innoMap._updates.find(meas._id);
      if (inno != innoMap._updates.end())
      {
        double cost_assignment = -(logPD + inno->second.log_likelihood);
        negative_log_cost_matrix(measIndex, trackIndex) = cost_assignment;
        if (cost_assignment > max_cost)
        {
          max_cost = cost_assignment;
        }
        double external_sources_density = inno->second.clutter_intensity + birth_track_density;
        LOG_DEB("clutter intensity " << inno->second.clutter_intensity);
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
  if (!_manager->params().filter.gnn.costMatrixParams.use_external_sources_density)
  {
    for (auto const& [measIndex, meas] : std::views::enumerate(splitMeasurements._data))
    {
      negative_log_cost_matrix(measIndex, tracks.size() + measIndex) = max_cost + 1;
    }
  }
  std::string costMatrixString;

  for (auto const& [measIndex, meas] : std::views::enumerate(splitMeasurements._data))
  {
    for (int trackIndex = 0; trackIndex < tracks.size() + splitMeasurements._data.size(); trackIndex++)
    {
      costMatrixString += std::to_string(negative_log_cost_matrix(measIndex, trackIndex)) + " ";
    }
    costMatrixString += "\n";
  }

  LOG_DEB("cost matrix: \n" << costMatrixString);
  LOG_DEB("Finished building cost matrix");
  return negative_log_cost_matrix;
}

std::vector<State> GNN_Tracker::getEstimate() const
{
  std::vector<State> estimate;
  LOG_DEB("num tracks (GNN) before getEstimate(): " << _gnn_tracks.size());
  if (_manager->params().filter.gnn.trackManagementParams.output_only_confirmed)
  {
    for (auto& track : _gnn_tracks)
    {
      if (track._stage == STAGE::CONFIRMED)
      {
        estimate.push_back(track);
      }
    }
  }
  else
  {
    estimate = _gnn_tracks;
  }
  LOG_DEB("num tracks (GNN) after getEstimate(): " << estimate.size());
  return estimate;
}

void GNN_Tracker::reset()
{
  LOG_DEB("Reset GNN Filter");
  _gnn_tracks = std::vector<State>();
  _time = Time{ 0s };
}

void GNN_Tracker::addBirthTracks(MeasurementContainer const& measurementContainer,
                                 std::map<MeasurementId, double> const& rzMap)
{
  LOG_DEB("Start adding birth tracks");
  LOG_DEB("Type (birth model): " << to_string(_manager->getBirthModel().type()));
  BaseBirthModel& birthModel = _manager->getBirthModel();
  std::vector<State> birth_tracks = birthModel.getBirthTracks(measurementContainer, rzMap, _gnn_tracks);
  for (auto& track : birth_tracks)
  {
    LOG_DEB("Start tracks are born");
    if (!_manager->params()
             .filter.gnn.trackManagementParams.use_existence_prob_from_birth_model_for_score_initialisation and
        _manager->getBirthModel().type() == BIRTH_MODEL_TYPE::DYNAMIC)
    {
      LOG_DEB("Start calculating track score for dynamic birth tracks");
      Innovation const& innoMap = track._innovation.at(measurementContainer._id);
      double PD = innoMap._detectionProbability;
      auto measurements = measurementContainer._data;
      const auto measIt = find_if(
          measurementContainer._data.begin(), measurementContainer._data.end(), [&](const Measurement& measurement) {
            return measurement._id == track._meta_data._lastAssociatedMeasurement;
          });
      double false_target_density = innoMap._updates.find(measIt->_id)->second.clutter_intensity;
      if (_manager->params().filter.gnn.costMatrixParams.calculate_birth_track_density)
      {
        LOG_DEB("calculate birth_track_density");
        double fov_size = measurementContainer._sensorInfo._sensor_fov.value().area(measIt->_meas_comps);
        double mean_num_birth = _manager->params().birth_model->dynamic_model.mean_num_birth;
        LOG_DEB("fov_size " << fov_size);
        birth_track_density = mean_num_birth / fov_size;
      }
      else
      {
        LOG_DEB("use default (given in configs) birth_track_density");
        birth_track_density = _manager->params().filter.gnn.costMatrixParams.birth_track_density;
      }
      LOG_DEB("birth_track_density: " << birth_track_density);
      track._score = log(birth_track_density) - log(false_target_density) + log(PD);
    }
    _gnn_tracks.push_back(std::move(track));
    LOG_DEB("Initial track score of Track " << track._label << ": " << track._score);
  }
  LOG_DEB("Done adding " << birth_tracks.size() << " birth tracks");
}

void GNN_Tracker::postProcessUpdate(std::size_t num_sensors)
{
  int tracks_before = _gnn_tracks.size();
  int tentative_tracks = 0;
  int preliminary_tracks = 0;
  int confirmed_tracks = 0;
  int dead_tracks = 0;

  auto to_remove = [&](State& track) {
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
