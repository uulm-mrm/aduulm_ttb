#include "tracking_lib/Trackers/LMB_FPM_Tracker.h"
#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/Measurements/MeasurementContainer.h"

#include <memory>
#include <utility>
#include <ranges>
#include <tracy/tracy/Tracy.hpp>

/// This file contains the implementation of
/// C. Hermann, M. Herrmann, T. Griebel, M. Buchholz and K. Dietmayer, "The Fast Product Multi-Sensor Labeled
/// Multi-Bernoulli Filter," 2023 26th International Conference on Information Fusion (FUSION), Charleston, SC, USA,
/// 2023, pp. 1-8, doi: 10.23919/FUSION52260.2023.10224189 (https://ieeexplore.ieee.org/document/10224189)
namespace ttb
{
constexpr auto tracy_color = tracy::Color::Indigo;
std::string to_string(LMB_FPM_Tracker_ProfilerData const& data)
{
  std::string out = "LMB_FPM_Tracker Cycle\n";
  out += "\tTime: " + std::to_string(to_nanoseconds(data._time.time_since_epoch())) + "ns\n";
  out += "\tCycle Duration: " + std::to_string(to_milliseconds(data._cycleDuration)) + "ms\n";
  out += "\tStatic Birth Duration: " + std::to_string(to_milliseconds(data._staticBirthDuration)) + "ms\n";
  out += "\tPrediction Duration: " + std::to_string(to_milliseconds(data._predictionDuration)) + "ms\n";
  for (auto const& [id, duration] : data._innovationDuration)
  {
    out += "\tMeasModel: " + id.value_ + " Innovation Duration: " + std::to_string(to_milliseconds(duration)) + "ms\n";
  }
  for (auto const& [id, stats] : data._measCycle)
  {
    out += "\tMeasModel: " + id.value_ + " MeasCycle Duration: " + std::to_string(to_milliseconds(stats._totDuration)) +
           "ms\n";
  }
  out += "\tFusion Duration Total: " + std::to_string(to_milliseconds(data._fusionDurationTotal)) + "ms\n";
  out += "\tFPM Fusion Duration Total: " + std::to_string(to_milliseconds(data._fusionDuration)) + "ms\n";
  out +=
      "\tFusion Dynamic Birth Duration: " + std::to_string(to_milliseconds(data._fusionDynamicBirthDuration)) + "ms\n";
  out += "\tPruneFusion Duration: " + std::to_string(to_milliseconds(data._postProcessUpdateDuration)) + "ms\n";
  out += "\t#Tracks: " + std::to_string(data._numTracks) + "\n";
  out += "\t#EstimatedTracks: " + std::to_string(data._numEstimatedTracks) + "\n";
  return out;
}

std::string to_stringStatistics(std::vector<LMB_FPM_Tracker_ProfilerData> const& datas)
{
  std::string out = "LMB_FPM_Tracker Cycle Statistics\n";
  struct FusionCenterCycleData
  {
    double total_cycle_mean{};  // time of complete cycle!
    double total_cycle_max{};
    double prediction_mean{};
    double prediction_max{};
    double staticBirth_mean{};
    double staticBirth_max{};
    double fusion_total_mean{};
    double fusion_total_max{};
    double fusion_birth_total_mean{};
    double fusion_birth_total_max{};
    double fpm_fusion_total_mean{};
    double fpm_fusion_total_max{};
    double postProcessUpdate_mean{};
    double postProcessUpdate_max{};
  };
  struct NodesCycleData
  {
    double innovation_mean{};
    double innovation_max{};
    double update_cycle_total_mean{};
    double update_cycle_total_max{};
    double grouping_mean{};
    double grouping_max{};
    double update_mean{};
    double update_max{};
    double merge_mean{};
    double merge_max{};
    double dynamicBirth_mean{};
    double dynamicBirth_max{};
  };
  std::map<MeasModelId, NodesCycleData> node_stats_map;
  FusionCenterCycleData fc_stats;
  std::size_t totalNumTracks = 0;
  std::size_t totalNumEstimatedTracks = 0;
  std::size_t maxNumTracks = 0;
  std::size_t maxEstimated = 0;
  for (auto const& data : datas)
  {
    fc_stats.total_cycle_mean += to_milliseconds(data._cycleDuration) / static_cast<double>(datas.size());
    fc_stats.total_cycle_max = std::max(to_milliseconds(data._cycleDuration), fc_stats.total_cycle_max);
    for (auto const& [id, duration] : data._innovationDuration)
    {
      NodesCycleData tmp;
      node_stats_map.emplace(id, tmp);
      node_stats_map[id].innovation_mean += to_milliseconds(duration) / static_cast<double>(datas.size());
      node_stats_map[id].innovation_max = std::max(to_milliseconds(duration), node_stats_map[id].innovation_max);
    }
    for (auto const& [id, stats] : data._measCycle)
    {
      node_stats_map[id].update_cycle_total_mean +=
          to_milliseconds(stats._totDuration) / static_cast<double>(datas.size());
      node_stats_map[id].update_cycle_total_max =
          std::max(to_milliseconds(stats._totDuration), node_stats_map[id].update_cycle_total_max);
      node_stats_map[id].grouping_mean += to_milliseconds(stats._groupingDuration) / static_cast<double>(datas.size());
      node_stats_map[id].grouping_max =
          std::max(to_milliseconds(stats._groupingDuration), node_stats_map[id].grouping_max);
      node_stats_map[id].update_mean += to_milliseconds(stats._updateDuration) / static_cast<double>(datas.size());
      node_stats_map[id].update_max = std::max(to_milliseconds(stats._updateDuration), node_stats_map[id].update_max);
      node_stats_map[id].merge_mean += to_milliseconds(stats._mergeDuration) / static_cast<double>(datas.size());
      node_stats_map[id].merge_max = std::max(to_milliseconds(stats._mergeDuration), node_stats_map[id].merge_max);
      node_stats_map[id].dynamicBirth_mean +=
          to_milliseconds(stats._dynamicBirthDuration) / static_cast<double>(datas.size());
      node_stats_map[id].dynamicBirth_max =
          std::max(to_milliseconds(stats._dynamicBirthDuration), node_stats_map[id].dynamicBirth_max);
    }
    fc_stats.fusion_total_mean += to_milliseconds(data._fusionDurationTotal) / static_cast<double>(datas.size());
    fc_stats.fusion_total_max = std::max(to_milliseconds(data._fusionDurationTotal), fc_stats.fusion_total_max);
    fc_stats.fusion_birth_total_mean +=
        to_milliseconds(data._fusionDynamicBirthDuration) / static_cast<double>(datas.size());
    fc_stats.fusion_birth_total_max =
        std::max(to_milliseconds(data._fusionDynamicBirthDuration), fc_stats.fusion_birth_total_max);
    fc_stats.fpm_fusion_total_max += to_milliseconds(data._fusionDuration) / static_cast<double>(datas.size());
    fc_stats.fpm_fusion_total_mean = std::max(to_milliseconds(data._fusionDuration), fc_stats.fpm_fusion_total_max);
    fc_stats.staticBirth_mean += to_milliseconds(data._staticBirthDuration) / static_cast<double>(datas.size());
    fc_stats.staticBirth_max = std::max(to_milliseconds(data._staticBirthDuration), fc_stats.staticBirth_max);
    fc_stats.postProcessUpdate_mean +=
        to_milliseconds(data._postProcessUpdateDuration) / static_cast<double>(datas.size());
    fc_stats.postProcessUpdate_max =
        std::max(to_milliseconds(data._postProcessUpdateDuration), fc_stats.postProcessUpdate_max);
    fc_stats.prediction_mean += to_milliseconds(data._predictionDuration) / static_cast<double>(datas.size());
    fc_stats.prediction_max = std::max(to_milliseconds(data._predictionDuration), fc_stats.prediction_max);
    totalNumTracks += data._numTracks;
    totalNumEstimatedTracks += data._numEstimatedTracks;
    maxNumTracks = std::max(data._numTracks, maxNumTracks);
    maxEstimated = std::max(data._numEstimatedTracks, maxEstimated);
  }

  double tmpInnoTime = node_stats_map.begin()->second.innovation_mean;
  double tmpUpdTotTime = node_stats_map.begin()->second.update_cycle_total_mean;
  for (auto const& [id, nodes_stats] : node_stats_map)
  {
    tmpInnoTime = std::max(nodes_stats.innovation_mean, tmpInnoTime);
    tmpUpdTotTime = std::max(nodes_stats.update_cycle_total_mean, tmpUpdTotTime);
  }
  double meanCalcTotal = tmpInnoTime + tmpUpdTotTime;
  meanCalcTotal += fc_stats.fusion_total_mean;
  meanCalcTotal += fc_stats.staticBirth_mean;
  meanCalcTotal += fc_stats.postProcessUpdate_mean;
  meanCalcTotal += fc_stats.prediction_mean;

  out += "\tMean/Max Cycle Duration: " + std::to_string(fc_stats.total_cycle_mean) + " / " +
         std::to_string(fc_stats.total_cycle_max) + "ms\n";
  out += "\tMean Calculated Cycle Duration: " + std::to_string(meanCalcTotal) + "ms\n";
  out += "\tMean/Max Prediction Duration: " + std::to_string(fc_stats.prediction_mean) + " / " +
         std::to_string(fc_stats.prediction_max) + "ms\n";
  out += "\tMean/Max Static Birth Duration: " + std::to_string(fc_stats.staticBirth_mean) + " / " +
         std::to_string(fc_stats.staticBirth_max) + "ms\n";
  for (auto const& [id, nodes_stats] : node_stats_map)
  {
    out +=
        "\tMean MeasModel: " + id.value_ + "\n\t\tMean/Max Innovation Duration:                               " +
        std::to_string(nodes_stats.innovation_mean) + " / " + std::to_string(nodes_stats.innovation_max) + "ms\n" +
        "\t\tMean/Max Total Cycle Duration of Update without Innovation:   " +
        std::to_string(nodes_stats.update_cycle_total_mean) + " / " +
        std::to_string(nodes_stats.update_cycle_total_max) + "ms\n" +
        "\t\tMean/Max Grouping Duration:                                   " +
        std::to_string(nodes_stats.grouping_mean) + " / " + std::to_string(nodes_stats.grouping_max) + "ms\n" +
        "\t\tMean/Max Update Duration:                                     " + std::to_string(nodes_stats.update_mean) +
        " / " + std::to_string(nodes_stats.update_max) + "ms\n" +
        "\t\tMean/Max Merge Duration:                                      " + std::to_string(nodes_stats.merge_mean) +
        " / " + std::to_string(nodes_stats.merge_max) + "ms\n" +
        "\t\tMean/Max Dynamic Birth Duration:                             " +
        std::to_string(nodes_stats.dynamicBirth_mean) + " / " + std::to_string(nodes_stats.dynamicBirth_max) + "ms\n";
  }
  out += "\tMean/Max Fusion Total Duration: " + std::to_string(fc_stats.fusion_total_mean) + " / " +
         std::to_string(fc_stats.fusion_total_max) + "ms\n";
  out += "\tMean/Max FPM Fusion Duration: " + std::to_string(fc_stats.fpm_fusion_total_mean) + " / " +
         std::to_string(fc_stats.fpm_fusion_total_max) + "ms\n";
  out += "\tMean/Max Fusion Dynamic Birth Duration: " + std::to_string(fc_stats.fusion_birth_total_mean) + " / " +
         std::to_string(fc_stats.fusion_birth_total_max) + "ms\n";
  out += "\tMean/Max Post Process Update Duration: " + std::to_string(fc_stats.postProcessUpdate_mean) + " / " +
         std::to_string(fc_stats.postProcessUpdate_max) + "ms\n";
  out += "\tTotal #Tracks: " + std::to_string(totalNumTracks) + "\n";
  out += "\tTotal #EstimatedTracks: " + std::to_string(totalNumEstimatedTracks) + "\n";
  out += "\tMean/Max #Tracks: " + std::to_string(static_cast<double>(totalNumTracks) / datas.size()) + " / " +
         std::to_string(maxNumTracks) + "\n";
  out += "\tMean/Max #EstimatedTracks: " + std::to_string(static_cast<double>(totalNumEstimatedTracks) / datas.size()) +
         " / " + std::to_string(maxEstimated) + "\n";
  return out;
}

LMB_FPM_Tracker::LMB_FPM_Tracker(TTBManager* manager)
  : _manager{ manager }, _dist_LMB(_manager), _parameter_estimation{ _manager }
{
}

FILTER_TYPE LMB_FPM_Tracker::type() const
{
  return FILTER_TYPE::LMB_FPM;
}

TTBManager* LMB_FPM_Tracker::manager() const
{
  return _manager;
}

Time LMB_FPM_Tracker::time() const
{
  return _time;
}

void LMB_FPM_Tracker::addStaticBirthTracks(MeasurementContainer const& measurementContainer,
                                           std::map<MeasurementId, double> const& rzMap)
{
  BaseBirthModel& birthModel = _manager->getBirthModel();
  std::vector<State> birthTracks = birthModel.getBirthTracks(measurementContainer, rzMap, _dist_LMB._tracks);
  for (State& track : birthTracks)
  {
    _dist_LMB._tracks.emplace_back(std::move(track));
  }
  LOG_DEB("Inserted " << birthTracks.size() << " Birth Tracks");
}

tttfusion::TracksSensor LMB_FPM_Tracker::addDynamicBirthTracks(const LMBDistribution& updatedLmbDist,
                                                               MeasurementContainer const& measurementContainer,
                                                               std::map<MeasurementId, double> const& rzMap) const
{
  BaseBirthModel& birthModel = _manager->getBirthModel();
  std::vector<State> birthTracks = birthModel.getBirthTracks(measurementContainer, rzMap, updatedLmbDist._tracks);
  std::map<StateId, State> birthTrackMap;

  for (State& track : birthTracks)
  {
    birthTrackMap.emplace(track._id, std::move(track));
  }
  LOG_DEB("Inserted " << birthTracks.size() << " Birth Tracks");
  return { ._id = measurementContainer._id,
           ._trackMapSensor = std::move(birthTrackMap),
           ._pD = _manager->meas_model_params(measurementContainer._id).detection.prob,
           ._sensorInfo = std::move(measurementContainer._sensorInfo) };
}

LMBDistribution LMB_FPM_Tracker::multiSensorDynamicBirth(std::vector<tttfusion::TracksSensor>&& birth_lmbs,
                                                         Duration& dynamicBirthTime) const
{
  ZoneScopedN("LMB_FPM_Tracker::multiSensorDynamicBirth");
  LMBDistribution fusedBirthLMB(_manager);
  if (birth_lmbs.size() == 1)
  {
    auto startTime = std::chrono::high_resolution_clock::now();
    for (const auto& [birth_id, birthTrack] : birth_lmbs.begin()->_trackMapSensor)
    {
      fusedBirthLMB._tracks.emplace_back(std::move(birthTrack));
    }
    dynamicBirthTime = std::chrono::high_resolution_clock::now() - startTime;
    return fusedBirthLMB;
  }

  auto startTime = std::chrono::high_resolution_clock::now();
  if (!birth_lmbs.empty())
  {
    ZoneScopedN("LMB_FPM_Tracker::Multi_Sensor_DynamicBirth");
    LOG_DEB("Birth tracks are not empty! Size: " << birth_lmbs.size());
    if (birth_lmbs.size() == 1)
    {
      throw std::runtime_error("only birth tracks of one sensor?!?");
    }
    uncorrelated_t2t_fusion::TTTUncorrelatedTracks uncorrT2Tfusion(
        _manager, std::move(_manager->params().filter.lmb_fpm.dynamicBirth), std::move(birth_lmbs));
    LOG_DEB("created T2Tfusion class");
    fusedBirthLMB = uncorrT2Tfusion.fuseTracksOfDifferentSensors();
  }
  dynamicBirthTime = std::chrono::high_resolution_clock::now() - startTime;

  return fusedBirthLMB;
}

void LMB_FPM_Tracker::postProcessStateDist()
{
  ZoneScopedNC("LMB_FPM_Tracker::postProcessStateDist", tracy_color);
  prune_if([](State const& state) { return state.isEmpty(); });
  if (_manager->params().thread_pool_size > 0)
  {
    _manager->thread_pool().detach_loop(
        std::size_t{ 0 }, _dist_LMB._tracks.size(), [&](std::size_t i) { _dist_LMB._tracks.at(i).postProcess(); });
    _manager->thread_pool().wait();
  }
  else
  {
    std::for_each(_dist_LMB._tracks.begin(), _dist_LMB._tracks.end(), [&](State& track) { track.postProcess(); });
  }
  prune_if([](State const& state) { return state.isEmpty(); });
}

void LMB_FPM_Tracker::cycle(Time time, std::vector<MeasurementContainer>&& measContainerList)
{
  ZoneScopedNC("LMB_FPM_Tracker::cycle", tracy_color);
  assert(_dist_LMB.isValid() and "Invalid LMB Dist");
  static profiling::GeneralDataProfiler<LMB_FPM_Tracker_ProfilerData> profiler("LMB_FPM_Tracker Cycle");

  LOG_INF("LMB_FPM CYCLE with " << measContainerList.size() << " Measurement Container");
  if (measContainerList.empty())
  {
    _dist_LMB.predict(time - _time, EgoMotionDistribution::zero());
    _time = time;
    return;
  }

  if (_manager->params().filter.lmb_fpm.sensor_number != measContainerList.size())
  {
    LOG_ERR("LMB_FPM_Tracker expects measurements of " << _manager->params().filter.lmb_fpm.sensor_number
                                                       << " sensors but only receives measurements of "
                                                       << measContainerList.size() << " sensors!");
  }
  auto const cycleStart = std::chrono::high_resolution_clock::now();
  LMB_FPM_Tracker_ProfilerData profilerData;
  profilerData._time = _time;

  //  bool sameMeasTime = true;
  Time measContainerTime = measContainerList.begin()->_time;
  for (auto it = std::next(measContainerList.begin()); it != measContainerList.end(); it++)
  {
    if (it->_time != measContainerTime)
    {
      LOG_ERR("Time stamps of measurements is not the same! Measurements should be synchronised for the use of PM "
              "trackers!");
    }
  }
  Duration deltaT = measContainerList.begin()->_time - _time;

  std::map<StateModelId, double> distID2mergeDist;  // save values of yaml file for later
  for (auto const& [state_model_id, state_model] : _manager->getStateModelMap())
  {
    distID2mergeDist.emplace(state_model_id,
                             _manager->state_model_params(state_model_id).distribution.post_process.merging_distance);
  }

  // Prediction of multi object distributions
  LOG_DEB("Predict");
  auto startTime = std::chrono::high_resolution_clock::now();
  _dist_LMB.predict(deltaT, measContainerList.begin()->_egoMotion);
  if (_manager->params().filter.lmb_fpm.do_profiling)
  {
    profilerData._predictionDuration = std::chrono::high_resolution_clock::now() - startTime;
  }
  LOG_DEB("Print LMB Density after prediction: " << _dist_LMB.toString());

  // static birth
  if (_manager->getBirthModel().type() == BIRTH_MODEL_TYPE::STATIC)
  {
    LOG_DEB("Add Static Birth Tracks");
    startTime = std::chrono::high_resolution_clock::now();
    addStaticBirthTracks(*measContainerList.begin(),
                         {});  // can_init of first measurement model must be set to true, the others do not matter here
    if (_manager->params().filter.lmb_fpm.do_profiling)
    {
      profilerData._staticBirthDuration = std::chrono::high_resolution_clock::now() - startTime;
    }
    LOG_DEB("Print LMB Density after static birth: " << _dist_LMB.toString());
  }

  // Calculation of the innovations
  // deactivate merging of state distributions for single sensor updates
  for (auto const& [state_model_id, state_model] : _manager->getStateModelMap())
  {
    _manager->state_model_next_params(state_model_id).distribution.post_process.merging_distance = 0;
  }
  _manager->update_params();
  for (auto const& measurementContainer : measContainerList)
  {
    LOG_DEB("Processing Msg. of sensor " + measurementContainer._id.value_
            << " with " << measurementContainer._data.size() << " Detections");

    LOG_DEB("Calculate Innovation");
    startTime = std::chrono::high_resolution_clock::now();
    _dist_LMB.calcInnovation(measurementContainer);
    if (_manager->params().filter.lmb_fpm.do_profiling)
    {
      profilerData._innovationDuration[measurementContainer._id] =
          std::chrono::high_resolution_clock::now() - startTime;
    }
  }
  // activate merging of state distributions for fusion, in update this should not be relevant anymore, since
  // GaussianMixtureDistributions are already calculated
  for (auto const& [state_model_id, state_model] : _manager->getStateModelMap())
  {
    _manager->state_model_next_params(state_model_id).distribution.post_process.merging_distance =
        distID2mergeDist.at(state_model_id);
  }
  LOG_DEB("Print predicted LMB Density at begin measurement cycle after innovation: " << _dist_LMB.toString());
  // perform multi-sensor update of fpm-lmb
  switch (_manager->params().filter.lmb_fpm.multi_sensor_update_method)
  {
    case MULTI_SENSOR_UPDATE_METHOD::FPM:
      fpm_update(std::move(measContainerList), profilerData);
      break;
    case MULTI_SENSOR_UPDATE_METHOD::PM:
      pm_update(std::move(measContainerList), profilerData);
      break;
    default:
      throw std::runtime_error("This pm update methode is not known... Multi-sensor update can not be performed");
  }

  // postprocessing
  assert(_dist_LMB.isValid() && "before postprocessing");
  LOG_DEB("Nr. Tracks: " << _dist_LMB._tracks.size());
  LOG_DEB("Postprocessing");
  LOG_DEB("Print LMB Density after fusion: " << _dist_LMB.toString());
  startTime = std::chrono::high_resolution_clock::now();
  _dist_LMB.postProcessUpdate();
  postProcessStateDist();
  _dist_LMB.resetPriorId();
  profilerData._postProcessUpdateDuration = std::chrono::high_resolution_clock::now() - startTime;
  LOG_DEB("Print LMB Density after postProcessUpdate: " << _dist_LMB.toString());
  assert(_dist_LMB.isValid() && "end of cycle");
  LOG_DEB("Nr. Tracks: " << _dist_LMB._tracks.size());

  if (_manager->params().filter.lmb_fpm.do_profiling)
  {
    profilerData._cycleDuration = std::chrono::high_resolution_clock::now() - cycleStart;
    profilerData._numTracks = _dist_LMB._tracks.size();
    profilerData._numEstimatedTracks = getEstimate().size();
    profiler.addData(std::move(profilerData));
  }
}

void LMB_FPM_Tracker::fpm_update(std::vector<MeasurementContainer>&& measContainerList,
                                 LMB_FPM_Tracker_ProfilerData& profilerData)
{
  ZoneScopedN("LMB_FPM_Tracker::fpm_update");
  // calculate single sensor updates for each sensor independently
  std::vector<LMBDistribution> singleSensorLMBUpdates;
  std::vector<tttfusion::TracksSensor> birthLMBs;
  std::mutex updated_single_sensors_lmbs_mutex;
  auto startTime = std::chrono::high_resolution_clock::now();
  if (_manager->params().thread_pool_size > 0)
  {
    _manager->next_params().lmb_distribution.calculate_single_sensor_group_updates_parallel = false;
    _manager->update_params();
    ZoneScopedN("LMB_FPM_Tracker::singleSensorUpdates");
    _manager->thread_pool().detach_loop(std::size_t{ 0 }, measContainerList.size(), [&](std::size_t i) {
      LOG_DEB("Start single sensor updates!");
      MeasurementCycleData measCycleData{};
      auto const MeasCycleStart = std::chrono::high_resolution_clock::now();
      LOG_DEB("Processing Msg. of sensor " + measContainerList.at(i)._id.value_
              << " with " << measContainerList.at(i)._data.size() << " Detections");
      LMBDistribution prior_lmb = _dist_LMB;
      prior_lmb.update(measContainerList.at(i));  // deactivate postprocessing of state distribution
      _parameter_estimation.update(
          measContainerList.at(i), prior_lmb.clutter_distribution(), prior_lmb.detection_distribution());
      LOG_DEB("update done");
      if (_manager->getBirthModel().type() == BIRTH_MODEL_TYPE::DYNAMIC)
      {
        //      LOG_FATAL("Add Dynamic Birth Tracks not implemented!!");
        startTime = std::chrono::high_resolution_clock::now();
        tttfusion::TracksSensor birth =
            addDynamicBirthTracks(prior_lmb, measContainerList.at(i), prior_lmb.meas_assignment_prob());
        {
          std::unique_lock lock(updated_single_sensors_lmbs_mutex);
          birthLMBs.push_back(std::move(birth));
        }
        measCycleData._dynamicBirthDuration = std::chrono::high_resolution_clock::now() - startTime;
      }
      //    LOG_DEB("Print LMB Density after postProcessUpdate: " << updatedLMBDist.toString());
      assert(prior_lmb.isValid() && "end of measurement update");
      LOG_DEB("Nr. Tracks: " << prior_lmb._tracks.size());
      {
        std::unique_lock lock(updated_single_sensors_lmbs_mutex);
        singleSensorLMBUpdates.push_back(std::move(prior_lmb));
        _time = measContainerList.at(i)._time;
        if (_manager->params().filter.lmb_fpm.do_profiling)
        {
          measCycleData._totDuration = std::chrono::high_resolution_clock::now() - MeasCycleStart;
          profilerData._measCycle[measContainerList.at(i)._id] = std::move(measCycleData);
        }
      }
    });
    _manager->thread_pool().wait();
    _manager->next_params().lmb_distribution.calculate_single_sensor_group_updates_parallel = true;
    _manager->update_params();
  }
  else
  {
    for (auto const& measurementContainer : measContainerList)
    {
      LOG_DEB("Start single sensor updates!");
      MeasurementCycleData measCycleData{};
      auto const MeasCycleStart = std::chrono::high_resolution_clock::now();
      LOG_INF("Processing Msg. of sensor " + measurementContainer._id.value_
              << " with " << measurementContainer._data.size() << " Detections");
      LMBDistribution prior_copy = _dist_LMB;
      prior_copy.update(measurementContainer);
      _parameter_estimation.update(
          measurementContainer, prior_copy.clutter_distribution(), prior_copy.detection_distribution());
      if (_manager->getBirthModel().type() == BIRTH_MODEL_TYPE::DYNAMIC)
      {
        startTime = std::chrono::high_resolution_clock::now();
        tttfusion::TracksSensor birth =
            addDynamicBirthTracks(prior_copy, measurementContainer, prior_copy.meas_assignment_prob());
        birthLMBs.push_back(std::move(birth));
        measCycleData._dynamicBirthDuration = std::chrono::high_resolution_clock::now() - startTime;
      }
      assert(prior_copy.isValid() && "end of measurement update");
      LOG_DEB("Nr. Tracks: " << prior_copy._tracks.size());
      singleSensorLMBUpdates.push_back(std::move(prior_copy));
      _time = measurementContainer._time;
      if (_manager->params().filter.lmb_fpm.do_profiling)
      {
        measCycleData._totDuration = std::chrono::high_resolution_clock::now() - MeasCycleStart;
        profilerData._measCycle[measurementContainer._id] = std::move(measCycleData);
      }
    }
  }
  _manager->update_params();
  // FPM fusion of single-sensor updates!
  startTime = std::chrono::high_resolution_clock::now();
  Duration dynamicBirth;
  Duration fpmFusionDuration;
  auto fpmFusionStartTime = std::chrono::high_resolution_clock::now();
  _dist_LMB.fpm_fusion(std::move(singleSensorLMBUpdates), false);
  fpmFusionDuration = std::chrono::high_resolution_clock::now() - fpmFusionStartTime;

  LMBDistribution fusedBirthLMB = multiSensorDynamicBirth(std::move(birthLMBs), dynamicBirth);

  // merge birth distribution into final fused distribution!
  _dist_LMB.merge(std::move(fusedBirthLMB));

  if (_manager->params().filter.lmb_fpm.do_profiling)
  {
    profilerData._fusionDurationTotal = std::chrono::high_resolution_clock::now() - startTime;
    profilerData._fusionDuration = fpmFusionDuration;
    profilerData._fusionDynamicBirthDuration = dynamicBirth;
  }
  LOG_DEB("Fused dist after fpm fusion and multi-sensor birth: " << _dist_LMB.toString());
  assert(_dist_LMB.isValid() && "Fusion is not valid after fpm fusion and multi-sensor dynamic birth!");
}

void LMB_FPM_Tracker::pm_update(std::vector<MeasurementContainer>&& measContainerList,
                                LMB_FPM_Tracker_ProfilerData& profilerData)
{
  ZoneScopedN("LMB_FPM_Tracker::pm_update");
  // calculate single sensor updates for each sensor independently
  std::vector<GLMBDistribution> singleSensorGLMBUpdates;
  std::vector<tttfusion::TracksSensor> birthLMBs;
  std::mutex updated_single_sensors_glmbs_mutex;
  auto startTime = std::chrono::high_resolution_clock::now();
  GLMBDistribution glmb_dist(_manager, std::move(_dist_LMB._tracks));
  glmb_dist.generateHypotheses();
  glmb_dist.postProcessPrediction();
  if (_manager->params().thread_pool_size > 0)
  {
    _manager->next_params().lmb_distribution.calculate_single_sensor_group_updates_parallel = false;
    _manager->update_params();
    ZoneScopedN("LMB_FPM_Tracker::singleSensorUpdates");
    _manager->thread_pool().detach_loop(std::size_t{ 0 }, measContainerList.size(), [&](std::size_t i) {
      LOG_DEB("Start single sensor updates!");
      MeasurementCycleData measCycleData{};
      auto const MeasCycleStart = std::chrono::high_resolution_clock::now();
      LOG_DEB("Processing Msg. of sensor " + measContainerList.at(i)._id.value_
              << " with " << measContainerList.at(i)._data.size() << " Detections");
      GLMBDistribution prior_glmb = glmb_dist;  // copy needed...
      prior_glmb.update(measContainerList.at(i));
      _parameter_estimation.update(
          measContainerList.at(i),
          prior_glmb.clutter_distribution(static_cast<Index>(measContainerList.at(i)._data.size())),
          prior_glmb.detection_distribution());
      LOG_DEB("update done");
      if (_manager->getBirthModel().type() == BIRTH_MODEL_TYPE::DYNAMIC)
      {
        //      LOG_FATAL("Add Dynamic Birth Tracks not implemented!!");
        startTime = std::chrono::high_resolution_clock::now();
        LMBDistribution updatedLMB(_manager);
        updatedLMB.convertGLMB2LMB(prior_glmb);  // todo(hermann) this must be done different for faster runtime...!!
        tttfusion::TracksSensor birth =
            addDynamicBirthTracks(updatedLMB, measContainerList.at(i), updatedLMB.meas_assignment_prob());
        {
          std::unique_lock lock(updated_single_sensors_glmbs_mutex);
          birthLMBs.push_back(std::move(birth));
        }
        measCycleData._dynamicBirthDuration = std::chrono::high_resolution_clock::now() - startTime;
      }
      //    LOG_DEB("Print LMB Density after postProcessUpdate: " << updatedLMBDist.toString());
      assert(prior_glmb.isValid() && "end of measurement update");
      {
        std::unique_lock lock(updated_single_sensors_glmbs_mutex);
        singleSensorGLMBUpdates.push_back(std::move(prior_glmb));
        _time = measContainerList.at(i)._time;
        if (_manager->params().filter.lmb_fpm.do_profiling)
        {
          measCycleData._totDuration = std::chrono::high_resolution_clock::now() - MeasCycleStart;
          profilerData._measCycle[measContainerList.at(i)._id] = std::move(measCycleData);
        }
      }
    });
    _manager->thread_pool().wait();
    _manager->next_params().lmb_distribution.calculate_single_sensor_group_updates_parallel = true;
    _manager->update_params();
  }
  else
  {
    for (auto const& measurementContainer : measContainerList)
    {
      LOG_DEB("Start single sensor updates!");
      MeasurementCycleData measCycleData{};
      auto const MeasCycleStart = std::chrono::high_resolution_clock::now();
      LOG_INF("Processing Msg. of sensor " + measurementContainer._id.value_
              << " with " << measurementContainer._data.size() << " Detections");
      GLMBDistribution prior_glmb = glmb_dist;  // copy needed...
      prior_glmb.update(measurementContainer);
      _parameter_estimation.update(
          measurementContainer,
          prior_glmb.clutter_distribution(static_cast<Index>(measurementContainer._data.size())),
          prior_glmb.detection_distribution());
      if (_manager->getBirthModel().type() == BIRTH_MODEL_TYPE::DYNAMIC)
      {
        startTime = std::chrono::high_resolution_clock::now();
        LMBDistribution updatedLMB(_manager);
        updatedLMB.convertGLMB2LMB(prior_glmb);  // todo(hermann) this must be done different for faster runtime...!!
        tttfusion::TracksSensor birth =
            addDynamicBirthTracks(updatedLMB, measurementContainer, updatedLMB.meas_assignment_prob());
        birthLMBs.push_back(std::move(birth));
        measCycleData._dynamicBirthDuration = std::chrono::high_resolution_clock::now() - startTime;
      }
      assert(prior_glmb.isValid() && "end of measurement update");
      singleSensorGLMBUpdates.push_back(std::move(prior_glmb));
      _time = measurementContainer._time;
      if (_manager->params().filter.lmb_fpm.do_profiling)
      {
        measCycleData._totDuration = std::chrono::high_resolution_clock::now() - MeasCycleStart;
        profilerData._measCycle[measurementContainer._id] = std::move(measCycleData);
      }
    }
  }
  _manager->update_params();
  // FPM fusion of single-sensor updates!
  startTime = std::chrono::high_resolution_clock::now();
  Duration dynamicBirth;
  Duration pmFusionDuration;
  auto fpmFusionStartTime = std::chrono::high_resolution_clock::now();
  glmb_dist.pm_fusion(std::move(singleSensorGLMBUpdates));  // todo(hermann): implement in GLMBDistribution...
  pmFusionDuration = std::chrono::high_resolution_clock::now() - fpmFusionStartTime;
  // convert GLMB to LMB density
  _dist_LMB.convertGLMB2LMB(std::move(glmb_dist));
  // dynamic birth
  LMBDistribution fusedBirthLMB = multiSensorDynamicBirth(std::move(birthLMBs), dynamicBirth);
  // merge birth distribution into final fused distribution!
  _dist_LMB.merge(std::move(fusedBirthLMB));

  if (_manager->params().filter.lmb_fpm.do_profiling)
  {
    profilerData._fusionDurationTotal = std::chrono::high_resolution_clock::now() - startTime;
    profilerData._fusionDuration = pmFusionDuration;
    profilerData._fusionDynamicBirthDuration = dynamicBirth;
  }
  LOG_DEB("Fused dist after fpm fusion and multi-sensor birth: " << _dist_LMB.toString());
  assert(_dist_LMB.isValid() && "Fusion is not valid after fpm fusion and multi-sensor dynamic birth!");
}

std::vector<State> LMB_FPM_Tracker::getEstimate() const
{
  std::vector<State> tracks = _dist_LMB.getEstimate();
  for (State& track : tracks)
  {
    track._misc["origin"] = std::string("LMB_FPM");
  }
  return tracks;
}

void LMB_FPM_Tracker::reset()
{
  LOG_ERR("Reset LMB_FPM Filter");
  _dist_LMB = LMBDistribution(_manager);
  _time = Time{ 0s };
}

}  // namespace ttb
