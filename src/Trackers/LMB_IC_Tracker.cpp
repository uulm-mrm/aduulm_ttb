#include "tracking_lib/Trackers/LMB_IC_Tracker.h"
// #####################################################################################################################
#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/Measurements/MeasurementContainer.h"
#include "tracking_lib/BirthModels/StaticBirthModel.h"
#include "tracking_lib/States/Innovation.h"
#include "tracking_lib/SelfAssessment/SelfAssessment.h"

#include <memory>
#include <utility>
#include <boost/math/distributions/beta.hpp>
#include <boost/math/distributions/gamma.hpp>
#include <tracy/tracy/Tracy.hpp>

namespace ttb
{
constexpr auto tracy_color = tracy::Color::Goldenrod;

std::string to_string(LMB_IC_Tracker_ProfilerData const& data)
{
  std::string out = "LMB_IC_Tracker Cycle\n";
  out += "\tTime: " + std::to_string(to_nanoseconds(data._time.time_since_epoch())) + "ns\n";
  out += "\tCycle Duration: " + std::to_string(to_milliseconds(data._cycleDuration)) + "ms\n";
  for (auto const& [id, stats] : data._measCycle)
  {
    out += "\tMeasModel: " + id.value_ + " Cycle Duration: " + std::to_string(to_milliseconds(stats._totDuration)) +
           "ms\n";
  }
  out += "\t#Tracks: " + std::to_string(data._numTracks) + "\n";
  out += "\t#EstimatedTracks: " + std::to_string(data._numEstimatedTracks) + "\n";
  return out;
}

std::string to_stringStatistics(std::vector<LMB_IC_Tracker_ProfilerData> const& datas)
{
  std::string out = "LMB_IC_Tracker Cycle Statistics\n";
  Duration meanCycleDuration(0);
  struct CycleStats
  {
    double total_mean{};
    double total_max{};
    double prediction_mean{};
    double prediction_max{};
    double postProcessPrediction_mean{};
    double postProcessPrediction_max{};
    double staticBirth_mean{};
    double staticBirth_max{};
    double innovation_mean{};
    double innovation_max{};
    double grouping_mean{};
    double grouping_max{};
    double update_mean{};
    double update_max{};
    double dynamicBirth_mean{};
    double dynmaicBirth_max{};
    double postProcessUpdate_mean{};
    double postProcessUpdate_max{};
    double clutterRate_mean{};
    double clutterRate_max{};
    double detectionProb_mean{};
    double detectionProb_max{};
  };
  std::map<MeasModelId, CycleStats> stats_map;
  std::size_t totalNumTracks = 0;
  std::size_t totalNumEstimatedTracks = 0;
  std::size_t maxNumTracks = 0;
  std::size_t maxEstimated = 0;
  for (auto const& data : datas)
  {
    meanCycleDuration += data._cycleDuration / datas.size();
    for (auto const& [id, stats] : data._measCycle)
    {
      CycleStats tmp;
      stats_map.emplace(id, tmp);
      stats_map[id].total_mean += to_milliseconds(stats._totDuration) / datas.size();
      stats_map[id].total_max = std::max(to_milliseconds(stats._totDuration), stats_map[id].total_max);
      stats_map[id].prediction_mean += to_milliseconds(stats._predicationDuration) / datas.size();
      stats_map[id].prediction_max =
          std::max(to_milliseconds(stats._predicationDuration), stats_map[id].prediction_max);
      stats_map[id].postProcessPrediction_mean += to_milliseconds(stats._postProcessPredictionDuration) / datas.size();
      stats_map[id].postProcessPrediction_max =
          std::max(to_milliseconds(stats._postProcessPredictionDuration), stats_map[id].postProcessPrediction_max);
      stats_map[id].staticBirth_mean += to_milliseconds(stats._staticBirthDuration) / datas.size();
      stats_map[id].staticBirth_max =
          std::max(to_milliseconds(stats._staticBirthDuration), stats_map[id].staticBirth_max);
      stats_map[id].innovation_mean += to_milliseconds(stats._calcInnovationDuration) / datas.size();
      stats_map[id].innovation_max =
          std::max(to_milliseconds(stats._calcInnovationDuration), stats_map[id].innovation_max);
      stats_map[id].grouping_mean += to_milliseconds(stats._groupingDuration) / datas.size();
      stats_map[id].grouping_max = std::max(to_milliseconds(stats._groupingDuration), stats_map[id].grouping_max);
      stats_map[id].update_mean += to_milliseconds(stats._updateDuration) / datas.size();
      stats_map[id].update_max = std::max(to_milliseconds(stats._updateDuration), stats_map[id].update_max);
      stats_map[id].dynamicBirth_mean += to_milliseconds(stats._dynmicBirthDuration) / datas.size();
      stats_map[id].dynmaicBirth_max =
          std::max(to_milliseconds(stats._dynmicBirthDuration), stats_map[id].dynmaicBirth_max);
      stats_map[id].postProcessUpdate_mean += to_milliseconds(stats._postProcessUpdateDuration) / datas.size();
      stats_map[id].postProcessUpdate_max =
          std::max(to_milliseconds(stats._postProcessUpdateDuration), stats_map[id].postProcessUpdate_max);
      stats_map[id].clutterRate_mean += to_milliseconds(stats._clutterRateDuration) / datas.size();
      stats_map[id].clutterRate_max =
          std::max(to_milliseconds(stats._clutterRateDuration), stats_map[id].clutterRate_max);
      stats_map[id].detectionProb_mean += to_milliseconds(stats._detectionProbDuration) / datas.size();
      stats_map[id].detectionProb_max =
          std::max(to_milliseconds(stats._detectionProbDuration), stats_map[id].detectionProb_max);
    }
    totalNumTracks += data._numTracks;
    totalNumEstimatedTracks += data._numEstimatedTracks;
    maxNumTracks = std::max(data._numTracks, maxNumTracks);
    maxEstimated = std::max(data._numEstimatedTracks, maxEstimated);
  }
  out += "\tMean Cycle Duration: " + std::to_string(to_milliseconds(meanCycleDuration)) + "ms\n";
  for (auto const& [id, stats] : stats_map)
  {
    out += "\tMeasModel: " + id.value_ +
           "\n\t\tMean/Max Total Cycle Duration:           " + std::to_string(stats.total_mean) + " / " +
           std::to_string(stats.total_max) + " ms\n" +
           "\t\tMean/Max Prediction Duration:                " + std::to_string(stats.prediction_mean) + " / " +
           std::to_string(stats.prediction_max) + " ms\n" +
           "\t\tMean/Max Post Process Prediction Duration:   " + std::to_string(stats.postProcessPrediction_mean) +
           " / " + std::to_string(stats.postProcessPrediction_max) + " ms\n" +
           "\t\tMean/Max Static Birth Duration:              " + std::to_string(stats.staticBirth_mean) + " / " +
           std::to_string(stats.staticBirth_max) + " ms\n" +
           "\t\tMean/Max Calc Innovation Duration:           " + std::to_string(stats.innovation_mean) + " / " +
           std::to_string(stats.innovation_max) + " ms\n" +
           "\t\tMean/Max Grouping Duration:                  " + std::to_string(stats.grouping_mean) + " / " +
           std::to_string(stats.grouping_max) + " ms\n" +
           "\t\tMean/Max Update Duration:                    " + std::to_string(stats.update_mean) + " / " +
           std::to_string(stats.update_max) + " ms\n" +
           "\t\tMean/Max Dynamic Birth Duration:             " + std::to_string(stats.dynamicBirth_mean) + " / " +
           std::to_string(stats.dynmaicBirth_max) + " ms\n" +
           "\t\tMean/Max Post Process Update Duration:       " + std::to_string(stats.postProcessUpdate_mean) + " / " +
           std::to_string(stats.postProcessUpdate_max) + " ms\n" +
           "\t\tMean/Max Clutter Rate Estimation Duration:   " + std::to_string(stats.clutterRate_mean) + " / " +
           std::to_string(stats.clutterRate_max) + " ms\n" +
           "\t\tMean/Max Detection Prob Estimation Duration: " + std::to_string(stats.detectionProb_mean) + " / " +
           std::to_string(stats.detectionProb_max) + " ms\n";
  }
  out += "\tTotal #Tracks: " + std::to_string(totalNumTracks) + "\n";
  out += "\tMean/Max #Tracks: " + std::to_string(static_cast<double>(totalNumTracks) / datas.size()) + " / " +
         std::to_string(maxNumTracks) + "\n";
  out += "\tMean/Max #EstimatedTracks: " + std::to_string(static_cast<double>(totalNumEstimatedTracks) / datas.size()) +
         " / " + std::to_string(maxEstimated) + "\n";
  return out;
}

LMB_IC_Tracker::LMB_IC_Tracker(TTBManager* manager)
  : _manager{ manager }, _dist_LMB{ _manager }, _parameter_estimation{ _manager }
{
}

FILTER_TYPE LMB_IC_Tracker::type() const
{
  return FILTER_TYPE::LMB_IC;
}

TTBManager* LMB_IC_Tracker::manager() const
{
  return _manager;
}

Time LMB_IC_Tracker::time() const
{
  return _time;
}

void LMB_IC_Tracker::addBirthTracks(MeasurementContainer const& measurementContainer,
                                    std::map<MeasurementId, double> const& rzMap)
{
  BaseBirthModel& birthModel = _manager->getBirthModel();
  std::vector<State> birthTracks = birthModel.getBirthTracks(measurementContainer, rzMap, _dist_LMB._tracks);
  for (State& track : birthTracks)
  {
    _dist_LMB._tracks.push_back(std::move(track));
  }
  LOG_DEB("Inserted " << birthTracks.size() << " Birth Tracks");
}

void LMB_IC_Tracker::cycle(Time time, std::vector<MeasurementContainer>&& measContainerList)
{
  ZoneScopedNC("LMB_IC_Tracker::cycle", tracy_color);
  assert(_dist_LMB.isValid() and "Invalid LMB Dist");
  static profiling::GeneralDataProfiler<LMB_IC_Tracker_ProfilerData> filterProfiler("/tmp/lmb_ic_tracker");
  LOG_DEB("LMB_IC CYCLE with " << measContainerList.size() << " Measurement Container");
  if (measContainerList.empty())
  {
    _dist_LMB.predict(time - _time, EgoMotionDistribution::zero());
    _time = time;
  }
  Time const start_lmb_ic_cycle_time = _time;
  auto const start_lmb_ic = std::chrono::high_resolution_clock::now();
  for (auto const& measurementContainer : measContainerList)
  {
    ZoneScopedNC("LMB_IC_Tracker::ic_cycle", tracy_color);
    ZoneText(measurementContainer._id.value_.c_str(), measurementContainer._id.value_.size());
    auto const meas_update_start = std::chrono::high_resolution_clock::now();
    assert(_dist_LMB.isValid() && "cycle start");
    Duration deltaT = measurementContainer._time - _time;
    if (deltaT < 0ms)
    {
      LOG_FATAL("Receiving Measurement Container from the ancient past .... - MUST NOT HAPPEN");
      LOG_FATAL("Filter Time: " + to_string(_time) +
                " Measurement Container Time: " + to_string(measurementContainer._time));
      LOG_FATAL("MeasurementContainer: " << measurementContainer._id.value_);
      throw std::runtime_error("Receiving Measurement Container from the ancient past .... - MUST NOT HAPPEN");
    }
    LOG_DEB("Processing Msg. of sensor " + measurementContainer._id.value_
            << " with " << measurementContainer._data.size() << " Detections");
    _dist_LMB.predict(deltaT, measurementContainer._egoMotion);
    _time = measurementContainer._time;

    if (_manager->getBirthModel().type() == BIRTH_MODEL_TYPE::STATIC)
    {
      LOG_DEB("Add Static Birth Tracks");
      addBirthTracks(measurementContainer, {});
    }
    _dist_LMB.calcInnovation(measurementContainer);
    _dist_LMB.update(measurementContainer);
    _parameter_estimation.update(
        measurementContainer, _dist_LMB.clutter_distribution(), _dist_LMB.detection_distribution());
    if (_manager->getBirthModel().type() == BIRTH_MODEL_TYPE::DYNAMIC)
    {
      LOG_DEB("Add Dynamic Birth Tracks");
      addBirthTracks(measurementContainer, _dist_LMB.meas_assignment_prob());
    }
    assert(_dist_LMB.isValid() && "before postprocessing");
    _time = measurementContainer._time;
    if (_manager->params().show_gui)
    {
      std::lock_guard lock(_manager->vizu().add_data_mutex);
      _manager->vizu()._meas_model_data[measurementContainer._id]._computation_time.emplace_back(
          start_lmb_ic_cycle_time, std::chrono::high_resolution_clock::now() - meas_update_start);
    }
  }
  filterProfiler.addData({
      ._time = _time,
      ._cycleDuration = std::chrono::high_resolution_clock::now() - start_lmb_ic,
      ._numTracks = _dist_LMB._tracks.size(),
  });
}

std::vector<State> LMB_IC_Tracker::getEstimate() const
{
  std::vector<State> tracks = _dist_LMB.getEstimate();
  for (State& track : tracks)
  {
    track._misc["origin"] = std::string("LMB_IC");
  }
  return tracks;
}

void LMB_IC_Tracker::reset()
{
  LOG_DEB("Reset LMB_IC Filter");
  _dist_LMB = LMBDistribution(_manager);
  _parameter_estimation = sa::ParameterEstimation(_manager);
  _time = Time{ 0s };
}

}  // namespace ttb
