#include "tracking_lib/Trackers/NN_Tracker.h"
// #####################################################################################################################
#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/Measurements/MeasurementContainer.h"
#include "tracking_lib/MultiObjectStateDistributions/LMBDistribution.h"
#include "tracking_lib/States/Innovation.h"

namespace ttb
{
NN_Tracker::NN_Tracker(TTBManager* manager) : _manager{ manager }
{
}

FILTER_TYPE NN_Tracker::type() const
{
  return FILTER_TYPE::NN;
}

TTBManager* NN_Tracker::manager() const
{
  return _manager;
}

Time NN_Tracker::time() const
{
  return _time;
}

void NN_Tracker::cycle(Time time, std::vector<MeasurementContainer>&& measContainerList)
{
  LOG_DEB("Update " + std::to_string(_tracks.size()) + " current Tracks");
  std::vector<MeasurementId> used_to_update;

  for (auto const& measContainer : measContainerList)
  {
    LMBDistribution lmb(_manager);
    lmb._tracks = _tracks;
    LOG_DEB("Predict LMB " + std::to_string(to_seconds(measContainer._time - _time)) + "s into the future");
    lmb.predict(measContainer._time - _time, measContainer._egoMotion);
    if (true)  // post process
    {
      LOG_DEB("Post Process the Prediction");
      lmb.postProcessPrediction();
    }

    LOG_DEB("Calc Innovation");
    lmb.calcInnovation(measContainer);
    // replace tracks with updated lmb tracks with the highest weight
    std::vector<State> updated_tracks;
    std::vector<MeasurementId> used_meas;
    for (auto& updated_track : lmb._tracks)
    {
      auto best = std::ranges::max_element(
          updated_track._innovation.at(measContainer._id)._updates,
          [](auto const& a, auto const& b) { return a.second.log_likelihood < b.second.log_likelihood; });

      State track = [&] {
        if (best != updated_track._innovation.at(measContainer._id)._updates.end())
        {
          used_to_update.push_back(best->first);
          return best->second.updated_dist;
        }
        return updated_track;
      }();
      track._time = measContainer._time;
      updated_tracks.push_back(std::move(track));
      _time = std::max(_time, measContainer._time);
    }
    _tracks = std::move(updated_tracks);
  }
  LOG_DEB("Updated " + std::to_string(_tracks.size()) + " Tracks");
  if (true)  // post_process
  {
    LOG_DEB("Post Process Update");
    LMBDistribution lmb(_manager);
    lmb._tracks = std::move(_tracks);
    lmb.postProcessUpdate();
    _tracks = std::move(lmb._tracks);
  }
  LOG_DEB(std::to_string(_tracks.size()) + " Tracks left");

  LOG_DEB("Insert non associated Detections as new Tracks");
  for (auto const& measurementContainer : measContainerList)
  {
    std::map<MeasurementId, double> assignment_prob;
    for (auto const& measurement : measurementContainer._data)
    {
      if (std::ranges::find(used_to_update, measurement._id) == used_to_update.end())
      {
        assignment_prob.emplace(measurement._id, 0);
      }
    }
    auto birth_tracks = _manager->getBirthModel().getBirthTracks(measurementContainer, assignment_prob, _tracks);
    _tracks.insert(_tracks.end(), birth_tracks.begin(), birth_tracks.end());
    _time = std::max(_time, measurementContainer._time);
  }
  LOG_DEB("Now there are " + std::to_string(_tracks.size()) + " Tracks");
}

std::vector<State> NN_Tracker::getEstimate() const
{
  return _tracks;
}

void NN_Tracker::reset()
{
}

}  // namespace ttb