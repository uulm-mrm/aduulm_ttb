#include "tracking_lib/TTTFilters/TransTTTFilter.h"

#include "tracking_lib/Transformations/Transformation.h"
#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/Misc/SeparatingAxisTest.h"
#include "tracking_lib/TTTFilters/TTTHelpers.h"

#include <tracy/tracy/Tracy.hpp>

auto constexpr tracy_color = tracy::Color::Aqua;

namespace ttb
{
TransTTTFilter::TransTTTFilter(TTBManager* manager) : _manager{ manager }
{
  LOG_DEB("TransTTTFilter constructed");
}

std::vector<State> TransTTTFilter::getEstimate() const
{
  return _tracks;
}

TTBManager* TransTTTFilter::manager() const
{
  return _manager;
}

Time TransTTTFilter::time() const
{
  return _time;
}

TTT_FILTER_TYPE TransTTTFilter::type() const
{
  return TTT_FILTER_TYPE::TRANS;
}

void TransTTTFilter::reset()
{
  _tracks.clear();
  _time = Time{ 0s };
}

void TransTTTFilter::cycle(Time time, std::vector<StateContainer> trackContainers)
{
  LOG_DEB("TransTTTFilter::cycle");
  ZoneScopedNC("TransTTTFilter::cycle", tracy_color);
  if (trackContainers.empty())
  {
    std::vector<State> predicted;
    for (State& state : _tracks)
    {
      if (state._meta_data._durationSinceLastAssociation <
          std::chrono::milliseconds(_manager->params().ttt_filter.trans.max_prediction_duration_ms))
      {
        // predict only non cam/vam tracks
        if (auto const cam_it = state._misc.find("planner"); cam_it == state._misc.end())
        {
          state.predict(time - state._time, EgoMotionDistribution::zero());
        }
        else
        {
          state._time = time;
        }
        if (not state.isEmpty())
        {
          predicted.push_back(std::move(state));
        }
      }
    }
    _tracks = std::move(predicted);
    _time = std::max(_time, time);
    return;
  }
  for (StateContainer& stateContainer : trackContainers)
  {
    Time const cycle_start = std::chrono::high_resolution_clock::now();
    std::vector<State> predicted;
    for (State& state : _tracks)
    {
      if (state._meta_data._durationSinceLastAssociation <
          std::chrono::milliseconds(_manager->params().ttt_filter.trans.max_prediction_duration_ms))
      {
        if (auto const cam_it = state._misc.find("planner"); cam_it == state._misc.end())
        {
          state.predict(stateContainer._time - state._time, stateContainer._egoMotion);
        }
        else
        {
          state._time = stateContainer._time;
        }
        if (not state.isEmpty())
        {
          predicted.push_back(std::move(state));
        }
      }
    }
    _tracks = std::move(predicted);
    _time = std::max(_time, stateContainer._time);
    tttfusion::AssignmentSol assignment = tttfusion::compute_assignment(
        _tracks,
        stateContainer._data,
        std::vector<std::pair<tttfusion::Metric, double> >{ std::pair{
            [&](State const& first, State const& second) {
              auto [modelId, currentStateDist] = first.getEstimate();
              Vector const currentMean = currentStateDist->mean();

              auto [newModelId, newDist] = second.getEstimate();
              Vector const newMean = newDist->mean();
              return static_cast<double>(not sat::is_overlapping(currentMean,
                                                                 _manager->getStateModel(modelId).state_comps(),
                                                                 newMean,
                                                                 _manager->getStateModel(newModelId).state_comps()));
            },
            0.5 } });
    std::vector<State> fused_tracks;
    for (auto& [first, second] : assignment.assigned)
    {
      std::optional<State> fused = tttfusion::cross_covariance_intersection(first, second, _manager);
      if (fused.has_value())
      {
        if (stateContainer._id.value_.find("CAM") != std::string::npos or
            stateContainer._id.value_.find("VAM") != std::string::npos)
        {
          fused.value()._misc["planner"] = std::string{ "associated with " + stateContainer._id.value_ };
          fused.value()._state_dist = std::move(second._state_dist);
          // all other tracks are not associated to this id
          for (State& state : fused_tracks)
          {
            if (auto const it = state._misc.find("planner");
                it != state._misc.end() and
                std::any_cast<std::string>(it->second) == std::string{ "associated with " + stateContainer._id.value_ })
            {
              state._misc.erase(it);
            }
          }
          for (State& state : assignment.non_assigned_first)
          {
            if (auto const it = state._misc.find("planner");
                it != state._misc.end() and
                std::any_cast<std::string>(it->second) == std::string{ "associated with " + stateContainer._id.value_ })
            {
              state._misc.erase(it);
            }
          }
        }
        else if (auto const cam_it = fused.value()._misc.find("planner"); cam_it != fused.value()._misc.end())
        {
          fused.value()._state_dist = std::move(first._state_dist);
        }
        fused_tracks.push_back(std::move(fused.value()));
      }
    }
    for (State& old_non_assigned : assignment.non_assigned_first)
    {
      fused_tracks.push_back(std::move(old_non_assigned));
    }
    for (State& new_non_assigned : assignment.non_assigned_second)
    {
      auto const& [id, dist] = new_non_assigned.getEstimate();
      auto trans = transformation::transform(
          dist->mean(),
          dist->covariance(),
          _manager->getStateModel(id).state_comps(),
          _manager->getStateModel(_manager->params().state.estimation.output_state_model).state_comps());
      if (not trans.has_value())
      {
        LOG_WARN("Can not transform Track " + new_non_assigned.toString() +
                 " to desired output state model. Ignore this track.");
        continue;
      }
      State state = _manager->createState();
      state._state_dist.emplace(
          _manager->params().state.estimation.output_state_model,
          std::make_unique<GaussianDistribution>(std::move(trans.value().mean), std::move(trans.value().cov)));
      state._time = new_non_assigned._time;
      state._meta_data._durationSinceLastAssociation = 0s;
      state._classification = new_non_assigned._classification;
      state._existenceProbability = new_non_assigned._existenceProbability;
      if (stateContainer._id.value_.find("CAM") != std::string::npos or
          stateContainer._id.value_.find("VAM") != std::string::npos)
      {
        for (State& prev_state : fused_tracks)
        {
          if (auto const it = prev_state._misc.find("planner");
              it != prev_state._misc.end() and
              std::any_cast<std::string>(it->second) == std::string{ "associated with " + stateContainer._id.value_ })
          {
            prev_state._misc.erase(it);
          }
        }
        state._misc["planner"] = std::string{ "associated with " + stateContainer._id.value_ };
      }
      fused_tracks.push_back(std::move(state));
    }
    _tracks = std::move(fused_tracks);
    assert([&] {  // NOLINT
      std::vector<std::string> planner;
      for (State const& state : _tracks)
      {
        if (auto const it = state._misc.find("planner"); it != state._misc.end())
        {
          if (auto const already_assoc_it = std::ranges::find(planner, std::any_cast<std::string>(it->second));
              already_assoc_it != planner.end())
          {
            return false;
            LOG_FATAL("Same CAM/VAM id assoc to multiple states: " << std::any_cast<std::string>(it->second));
          }
          planner.push_back(std::any_cast<std::string>(it->second));
        }
      }
      return true;
    }());
    if (_manager->params().show_gui)
    {
      _manager->vizu()._meas_model_data[MeasModelId{ stateContainer._id.value_ }]._computation_time.emplace_back(
          _time, std::chrono::high_resolution_clock::now() - cycle_start);
    }
  }
}

}  // namespace ttb