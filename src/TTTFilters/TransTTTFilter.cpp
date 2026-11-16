#include "tracking_lib/TTTFilters/TransTTTFilter.h"

#include "tracking_lib/Transformations/Transformation.h"
#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/Misc/SeparatingAxisTest.h"
#include "tracking_lib/TTTFilters/TTTHelpers.h"

#include <boost/math/distributions/chi_squared.hpp>
#include <set>

namespace ttb
{

template class TransTTTFilter<double>;
template class TransTTTFilter<AD>;

template <class UDT>
TransTTTFilter<UDT>::TransTTTFilter(TTBManager<UDT>* manager) : _manager{ manager }
{
  LOG_DEB("TransTTTFilter constructed");
}

template <class UDT>
std::vector<BernoulliState<UDT>> TransTTTFilter<UDT>::get_estimate() const
{
  return _tracks;
}

template <class UDT>
std::vector<StateId> TransTTTFilter<UDT>::all_tracks() const
{
  return {};
}

template <class UDT>
TTBManager<UDT>* TransTTTFilter<UDT>::manager() const
{
  return _manager;
}

template <class UDT>
Time TransTTTFilter<UDT>::time() const
{
  return _time;
}

template <class UDT>
TTT_FILTER_TYPE TransTTTFilter<UDT>::type() const
{
  return TTT_FILTER_TYPE::TRANS;
}

template <class UDT>
void TransTTTFilter<UDT>::reset()
{
  _tracks.clear();
  _time = Time{ 0s };
}

template <class UDT>
void TransTTTFilter<UDT>::cycle(std::vector<DataSourceContainer<UDT>> trackContainers)
{
  LOG_DEB("TransTTTFilter::cycle");
  for (DataSourceContainer<UDT>& stateContainer : trackContainers)
  {
    Time const cycle_start = std::chrono::high_resolution_clock::now();
    if (stateContainer._source_id.value_.find("CAM") != std::string::npos or
        stateContainer._source_id.value_.find("VAM") != std::string::npos)
    {
      _cam_vam_source_to_label.emplace(stateContainer._source_id, State<UDT>::_labelGenerator.getID());
      if (stateContainer._tracks.size() != 1)
      {
        LOG_WARN("received vam/vam data container with data size != 1. Ignoring this beast...");
        LOG_WARN(stateContainer.toString());
        continue;
      }
    }
    std::vector<BernoulliState<UDT>> predicted;
    for (BernoulliState<UDT>& state : _tracks)
    {
      if (state.state._meta_data._durationSinceLastAssociation <
          std::chrono::milliseconds(_manager->params()->ttt_filter.trans.max_prediction_duration_ms))
      {
        state.state.predict(stateContainer._time - state.state._time, stateContainer._predictionInfo._egoMotion);

        if (not state.state.isEmpty())
        {
          predicted.push_back(std::move(state));
        }
        else
        {
          LOG_DEB("state is empty");
        }
      }
    }
    _tracks = std::move(predicted);
    _time = std::max(_time, stateContainer._time);
    tttfusion::AssignmentSol assignment = [&] {
      if (_manager->params()->ttt_filter.trans.reuse_source_ids)
      {
        tttfusion::AssignmentSol<UDT> label_assignment;
        std::set<Label> used_second_labels;
        for (BernoulliState<UDT> const& first : _tracks)
        {
          if (auto const it = std::ranges::find_if(
                  stateContainer._tracks,
                  [&](BernoulliState<UDT> const& track) { return track.state._label == first.state._label; });
              it != stateContainer._tracks.end() and not used_second_labels.contains(it->state._label))
          {
            label_assignment.assigned.emplace_back(first, *it);
            used_second_labels.insert(it->state._label);
          }
          else
          {
            label_assignment.non_assigned_first.push_back(first);
          }
        }
        for (BernoulliState<UDT> const& second : stateContainer._tracks)
        {
          if (not used_second_labels.contains(second.state._label))
          {
            label_assignment.non_assigned_second.push_back(second);
          }
        }
        return label_assignment;
      }
      return tttfusion::compute_assignment(
          _tracks,
          stateContainer._tracks,
          std::vector<std::pair<tttfusion::Metric<UDT>, UDT>>{
              std::pair{ [&](BernoulliState<UDT> const& first, BernoulliState<UDT> const& second) {
                          auto [modelId, currentStateDist] = first.state.getEstimate();
                          Vector<UDT> const currentMean = currentStateDist.mean();

                          auto [newModelId, newDist] = second.state.getEstimate();
                          Vector<UDT> const newMean = newDist.mean();
                          return static_cast<double>(not sat::is_overlapping(
                              currentMean, _manager->state_comps(modelId), newMean, _manager->state_comps(newModelId)));
                        },
                         0.5 },  /// merge if the tracks are overlapping
              std::pair{ [&](BernoulliState<UDT> const& first, BernoulliState<UDT> const& second) -> UDT {
                          auto [modelId, currentStateDist] = first.state.getEstimate();
                          auto [newModelId, newDist] = second.state.getEstimate();
                          Vector<UDT> const currentMean = currentStateDist.mean();
                          auto const first_inds =
                              _manager->state_comps(modelId).indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y });
                          auto const second_inds =
                              _manager->state_comps(newModelId).indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y });
                          if (not(first_inds.has_value() and second_inds.has_value()))
                          {
                            return 0.0;
                          }
                          Vector<UDT> const diff =
                              currentStateDist.mean()(first_inds.value()) - newDist.mean()(second_inds.value());
                          Matrix<UDT> const cov = currentStateDist.covariance()(first_inds.value(), first_inds.value()) +
                                             newDist.covariance()(second_inds.value(), second_inds.value());
                          UDT const mhd2 = diff.transpose() * cov.llt().solve(diff);
                          return mhd2;
                        },
                         boost::math::quantile(boost::math::chi_squared_distribution(2.0),
                                               0.9) }  // merge if the tracks are close with respect to their covariance
          },
          false);
    }();
    std::vector<BernoulliState<UDT>> fused_tracks;
    for (auto& [first, second] : assignment.assigned)
    {
      std::optional<BernoulliState<UDT>> fused = [&] -> std::optional<BernoulliState<UDT>> {
        if (_manager->params()->ttt_filter.trans.track_to_track_fusion ==
            TRACK_TO_TRACK_FUSION_STRATEGY::CROSS_COVARIANCE_INTERSECTION)
        {
          return tttfusion::cross_covariance_intersection(first, second, _manager);
        }
        else if (_manager->params()->ttt_filter.trans.track_to_track_fusion ==
                 TRACK_TO_TRACK_FUSION_STRATEGY::TAKE_BEST)
        {
          if (first.state.getEstimate().second.covariance().trace() <
              second.state.getEstimate().second.covariance().trace())
          {
            return first;
          }
          return second;
        }
        LOG_FATAL("error");
        assert(false);
        std::unreachable();
      }();

      if (fused.has_value())
      {
        if (_manager->params()->ttt_filter.trans.reuse_source_ids)
        {
          fused.value().state._label = second.state._label;
        }
        if (not fused.value().state.isEmpty())
        {
          fused_tracks.push_back(std::move(fused.value()));
        }
        else
        {
          LOG_DEB("state is empty");
        }
      }
    }
    for (BernoulliState<UDT>& old_non_assigned : assignment.non_assigned_first)
    {
      fused_tracks.push_back(std::move(old_non_assigned));
    }
    for (BernoulliState<UDT>& new_non_assigned : assignment.non_assigned_second)
    {
      std::map<StateModelId, std::vector<std::pair<UDT, GaussianDistribution<UDT>>>> state_dist;
      for (auto const& [model, dist] : new_non_assigned.state._state_dist)
      {
        for (auto const& [weight, gauss] : dist.second.to_gaussian_components())
        {
          auto trans =
              transformation::transform(gauss.mean(),
                                        gauss.covariance(),
                                        _manager->state_comps(model),
                                        _manager->state_comps(_manager->params()->state.estimation.output_state_model));
          if (trans.has_value())
          {
            state_dist[_manager->params()->state.estimation.output_state_model].emplace_back(
                weight * dist.first, GaussianDistribution(std::move(trans.value().mean), std::move(trans.value().cov)));
          }
        }
      }
      if (state_dist.empty())
      {
        LOG_WARN(
            std::format("Can not transform Track {} to desired state model {} with components {}",
                        new_non_assigned.state.toString(),
                        _manager->params()->state.estimation.output_state_model.value_,
                        _manager->state_comps(_manager->params()->state.estimation.output_state_model).toString()));
        continue;
      }
      State state(_manager, std::move(state_dist));
      if (_manager->params()->ttt_filter.trans.reuse_source_ids)
      {
        state._label = new_non_assigned.state._label;
      }
      state._time = new_non_assigned.state._time;
      state._meta_data._durationSinceLastAssociation = 0s;
      state._classification = new_non_assigned.state._classification;
      if (not state.isEmpty())
      {
        fused_tracks.emplace_back(std::move(state), new_non_assigned.existence_probability);
      }
    }
    _tracks = std::move(fused_tracks);
    assert([&] {  // NOLINT
      std::vector<Label> labels;
      for (BernoulliState<UDT> const& state : _tracks)
      {
        if (std::ranges::find(labels, state.state._label) != labels.end())
        {
          LOG_FATAL("same label");
          LOG_FATAL(state.state.toString());
          return false;
        }
        labels.push_back(state.state._label);
        if (not state.state.isValid())
        {
          LOG_FATAL("invalid state");
          LOG_FATAL(state.state.toString());
          return false;
        }
      }
      return true;
    }());
    if (_manager->params()->show_gui)
    {
      _manager->vizu()
          ._meas_model_data.wlock()
          .data[SourceId{ stateContainer._source_id.value_ }]
          ._computation_time.emplace_back(_time, std::chrono::high_resolution_clock::now() - cycle_start);
    }
  }
}

}  // namespace ttb
