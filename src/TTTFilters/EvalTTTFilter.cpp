#include "tracking_lib/TTTFilters/EvalTTTFilter.h"

#include "tracking_lib/Transformations/Transformation.h"
#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/TTTFilters/TTTHelpers.h"
#include "tracking_lib/Misc/Profiler.h"
#include <unsupported/Eigen/MatrixFunctions>

namespace ttb
{

std::string to_string(EvalData const& data)
{
  return {};
}
std::string to_stringStatistics(std::vector<EvalData> const& data)
{
  return {};
}

EvalTTTFilter::EvalTTTFilter(TTBManager* manager) : _manager{ manager }
{
}

TTBManager* EvalTTTFilter::manager() const
{
  return _manager;
}

Time EvalTTTFilter::time() const
{
  return _time;
}

TTT_FILTER_TYPE EvalTTTFilter::type() const
{
  return TTT_FILTER_TYPE::EVAL;
}

std::vector<State> EvalTTTFilter::getEstimate() const
{
  return {};
}

void EvalTTTFilter::reset()
{
  _time = Time{ 0s };
  _reference_tracks.clear();
  _eval.clear();
}

void EvalTTTFilter::cycle(Time time, std::vector<StateContainer> trackContainers)
{
  static std::map<std::pair<SourceId, SourceId>, profiling::GeneralDataProfiler<EvalData>> profilers;
  LOG_DEB("NoTTTFilter::cycle");
  _time = time;
  for (StateContainer& trackContainer : trackContainers)
  {
    _time = trackContainer._time;
    // update reference
    if (trackContainer._id.value_.find(_manager->params().ttt_filter.eval.reference) != std::string::npos)
    {
      _reference_tracks[trackContainer._id] = std::move(trackContainer);
    }
    // evaluate against reference
    else
    {
      for (auto& [model_id, reference_container] : _reference_tracks)
      {
        // predict reference
        std::vector<State> predicted;
        for (State& state : reference_container._data)
        {
          if (state._meta_data._durationSinceLastAssociation <
              std::chrono::milliseconds(_manager->params().ttt_filter.eval.max_reference_prediction_duration_ms))
          {
            state.predict(reference_container._time - state._time, reference_container._egoMotion);
            if (not state.isEmpty())
            {
              predicted.push_back(std::move(state));
            }
          }
        }
        reference_container._data = std::move(predicted);
        tttfusion::AssignmentSol assignment = tttfusion::compute_assignment(
            trackContainer._data,
            reference_container._data,
            std::vector<std::pair<tttfusion::Metric, double>>{
                std::pair{ [&](State const& first, State const& second) {
                            auto [modelId, currentStateDist] = first.getEstimate();
                            Vector2 const currentMean =
                                currentStateDist->mean()(_manager->getStateModel(modelId)
                                                             .state_comps()
                                                             .indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y })
                                                             .value());

                            auto [newModelId, newDist] = second.getEstimate();
                            Vector2 const newMean = newDist->mean()(_manager->getStateModel(newModelId)
                                                                        .state_comps()
                                                                        .indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y })
                                                                        .value());
                            return (currentMean - newMean).norm();
                          },
                           _manager->params().ttt_filter.eval.max_distance } });
        EvalData data{
          .time = trackContainer._time,
          .distances =
              [&] {
                std::vector<std::tuple<Vector2, Vector2, double>> distances;
                for (auto const& [first, second] : assignment.assigned)
                {
                  auto const& [first_id, first_dist] = first.getEstimate();
                  auto const first_xy_ind =
                      _manager->getStateModel(first_id).state_comps().indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y });
                  if (not first_xy_ind.has_value())
                  {
                    continue;
                  }
                  Vector2 first_mean = first_dist->mean()(first_xy_ind.value());
                  auto const& [second_id, second_dist] = second.getEstimate();
                  auto const second_xy_ind =
                      _manager->getStateModel(second_id).state_comps().indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y });
                  if (not second_xy_ind.has_value())
                  {
                    continue;
                  }
                  Vector2 second_mean = second_dist->mean()(second_xy_ind.value());
                  double vabs = std::abs(transformation::transform(first_dist->mean(),
                                                                   _manager->getStateModel(first_id).state_comps(),
                                                                   Components({ COMPONENT::VEL_ABS }))
                                             .value_or(Vector::Zero(1))(0) -
                                         transformation::transform(second_dist->mean(),
                                                                   _manager->getStateModel(second_id).state_comps(),
                                                                   Components({ COMPONENT::VEL_ABS }))
                                             .value_or(Vector::Zero(1))(0));
                  distances.emplace_back(first_mean - second_mean,
                                         first_dist->covariance()(first_xy_ind.value(), first_xy_ind.value())
                                             .sqrt()
                                             .llt()
                                             .solve(first_mean - second_mean),
                                         vabs);
                }
                return distances;
              }(),
          .unassigned_reference_tracks = assignment.non_assigned_second.size(),
          .unassigned_eval_tracks = assignment.non_assigned_first.size()
        };
        _eval[{ trackContainer._id, reference_container._id }].push_back(data);
        if (_manager->params().show_gui)
        {
          std::lock_guard lock(_manager->vizu().add_data_mutex);
          _manager->vizu()
              ._meas_model_data[MeasModelId{ trackContainer._id.value_ }]
              ._distance_to_reference[MeasModelId{ reference_container._id.value_ }]
              .emplace_back(trackContainer._time, [&] -> std::tuple<Vector2, Vector2, double> {
                Vector2 tot_dist = Vector2::Zero();
                Vector2 tot_normed_dist = Vector2::Zero();
                double tot_vabs = 0;
                for (auto const& [dist, normed_dist, vabs] : data.distances)
                {
                  tot_dist += dist;
                  tot_normed_dist += normed_dist;
                  tot_vabs += vabs;
                }
                return { tot_dist, tot_normed_dist, tot_vabs };
              }());
        }

        if (not profilers.contains({ trackContainer._id, reference_container._id }))
        {
          profilers.emplace(std::pair{ trackContainer._id, reference_container._id },
                            "/tmp/" + trackContainer._id.value_ + "__" + reference_container._id.value_);
        }
        profilers.at({ trackContainer._id, reference_container._id }).addData(std::move(data));
      }
    }
  }
}

}  // namespace ttb