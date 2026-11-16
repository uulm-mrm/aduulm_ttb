#pragma once

#include "tracking_lib/TTBTypes/Concepts.h"
#include "tracking_lib/TTTFilters/TransTTTFilter.h"

#include "tracking_lib/BirthModels/BaseBirthModel.h"
#include <tracy/tracy/Tracy.hpp>
namespace ttb
{

template <class UDT, class MODist, class Manager>
  requires CPTCommonMODist<MODist, UDT> && CPTMSUpdate<MODist, UDT>
class MSTracker
{
public:
  explicit MSTracker(Manager* manager) : _manager{ manager }, _mo_dist{ manager }
  {
  }

  [[nodiscard]] std::vector<StateId> all_tracks() const
  {
    return _mo_dist.all_tracks();
  }

  void add_birth_tracks(std::vector<DataSourceContainer<UDT>> const& data_source_containers, bool static_birth)
  {
    // ZoneScopedN("MSTracker::add_birth_tracks");
    std::vector<DataSourceContainer<UDT>> meas_birth_containers;
    std::vector<DataSourceContainer<UDT>> state_birth_containers;
    for (DataSourceContainer<UDT> const& data_source_container : data_source_containers)
    {
      BirthTracks birth_tracks = [&] {
        if (static_birth)
        {
          return _manager->getBirthModel().getBirthTracks(data_source_container,
                                                          {},
                                                          _mo_dist.get_estimate() |
                                                              member<&BernoulliState<UDT>::state> |
                                                              std::ranges::to<std::vector<State<UDT>>>());
        }
        return _manager->getBirthModel().getBirthTracks(data_source_container,
                                                        _mo_dist.meas_assignment_prob(data_source_container),
                                                        _mo_dist.get_estimate() | member<&BernoulliState<UDT>::state> |
                                                            std::ranges::to<std::vector<State<UDT>>>());
      }();
      state_birth_containers.push_back({ ._source_id = data_source_container._source_id,
                                         ._tracks = birth_tracks.birth_tracks,
                                         ._time = data_source_container._time,
                                         ._sensorInfo = data_source_container._sensorInfo,
                                         ._predictionInfo = data_source_container._predictionInfo });
      meas_birth_containers.push_back({ ._source_id = data_source_container._source_id,
                                        ._measurements =
                                            [&] {
                                              std::vector<State<UDT>> meas;
                                              for (std::optional<State<UDT>>& m : birth_tracks.birth_measurements)
                                              {
                                                if (m.has_value())
                                                {
                                                  meas.push_back(std::move(m.value()));
                                                }
                                              }
                                              return meas;
                                            }(),
                                        ._time = data_source_container._time,
                                        ._sensorInfo = data_source_container._sensorInfo,
                                        ._predictionInfo = data_source_container._predictionInfo });
    }
    TransTTTFilter ttt_filter(_manager);
    ttt_filter.cycle(std::move(state_birth_containers));
    std::vector<BernoulliState<UDT>> birth_tracks = ttt_filter.get_estimate();
    for (BernoulliState<UDT>& state : birth_tracks)
    {
      state.state._type = STATE_TYPE::BIRTH;
    }
    _mo_dist.add_tracks(std::move(birth_tracks));
  }

  void cycle(std::vector<DataSourceContainer<UDT>> data_source_containers)
  {
    // ZoneScopedN("MSTracker::cycle");
    assert([&] {  // NOLINT
      if (not _mo_dist.is_valid())
      {
        LOG_FATAL("Invalid MO Distribution before cycle");
        return false;
      }
      return true;
    }());
    if (data_source_containers.empty())
    {
      return;
    }
    assert(data_source_containers.back()._time >= data_source_containers.front()._time);
    assert([&] {
      if (Duration const diff = data_source_containers.back()._time - data_source_containers.front()._time;
          diff > std::chrono::milliseconds(_manager->params()->buffer.max_delta_ms))
      {
        LOG_FATAL(std::format("Time diff {}ms bigger than allowed diff {}",
                              to_milliseconds(diff),
                              _manager->params()->buffer.max_delta_ms));
        for (DataSourceContainer<UDT> const& data_container : data_source_containers)
        {
          LOG_FATAL(std::format("Time: {}", data_container._time));
        }
        return false;
      }
      return true;
    }());
    Time const mean_container_time = data_source_containers.front()._time +
                                     (data_source_containers.back()._time - data_source_containers.front()._time) / 2;

    Duration const deltaT = mean_container_time - _time;
    if (deltaT < 0ms)
    {
      LOG_ERR(std::format("Received Data Container from the past. Ignoring this data. Filter Time: {}. Data Time: "
                          "{}. "
                          "Blame the data buffer. Should not happen",
                          _time,
                          data_source_containers.front()._time));
      return;
    }
    _mo_dist.predict(deltaT, data_source_containers.front()._predictionInfo);
    if (_manager->getBirthModel().type() == BIRTH_MODEL_TYPE::STATIC)
    {
      add_birth_tracks(data_source_containers, true);
    }
    _mo_dist.update(data_source_containers);
    if (_manager->params()->show_gui)
    {
      // ZoneScopedN("Tracker::update::gui");
      for (DataSourceContainer<UDT> const& measurementContainer : data_source_containers)
      {
        UDT assoc = 0;
        for (auto const& [meas_id, assoc_prob] : _mo_dist.meas_assignment_prob(measurementContainer))
        {
          assoc += assoc_prob;
        }
        {
          _manager->vizu()
              ._meas_model_data.wlock()
              .data[measurementContainer._source_id]
              ._num_assoc_measurements.emplace_back(measurementContainer._time, to_val(assoc));
        }
        if constexpr (requires {
                        { _mo_dist.cardinality_distribuion() } -> std::convertible_to<Vector<UDT>>;
                      })
        {
          _manager->vizu()
              ._meas_model_data.wlock()
              .data[measurementContainer._source_id]
              ._cardinality_distribution.emplace_back(measurementContainer._time,
                                                      to_val(_mo_dist.cardinality_distribution()));
        }
        _manager->vizu()
            ._meas_model_data.wlock()
            .data[measurementContainer._source_id]
            ._states_in_sensor_cs.emplace_back(measurementContainer._time, [&] {
              std::vector<State<UDT>> state_sc;
              for (State<UDT> const& state : _mo_dist.get_estimate() | member<&BernoulliState<UDT>::state>)
              {
                SE3Trafo to_sensor =
                    default_tracking_cs_to_default_sensor_cs(measurementContainer._source_id).inverse() *
                    measurementContainer._sensorInfo._to_sensor_cs;
                if (State trafo = state.transform(to_sensor); not trafo.isEmpty())
                {
                  state_sc.push_back(std::move(trafo));
                }
              }
              return state_sc;
            }());
      }
    }
    if constexpr (CPTSelfAssessableMODist<MODist, UDT>)
    {
      // ZoneScopedN("MSTracker::update_self_assessment");
      if (_manager->params()->self_assessment.enable)
      {
        for (DataSourceContainer<UDT> const& data_source_container : data_source_containers)
        {
          if (_manager->params()->self_assessment.detection_prob_estimation.enable)
          {
            _manager->assessment()._model_assessment._detection_model._estimation_method.update_detection_estimate(
                data_source_container, _mo_dist.detection_distribution(data_source_container));
          }
          if (_manager->params()->self_assessment.bayes_factor.detection.enable)
          {
            _manager->assessment()._model_assessment._detection_model._bf_method.update_detection_bf(
                data_source_container,
                _mo_dist.bayes_factor_statistics_measurement_likelihood_without_detection(data_source_container),
                _mo_dist.bayes_factor_statistics_measurement_likelihood(data_source_container));
          }
          if (_manager->params()->self_assessment.clutter_rate_estimation.enable)
          {
            _manager->assessment()._model_assessment._clutter_model._estimation_method.update_clutter_estimate(
                data_source_container, _mo_dist.clutter_distribution(data_source_container));
          }
          if (_manager->params()->self_assessment.bayes_factor.clutter.enable)
          {
            _manager->assessment()._model_assessment._clutter_model._bf_method.update_clutter_bf(
                data_source_container,
                _mo_dist.bayes_factor_statistics_measurement_likelihood_without_clutter(data_source_container),
                _mo_dist.bayes_factor_statistics_measurement_likelihood(data_source_container));
          }
        }
        std::vector<State<UDT>> const estimated_tracks =
            _mo_dist.get_estimate() | member<&BernoulliState<UDT>::state> | std::ranges::to<std::vector<State<UDT>>>();
        for (DataSourceContainer<UDT> const& data_source_container : data_source_containers)
        {
          if (_manager->params()->self_assessment.enable)
          {
            _manager->assessment()._model_assessment.update_assessment_opinions(data_source_container,
                                                                                estimated_tracks);
          }
        }
        _manager->assessment().compute_track_assessment(
            _mo_dist.all_tracks(),
            std::chrono::milliseconds(_manager->params()->self_assessment.track_assessment.window_length_ms));
      }
    }
    if (_manager->getBirthModel().type() == BIRTH_MODEL_TYPE::DYNAMIC)
    {
      add_birth_tracks(data_source_containers, false);
    }
    _time = mean_container_time;
    if (_manager->params()->self_assessment.enable)
    {
      // ZoneScopedN("MSTracker::update_self_assessment");
      _manager->assessment()._unmodeled_aspects.evaluate_overlapping(
          _mo_dist.get_estimate() | member<&BernoulliState<UDT>::state> | std::ranges::to<std::vector<State<UDT>>>());
      if (_manager->params()->self_assessment.persistence_prob_estimation.enable)
      {
        _manager->assessment()._model_assessment._persistence_model_assessment.update(
            _mo_dist.surviving_distribution());
      }
      _manager->assessment().compute_implementation_opinion();
    }
    assert([&] {  // NOLINT
      if (not _mo_dist.is_valid())
      {
        LOG_FATAL("MO distribution is invalid after update with");
        return false;
      }
      return true;
    }());
  }

  [[nodiscard]] std::vector<BernoulliState<UDT>> get_estimate() const
  {
    return _mo_dist.get_estimate();
  }

  void reset()
  {
    _mo_dist = MODist{ _manager };
    _time = Time{ 0s };
  }

  [[nodiscard]] Time time() const
  {
    return _time;
  }

  Manager* _manager;
  Time _time{ 0s };
  MODist _mo_dist;
};

}  // namespace ttb
