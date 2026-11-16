#pragma once

#include "tracking_lib/MeasurementModels/GaussianMeasurementModel.h"
#include "tracking_lib/TTBTypes/Concepts.h"

#include <tracking_lib/BirthModels/BaseBirthModel.h>
#include <tracy/tracy/Tracy.hpp>

namespace ttb
{

template <class UDT, class MODist, class Manager>
// requires CPTCommonMODist<MODist, UDT> && CPTICUpdate<MODist, UDT>
class ICTracker
{
public:
  explicit ICTracker(Manager* manager) : _manager{ manager }, _mo_dist{ manager }
  {
  }

  [[nodiscard]] std::vector<StateId> all_tracks() const
  {
    return _mo_dist.all_tracks();
  }

  void cycle(std::vector<DataSourceContainer<UDT>> const& data_containers)
  {
    if (not _tape.has_value())
    {
      _tape.emplace();
    }
    // ZoneScopedN("ICTracker::cycle");
    assert([&] {  // NOLINT
      if (not _mo_dist.is_valid())
      {
        LOG_FATAL("Invalid MO Distribution before cycle");
        return false;
      }
      return true;
    }());
    LOG_DEB("IC CYCLE with " << data_containers.size() << " Measurement Container");
    Time const filter_time_start_cycle = _time;
    auto const thread_id = std::this_thread::get_id();
    for (DataSourceContainer<UDT> const& measurementContainer : data_containers)
    {
      // ZoneScopedN("ICTracker::ic_cycle");
      auto const system_time_start_ic_cycle = std::chrono::high_resolution_clock::now();

      if constexpr (std::is_same_v<UDT, AD>)
      {
        _tape.value().newRecording();
        if (_manager->params()->meas_model_params(measurementContainer._source_id).detection.learn_model)
        {
          _tape.value().registerInputs(std::get<GaussianMeasurementModel<UDT>>(
                                           _manager->getMeasModel(measurementContainer._source_id)._meas_model)
                                           ._detection_model_params.begin(),
                                       std::get<GaussianMeasurementModel<UDT>>(
                                           _manager->getMeasModel(measurementContainer._source_id)._meas_model)
                                           ._detection_model_params.end());
        }
      }

      Duration const deltaT = measurementContainer._time - _time;
      if (deltaT < 0ms)
      {
        LOG_ERR(std::format("Received Data Container from the past. Ignoring this data. Filter Time: {}. Data Time: "
                            "{}. "
                            "Blame the data buffer. Should not happen",
                            _time,
                            measurementContainer._time));
        continue;
      }
      LOG_DEB("Processing Msg. of sensor " + measurementContainer._source_id.value_
              << " with " << measurementContainer._measurements.size() << " Detections");
      _mo_dist.predict(deltaT, measurementContainer._predictionInfo);
      _time = measurementContainer._time;

      if (_manager->getBirthModel().type() == BIRTH_MODEL_TYPE::STATIC ||
          _manager->getBirthModel().type() == BIRTH_MODEL_TYPE::ROI)
      {
        _mo_dist.add_tracks(_manager->getBirthModel()
                                .getBirthTracks(measurementContainer,
                                                {},
                                                _mo_dist.get_estimate() | member<&BernoulliState<UDT>::state> |
                                                    std::ranges::to<std::vector<State<UDT>>>())
                                .birth_tracks);
      }
      _mo_dist.update(measurementContainer);
      if constexpr (std::is_same_v<UDT, AD> and std::is_same_v<MODist, LMBDistribution<AD>>)
      {
        if (_manager->params()->meas_model_params(measurementContainer._source_id).detection.learn_model)
        {
          assert(thread_id == std::this_thread::get_id());
          UDT loss = -_mo_dist.bayes_factor_statistics_measurement_likelihood(measurementContainer);
          _tape.value().registerOutput(loss);
          derivative(loss) = 1.0;
          _tape.value().computeAdjoints();
          Vector<UDT>& params = std::get<GaussianMeasurementModel<UDT>>(
                                    _manager->getMeasModel(measurementContainer._source_id)._meas_model)
                                    ._detection_model_params;
          Vector<double> const grad = [&] {
            Vector<double> g = Vector<double>::Zero(params.rows());
            for (Index i = 0; i < params.rows(); ++i)
            {
              g(i) = params(i).derivative();
            }
            return g;
          }();
          double const lr = 5e-3;
          if (to_val(grad).array().isFinite().all())
          {
            for (Index i = 0; i < params.rows(); ++i)
            {
              params(i) -= lr * grad(i);
            }
          }
        }
      }
      if (_manager->params()->show_gui)
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
                if (State<UDT> trafo = state.transform(to_sensor); not trafo.isEmpty())
                {
                  state_sc.push_back(std::move(trafo));
                }
              }
              return state_sc;
            }());
      }
      if constexpr (CPTSelfAssessableMODist<MODist, UDT>)
      {
        // ZoneScopedN("ICTracker::self_assessment");
        if (_manager->params()->self_assessment.enable)
        {
          if (_manager->params()->self_assessment.detection_prob_estimation.enable)
          {
            _manager->assessment()._model_assessment._detection_model._estimation_method.update_detection_estimate(
                measurementContainer, _mo_dist.detection_distribution(measurementContainer));
          }
          if (_manager->params()->self_assessment.bayes_factor.detection.enable)
          {
            _manager->assessment()._model_assessment._detection_model._bf_method.update_detection_bf(
                measurementContainer,
                _mo_dist.bayes_factor_statistics_measurement_likelihood_without_detection(measurementContainer),
                _mo_dist.bayes_factor_statistics_measurement_likelihood(measurementContainer));
          }
          if (_manager->params()->self_assessment.clutter_rate_estimation.enable)
          {
            _manager->assessment()._model_assessment._clutter_model._estimation_method.update_clutter_estimate(
                measurementContainer, _mo_dist.clutter_distribution(measurementContainer));
          }
          if (_manager->params()->self_assessment.bayes_factor.clutter.enable)
          {
            _manager->assessment()._model_assessment._clutter_model._bf_method.update_clutter_bf(
                measurementContainer,
                _mo_dist.bayes_factor_statistics_measurement_likelihood_without_clutter(measurementContainer),
                _mo_dist.bayes_factor_statistics_measurement_likelihood(measurementContainer));
          }
          _manager->assessment()._model_assessment.update_assessment_opinions(
              measurementContainer,
              _mo_dist.get_estimate() | member<&BernoulliState<UDT>::state> |
                  std::ranges::to<std::vector<State<UDT>>>());
          _manager->assessment().compute_track_assessment(
              _mo_dist.all_tracks(),
              std::chrono::milliseconds(_manager->params()->self_assessment.track_assessment.window_length_ms));
        }
      }

      if (_manager->getBirthModel().type() == BIRTH_MODEL_TYPE::DYNAMIC)
      {
        LOG_DEB("Add Dynamic Birth Tracks");
        _mo_dist.add_tracks(_manager->getBirthModel()
                                .getBirthTracks(measurementContainer,
                                                _mo_dist.meas_assignment_prob(measurementContainer),
                                                _mo_dist.get_estimate() | member<&BernoulliState<UDT>::state> |
                                                    std::ranges::to<std::vector<State<UDT>>>())
                                .birth_tracks);
        LOG_DEB("Add Dynamic Birth Tracks");
      }
      _time = measurementContainer._time;
      if (_manager->params()->show_gui)
      {
        LOG_DEB("Show GUI");
        _manager->vizu()._meas_model_data.wlock().data[measurementContainer._source_id]._computation_time.emplace_back(
            filter_time_start_cycle, std::chrono::high_resolution_clock::now() - system_time_start_ic_cycle);
        std::ranges::sort(
            _manager->vizu()._meas_model_data.wlock().data[measurementContainer._source_id]._computation_time);
        for (StateId track_id : _mo_dist.all_tracks())
        {
          auto const track = _manager->try_get_state(track_id);
          if (track.has_value())
          {
            _manager->vizu()
                ._meas_model_data.wlock()
                .data[measurementContainer._source_id]
                ._tracks[track.value().get()._label]
                .push_back(track.value().get());
          }
        }
        LOG_DEB("Show GUI");
      }
      LOG_DEB("IC cycle done");
      assert([&] {  // NOLINT
        if (not _mo_dist.is_valid())
        {
          LOG_FATAL("MO distribution is invalid after update with " << measurementContainer._id);
          return false;
        }
        return true;
      }());
    }
    if (_manager->params()->self_assessment.enable)
    {
      LOG_DEB("Self Assessment");
      _manager->assessment().compute_implementation_opinion();
      if constexpr (CPTSelfAssessableMODist<MODist, UDT>)
      {
        if (_manager->params()->self_assessment.persistence_prob_estimation.enable)
        {
          _manager->assessment()._model_assessment._persistence_model_assessment.update(
              _mo_dist.surviving_distribution());
        }
      }
      _manager->assessment()._unmodeled_aspects.evaluate_overlapping(
          _mo_dist.get_estimate() | member<&BernoulliState<UDT>::state> | std::ranges::to<std::vector<State<UDT>>>());
      LOG_DEB("Self Assessment");
    }
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
  /// Automatic differentiation tape
  std::optional<xad::Tape<double>> _tape;
};

}  // namespace ttb
