#include "tracking_lib/TTBManager/TTBManager.h"
// ######################################################################################################################
#include "tracking_lib/PersistenceModels/ConstantPersistenceModel.h"
#include "tracking_lib/PersistenceModels/ROIPersistenceModel.h"
#include "tracking_lib/StateTransition/MarkovTransition.h"
#include "tracking_lib/BirthModels/BaseBirthModel.h"
#include "tracking_lib/StateModels/BaseStateModel.h"
#include "tracking_lib/MeasurementModels/BaseMeasurementModel.h"
#include "tracking_lib/MeasurementModels/GaussianMeasurementModel.h"
#include "tracking_lib/BirthModels/DynamicBirthModel.h"
#include "tracking_lib/BirthModels/StaticBirthModel.h"
#include <tracking_lib/BirthModels/ROIBirthModel.h>
#include "tracking_lib/StateModels/CPStateModel.h"
#include "tracking_lib/StateModels/CTP.h"
#include "tracking_lib/StateModels/CVStateModel.h"
#include "tracking_lib/StateModels/CAStateModel.h"
#include "tracking_lib/StateModels/ISCATRStateModel.h"
#include "tracking_lib/StateModels/CTRAStateModel.h"
#include "tracking_lib/StateModels/CTRVStateModel.h"
#include "tracking_lib/Misc/AngleNormalization.h"
#include "tracking_lib/Misc/Ospa.h"
#include "tracking_lib/TTBTypes/Components.h"
#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/Transformations/Transformation.h"

#include <aduulm_logger/aduulm_logger.hpp>
#include <figcone/configreader.h>
#include <stacktrace>
#include <csignal>
#include <tracy/tracy/TracyC.h>

namespace ttb
{
std::string to_string(GroundTruthProfiling const& gt_data)
{
  return std::format("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n",
                     to_seconds(gt_data.time.time_since_epoch()),
                     gt_data.gospa.false_cost,
                     gt_data.gospa.missed_cost,
                     gt_data.gospa.loc_cost,
                     gt_data.gospa.gospa,
                     gt_data.ospa.ospa,
                     gt_data.ospa2.ospa2,
                     gt_data.gospa.card_gt,
                     gt_data.gospa.card_est,
                     gt_data.cycle_time.count());
}

template class TTBManager<double>;
template class TTBManager<AD>;

template <class UDT>
TTBManager<UDT>::TTBManager(std::filesystem::path const& config_file)
  : TTBManager<UDT>([&] -> Params {
    figcone::ConfigReader reader{};
    try
    {
      LOG_INF(std::format("Config file {}", config_file.c_str()));
      auto params = reader.readYamlFile<Params>(config_file);  // NOLINT
      if (param_version != params.version)
      {
        LOG_DEB("You are using Param version: " << params.version << ". The tracking_lib version is " << param_version
                                                << ". The versions are compatible.");
      }
      return params;
    }
    catch (std::exception const& excp)
    {
      LOG_FATAL(std::format("Can not parse the parameter file: {}", config_file.generic_string()));
      LOG_FATAL("Error: " << excp.what());
      std::ifstream file(config_file);
      std::stringstream sstream;
      sstream << file.rdbuf();
      LOG_FATAL("Given parameters:\n" << sstream.str());
      throw;
    }
  }())
{
}

template <class UDT>
TTBManager<UDT>::TTBManager(Params input_params)
  : _params{ input_params }
  , _original_params{ std::move(input_params) }
  , _tracker{ this }
  , _measModels{ [&] {
    std::map<SourceId, SingleObjectMeasurementModel<UDT>> meas_models;
    auto const meas_params = params();
    for (auto const& gaussianModel : meas_params->meas_models.gaussian_models)
    {
      SourceId const id = gaussianModel.id;
      if (meas_models.contains(id))
      {
        LOG_FATAL("MeasurementModels with the same ID detected");
        throw std::runtime_error("MeasurementModels with the same ID detected - FIX config");
      }
      SingleObjectMeasurementModel<UDT> so_model{ *this, GaussianMeasurementModel<UDT>{ this, id } };
      meas_models.emplace(id, so_model);
    }
    if (meas_models.empty() and params()->filter.enable)
    {
      LOG_WARN("No Measurement Model is created!");
    }
    return meas_models;
  }() }
  , _state_comps{ [&] {
    StateIdCompsContainer state_comps;
    for (StateModelParams const& stateModelParams : params()->state_models)
    {
      state_comps.emplace(stateModelParams.id, [&] {
        switch (stateModelParams.extent)
        {
          case EXTENT::NONE:
            return state_model_type_2_state_comps_map.at(stateModelParams.type);
          case EXTENT::RECTANGULAR:
            return state_model_type_2_state_comps_map.at(stateModelParams.type).merge(Components(EXTENT_RECTANGULAR));
          case EXTENT::CIRCULAR:
            return state_model_type_2_state_comps_map.at(stateModelParams.type).merge(Components(EXTENT_CIRCULAR));
          default:
            assert(false);
            std::unreachable();
        }
      }());
    }
    return state_comps;
  }() }
  , _noise_comps{ [&] {
    StateIdCompsContainer noise_comps;
    for (StateModelParams const& stateModelParams : params()->state_models)
    {
      noise_comps.emplace(stateModelParams.id, [&] {
        switch (stateModelParams.extent)
        {
          case EXTENT::NONE:
            return state_model_type_2_noise_comps_map.at(stateModelParams.type);
          case EXTENT::RECTANGULAR:
            return state_model_type_2_noise_comps_map.at(stateModelParams.type)
                .merge(Components(EXTENT_RECTANGULAR_NOISE));
          case EXTENT::CIRCULAR:
            return state_model_type_2_noise_comps_map.at(stateModelParams.type)
                .merge(Components(EXTENT_CIRCULAR_NOISE));
          default:
            assert(false);
            std::unreachable();
        }
      }());
    }
    return noise_comps;
  }() }
  , _stateModels{ [&] {
    std::map<StateModelId, StateModel<UDT>> models;
    for (StateModelParams const& stateModelParams : params()->state_models)
    {
      StateModel<UDT> stateModel{ [&]() -> StateModel<UDT> {
        switch (stateModelParams.type)
        {
          case STATE_MODEL_TYPE::CP:
            return { *this, stateModelParams.id, CPStateModel<UDT>{ *this, stateModelParams.id } };
          case STATE_MODEL_TYPE::CTP:
            return { *this, stateModelParams.id, CTP<UDT>(*this, stateModelParams.id) };
          case STATE_MODEL_TYPE::CV:
            return { *this, stateModelParams.id, CVStateModel<UDT>(*this, stateModelParams.id) };
          case STATE_MODEL_TYPE::CA:
            return { *this, stateModelParams.id, CAStateModel<UDT>(*this, stateModelParams.id) };
          case STATE_MODEL_TYPE::ISCATR:
            return { *this, stateModelParams.id, ISCATRStateModel<UDT>(*this, stateModelParams.id) };
          case STATE_MODEL_TYPE::CTRV:
            return { *this, stateModelParams.id, CTRVStateModel<UDT>(*this, stateModelParams.id) };
          case STATE_MODEL_TYPE::CTRA:
            return { *this, stateModelParams.id, CTRAStateModel<UDT>(*this, stateModelParams.id) };
        }
        assert(false);
        std::unreachable();
      }() };
      models.emplace(stateModelParams.id, std::move(stateModel));
    }
    if (models.empty())
    {
      LOG_WARN("No state model was created!");
    }
    return models;
  }() }
  , _markovTransition{ [&] { return MarkovTransition<UDT>(*this); }() }
  , _persistence_model{ [&] -> PersistenceModel<UDT> {
    switch (params()->persistence_model.type)
    {
      case PERSISTENCE_MODEL_TYPE::CONSTANT:
        return { ConstantPersistenceModel<UDT>(this) };
      case PERSISTENCE_MODEL_TYPE::ROI:
        return { ROIPersistenceModel<UDT>(this) };
    }
    assert(false);
    std::unreachable();
  }() }
  , _birth_model{ [&] -> BirthModel<UDT> {
    if (not params()->birth_model.has_value())
    {
      return { StaticBirthModel<UDT>(this) };
    }
    switch (params()->birth_model.value().type)
    {
      case BIRTH_MODEL_TYPE::DYNAMIC:
        return { DynamicBirthModel<UDT>(this) };
      case BIRTH_MODEL_TYPE::STATIC:
        return { StaticBirthModel<UDT>(this) };
      case BIRTH_MODEL_TYPE::ROI:
        return { ROIBirthModel<UDT>(this) };
    }
    assert(false);
    std::unreachable();
  }() }
  , _assessment(this)
  , _dataBuffer{ typename DataBuffer::Params{
        .mode = minimal_latency_buffer::BufferMode::BATCH,
        .reset_threshold = 10s,
        .measurement_confidence_quantile = params()->buffer.max_wait_duration_quantile,
        .max_abs_measurement_jitter = std::chrono::milliseconds(params()->buffer.max_wait_duration_ms),
        .wait_confidence_quantile = params()->buffer.max_wait_duration_quantile,
        .max_abs_wait_jitter = std::chrono::milliseconds(params()->buffer.max_wait_duration_ms),
        .max_total_wait_time = std::chrono::milliseconds(params()->buffer.max_wait_duration_ms),
        .batch = { .max_delta = std::chrono::milliseconds(params()->buffer.max_delta_ms) } } }

  , _viz{ Vizu<UDT>(this) }

{
  assert(hpx::is_running());
  if (params()->show_gui)
  {
#if __has_include("minimal_latency_buffer/mlb_visu.hpp")
    _dataBuffer.setCreateCopy(true);
    _data_buffer_visualizer = minimal_latency_buffer::MLBVisualizer([&] { return _dataBuffer.getShallowData(); });
#endif

    show_gui();
  }

  LOG_DEB("created TTBManager");
  TracySetProgramName(params()->name.c_str());
}

template <class UDT>
Database<UDT> const& TTBManager<UDT>::database() const
{
  return _database;
}

template <class UDT>
Database<UDT>& TTBManager<UDT>::database()
{
  return _database;
}

template <class UDT>
PersistenceModel<UDT> const& TTBManager<UDT>::getPersistenceModel() const
{
  return _persistence_model;
}

template <class UDT>
PersistenceModel<UDT>& TTBManager<UDT>::getPersistenceModel()
{
  return _persistence_model;
}

template <class UDT>
Tracker<UDT> const& TTBManager<UDT>::getFilter() const
{
  return _tracker;
}

template <class UDT>
Tracker<UDT>& TTBManager<UDT>::getFilter()
{
  return _tracker;
}

template <class UDT>
BirthModel<UDT> const& TTBManager<UDT>::getBirthModel() const
{
  return _birth_model;
}

template <class UDT>
BirthModel<UDT>& TTBManager<UDT>::getBirthModel()
{
  return _birth_model;
}

template <class UDT>
MarkovTransition<UDT> const& TTBManager<UDT>::getTransition() const
{
  return _markovTransition;
}

template <class UDT>
MarkovTransition<UDT>& TTBManager<UDT>::getTransition()
{
  return _markovTransition;
}

template <class UDT>
SingleObjectMeasurementModel<UDT> const& TTBManager<UDT>::getMeasModel(SourceId const& id) const
{
  assert(params()->have_measurement_model(id));
  for (auto const& [model_id, meas_model] : _measModels)
  {
    if (model_id == id or (model_id.value_.back() == '*' and model_id.value_.starts_with(id.value_)))
    {
      return _measModels.at(model_id);
    }
  }
  assert(false);
  std::unreachable();
}

template <class UDT>
SingleObjectMeasurementModel<UDT>& TTBManager<UDT>::getMeasModel(SourceId const& id)
{
  assert(params()->have_measurement_model(id));
  for (auto const& [model_id, meas_model] : _measModels)
  {
    if (model_id == id or (model_id.value_.back() == '*' and model_id.value_.starts_with(id.value_)))
    {
      return _measModels.at(model_id);
    }
  }
  assert(false);
  std::unreachable();
}

template <class UDT>
std::map<StateModelId, StateModel<UDT>> const& TTBManager<UDT>::getStateModelMap() const
{
  return _stateModels;
}

template <class UDT>
std::map<StateModelId, StateModel<UDT>>& TTBManager<UDT>::getStateModelMap()
{
  return _stateModels;
}

template <class UDT>
StateModel<UDT> const& TTBManager<UDT>::getStateModel(StateModelId id) const
{
  assert(_stateModels.contains(id));
  return _stateModels.at(id);
}

template <class UDT>
StateModel<UDT>& TTBManager<UDT>::getStateModel(StateModelId id)
{
  assert(_stateModels.contains(id));
  return _stateModels.at(id);
}

template <class UDT>
[[nodiscard]] Components const& TTBManager<UDT>::state_comps(StateModelId id) const
{
  assert([&] {  // NOLINT
    if (not _state_comps.contains(id))
    {
      LOG_FATAL(std::format("state model {} is not known", id));
      LOG_FATAL(std::stacktrace::current());
      return false;
    }
    return true;
  }());
  return _state_comps.find(id)->comps;
}

template <class UDT>
Components const& TTBManager<UDT>::noise_comps(StateModelId id) const
{
  assert([&] {  // NOLINT
    if (not _noise_comps.contains(id))
    {
      LOG_FATAL(std::format("Dont have noise components for state model model {}", id));
      LOG_FATAL(std::stacktrace::current());
      return false;
    }
    return true;
  }());
  return _noise_comps.find(id)->comps;
}

template <class UDT>
StateModelId TTBManager<UDT>::state_model_id(Components const& comps) const
{
  {
    if (auto const it = _state_comps.template get<Components>().find(comps);
        it != _state_comps.template get<Components>().end())
    {
      return it->id;
    }
  }
  StateModelId const new_model_id = [&] {
    if (_state_comps.empty())
    {
      return StateModelId{ 1 };
    }
    return StateModelId{ std::ranges::max(_state_comps, {}, &StateIdComps::id).id.value_ + 1 };
  }();
  _state_comps.emplace(new_model_id, comps);
  return new_model_id;
}

template <class UDT>
StateModel<UDT> const& TTBManager<UDT>::getStateModel(STATE_MODEL_TYPE type, EXTENT extent) const
{
  for (auto const& [id, model] : _stateModels)
  {
    if (model.type() == type and params()->state_model_params(model.id()).extent == extent)
    {
      return model;
    }
  }
  LOG_FATAL("State Model with type: " << to_string(type) << " not found");
  throw std::runtime_error("State Model with type: " + to_string(type) + " not found");
}

template <class UDT>
StateModel<UDT>& TTBManager<UDT>::getStateModel(STATE_MODEL_TYPE type, EXTENT extent)
{
  for (auto const& [id, model] : _stateModels)
  {
    if (model.type() == type and params()->state_model_params(model.id()).extent == extent)
    {
      return _stateModels.at(id);
    }
  }
  LOG_FATAL("State Model with type: " << to_string(type) << " not found");
  throw std::runtime_error("State Model with type: " + to_string(type) + " not found");
}

template <class UDT>
State<UDT> const& TTBManager<UDT>::get_state(StateId id) const
{
  auto const it = database().states.find(id);
  assert([&] {  // NOLINT
    if (it == database().states.end())
    {
      LOG_FATAL(std::format("No State with Id {}", id));
      LOG_FATAL(std::stacktrace::current());
      return false;
    }
    return true;
  }());
  return *it;
}

template <class UDT>
UDT TTBManager<UDT>::get_state_existence_probability(StateId id) const
{
  auto const it = database().state_existences.find(id);
  assert([&] {  // NOLINT
    if (it == database().state_existences.end())
    {
      LOG_FATAL(std::format("No State with Id {}", id));
      LOG_FATAL(std::stacktrace::current());
      return false;
    }
    return true;
  }());
  return it->existence_probability;
}

template <class UDT>
std::optional<std::reference_wrapper<State<UDT> const>> TTBManager<UDT>::try_get_state(StateId id) const
{
  auto const it = database().states.find(id);
  if (it == database().states.end())
  {
    return std::nullopt;
  }
  return { *it };
}

template <class UDT>
TrackingAssessment<UDT>& TTBManager<UDT>::assessment()
{
  return _assessment;
}

template <class UDT>
void TTBManager<UDT>::cycle(Time time)
{
  auto [buffer_time, containers] = get_data_containers(time);
  if (params()->sort_measurement_containers)
  {
    // make sure that containers always have the same order!
    std::sort(
        containers.begin(), containers.end(), [](const DataSourceContainer<UDT>& a, const DataSourceContainer<UDT>& b) {
          return a._source_id.value_ < b._source_id.value_;
        });
  }
  if (params()->filter.enable or params()->ttt_filter.enable)
  {
    cycle(buffer_time, std::move(containers), false);
  }
  else
  {
    LOG_FATAL("No Filter or TTT Filter enabled. False Config");
    throw std::runtime_error("No Filter or TTT Filter enabled. False Config");
  }
  database().cut_data_containers_to_max_size();
}

template <class UDT>
void save_viz_data_containers(
    TTBManager<UDT> const* manager,
    Time time,
    Time buffer_time,
    std::vector<minimal_latency_buffer::TimeData<SourceId, DataSourceContainer<UDT>>> const& data,
    std::vector<minimal_latency_buffer::TimeData<SourceId, DataSourceContainer<UDT>>> const& discarded_data,
    std::vector<DataSourceContainer<UDT>> const& containers,
    std::size_t num_queued_elems)
{
  if (manager->params()->show_gui)
  {
    if (not containers.empty())
    {
      if (Duration const filter_delay = time - containers.front()._time; filter_delay < Duration(1h))
      {
        manager->vizu()._cycle_data.wlock().data._filter_delay_first.emplace_back(time, filter_delay);
        manager->vizu()._cycle_data.wlock().data._filter_delay_last.emplace_back(time, time - containers.back()._time);
      }
    }
    if (time - buffer_time < Duration(1h))
    {
      manager->vizu()._cycle_data.wlock().data._buffer_delay.emplace_back(time, time - buffer_time);
    }
    manager->vizu()._cycle_data.wlock().data._buffer_size.emplace_back(time, num_queued_elems);
    std::ranges::sort(manager->vizu()._cycle_data.wlock().data._buffer_size);
    for (auto const& timed_data : data)
    {
      manager->vizu()._meas_model_data.wlock().data[timed_data.id]._duration_in_buffer.emplace_back(
          time, time - timed_data.receipt_time);
    }
    for (auto const& timed_data : discarded_data)
    {
      manager->vizu()._meas_model_data.wlock().data[timed_data.id]._duration_in_buffer.emplace_back(
          time, time - timed_data.receipt_time);
    }
    for (auto& [model_id, vizu_data] : manager->vizu()._meas_model_data.wlock().data)
    {
      if (std::ranges::find_if(discarded_data, [&](auto const& time_data) { return time_data.id == model_id; }) !=
          discarded_data.end())
      {
        vizu_data._discarded_data.emplace_back(buffer_time);
      }
      if (auto it = std::ranges::find_if(
              containers, [&](DataSourceContainer<UDT> const& container) { return container._source_id == model_id; });
          it != containers.end())
      {
        vizu_data._used_in_cycle.emplace_back(it->_time);
      }
    }
    manager->vizu()._cycle_data.wlock().data._trigger_time.emplace_back(buffer_time);
    if (not containers.empty())
    {
      manager->vizu()._cycle_data.wlock().data._num_sources.emplace_back(time, containers.size());
      manager->vizu()._cycle_data.wlock().data._num_measurements.emplace_back(
          time, std::accumulate(containers.begin(), containers.end(), 0, [](int old, auto const& container) {
            return old + container._tracks.size() + container._measurements.size();
          }));
    }
    for (auto const& container : containers)
    {
      manager->vizu()._meas_model_data.wlock().data[container._source_id]._measurements.emplace_back(container._time,
                                                                                                     container);
    }
  }
}

template <class UDT>
std::pair<Time, std::vector<DataSourceContainer<UDT>>> TTBManager<UDT>::get_data_containers(Time time)
{
  minimal_latency_buffer::PopReturn data = _dataBuffer.pop(time);
  std::vector<DataSourceContainer<UDT>> containers;
  for (auto& buffer_data : data.data)
  {
    Duration const dt = buffer_data.data.value()._time - filter_time();
    if (dt < 0s)
    {
      LOG_FATAL(std::format("Received Measurement from the past ({}ms ago than the curren tracking time). Must "
                            "not happen. Check the data buffer.\nCurrent Filter Time: {}\nRequested Data for time: "
                            "{}\nReceived Data for time: {}",
                            to_milliseconds(dt),
                            filter_time(),
                            time,
                            buffer_data.data.value()._time));
      std::terminate();
    }
    if (dt > 100s and filter_time() > Time{ 0s })
    {
      LOG_WARN(std::format("Detected forward jump of {}s. Full Reset.", to_seconds(dt)));
      reset();
      return {};
    }
    containers.emplace_back(std::move(buffer_data.data.value()));
  }
  save_viz_data_containers(this,
                           time,
                           data.buffer_time,
                           data.data,
                           data.discarded_data,
                           containers,
                           _dataBuffer.getNumberOfQueuedElements());
  if (not containers.empty() and not _max_delays.empty())
  {
    if (Duration const max_delay = std::ranges::max(_max_delays | std::views::values);
        time - containers.back()._time > 2 * max_delay)
    {
      LOG_WARN(std::format("Buffer delay {} is 2 times greater than the largest stream delay {}. This should not "
                           "happen. Reset buffer.",
                           time - containers.back()._time,
                           max_delay));
      _dataBuffer.reset();
    }
    if (Duration const abs_delay = time - containers.back()._time;
        abs_delay > 1.5 * std::chrono::milliseconds(params()->buffer.max_wait_duration_ms))
    {
      LOG_WARN(std::format("Requested data for time: {}. Got data from the buffer for time {}. Delay of: {}ms. Note "
                           "that this is more than 1.5 times bigger than the max_total_wait_time: {}ms of the buffer. "
                           "Check latencies of the tracking input streams."
                           "Should not happen.",
                           time,
                           containers.back()._time,
                           to_milliseconds(abs_delay),
                           params()->buffer.max_wait_duration_ms));
      for (auto const& [source_id, max_latency] : _max_delays)
      {
        LOG_WARN(std::format("Input {} has max_latency of {}ms", source_id, to_milliseconds(max_latency)));
      }
    }
  }
  return { data.buffer_time, std::move(containers) };
}

template <class UDT>
void TTBManager<UDT>::cycle(Time time, std::vector<DataSourceContainer<UDT>> data_containers, bool save_gui_data)
{
  TracyFiberEnter("TTBManager::cycle");
  TracyCZoneN(ctx, "TTBManager::cycle", 1);
  Params next_params = *params();
  for (DataSourceContainer<UDT> const& data_source_container : data_containers)
  {
    if (auto const it = std::ranges::find(params()->oracle_model_parameters.meas_models,
                                          data_source_container._source_id,
                                          &OracleMeasModelParams::id);
        it != params()->oracle_model_parameters.meas_models.end())
    {
      assert(data_source_container._ground_truth_info.has_value());
      if (it->detection_model)
      {
        assert(data_source_container._ground_truth_info.value().meas_model_info.detection_prob <= 1.0 and
               data_source_container._ground_truth_info.value().meas_model_info.detection_prob >= 0 and
               std::isfinite(data_source_container._ground_truth_info.value().meas_model_info.detection_prob));
        next_params.meas_model_params(data_source_container._source_id).detection.prob =
            data_source_container._ground_truth_info.value().meas_model_info.detection_prob;
      }
      if (it->clutter_model)
      {
        assert(data_source_container._ground_truth_info.value().meas_model_info.clutter_rate > 0 and
               std::isfinite(data_source_container._ground_truth_info.value().meas_model_info.clutter_rate));
        next_params.meas_model_params(data_source_container._source_id).clutter.rate.value() =
            data_source_container._ground_truth_info.value().meas_model_info.clutter_rate;
      }
    }
  }
  if (params()->oracle_model_parameters.persistence_prob)
  {
    auto const& gt_container =
        database().data_containers.template get<SourceIdTime>().find(std::tuple{ SourceId{ "GROUND_TRUTH" }, time });
    assert(gt_container != database().data_containers.template get<SourceIdTime>().end());
    assert(gt_container->_ground_truth_info.has_value());
    assert(std::isfinite(gt_container->_ground_truth_info.value().state_model_info.persistence_prob));
    assert(params()->persistence_model.type == PERSISTENCE_MODEL_TYPE::CONSTANT);
    next_params.persistence_model.constant.persistence_prob =
        gt_container->_ground_truth_info.value().state_model_info.persistence_prob;
  }
  if (params()->oracle_model_parameters.birth_model)
  {
    auto const& gt_container =
        database().data_containers.template get<SourceIdTime>().find(std::tuple{ SourceId{ "GROUND_TRUTH" }, time });
    assert(gt_container != database().data_containers.template get<SourceIdTime>().end());
    assert(gt_container->_ground_truth_info.has_value());
    assert(params()->birth_model->type == BIRTH_MODEL_TYPE::STATIC);
    next_params.birth_model->static_model.locations.clear();
    next_params.birth_model->default_birth_existence_prob =
        gt_container->_ground_truth_info.value().birth_model_info.birth_probability;
    next_params.birth_model->allow_overlapping = true;
    for (auto const& possible_state : gt_container->_ground_truth_info->birth_model_info.poss_birth_state)
    {
      StaticBirthLocation loc;
      for (auto const& [comp, pos_var] : possible_state)
      {
        loc.mean[to_string(comp)] = pos_var.first;
        loc.var[to_string(comp)] = pos_var.second;
      }
      next_params.birth_model->static_model.locations.push_back(std::move(loc));
    }
  }
  if (params()->oracle_model_parameters.state_model_noise)
  {
    auto const& gt_container =
        database().data_containers.template get<SourceIdTime>().find(std::tuple{ SourceId{ "GROUND_TRUTH" }, time });
    assert(gt_container != database().data_containers.template get<SourceIdTime>().end());
    assert(gt_container->_ground_truth_info.has_value());
    assert(std::isfinite(gt_container->_ground_truth_info.value().state_model_info.noise_dev));
    assert(params()->state.multi_model.use_state_models.size() == 1);
    assert(getStateModel(params()->state.multi_model.use_state_models.front()).type() == STATE_MODEL_TYPE::CV);
    next_params.state_model_params(params()->state.multi_model.use_state_models.front())
        .model_noise_std_dev.at(to_string(COMPONENT::ACC_X)) =
        gt_container->_ground_truth_info.value().state_model_info.noise_dev;
    next_params.state_model_params(params()->state.multi_model.use_state_models.front())
        .model_noise_std_dev.at(to_string(COMPONENT::ACC_Y)) =
        gt_container->_ground_truth_info.value().state_model_info.noise_dev;
  }
  update_params(std::move(next_params));
  if (save_gui_data)
  {
    save_viz_data_containers(this, time, time, {}, {}, data_containers, 0);
  }
  Time const cycle_start = std::chrono::high_resolution_clock::now();
  Time const filter_time_start = filter_time();
  if (not data_containers.empty())
  {
    std::vector<std::vector<DataSourceContainer<UDT>>> batched_container(1);
    for (DataSourceContainer<UDT> const& container : data_containers)
    {
      if (batched_container.back().empty() or container._time - batched_container.back().back()._time <
                                                  std::chrono::milliseconds(params()->buffer.max_delta_ms))
      {
        batched_container.back().push_back(container);
      }
      else
      {
        batched_container.push_back({});
        batched_container.back().push_back(container);
      }
    }
    for (std::vector<DataSourceContainer<UDT>> const& batch : batched_container)
    {
      _tracker.cycle(batch);
    }
  }

  Time const cycle_end = std::chrono::high_resolution_clock::now();
  std::tuple<std::optional<GospaResult>, std::optional<OspaResult>, std::optional<Ospa2Result>> ground_truth_metrics =
      [&] -> std::tuple<std::optional<GospaResult>, std::optional<OspaResult>, std::optional<Ospa2Result>> {
    if (auto const gt = database().data_containers.template get<SourceIdTime>().find(
            std::tuple{ SourceId{ "GROUND_TRUTH" }, _tracker.time() });
        gt != database().data_containers.template get<SourceIdTime>().end())
    {
      std::vector<BernoulliState<UDT>> const estimate = getEstimate();
      auto gospa = [&] -> std::optional<GospaResult> {
        if (params()->ground_truth_metrics.gospa.evaluate)
        {
          GospaResult gospa_result =
              eval_gospa(*this,
                         estimate | member<&BernoulliState<UDT>::state> | std::ranges::to<std::vector<State<UDT>>>(),
                         gt->_tracks | member<&BernoulliState<UDT>::state> | std::ranges::to<std::vector<State<UDT>>>(),
                         params()->ground_truth_metrics.gospa.c,
                         params()->ground_truth_metrics.gospa.p);
          return gospa_result;
        }
        return {};
      }();
      auto ospa = [&] -> std::optional<OspaResult> {
        if (params()->ground_truth_metrics.ospa.evaluate)
        {
          return eval_ospa(*this,
                           estimate | member<&BernoulliState<UDT>::state> | std::ranges::to<std::vector<State<UDT>>>(),
                           gt->_tracks | member<&BernoulliState<UDT>::state> |
                               std::ranges::to<std::vector<State<UDT>>>(),
                           params()->ground_truth_metrics.ospa.c,
                           params()->ground_truth_metrics.ospa.p);
        }
        return {};
      }();
      auto ospa2 = [&] -> std::optional<Ospa2Result> {
        if (params()->ground_truth_metrics.ospa2.evaluate)
        {
          assert(filter_time() == gt->_time);
          for (State<UDT> const& state : gt->_tracks | member<&BernoulliState<UDT>::state>)
          {
            assert(state._time == gt->_time);
            if (not gt_tracks.contains(state._label))
            {
              gt_tracks.emplace(state._label,
                                SlidingWindow<std::optional<State<UDT>>>(
                                    std::chrono::milliseconds(params()->trajectory.sliding_window_length_ms)));
            }
            gt_tracks.at(state._label).insert(state._time, state);
          }
          for (auto& [label, states] : gt_tracks)
          {
            if (not std::ranges::contains(
                    gt->_tracks | member<&BernoulliState<UDT>::state>, label, &State<UDT>::_label))
            {
              states.insert(gt->_time, {});
            }
          }
          std::erase_if(gt_tracks, [](std::pair<Label, SlidingWindow<std::optional<State<UDT>>>> const& entry) {
            return std::ranges::all_of(entry.second.data_buffer, [](TimedData<std::optional<State<UDT>>> const& data) {
              return not data.data.has_value();
            });
          });
          return eval_ospa2(*this, gt_tracks, _estimated_tracks);
        }
        return {};
      }();
      std::string save_name = "/tmp/metric_results";
      auto [profiler, _] = _profilers.try_emplace("metric_results", save_name);
      Duration cycle_time = cycle_end - cycle_start;
      if (gospa.has_value() and ospa2.has_value() and ospa.has_value())
      {
        profiler->second.addData(
            to_string(GroundTruthProfiling{ time, gospa.value(), ospa.value(), ospa2.value(), cycle_time }));
      }
      return std::tuple{ gospa, ospa, ospa2 };
    }
    return {};
  }();
  if (filter_time_start > Time{ 0s } and not data_containers.empty())
  {
    LOG_INF(std::format("Comp Time: {}", to_milliseconds(cycle_end - cycle_start)));
  }
  if (params()->show_gui)
  {
    if (filter_time_start > Time{ 0s } and not data_containers.empty())
    {
      vizu()._cycle_data.wlock().data.computation_time.emplace_back(filter_time_start, cycle_end - cycle_start);
      std::ranges::sort(vizu()._cycle_data.wlock().data.computation_time);
    }
    auto const& [gospa, ospa, ospa2] = ground_truth_metrics;
    if (gospa.has_value())
    {
      vizu()._cycle_data.wlock().data.gospa.emplace_back(_tracker.time(), std::move(gospa.value()));
    }
    if (ospa.has_value())
    {
      vizu()._cycle_data.wlock().data.ospa.emplace_back(_tracker.time(), std::move(ospa.value()));
    }
    if (ospa2.has_value())
    {
      vizu()._cycle_data.wlock().data.ospa2.emplace_back(_tracker.time(), std::move(ospa2.value()));
    }
  }
  ++_num_cycles;
  TracyFiberEnter("TTBManager::cycle");
  TracyCZoneEnd(ctx);
}

template <class UDT>
std::vector<BernoulliState<UDT>> TTBManager<UDT>::getEstimate(Time time, EgoMotionDistribution egoMotion)
{
  if (time < filter_time())
  {
    LOG_WARN("Reset.");
    reset();
    return {};
  }
  std::vector<BernoulliState<UDT>> tracks = getEstimate();
  if (params()->state.estimation.perform_prediction)
  {
    std::vector<BernoulliState<UDT>> predicted_tracks;
    for (BernoulliState<UDT>& track : tracks)
    {
      auto const dt = time - track.state._time;
      if (dt > 1s)
      {
        LOG_WARN_THROTTLE(1,
                          "Requested Estimation of the far future in " + std::to_string(ttb::to_milliseconds(dt)) +
                              "ms. Ignoring this request and return the non-predicted track.");
        continue;
      }
      track.state.predict(dt, egoMotion);
      if (not track.state.isEmpty())
      {
        auto _ = track.state.getEstimate();  // this caches the result
        predicted_tracks.push_back(std::move(track));
      }
    }
    return predicted_tracks;
  }
  return tracks;
}

template <class UDT>
std::vector<BernoulliState<UDT>> TTBManager<UDT>::getEstimate()
{
  TracyFiberEnter("TTBManager::getEstimate");
  TracyCZoneN(ctx, "TTBManager::getEstimate", 1);
  std::vector<BernoulliState<UDT>> tracks = _tracker.get_estimate();
  for (BernoulliState<UDT>& track : tracks)
  {
    auto [model, dist] = track.state.getEstimate();
    STATE_MODEL_TYPE const type = params()->state_model_params(model).type;
    if (type == STATE_MODEL_TYPE::CTRV)
    {
      Components const& output_comps = getStateModel(model).state_comps();
      if (auto ind = output_comps.indexOf({ COMPONENT::VEL_ABS, COMPONENT::ROT_Z }); ind.has_value())
      {
        Vector<UDT> mean = dist.mean();
        if (mean(ind.value()(0)) < 0)
        {
          mean(ind.value()(0)) *= -1;
          mean(ind.value()(1)) += std::numbers::pi;
          angles::normalizeAngle(mean(ind.value()(1)));
          dist.mean() = std::move(mean);
          track.state._state_dist.clear();
          track.state._state_dist.emplace(model, std::pair{ 1.0, Distribution<UDT>(std::move(dist)) });
        }
      }
    }
  }
  if (params()->static_tracks.enable)
  {
    for (StaticTrackParams const& static_track : params()->static_tracks.tracks)
    {
      std::vector<COMPONENT> comps;
      for (auto const& [str_comp, val] : static_track.mean)
      {
        auto comp = to_COMPONENT(str_comp);
        if (not comp.has_value())
        {
          LOG_WARN("Unknown comp in static track: " + str_comp);
          continue;
        }
        comps.push_back(comp.value());
      }
      Components meas_comps(comps);
      Vector<UDT> mean(comps.size());
      for (COMPONENT comp : meas_comps._comps)
      {
        mean(meas_comps.indexOf(comp).value()) = static_track.mean.at(to_string(comp));
      }
      Matrix<UDT> cov =
          1e-7 * Matrix<UDT>::Identity(static_cast<Index>(comps.size()), static_cast<Index>(comps.size()));
      std::optional<transformation::Transformed<UDT>> trafo = transformation::transform(
          mean, cov, meas_comps, getStateModel(params()->state.estimation.output_state_model).state_comps());
      if (trafo.has_value())
      {
        std::map<StateModelId, std::vector<std::pair<UDT, GaussianDistribution<UDT>>>> state_dist;
        state_dist.emplace(
            params()->state.estimation.output_state_model,
            std::vector{
                std::pair{ UDT{ 1.0 }, GaussianDistribution<UDT>(std::move(trafo->mean), std::move(trafo->cov)) } });
        State<UDT> track(this, std::move(state_dist));
        track.postProcess();
        track._misc["origin"] = std::string("Static Object");
        track._time = filter_time();
        if (not track.isEmpty())
        {
          tracks.emplace_back(std::move(track), 1);
        }
        else
        {
          LOG_WARN("Static track is empty, did not survive the post processing");
        }
      }
      else
      {
        LOG_WARN("Can not create Static track");
      }
    }
  }
  std::ranges::sort(tracks, [](BernoulliState<UDT> const& first, BernoulliState<UDT> const& second) {
    return first.state._label < second.state._label;
  });
  for (State<UDT> const& state : tracks | member<&BernoulliState<UDT>::state>)
  {
    if (not _estimated_tracks.contains(state._label))
    {
      _estimated_tracks.emplace(state._label,
                                SlidingWindow<std::optional<State<UDT>>>(std::chrono::milliseconds(
                                    params()->ground_truth_metrics.ospa2.sliding_window_length_ms)));
    }
    _estimated_tracks.at(state._label).insert(state._time, state);
  }
  for (auto& label_states : _estimated_tracks)
  {
    if (not std::ranges::contains(
            tracks | member<&BernoulliState<UDT>::state>, label_states.first, &State<UDT>::_label))
    {
      label_states.second.insert(filter_time(), {});
    }
  }
  std::erase_if(_estimated_tracks, [](std::pair<Label, SlidingWindow<std::optional<State<UDT>>>> const& entry) {
    return std::ranges::all_of(entry.second.data_buffer, [](TimedData<std::optional<State<UDT>>> const& data) {
      return not data.data.has_value();
    });
  });

  if (params()->show_gui)
  {
    for (State<UDT> const& track : tracks | member<&BernoulliState<UDT>::state>)
    {
      vizu()._cycle_data.wlock().data._tracks[track._label].emplace_back(track);
    }
    for (StateId track : _tracker.all_tracks())
    {
      State<UDT> const& state = get_state(track);
      vizu()._cycle_data.wlock().data._all_tracks[state._label].emplace_back(state);
    }
    vizu()._cycle_data.wlock().data._num_tracks.emplace_back(filter_time(), tracks.size());
    std::vector<State<UDT>> vanished;
    for (auto const& [label, states] : _estimated_tracks)
    {
      if (states.data_buffer.size() >= 2)
      {
        std::optional<State<UDT>> const& prev_state = states.data_buffer.at(states.data_buffer.size() - 2).data;
        if (not states.data_buffer.at(states.data_buffer.size() - 1).data.has_value() and prev_state.has_value())
        {
          vanished.push_back(prev_state.value());
        }
      }
    }
    vizu()._cycle_data.wlock()->_vanished_tracks.emplace_back(filter_time(), std::move(vanished));
  }
  TracyFiberEnter("TTBManager::getEstimate");
  TracyCZoneEnd(ctx);
  return tracks;
}

template <class UDT>
void TTBManager<UDT>::addData(DataSourceContainer<UDT> meas, Time receive_time)
{
  TracyFiberEnter("TTBManager::addData");
  TracyCZoneN(ctx, "TTBManager::addData", 1);
  LOG_DEB("TTBManager::addData");
  LOG_DEB("Data Source " << meas._source_id);
  if (meas._source_id == SourceId{ "GROUND_TRUTH" })
  {
    if (params()->show_gui)
    {
      for (State<UDT> const& state : meas._tracks | member<&BernoulliState<UDT>::state>)
      {
        vizu()._cycle_data.wlock().data._gt_tracks[state._label].push_back(state);
      }
    }
    _database.data_containers.insert(std::move(meas));
    TracyFiberEnter("TTBManager::addData");
    TracyCZoneEnd(ctx);
    return;
  }
  if (not params()->have_measurement_model(meas._source_id) and not params()->ttt_filter.enable)
  {
    LOG_WARN_THROTTLE(10, "Received MeasurementContainer from unknown model: " + meas._source_id.value_ + ". Ignoring");
    TracyFiberEnter("TTBManager::addData");
    TracyCZoneEnd(ctx);
    return;
  }
  if (receive_time < _current_time)
  {
    LOG_WARN(std::format("Measurements are not delivered in order of receiving time.\nCurrent Time: {}\nMsg receive "
                         "time: {}\nHighly suspicious.",
                         _current_time,
                         receive_time));
  }
  _current_time = std::max(_current_time, receive_time);

  if (meas._time > receive_time)
  {
    LOG_WARN(std::format("Received measurement from the future!!!\nMeasurement acquisition time is {}ms after the "
                         "receive time.\nData Source {}\nSet acquisition time to receive time. Highly suspicious.",
                         to_milliseconds(meas._time - receive_time),
                         meas._source_id));
    meas._time = receive_time;
  }

  {
    LOG_DEB(meas.toString());
    _database.data_containers.insert(meas);
    for (State<UDT> const& state_meas : meas._measurements)
    {
      _database.states.insert(state_meas);
      _database.meas_to_data_container.insert({ state_meas._id, meas._id });
    }
    for (BernoulliState<UDT> const& state_meas : meas._tracks)
    {
      _database.states.insert(state_meas.state);
      _database.state_existences.emplace(state_meas.state._id, state_meas.existence_probability);
      _database.meas_to_data_container.insert({ state_meas.state._id, meas._id });
    }
  }

  if (params()->show_gui)
  {
    vizu()._meas_model_data.wlock().data[meas._source_id]._delay.emplace_back(meas._time, receive_time - meas._time);
    std::ranges::sort(vizu()._meas_model_data.wlock().data[meas._source_id]._delay);
    vizu()._meas_model_data.wlock().data[meas._source_id]._estimated_delay.emplace_back(
        meas._time, _dataBuffer.getEstimatedLatency(meas._source_id));
    std::ranges::sort(vizu()._meas_model_data.wlock().data[meas._source_id]._estimated_delay);
  }
  SourceId const id = meas._source_id;
  Time const meas_time = meas._time;
  _newestMeasurementTime = std::max(_newestMeasurementTime, meas_time);
  _max_delays[meas._source_id] = std::max(_max_delays[meas._source_id], receive_time - meas._time);
  minimal_latency_buffer::PushReturn const rv = _dataBuffer.push(id, receive_time, meas_time, std::move(meas));
  if (rv == minimal_latency_buffer::PushReturn::RESET)
  {
    LOG_WARN("Full reset because the Buffer detected some anomaly.");
    reset();
  }
  LOG_DEB("TTBManager::addData done");
  TracyFiberEnter("TTBManager::addData");
  TracyCZoneEnd(ctx);
}

template <class UDT>
void TTBManager<UDT>::reset()
{
  TracyFiberEnter("TTBManager::reset");
  TracyCZoneN(ctx, "TTBManager::reset", 1);
  LOG_DEB("Reset TTB Manager");
  _last_reset_time = _current_time;
  _current_time = Time{ 0s };
  _newestMeasurementTime = Time{ 0s };
  _dataBuffer.reset();
  _tracker.reset();
  _database.reset();
  _birth_model.reset();
  _assessment.reset();
  _num_cycles = 0;
  _viz.reset();
  gt_tracks.clear();
  _estimated_tracks.clear();
  TracyFiberEnter("TTBManager::reset");
  TracyCZoneEnd(ctx);
}

template <class UDT>
void TTBManager<UDT>::show_gui()
{
  Params next_params = *params();
  next_params.show_gui = true;
  update_params(std::move(next_params));
  const std::string viz_pool = "viz_pool";
  if (hpx::resource::pool_exists(viz_pool))
  {
    LOG_WARN("This may crash. Whether this works is highly platform dependent.");
    LOG_WARN("Should only be printed if you run the vizu along with the Python Bindings.");
    LOG_WARN("Problem is that glfw should only run in the main thread ...");
    hpx::execution::parallel_executor viz_exec(&hpx::resource::get_thread_pool(viz_pool));
    hpx::post(viz_exec, [&] {
      try
      {
        _viz.viz_loop();
      }
      catch (std::exception const& err)
      {
        LOG_WARN("Visualization crashed: " << err.what());
      }
      catch (...)
      {
        LOG_WARN("Visualization crashed");
      }
    });
  }
}

template <class UDT>
void TTBManager<UDT>::update_params(Params next_params)
{
  _params.update(std::move(next_params));
}

template <class UDT>
std::shared_ptr<Params const> TTBManager<UDT>::params() const
{
  return _params.get();
}

template <class UDT>
Params const& TTBManager<UDT>::original_params() const
{
  return _original_params;
}

template <class UDT>
Vizu<UDT>& TTBManager<UDT>::vizu() const
{
  return _viz;
}

template <class UDT>
Time TTBManager<UDT>::filter_time() const
{
  return _tracker.time();
}

}  // namespace ttb
