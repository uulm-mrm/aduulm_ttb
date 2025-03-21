#include "tracking_lib/TTBManager/TTBManager.h"
// ######################################################################################################################
#include "tracking_lib/Measurements/MeasurementContainer.h"
#include "tracking_lib/TTTFilters/BaseTTTFilter.h"
#include "tracking_lib/PersistenceModels/ConstantPersistenceModel.h"
#include "tracking_lib/StateTransition/MarkovTransition.h"
#include "tracking_lib/BirthModels/BaseBirthModel.h"
#include "tracking_lib/StateModels/BaseStateModel.h"
#include "tracking_lib/MeasurementModels/BaseMeasurementModel.h"
#include "tracking_lib/MeasurementModels/GaussianMeasurementModel.h"
#include "tracking_lib/BirthModels/DynamicBirthModel.h"
#include "tracking_lib/BirthModels/StaticBirthModel.h"
#include "tracking_lib/Trackers/GLMB_IC_Tracker.h"
#include "tracking_lib/Trackers/LMB_IC_Tracker.h"
#include "tracking_lib/Trackers/LMB_FPM_Tracker.h"
#include "tracking_lib/Trackers/NO_Tracker.h"
#include "tracking_lib/Trackers/NN_Tracker.h"
#include "tracking_lib/Trackers/Id_Tracker.h"
#include "tracking_lib/Trackers/GNN_Tracker.h"
#include "tracking_lib/Trackers/PHD_Tracker.h"
#include "tracking_lib/TTTFilters/NoTTTFilter.h"
#include "tracking_lib/TTTFilters/TransTTTFilter.h"
#include "tracking_lib/TTTFilters/EvalTTTFilter.h"
#include "tracking_lib/StateModels/CPStateModel.h"
#include "tracking_lib/StateModels/CTP.h"
#include "tracking_lib/StateModels/CVStateModel.h"
#include "tracking_lib/StateModels/CAStateModel.h"
#include "tracking_lib/StateModels/ISCATRStateModel.h"
#include "tracking_lib/StateModels/CTRAStateModel.h"
#include "tracking_lib/StateModels/CTRVStateModel.h"
#include "tracking_lib/Distributions/GaussianDistribution.h"
#include "tracking_lib/Distributions/MixtureDistribution.h"
#include "tracking_lib/States/Innovation.h"

#include <tracy/tracy/Tracy.hpp>
#include <figcone/configreader.h>

namespace ttb
{

constexpr auto tracy_color = tracy::Color::Gray80;

TTBManager::TTBManager(std::filesystem::path const& config_file)
  : TTBManager([&] -> Params {
    figcone::ConfigReader reader{};
    try
    {
      auto params = reader.readYamlFile<Params>(config_file);
      if (param_version != params.version)
      {
        LOG_WARN("You are using Param version: " << params.version << ". The tracking_lib version is " << param_version
                                                 << ". The versions are compatible.");
      }
      return params;
    }
    catch (std::exception const& excp)
    {
      LOG_FATAL("Can not read Parameter Config file. You are using an incompatible format. The current Parameter "
                "Specification Version is "
                << param_version);
      LOG_FATAL("Error: " << excp.what());
      std::ifstream file(config_file);
      std::stringstream sstream;
      sstream << file.rdbuf();
      LOG_FATAL("Config: " << sstream.str());
      throw;
    }
  }())
{
}

TTBManager::TTBManager(Params params)
  : _params{ std::move(params) }
  , _next_params{ _params }
  , _original_params{ _params }
  , _thread_pool{ BS::thread_pool(_params.thread_pool_size) }
  , _filter{ [&] -> std::unique_ptr<BaseTracker> {
    if (not _params.filter.enable)
    {
      return {};
    }
    switch (_params.filter.type)
    {
      case FILTER_TYPE::NO:
        return std::make_unique<NO_Tracker>(this);
      case FILTER_TYPE::NN:
        return std::make_unique<NN_Tracker>(this);
      case FILTER_TYPE::LMB_IC:
        return std::make_unique<LMB_IC_Tracker>(this);
      case FILTER_TYPE::LMB_PM:
        return {};
      case FILTER_TYPE::LMB_FPM:
        return std::make_unique<LMB_FPM_Tracker>(this);
      case FILTER_TYPE::GLMB_IC:
        return std::make_unique<GLMB_IC_Tracker>(this);
      case FILTER_TYPE::GLMB_PM:
        return {};
      case FILTER_TYPE::GNN:
        return std::make_unique<GNN_Tracker>(this);
      case FILTER_TYPE::ID_TRACKER:
        return std::make_unique<Id_Tracker>(this);
      case FILTER_TYPE::PHD:
        return std::make_unique<PHD_Tracker>(this);
    }
    assert(false);
    DEBUG_ASSERT_MARK_UNREACHABLE;
  }() }
  , _tttFilter{ [&] -> std::unique_ptr<BaseTTTFilter> {
    if (not _params.ttt_filter.enable)
    {
      return {};
    }
    switch (_params.ttt_filter.type)
    {
      case TTT_FILTER_TYPE::NO:
        return std::make_unique<NoTTTFilter>(this);
      case TTT_FILTER_TYPE::TRANS:
        return std::make_unique<TransTTTFilter>(this);
      case TTT_FILTER_TYPE::EVAL:
        return std::make_unique<EvalTTTFilter>(this);
    }
    assert(false);
    DEBUG_ASSERT_MARK_UNREACHABLE;
  }() }
  , _measModels{ [&] {
    std::map<MeasModelId, std::unique_ptr<BaseMeasurementModel>> meas_models;
    auto const& meas_params = _params.meas_models;
    for (auto const& gaussianModel : meas_params.gaussian_models)
    {
      MeasModelId const id = gaussianModel.id;
      if (meas_models.contains(id))
      {
        LOG_FATAL("MeasurementModels with the same ID detected");
        throw std::runtime_error("MeasurementModels with the same ID detected - FIX config");
      }
      meas_models[id] = std::make_unique<GaussianMeasurementModel>(this, id);
    }
    if (meas_models.empty() and _params.filter.enable)
    {
      LOG_WARN("No Measurement Model is created!");
    }
    return meas_models;
  }() }
  , _stateModels{ [&] {
    std::map<StateModelId, std::unique_ptr<BaseStateModel>> models;
    for (StateModelParams const& stateModelParams : _params.state_models)
    {
      std::unique_ptr<BaseStateModel> stateModel{ [&]() -> std::unique_ptr<BaseStateModel> {
        switch (stateModelParams.type)
        {
          case STATE_MODEL_TYPE::CP:
            return std::make_unique<CPStateModel>(this, stateModelParams.id);
          case STATE_MODEL_TYPE::CTP:
            return std::make_unique<CTP>(this, stateModelParams.id);
          case STATE_MODEL_TYPE::CV:
            return std::make_unique<CVStateModel>(this, stateModelParams.id);
          case STATE_MODEL_TYPE::CA:
            return std::make_unique<CAStateModel>(this, stateModelParams.id);
          case STATE_MODEL_TYPE::ISCATR:
            return std::make_unique<ISCATRStateModel>(this, stateModelParams.id);
          case STATE_MODEL_TYPE::CTRV:
            return std::make_unique<CTRVStateModel>(this, stateModelParams.id);
          case STATE_MODEL_TYPE::CTRA:
            return std::make_unique<CTRAStateModel>(this, stateModelParams.id);
        }
        assert(false);
        DEBUG_ASSERT_MARK_UNREACHABLE;
      }() };
      models.emplace(stateModelParams.id, std::move(stateModel));
    }
    if (models.empty())
    {
      LOG_WARN("No state model was created!");
    }
    return models;
  }() }
  , _markovTransition{ [&] { return MarkovTransition(this); }() }
  , _persistence_model{ [&] -> std::unique_ptr<BasePersistenceModel> {
    switch (_params.persistence_model.type)
    {
      case PERSISTENCE_MODEL_TYPE::CONSTANT:
        return std::make_unique<ConstantPersistenceModel>(this);
    }
    assert(false);
    DEBUG_ASSERT_MARK_UNREACHABLE;
  }() }
  , _birth_model{ [&] -> std::unique_ptr<BaseBirthModel> {
    if (not _params.birth_model.has_value())
    {
      return {};
    }
    switch (_params.birth_model.value().type)
    {
      case BIRTH_MODEL_TYPE::DYNAMIC:
        return std::make_unique<DynamicBirthModel>(this);
      case BIRTH_MODEL_TYPE::STATIC:
        return std::make_unique<StaticBirthModel>(this);
    }
    assert(false);
    DEBUG_ASSERT_MARK_UNREACHABLE;
  }() }
  , _measBuffer{ MeasurementBuffer(MeasurementBuffer::Params{
        .mode = minimal_latency_buffer::BufferMode::BATCH,
        .reset_threshold = std::chrono::seconds(10),
        .max_total_wait_time = std::chrono::milliseconds(_params.buffer.max_wait_duration_ms),
        .batch = { .max_delta = std::chrono::milliseconds(_params.buffer.max_delta_ms) } }) }
  , _trackBuffer{ TrackBuffer(
        TrackBuffer::Params{ .mode = minimal_latency_buffer::BufferMode::BATCH,
                             .reset_threshold = std::chrono::seconds(10),
                             .max_total_wait_time = std::chrono::milliseconds(_params.buffer.max_wait_duration_ms),
                             .batch = { .max_delta = std::chrono::milliseconds(_params.buffer.max_delta_ms) } }) }
  , _viz{ Vizu(this) }
  , _viz_thread{ [&] -> std::jthread {
    if (_params.show_gui)
    {
      return std::jthread([&] {
        try
        {
          _viz.viz_loop();
        }
        catch (std::exception const& err)
        {
          LOG_WARN("Visualization crashed: " << err.what());
          _params.show_gui = false;
        }
        catch (...)
        {
          LOG_WARN("Visualization crashed");
          _params.show_gui = false;
        }
      });
    }
    return {};
  }() }
{
  TracySetProgramName(_params.name.c_str());
  update_params();
}

BasePersistenceModel const& TTBManager::getPersistenceModel() const
{
  TracyPlotConfig("ManagerData", tracy::PlotFormatType::Number, true, false, tracy_color);
  TracyPlot("ManagerData", static_cast<int64_t>(1));
  TracyPlot("ManagerData", static_cast<int64_t>(0));
  assert(_persistence_model);
  return *_persistence_model;
}

BasePersistenceModel& TTBManager::getPersistenceModel()
{
  TracyPlotConfig("ManagerData", tracy::PlotFormatType::Number, true, false, tracy_color);
  TracyPlot("ManagerData", static_cast<int64_t>(1));
  TracyPlot("ManagerData", static_cast<int64_t>(0));
  assert(_persistence_model);
  return *_persistence_model;
}

BaseTracker const& TTBManager::getFilter() const
{
  TracyPlotConfig("ManagerData", tracy::PlotFormatType::Number, true, false, tracy_color);
  TracyPlot("ManagerData", static_cast<int64_t>(2));
  TracyPlot("ManagerData", static_cast<int64_t>(0));
  assert(_filter);
  return *_filter;
}

BaseTracker& TTBManager::getFilter()
{
  TracyPlotConfig("ManagerData", tracy::PlotFormatType::Number, true, false, tracy_color);
  TracyPlot("ManagerData", static_cast<int64_t>(2));
  TracyPlot("ManagerData", static_cast<int64_t>(0));
  assert(_filter);
  return *_filter;
}

BaseBirthModel const& TTBManager::getBirthModel() const
{
  TracyPlotConfig("ManagerData", tracy::PlotFormatType::Number, true, false, tracy_color);
  TracyPlot("ManagerData", static_cast<int64_t>(3));
  TracyPlot("ManagerData", static_cast<int64_t>(0));
  assert(_birth_model);
  return *_birth_model;
}

BaseBirthModel& TTBManager::getBirthModel()
{
  TracyPlotConfig("ManagerData", tracy::PlotFormatType::Number, true, false, tracy_color);
  TracyPlot("ManagerData", static_cast<int64_t>(3));
  TracyPlot("ManagerData", static_cast<int64_t>(0));
  assert(_birth_model);
  return *_birth_model;
}

MarkovTransition const& TTBManager::getTransition() const
{
  TracyPlotConfig("ManagerData", tracy::PlotFormatType::Number, true, false, tracy_color);
  TracyPlot("ManagerData", static_cast<int64_t>(4));
  TracyPlot("ManagerData", static_cast<int64_t>(0));
  return _markovTransition;
}

MarkovTransition& TTBManager::getTransition()
{
  TracyPlotConfig("ManagerData", tracy::PlotFormatType::Number, true, false, tracy_color);
  TracyPlot("ManagerData", static_cast<int64_t>(4));
  TracyPlot("ManagerData", static_cast<int64_t>(0));
  return _markovTransition;
}

std::map<MeasModelId, std::unique_ptr<BaseMeasurementModel>> const& TTBManager::getMeasModelMap() const
{
  TracyPlotConfig("ManagerData", tracy::PlotFormatType::Number, true, false, tracy_color);
  TracyPlot("ManagerData", static_cast<int64_t>(5));
  TracyPlot("ManagerData", static_cast<int64_t>(0));
  return _measModels;
}

std::map<MeasModelId, std::unique_ptr<BaseMeasurementModel>>& TTBManager::getMeasModelMap()
{
  TracyPlotConfig("ManagerData", tracy::PlotFormatType::Number, true, false, tracy_color);
  TracyPlot("ManagerData", static_cast<int64_t>(5));
  TracyPlot("ManagerData", static_cast<int64_t>(0));
  return _measModels;
}

BaseMeasurementModel const& TTBManager::getMeasModel(MeasModelId const& id) const
{
  TracyPlotConfig("ManagerData", tracy::PlotFormatType::Number, true, false, tracy_color);
  TracyPlot("ManagerData", static_cast<int64_t>(5));
  TracyPlot("ManagerData", static_cast<int64_t>(0));
  return *_measModels.at(id);
}

BaseMeasurementModel& TTBManager::getMeasModel(MeasModelId const& id)
{
  TracyPlotConfig("ManagerData", tracy::PlotFormatType::Number, true, false, tracy_color);
  TracyPlot("ManagerData", static_cast<int64_t>(5));
  TracyPlot("ManagerData", static_cast<int64_t>(0));
  return *_measModels.at(id);
}

GaussianMeasurementModelParams const& TTBManager::meas_model_params(MeasModelId const& id) const
{
  auto const it = std::ranges::find(_params.meas_models.gaussian_models, id, &GaussianMeasurementModelParams::id);
  assert(it != _params.meas_models.gaussian_models.end());
  return *it;
}

GaussianMeasurementModelParams& TTBManager::meas_model_next_params(MeasModelId const& id)
{
  auto const it = std::ranges::find(_next_params.meas_models.gaussian_models, id, &GaussianMeasurementModelParams::id);
  assert(it != _next_params.meas_models.gaussian_models.end());
  return *it;
}

std::map<StateModelId, std::unique_ptr<BaseStateModel>> const& TTBManager::getStateModelMap() const
{
  TracyPlotConfig("ManagerData", tracy::PlotFormatType::Number, true, false, tracy_color);
  TracyPlot("ManagerData", static_cast<int64_t>(7));
  TracyPlot("ManagerData", static_cast<int64_t>(0));
  return _stateModels;
}

std::map<StateModelId, std::unique_ptr<BaseStateModel>>& TTBManager::getStateModelMap()
{
  TracyPlotConfig("ManagerData", tracy::PlotFormatType::Number, true, false, tracy_color);
  TracyPlot("ManagerData", static_cast<int64_t>(7));
  TracyPlot("ManagerData", static_cast<int64_t>(0));
  return _stateModels;
}

BaseStateModel const& TTBManager::getStateModel(StateModelId id) const
{
  TracyPlotConfig("ManagerData", tracy::PlotFormatType::Number, true, false, tracy_color);
  TracyPlot("ManagerData", static_cast<int64_t>(7));
  TracyPlot("ManagerData", static_cast<int64_t>(0));
  return *_stateModels.at(id);
}

BaseStateModel& TTBManager::getStateModel(StateModelId id)
{
  TracyPlotConfig("ManagerData", tracy::PlotFormatType::Number, true, false, tracy_color);
  TracyPlot("ManagerData", static_cast<int64_t>(7));
  TracyPlot("ManagerData", static_cast<int64_t>(0));
  return *_stateModels.at(id);
}

BaseStateModel const& TTBManager::getStateModel(STATE_MODEL_TYPE type, EXTENT extent) const
{
  TracyPlotConfig("ManagerData", tracy::PlotFormatType::Number, true, false, tracy_color);
  TracyPlot("ManagerData", static_cast<int64_t>(7));
  TracyPlot("ManagerData", static_cast<int64_t>(0));
  for (auto const& [id, model] : _stateModels)
  {
    if (model->type() == type and state_model_params(model->id()).extent == extent)
    {
      return *model;
    }
  }
  LOG_FATAL("State Model with type: " << to_string(type) << " not found");
  throw std::runtime_error("State Model with type: " + to_string(type) + " not found");
}

BaseStateModel& TTBManager::getStateModel(STATE_MODEL_TYPE type, EXTENT extent)
{
  TracyPlotConfig("ManagerData", tracy::PlotFormatType::Number, true, false, tracy_color);
  TracyPlot("ManagerData", static_cast<int64_t>(7));
  TracyPlot("ManagerData", static_cast<int64_t>(0));
  for (auto const& [id, model] : _stateModels)
  {
    if (model->type() == type and state_model_params(model->id()).extent == extent)
    {
      return *model;
    }
  }
  LOG_FATAL("State Model with type: " << to_string(type) << " not found");
  throw std::runtime_error("State Model with type: " + to_string(type) + " not found");
}

StateModelParams const& TTBManager::state_model_params(StateModelId id) const
{
  auto const it = std::ranges::find(_params.state_models, id, &StateModelParams::id);
  assert(it != _params.state_models.end());
  return *it;
}

StateModelParams& TTBManager::state_model_next_params(StateModelId id)
{
  auto const it = std::ranges::find(_next_params.state_models, id, &StateModelParams::id);
  assert(it != _next_params.state_models.end());
  return *it;
}

State TTBManager::createState()
{
  std::lock_guard lock(_create_state_mutex);
  LOG_DEB("Creating New Empty State Distribution");
  if (_stateDistributionCache.has_value())
  {
    _stateDistributionCache.value()._id = State::_idGenerator.getID();
    _stateDistributionCache.value()._label = State::_labelGenerator.getID();
    return _stateDistributionCache.value();
  }
  std::map<StateModelId, std::unique_ptr<BaseDistribution>> sd_map;
  auto const& use_models = _params.state.multi_model.use_state_models;
  for (StateModelParams const& stateModelParams : _params.state_models)
  {
    if (std::ranges::find(use_models, stateModelParams.id) == use_models.end())
    {
      continue;
    }
    std::unique_ptr<BaseDistribution> dist;
    switch (stateModelParams.distribution.type)
    {
      case DISTRIBUTION_TYPE::GAUSSIAN:
        dist = std::make_unique<GaussianDistribution>();
        break;
      case DISTRIBUTION_TYPE::PARTICLE:
        // dist = std::make_unique<ParticleDistribution>();
        LOG_FATAL("Not supported");
        assert(false);
        break;
      case DISTRIBUTION_TYPE::MIXTURE:
        LOG_FATAL("The underlying Dist Type can not be MIXTURE - BUG ALERT");
        assert(false);
        break;
    }
    if (stateModelParams.distribution.mixture)  // construct mixture
    {
      dist = std::make_unique<MixtureDistribution>();
    }
    sd_map.emplace(stateModelParams.id, std::move(dist));
  }
  State bsd(this, std::move(sd_map));

  for (auto const& [id, meas_model] : getMeasModelMap())
  {
    bsd._innovation.emplace(id, Innovation{});
  }
  _stateDistributionCache = std::move(bsd);
  return _stateDistributionCache.value();
}

void TTBManager::cycle(Time time)
{
  FrameMarkStart("TTBManager::cycle");
  update_params();
  if (_params.filter.enable)
  {
    LOG_DEB("Tracking Cycle");
    auto [buffer_time, containers] = get_meas_containers(time);
    cycle(buffer_time, std::move(containers), false);
  }
  else if (_params.ttt_filter.enable)
  {
    LOG_DEB("TTT Cycle");
    auto [buffer_time, containers] = get_state_containers(time);
    cycle(buffer_time, std::move(containers), false);
  }
  else
  {
    LOG_FATAL("No Filter or TTT Filter enabled. False Config");
    throw std::runtime_error("No Filter or TTT Filter enabled. False Config");
  }
  FrameMarkEnd("TTBManager::cycle");
}

auto save_viz_data_containers(TTBManager* manager,
                              Time time,
                              Time buffer_time,
                              auto const& data,
                              auto const& discarded_data,
                              auto const& containers,
                              std::size_t num_queued_elems)
{
  if (manager->params().show_gui)
  {
    std::lock_guard lock(manager->vizu().add_data_mutex);
    if (time - buffer_time < Duration(1h))
    {
      manager->vizu()._cycle_data._buffer_delay.emplace_back(time, time - buffer_time);
    }
    manager->vizu()._cycle_data._buffer_size.emplace_back(time, num_queued_elems);
    for (auto& [queue_id, meas_time, receipt_time, earliest_expected, latest_receipt_time, timed_data, created_ph] :
         data)
    {
      manager->vizu()._meas_model_data[MeasModelId{ queue_id.value_ }]._duration_in_buffer.emplace_back(
          time, time - receipt_time);
    }
    for (auto& [model_id, vizu_data] : manager->vizu()._meas_model_data)
    {
      if (std::ranges::find_if(discarded_data, [&](auto const& time_data) {
            return time_data.id.value_ == model_id.value_;
          }) != discarded_data.end())
      {
        vizu_data._discarded_data.emplace_back(buffer_time);
      }
      if (auto it = std::ranges::find_if(
              containers, [&](auto const& container) { return container._id.value_ == model_id.value_; });
          it != containers.end())
      {
        vizu_data._used_in_cycle.emplace_back(it->_time);
      }
    }
    manager->vizu()._cycle_data._trigger_time.emplace_back(buffer_time);
    if (not containers.empty())
    {
      manager->vizu()._cycle_data._num_sources.emplace_back(time, containers.size());
      manager->vizu()._cycle_data._num_measurements.emplace_back(
          time, std::accumulate(containers.begin(), containers.end(), 0, [](int old, auto const& container) {
            return old + container._data.size();
          }));
    }
    for (auto const& container : containers)
    {
      manager->vizu()._meas_model_data[MeasModelId{ container._id.value_ }]._measurements.emplace_back(container._time,
                                                                                                       container);
    }
  }
}

std::pair<Time, std::vector<MeasurementContainer>> TTBManager::get_meas_containers(Time time)
{
  minimal_latency_buffer::PopReturn data = _measBuffer.pop(time);
  std::vector<MeasurementContainer> containers;
  for (auto& [queue_id, meas_time, receipt_time, earliest_expected, latest_receipt_time, timed_data, created_ph] :
       data.data)
  {
    Duration const dt = timed_data.value()._time - filter_time();
    if (dt < std::chrono::seconds(-1))
    {
      LOG_WARN("Received Measurement from the ancient past (" << to_milliseconds(dt) << "ms ago). Reset.");
      reset();
      return {};
    }
    if (dt < std::chrono::seconds(0))
    {
      LOG_INF("Received Measurement from the past (" << to_milliseconds(dt) << "ms ago). Ignore this Measurement.");
      continue;
    }
    if (dt > std::chrono::seconds(100) and filter_time() > Time{ 0s })
    {
      LOG_WARN("Detected jump in measurement time of " + std::to_string(to_seconds(dt)) + "sec. Reset");
      reset();
      return {};
    }
    containers.emplace_back(std::move(timed_data.value()));
  }
  save_viz_data_containers(this,
                           time,
                           data.buffer_time,
                           data.data,
                           data.discarded_data,
                           containers,
                           _measBuffer.getNumberOfQueuedElements());
  return { data.buffer_time, std::move(containers) };
}

std::pair<Time, std::vector<StateContainer>> TTBManager::get_state_containers(Time time)
{
  minimal_latency_buffer::PopReturn data = _trackBuffer.pop(time);
  LOG_DEB("Got Data from the buffer");
  std::vector<StateContainer> containers;
  for (auto& [queue_id, meas_time, receipt_time, earliest_expected, latest_receipt_time, timed_data, created_ph] :
       data.data)
  {
    containers.push_back(std::move(timed_data.value()));
  }
  save_viz_data_containers(this,
                           time,
                           data.buffer_time,
                           data.data,
                           data.discarded_data,
                           containers,
                           _trackBuffer.getNumberOfQueuedElements());
  return { data.buffer_time, std::move(containers) };
}

void TTBManager::cycle(Time time, std::vector<MeasurementContainer> data_containers, bool save_gui_data)
{
  ZoneScopedNC("TTBManager::cycle", tracy_color);
  std::vector<MeasurementContainer> filtered_containers;
  filtered_containers.reserve(data_containers.size());
  for (MeasurementContainer& container : data_containers)
  {
    if (not getMeasModelMap().contains(container._id))
    {
      LOG_WARN("Received MeasurementContainer from unknown model: " + container._id.value_ + ". Ignoring");
      continue;
    }
    filtered_containers.push_back(std::move(container));
  }
  update_params();
  if (save_gui_data)
  {
    save_viz_data_containers(this,
                             time,
                             time,
                             std::vector<minimal_latency_buffer::TimeData<MeasModelId, MeasurementContainer>>{},
                             std::vector<minimal_latency_buffer::TimeData<MeasModelId, MeasurementContainer>>{},
                             filtered_containers,
                             0);
  }
  Time const cycle_start = std::chrono::high_resolution_clock::now();
  Time const filter_time_start = filter_time();
  _filter->cycle(time, std::move(filtered_containers));
  if (params().show_gui)
  {
    std::lock_guard lock(vizu().add_data_mutex);
    vizu()._cycle_data.computation_time.emplace_back(filter_time_start,
                                                     std::chrono::high_resolution_clock::now() - cycle_start);
  }
  ++_num_cycles;
}

void TTBManager::cycle(Time time, std::vector<StateContainer> data_containers, bool save_gui_data)
{
  ZoneScopedNC("TTBManager::tttCycle", tracy_color);
  update_params();
  if (save_gui_data)
  {
    save_viz_data_containers(this,
                             time,
                             time,
                             std::vector<minimal_latency_buffer::TimeData<SourceId, StateContainer>>{},
                             std::vector<minimal_latency_buffer::TimeData<SourceId, StateContainer>>{},
                             data_containers,
                             0);
  }
  Time const cycle_start = std::chrono::high_resolution_clock::now();
  Time const filter_time_start = filter_time();
  _tttFilter->cycle(time, std::move(data_containers));
  if (params().show_gui)
  {
    std::lock_guard lock(vizu().add_data_mutex);
    vizu()._cycle_data.computation_time.emplace_back(filter_time_start,
                                                     std::chrono::high_resolution_clock::now() - cycle_start);
  }
  ++_num_cycles;
}

std::vector<State> TTBManager::getEstimate(Time time, EgoMotionDistribution egoMotion)
{
  FrameMarkStart("TTBManager::getEstimate");
  ZoneScopedNC("TTTBManager::getEstimate", tracy_color);
  if (time < filter_time())
  {
    LOG_WARN("Reset.");
    reset();
    return {};
  }
  if (_params.show_gui)
  {
    std::lock_guard lock(vizu().add_data_mutex);
    if (time - filter_time() < Duration(1h))
    {
      vizu()._cycle_data._filter_delay.emplace_back(time, time - filter_time());
    }
  }
  std::vector<State> tracks = getEstimate();

  if (_params.state.estimation.perform_prediction)
  {
    if (_params.thread_pool_size > 0)
    {
      thread_pool().detach_loop(std::size_t{ 0 }, tracks.size(), [&](std::size_t i) {
        auto const dt = time - tracks.at(i)._time;
        if (dt > 1s)
        {
          LOG_WARN_THROTTLE(
              1, "Requested Estimation of the far future in " + std::to_string(ttb::to_milliseconds(dt)) + "ms");
          return;
        }
        tracks.at(i).predict(dt, egoMotion);
        auto _ = tracks.at(i).getEstimate();  // this caches the result
      });
      thread_pool().wait();
    }
    else
    {
      for (State& track : tracks)
      {
        auto const dt = time - track._time;
        if (dt > 1s)
        {
          LOG_WARN_THROTTLE(
              1, "Requested Estimation of the far future in " + std::to_string(ttb::to_milliseconds(dt)) + "ms");
          continue;
        }
        track.predict(dt, egoMotion);
        auto _ = track.getEstimate();  // this caches the result
      }
    }
  }
  FrameMarkEnd("TTBManager::getEstimate");
  return tracks;
}

std::vector<State> TTBManager::getEstimate() const
{
  ZoneScopedNC("TTTBManager::getEstimate", tracy_color);
  std::vector<State> tracks{ [&] {
    if (_params.filter.enable)
    {
      return trackingEstimate();
    }
    if (_params.ttt_filter.enable)
    {
      return tttEstimate();
    }
    assert(false);
    DEBUG_ASSERT_MARK_UNREACHABLE;
  }() };
  if (_params.static_tracks.enable)
  {
    for (StaticTrackParams const& static_track : _params.static_tracks.tracks)
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
      Vector mean(comps.size());
      for (COMPONENT comp : meas_comps._comps)
      {
        mean(meas_comps.indexOf(comp).value()) = static_track.mean.at(to_string(comp));
      }
      Matrix cov = 1e-7 * Matrix::Identity(static_cast<Index>(comps.size()), static_cast<Index>(comps.size()));
      Measurement meas(std::make_unique<GaussianDistribution>(std::move(mean), std::move(cov)),
                       filter_time(),
                       std::move(meas_comps));
      std::optional<State> track = getMeasModel(_params.static_tracks.meas_model_id).createState(meas, true);
      track.value()._misc["origin"] = std::string("Static Object");
      track.value()._existenceProbability = 1;
      tracks.push_back(std::move(track.value()));
    }
  }
  std::ranges::sort(tracks, [](State const& first, State const& second) { return first._label < second._label; });
  TracyPlotConfig("# Estimated Tracks", tracy::PlotFormatType::Number, true, false, tracy_color);
  TracyPlot("# Estimated Tracks", static_cast<int64_t>(tracks.size()));
  std::string info_str = "#Estimated Tracks: " + std::to_string(tracks.size());
  ZoneText(info_str.c_str(), info_str.size());
  FrameMarkNamed("TTBManager");
  if (_params.show_gui)
  {
    std::lock_guard lock(vizu().add_data_mutex);
    for (State const& track : tracks)
    {
      vizu()._cycle_data._tracks[track._label].emplace_back(track);
    }
    vizu()._cycle_data._num_tracks.emplace_back(filter_time(), tracks.size());
  }
  return tracks;
}

std::vector<State> TTBManager::trackingEstimate() const
{
  return _filter->getEstimate();
}

std::vector<State> TTBManager::tttEstimate() const
{
  return _tttFilter->getEstimate();
}

void TTBManager::addData(MeasurementContainer meas, Time receive_time)
{
  if (not getMeasModelMap().contains(meas._id))
  {
    LOG_WARN("Received MeasurementContainer from unknown model: " + meas._id.value_ + ". Ignoring");
    return;
  }
  if (meas._time > receive_time)
  {
    LOG_WARN("Received measurement from the future!!!\nMeasurement is " +
             std::to_string(to_milliseconds(meas._time - receive_time)) +
             " ms before current time. Set measurement time to current time. \nData source " + meas._id.value_);
    meas._time = receive_time;
  }
  ZoneScopedNC("TTBManager::addMeasurementContainer", tracy_color);
  ZoneText(meas._id.value_.c_str(), meas._id.value_.size());
  ZoneText("Time", 4);
  auto const time_str = std::to_string(to_milliseconds(meas._time.time_since_epoch()));
  ZoneText(time_str.c_str(), time_str.size());
  TracyPlotConfig("TTBManager::MeasurementContainer", tracy::PlotFormatType::Number, true, false, tracy_color);
  TracyPlot("TTBManager::MeasurementContainer", static_cast<int64_t>(1));
  TracyPlot("TTBManager::MeasurementContainer", static_cast<int64_t>(0));
  LOG_DEB("Store incoming MeasurementContainer in the LIB");

  if (_params.show_gui)
  {
    std::lock_guard lock(vizu().add_data_mutex);
    vizu()._meas_model_data[meas._id]._delay.emplace_back(meas._time, receive_time - meas._time);
    vizu()._meas_model_data[meas._id]._estimated_delay.emplace_back(meas._time,
                                                                    _measBuffer.getEstimatedLatency(meas._id));
  }
  MeasModelId const id = meas._id;
  Time const meas_time = meas._time;
  if (receive_time - _measBuffer.getBufferTime() > std::chrono::milliseconds(_params.buffer.max_wait_duration_ms) and
      _measBuffer.total_size() > 0)
  {
    LOG_WARN("Buffer lag is large: " << to_milliseconds(receive_time - _measBuffer.getBufferTime()) << "ms. Reset.");
    reset();
  }
  minimal_latency_buffer::PushReturn const rv = _measBuffer.push(id, receive_time, meas_time, std::move(meas));
  if (rv == minimal_latency_buffer::PushReturn::RESET)
  {
    LOG_WARN("Reset Buffer because the Buffer detected some time anomaly.");
    reset();
  }
  _newestMeasurementTime = std::max(_newestMeasurementTime, meas_time);
}

void TTBManager::addData(StateContainer trackContainer, Time receive_time)
{
  LOG_DEB("add TrackContainer");
  if (trackContainer._time > receive_time)
  {
    LOG_WARN("Received measurement from the future!!!\nMeasurement is " +
             std::to_string(to_milliseconds(trackContainer._time - receive_time)) +
             " ms before current time. Set measurement time to current time. \nData source " +
             trackContainer._id.value_);
    trackContainer._time = receive_time;
  }
  SourceId const id = trackContainer._id;
  Time const track_time = trackContainer._time;
  if (_params.show_gui)
  {
    std::lock_guard lock(vizu().add_data_mutex);
    vizu()._meas_model_data[MeasModelId{ trackContainer._id.value_ }]._delay.emplace_back(
        trackContainer._time, receive_time - trackContainer._time);
    vizu()._meas_model_data[MeasModelId{ trackContainer._id.value_ }]._estimated_delay.emplace_back(
        trackContainer._time, _trackBuffer.getEstimatedLatency(trackContainer._id));
  }
  if (receive_time - _trackBuffer.getBufferTime() > std::chrono::milliseconds(_params.buffer.max_wait_duration_ms) and
      _trackBuffer.total_size() > 0)
  {
    LOG_WARN("Buffer lag is large: " << to_milliseconds(receive_time - _measBuffer.getBufferTime()) << "ms. Reset.");
    reset();
  }
  minimal_latency_buffer::PushReturn const rv =
      _trackBuffer.push(id, receive_time, track_time, std::move(trackContainer));
  if (rv == minimal_latency_buffer::PushReturn::RESET)
  {
    LOG_WARN("Reset Buffer because the Buffer detected some time anomaly.");
    reset();
  }
}

void TTBManager::reset()
{
  LOG_DEB("Reset TTB Manager");
  TracyPlotConfig("TTBManager::reset", tracy::PlotFormatType::Number, true, false, tracy_color);
  TracyPlot("TTBManager::reset", static_cast<int64_t>(1));
  if (_filter)
  {
    _filter->reset();
  }
  if (_tttFilter)
  {
    _tttFilter->reset();
  }
  if (_birth_model)
  {
    _birth_model->reset();
  }
  _num_cycles = 0;
  _newestMeasurementTime = Time{ 0s };
  _measBuffer.reset();
  _trackBuffer.reset();
  std::lock_guard lock(vizu().add_data_mutex);
  _viz.reset();
  TracyPlot("TTBManager::reset", static_cast<int64_t>(0));
}

void TTBManager::show_gui()
{
  _next_params.show_gui = true;
  update_params();
  _viz_thread = std::jthread([&] { _viz.viz_loop(); });
}

Params& TTBManager::next_params()
{
  return _next_params;
}

void TTBManager::update_params()
{
  std::lock_guard lock(_params_mutex);
  _params = _next_params;
  _stateDistributionCache.reset();
}

Params const& TTBManager::params() const
{
  return _params;
}

Params const& TTBManager::original_params() const
{
  return _original_params;
}

BS::thread_pool& TTBManager::thread_pool() const
{
  return _thread_pool;
}

Vizu& TTBManager::vizu() const
{
  return _viz;
}

Time TTBManager::filter_time() const
{
  if (_filter)
  {
    return _filter->time();
  }
  if (_tttFilter)
  {
    return _tttFilter->time();
  }
  LOG_WARN("No Filter set");
  return Time{ 0s };
}

}  // namespace ttb
