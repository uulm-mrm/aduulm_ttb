#pragma once

#include "tracking_lib/TTTFilters/BaseTTTFilter.h"
#include "tracking_lib/StateTransition/MarkovTransition.h"
#include "tracking_lib/States/StateContainer.h"
#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/Trackers/BaseTracker.h"
#include "tracking_lib/StateModels/BaseStateModel.h"
#include "tracking_lib/MeasurementModels/BaseMeasurementModel.h"
#include "tracking_lib/BirthModels/BaseBirthModel.h"
#include "tracking_lib/PersistenceModels/BasePersistenceModel.h"
#include "tracking_lib/TTBManager/Visualization.h"
#include "minimal_latency_buffer/minimal_latency_buffer.hpp"
#include "tracking_lib/External/thread-pool/include/BS_thread_pool.hpp"

namespace ttb
{

class Track;

/// This class manages all components and is the main handle of the tacking_lib
/// It is responsible for the creation and lifetime management of the components
class TTBManager final
{
public:
  explicit TTBManager(Params params);
  explicit TTBManager(std::filesystem::path const& config_file);
  /// can not copy / move this around ....
  TTBManager(TTBManager const& other) = delete;
  TTBManager(TTBManager&& other) noexcept = delete;
  TTBManager& operator=(TTBManager const& other) = delete;
  TTBManager& operator=(TTBManager&& other) = delete;
  /// access to the persistence model
  [[nodiscard]] BasePersistenceModel const& getPersistenceModel() const;
  [[nodiscard]] BasePersistenceModel& getPersistenceModel();
  /// access to the filter
  [[nodiscard]] BaseTracker const& getFilter() const;
  [[nodiscard]] BaseTracker& getFilter();
  /// access to the birth model
  [[nodiscard]] BaseBirthModel const& getBirthModel() const;
  [[nodiscard]] BaseBirthModel& getBirthModel();
  /// access to the Transition Model
  [[nodiscard]] MarkovTransition const& getTransition() const;
  [[nodiscard]] MarkovTransition& getTransition();
  /// access to all measurement Models
  [[nodiscard]] std::map<MeasModelId, std::unique_ptr<BaseMeasurementModel>> const& getMeasModelMap() const;
  [[nodiscard]] std::map<MeasModelId, std::unique_ptr<BaseMeasurementModel>>& getMeasModelMap();
  /// access to a single measurement Model
  [[nodiscard]] BaseMeasurementModel const& getMeasModel(MeasModelId const& id) const;
  [[nodiscard]] BaseMeasurementModel& getMeasModel(MeasModelId const& id);
  /// parameters of a single MeasurementModel
  [[nodiscard]] GaussianMeasurementModelParams const& meas_model_params(MeasModelId const& id) const;
  [[nodiscard]] GaussianMeasurementModelParams& meas_model_next_params(MeasModelId const& id);
  /// access to all state Models
  [[nodiscard]] std::map<StateModelId, std::unique_ptr<BaseStateModel>> const& getStateModelMap() const;
  [[nodiscard]] std::map<StateModelId, std::unique_ptr<BaseStateModel>>& getStateModelMap();
  /// access to the state Model with the given id
  [[nodiscard]] BaseStateModel const& getStateModel(StateModelId id) const;
  [[nodiscard]] BaseStateModel& getStateModel(StateModelId id);
  [[nodiscard]] StateModelParams const& state_model_params(StateModelId id) const;
  [[nodiscard]] StateModelParams& state_model_next_params(StateModelId id);
  /// access to a stateModel by type, return first if multiples
  [[nodiscard]] BaseStateModel const& getStateModel(STATE_MODEL_TYPE type, EXTENT extent) const;
  [[nodiscard]] BaseStateModel& getStateModel(STATE_MODEL_TYPE type, EXTENT extent);
  /// create a new State with the right number of state models with their distributions and innovation
  [[nodiscard]] State createState();
  /// Perform one cycle of the filter. The measurements are loaded from the buffer.
  void cycle(Time time);
  /// Perform one tracking cycle with the given data. The caller is responsible for ensuring the right sequence, etc
  /// .... For now, there are two overloads for tracking and track-to-track. I hope, we can unify them. The time
  /// specifies the maximum time the filter can progress, you can set it to the latest measurement container time.
  void cycle(Time time, std::vector<MeasurementContainer> data_containers, bool save_gui_data = true);
  void cycle(Time time, std::vector<StateContainer> data_containers, bool save_gui_data = true);

private:
  /// get the measurement containers from the buffer
  std::pair<Time, std::vector<MeasurementContainer>> get_meas_containers(Time time);
  /// get the state containers from the buffer
  std::pair<Time, std::vector<StateContainer>> get_state_containers(Time time);

public:
  /// return the predicted estimation of the filter
  [[nodiscard]] std::vector<State> getEstimate(Time Time, EgoMotionDistribution egoMotion);
  /// return the current estimation of the filter
  [[nodiscard]] std::vector<State> getEstimate() const;
  /// add a Measurement Container to the measurement Buffer
  void addData(MeasurementContainer meas, Time receive_time);
  /// add a Track Container to the buffer
  void addData(StateContainer trackContainer, Time receive_time);
  /// reset all data (filter, birth model, ...)
  void reset();
  /// show the gui
  void show_gui();
  /// get the time of the filter
  [[nodiscard]] Time filter_time() const;
  /// the current parameters, read only, overwritten by next_params after call to update params
  Params const& params() const;
  /// the original/first parameters
  Params const& original_params() const;
  /// the next effective parameters, read+write, but only effective after a call to update_params
  Params& next_params();
  /// set the params to the next_params, this is automatically called at the start of a cycle
  void update_params();
  /// access to the thread pool
  BS::thread_pool& thread_pool() const;
  /// visualization manager
  Vizu& vizu() const;

private:
  /// the current parameters, read only
  Params _params;
  /// the next params, only effective after call to update_params, automatically done before the next cycle
  Params _next_params;
  /// the original params
  Params _original_params;

public:
  /// mutex for updating params
  std::mutex _params_mutex;

private:
  /// thread pool
  mutable BS::thread_pool _thread_pool;
  /// estimation of the "normal" tracker
  [[nodiscard]] std::vector<State> trackingEstimate() const;
  /// estimation of the track-to-track filter
  [[nodiscard]] std::vector<State> tttEstimate() const;
  /// Multi-Object Multi-Sensor Filter
  std::unique_ptr<BaseTracker> _filter;
  /// Track-to-track filter
  std::unique_ptr<BaseTTTFilter> _tttFilter;
  /// Measurement Models
  std::map<MeasModelId, std::unique_ptr<BaseMeasurementModel>> _measModels;
  /// State Models
  std::map<StateModelId, std::unique_ptr<BaseStateModel>> _stateModels;
  /// Transition Model
  MarkovTransition _markovTransition;
  /// Persistence Model
  std::unique_ptr<BasePersistenceModel> _persistence_model;
  /// Birth Model
  std::unique_ptr<BaseBirthModel> _birth_model;
  /// Profiling Data
  Time _newestMeasurementTime = Time::min();
  std::size_t _num_cycles = 0;
  /// Data Buffer
  using MeasurementBuffer = minimal_latency_buffer::MinimalLatencyBuffer<MeasurementContainer, MeasModelId>;
  using TrackBuffer = minimal_latency_buffer::MinimalLatencyBuffer<StateContainer, SourceId>;
  MeasurementBuffer _measBuffer;
  TrackBuffer _trackBuffer;
  /// Visualisation
  mutable Vizu _viz;
  std::jthread _viz_thread;
  std::mutex _create_state_mutex;
  mutable std::optional<State> _stateDistributionCache;
};

}  // namespace ttb