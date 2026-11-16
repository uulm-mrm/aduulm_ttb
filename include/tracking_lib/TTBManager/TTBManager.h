#pragma once

#include "tracking_lib/StateTransition/MarkovTransition.h"
#include "tracking_lib/DataContainer/DataSourceContainer.h"
#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/Trackers/Tracker.h"
#include "tracking_lib/StateModels/BaseStateModel.h"
#include "tracking_lib/MeasurementModels/BaseMeasurementModel.h"
#include "tracking_lib/BirthModels/BaseBirthModel.h"
#include "tracking_lib/PersistenceModels/BasePersistenceModel.h"
#include "tracking_lib/TTBManager/Visualization.h"
#include "minimal_latency_buffer/minimal_latency_buffer.hpp"
#if __has_include("minimal_latency_buffer/mlb_visu.hpp")
#include "minimal_latency_buffer/mlb_visu.hpp"
#endif
#include "tracking_lib/DataContainer/Database.h"
#include "tracking_lib/SelfAssessment/SelfAssessment.h"
namespace ttb
{

struct GroundTruthProfiling
{
  Time time;
  GospaResult gospa;
  OspaResult ospa;
  Ospa2Result ospa2;
  Duration cycle_time;
};

[[nodiscard]] std::string to_string(GroundTruthProfiling const& data);

/// This class manages all components and is the main handle of the tacking_lib
/// It is responsible for the creation and lifetime management of the components
template <class UDT>
class TTBManager
{
public:
  explicit TTBManager(Params params);
  explicit TTBManager(std::filesystem::path const& config_file);
  /// can not copy / move this around ....
  TTBManager(TTBManager const& other) = delete;
  TTBManager(TTBManager&& other) noexcept = delete;
  TTBManager& operator=(TTBManager const& other) = delete;
  TTBManager& operator=(TTBManager&& other) = delete;
  /// access to the database
  [[nodiscard]] Database<UDT> const& database() const;
  [[nodiscard]] Database<UDT>& database();
  /// access to the persistence model
  [[nodiscard]] PersistenceModel<UDT> const& getPersistenceModel() const;
  [[nodiscard]] PersistenceModel<UDT>& getPersistenceModel();
  /// access to the filter
  [[nodiscard]] Tracker<UDT> const& getFilter() const;
  [[nodiscard]] Tracker<UDT>& getFilter();
  /// access to the birth model
  [[nodiscard]] BirthModel<UDT> const& getBirthModel() const;
  [[nodiscard]] BirthModel<UDT>& getBirthModel();
  /// access to the Transition Model
  [[nodiscard]] MarkovTransition<UDT> const& getTransition() const;
  [[nodiscard]] MarkovTransition<UDT>& getTransition();
  /// access to a single measurement Model
  [[nodiscard]] SingleObjectMeasurementModel<UDT> const& getMeasModel(SourceId const& id) const;
  [[nodiscard]] SingleObjectMeasurementModel<UDT>& getMeasModel(SourceId const& id);
  /// access to all state Models
  [[nodiscard]] std::map<StateModelId, StateModel<UDT>> const& getStateModelMap() const;
  [[nodiscard]] std::map<StateModelId, StateModel<UDT>>& getStateModelMap();
  /// access to the state Model with the given id
  [[nodiscard]] StateModel<UDT> const& getStateModel(StateModelId id) const;
  [[nodiscard]] StateModel<UDT>& getStateModel(StateModelId id);
  /// the components of state model
  /// State Models
  struct StateIdComps
  {
    StateModelId id;
    Components comps;
  };
  using StateIdCompsContainer = bmi::multi_index_container<
      StateIdComps,
      bmi::indexed_by<
          bmi::hashed_unique<bmi::tag<StateModelId>, bmi::member<StateIdComps, StateModelId, &StateIdComps::id>>,
          bmi::hashed_non_unique<bmi::tag<Components>, bmi::member<StateIdComps, Components, &StateIdComps::comps>>>>;
  [[nodiscard]] Components const& state_comps(StateModelId id) const;
  [[nodiscard]] Components const& noise_comps(StateModelId id) const;
  /// get the state_model_id of specific components, this can introduce a new state_model id for which no state model is
  /// defined. This can e.g. happen for measurements
  [[nodiscard]] StateModelId state_model_id(Components const& comps) const;
  /// access to a stateModel by type, return first if multiples
  [[nodiscard]] StateModel<UDT> const& getStateModel(STATE_MODEL_TYPE type, EXTENT extent) const;
  [[nodiscard]] StateModel<UDT>& getStateModel(STATE_MODEL_TYPE type, EXTENT extent);
  /// create a new State with the right number of state models with their distributions and innovation
  [[nodiscard]] State<UDT> const& get_state(StateId id) const;
  [[nodiscard]] UDT get_state_existence_probability(StateId id) const;
  [[nodiscard]] std::optional<std::reference_wrapper<State<UDT> const>> try_get_state(StateId id) const;

  /// get self-assessment
  [[nodiscard]] TrackingAssessment<UDT>& assessment();
  /// Perform one cycle of the filter. The measurements are loaded from the buffer.
  void cycle(Time time);
  /// Perform one tracking cycle with the given data. The caller is responsible for ensuring the right sequence, etc
  /// specifies the maximum time the filter can progress, you can set it to the latest measurement container time.
  void cycle(Time time, std::vector<DataSourceContainer<UDT>> data_containers, bool save_gui_data = true);

  /// a map for storing various loggers that write data to a file a destruction
  std::map<std::string, profiling::GeneralDataProfiler> _profilers;

private:
  /// get the measurement containers from the buffer
  std::pair<Time, std::vector<DataSourceContainer<UDT>>> get_data_containers(Time time);

public:
  /// return the predicted estimation of the filter
  [[nodiscard]] std::vector<BernoulliState<UDT>> getEstimate(Time Time, EgoMotionDistribution egoMotion);
  /// return the current estimation of the filter
  [[nodiscard]] std::vector<BernoulliState<UDT>> getEstimate();
  /// add a Measurement Container to the measurement Buffer
  void addData(DataSourceContainer<UDT> meas, Time receive_time);
  /// reset all data (filter, birth model, ...)
  void reset();
  /// show the gui
  void show_gui();
  /// get the time of the filter
  [[nodiscard]] Time filter_time() const;
  /// the current parameters, direct read access, must be synchronzed manually
  // [[nodiscard]] ConsistentStore<Params>::Handle params() const;
  /// the current parameters, direct read access, must be synchronzed manually
  [[nodiscard]] std::shared_ptr<Params const> params() const;
  /// the original/first parameters
  [[nodiscard]] Params const& original_params() const;
  /// the next effective parameters, read+write, but only effective after a call to update_params
  void update_params(Params next_params);
  /// visualization manager
  [[nodiscard]] Vizu<UDT>& vizu() const;

private:
  /// the current parameters, read only
  ConsistentStore<Params> _params;
  /// the original params
  Params _original_params;
  /// save all data
  Database<UDT> _database;
  /// optional gt_tracks
  std::map<Label, SlidingWindow<std::optional<State<UDT>>>> gt_tracks;
  /// all estimated tracks
  std::map<Label, SlidingWindow<std::optional<State<UDT>>>> _estimated_tracks;
  /// the actual tracker
  Tracker<UDT> _tracker;
  /// Measurement Models
  std::map<SourceId, SingleObjectMeasurementModel<UDT>> _measModels;

public:
  /// this is intentionally not Synchronized, write access only from the main thread THIS IS NOT AUTOMATICALLY
  /// CHECKED!!!
  mutable StateIdCompsContainer _state_comps;
  StateIdCompsContainer _noise_comps;
#if __has_include("minimal_latency_buffer/mlb_visu.hpp")
  /// Visualisation Data buffer
  type_safe::deferred_construction<minimal_latency_buffer::MLBVisualizer> _data_buffer_visualizer;
#endif
  std::size_t _num_cycles = 0;

private:
  std::map<StateModelId, StateModel<UDT>> _stateModels;

  /// Transition Model
  MarkovTransition<UDT> _markovTransition;
  /// Persistence Model
  PersistenceModel<UDT> _persistence_model;
  /// Birth Model
  BirthModel<UDT> _birth_model;
  /// Self-Assessment of different aspects of the models used by the tracker
  TrackingAssessment<UDT> _assessment;
  /// Profiling Data
  Time _newestMeasurementTime = Time::min();
  /// Data Buffer
  using DataBuffer = minimal_latency_buffer::MinimalLatencyBuffer<DataSourceContainer<UDT>, SourceId>;
  DataBuffer _dataBuffer;
  std::map<SourceId, Duration> _max_delays;  // max delay of a data source
  /// Visualisation
  mutable Vizu<UDT> _viz;
  hpx::jthread _viz_thread;
  /// the current (system) time
  Time _current_time = Time{ 0s };
  /// the last timestamp of a reset
  mutable Time _last_reset_time = Time{ 0s };
};

}  // namespace ttb
