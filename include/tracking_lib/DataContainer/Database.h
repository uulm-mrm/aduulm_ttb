#pragma once

#include "tracking_lib/DataContainer/DataSourceContainer.h"
#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/States/State.h"
#include "tracking_lib/Misc/LockedData.h"

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/composite_key.hpp>
#include <boost/multi_index/member.hpp>

namespace bmi = boost::multi_index;

namespace ttb
{

class SourceIdTime
{
};

class PriorStateId
{
};

template <class UDT>
struct StatePersistenceProbability
{
  StateId state_id = NO_PRIOR;
  UDT persistence_probability = std::numeric_limits<UDT>::quiet_NaN();
};

template <class UDT>
struct StateExistenceProbability
{
  StateId state_id = NO_PRIOR;
  UDT existence_probability = std::numeric_limits<UDT>::quiet_NaN();
};

template <class UDT>
struct StateDetectionProbability
{
  StateId state_id = NO_PRIOR;
  DataSourceContainerId data_source_container_id = DUMMY_CONTAINER;
  UDT detection_probability = std::numeric_limits<UDT>::quiet_NaN();
};

template <class UDT>
struct StateMeasLogLikelihood
{
  StateId state_id = NO_PRIOR;
  StateId meas_id = NO_PRIOR;
  UDT meas_log_likelihood = std::numeric_limits<UDT>::quiet_NaN();
};

template <class UDT>
struct MeasClutterIntensity
{
  StateId meas_id = NO_PRIOR;
  UDT clutter_intensity = std::numeric_limits<UDT>::quiet_NaN();
};

struct StateUpdate
{
  StateId prior_state_id = NO_PRIOR;
  DataSourceContainerId data_container_id = DUMMY_CONTAINER;
  StateId meas_id = NO_PRIOR;
  StateId updated_state_id = NO_PRIOR;
};

struct RelevantMeas
{
  StateId prior_state_id = NO_PRIOR;
  DataSourceContainerId data_container_id = DUMMY_CONTAINER;
  std::vector<StateId> relevant_meas_ids = {};
};

class StateIdContainerId
{
};

class StateMeasId
{
};

class PriorId
{
};

struct MeasData
{
  StateId meas_id;
  DataSourceContainerId data_source_container_id;
};

template <class UDT>
class Database
{
public:
  using MeasDataContainer = bmi::multi_index_container<
      MeasData,
      bmi::indexed_by<
          bmi::hashed_unique<bmi::tag<StateId>, bmi::member<MeasData, StateId, &MeasData::meas_id>>,
          bmi::hashed_non_unique<bmi::tag<DataSourceContainerId>,
                                 bmi::member<MeasData, DataSourceContainerId, &MeasData::data_source_container_id>>>>;

  using DataContainer = bmi::multi_index_container<
      DataSourceContainer<UDT>,
      bmi::indexed_by<
          bmi::hashed_non_unique<
              bmi::tag<SourceId>,
              bmi::member<DataSourceContainer<UDT>, SourceId, &DataSourceContainer<UDT>::_source_id>>,
          bmi::hashed_unique<
              bmi::tag<DataSourceContainerId>,
              bmi::member<DataSourceContainer<UDT>, DataSourceContainerId, &DataSourceContainer<UDT>::_id>>,
          bmi::ordered_non_unique<bmi::tag<Time>,
                                  bmi::member<DataSourceContainer<UDT>, Time, &DataSourceContainer<UDT>::_time>>,
          bmi::hashed_unique<
              bmi::tag<SourceIdTime>,
              bmi::composite_key<DataSourceContainer<UDT>,
                                 bmi::member<DataSourceContainer<UDT>, SourceId, &DataSourceContainer<UDT>::_source_id>,
                                 bmi::member<DataSourceContainer<UDT>, Time, &DataSourceContainer<UDT>::_time>>>>>;

  using StateContainer = bmi::multi_index_container<
      State<UDT>,
      bmi::indexed_by<
          bmi::hashed_unique<bmi::tag<StateId>, bmi::member<State<UDT>, StateId, &State<UDT>::_id>>,
          bmi::ordered_non_unique<bmi::tag<Time>, bmi::member<State<UDT>, Time, &State<UDT>::_time>>,
          bmi::hashed_non_unique<bmi::tag<Label>, bmi::member<State<UDT>, Label, &State<UDT>::_label>>,
          bmi::hashed_non_unique<bmi::tag<PriorId>, bmi::member<State<UDT>, StateId, &State<UDT>::_prior_id>>>>;

  using StateExistenceContainer = bmi::multi_index_container<
      StateExistenceProbability<UDT>,
      bmi::indexed_by<bmi::hashed_unique<
          bmi::tag<StateId>,
          bmi::member<StateExistenceProbability<UDT>, StateId, &StateExistenceProbability<UDT>::state_id>>>>;

  using StatePersistenceContainer = bmi::multi_index_container<
      StatePersistenceProbability<UDT>,
      bmi::indexed_by<bmi::hashed_unique<
          bmi::tag<StateId>,
          bmi::member<StatePersistenceProbability<UDT>, StateId, &StatePersistenceProbability<UDT>::state_id>>>>;

  using StateDetectionProbabilityContainer = bmi::multi_index_container<
      StateDetectionProbability<UDT>,
      bmi::indexed_by<
          bmi::hashed_unique<
              bmi::tag<StateIdContainerId>,
              bmi::composite_key<
                  StateDetectionProbability<UDT>,
                  bmi::member<StateDetectionProbability<UDT>, StateId, &StateDetectionProbability<UDT>::state_id>,
                  bmi::member<StateDetectionProbability<UDT>,
                              DataSourceContainerId,
                              &StateDetectionProbability<UDT>::data_source_container_id>>>,
          bmi::hashed_non_unique<
              bmi::tag<StateId>,
              bmi::member<StateDetectionProbability<UDT>, StateId, &StateDetectionProbability<UDT>::state_id>>,

          bmi::hashed_non_unique<bmi::tag<DataSourceContainerId>,
                                 bmi::member<StateDetectionProbability<UDT>,
                                             DataSourceContainerId,
                                             &StateDetectionProbability<UDT>::data_source_container_id>>>>;

  using StateMeasLogLikelihoodContainer = bmi::multi_index_container<
      StateMeasLogLikelihood<UDT>,
      bmi::indexed_by<
          bmi::hashed_unique<
              bmi::tag<StateMeasId>,
              bmi::composite_key<
                  StateMeasLogLikelihood<UDT>,
                  bmi::member<StateMeasLogLikelihood<UDT>, StateId, &StateMeasLogLikelihood<UDT>::state_id>,
                  bmi::member<StateMeasLogLikelihood<UDT>, StateId, &StateMeasLogLikelihood<UDT>::meas_id>>>,
          bmi::hashed_non_unique<
              bmi::tag<StateId>,
              bmi::member<StateMeasLogLikelihood<UDT>, StateId, &StateMeasLogLikelihood<UDT>::state_id>>>>;

  using MeasClutterIntensityContainer = bmi::multi_index_container<
      MeasClutterIntensity<UDT>,
      bmi::indexed_by<
          bmi::hashed_unique<bmi::tag<StateId>,
                             bmi::member<MeasClutterIntensity<UDT>, StateId, &MeasClutterIntensity<UDT>::meas_id>>>>;

  using StateUpdateContainer = bmi::multi_index_container<
      StateUpdate,
      bmi::indexed_by<
          bmi::ordered_unique<
              bmi::composite_key<StateUpdate,
                                 bmi::member<StateUpdate, StateId, &StateUpdate::prior_state_id>,
                                 bmi::member<StateUpdate, DataSourceContainerId, &StateUpdate::data_container_id>,
                                 bmi::member<StateUpdate, StateId, &StateUpdate::meas_id>>>,
          bmi::hashed_non_unique<bmi::tag<StateId>, bmi::member<StateUpdate, StateId, &StateUpdate::prior_state_id>>>>;

  using RelevantMeasContainer = bmi::multi_index_container<
      RelevantMeas,
      bmi::indexed_by<bmi::hashed_unique<bmi::composite_key<
                          RelevantMeas,
                          bmi::member<RelevantMeas, StateId, &RelevantMeas::prior_state_id>,
                          bmi::member<RelevantMeas, DataSourceContainerId, &RelevantMeas::data_container_id>>>,
                      bmi::hashed_non_unique<bmi::tag<StateId>,
                                             bmi::member<RelevantMeas, StateId, &RelevantMeas::prior_state_id>>>>;

  /// reduce the size of the containers
  void cut_data_containers_to_max_size();
  Database()
  {
    constexpr std::size_t RESERVE_SIZE = 1000000;
    data_containers.reserve(RESERVE_SIZE);
    states.reserve(RESERVE_SIZE);
    state_existences.reserve(RESERVE_SIZE);
    state_persistences.reserve(RESERVE_SIZE);
    state_detection_probabilities.reserve(RESERVE_SIZE);
    state_meas_log_likelihood.reserve(RESERVE_SIZE);
    meas_clutter_intensity.reserve(RESERVE_SIZE);
    relevant_meas.reserve(RESERVE_SIZE);
  }
  Duration keep_data = 5s;
  DataContainer data_containers;
  MeasDataContainer meas_to_data_container;
  StateContainer states;
  StateExistenceContainer state_existences;
  StatePersistenceContainer state_persistences;
  StateDetectionProbabilityContainer state_detection_probabilities;
  StateMeasLogLikelihoodContainer state_meas_log_likelihood;
  MeasClutterIntensityContainer meas_clutter_intensity;
  StateUpdateContainer state_updates;
  RelevantMeasContainer relevant_meas;
  void reset();
};

}  // namespace ttb
