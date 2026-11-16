#pragma once
#include "tracking_lib/TTBTypes/TTBTypes.h"
#include <any>
#include <concepts>

namespace ttb
{

template <class UDT>
struct BernoulliState;
class PredictionInformation;
class SensorInformation;
class Region;

template <class T>
concept CPTDistribution = requires(T dist) {
  { dist.to_string() } -> std::convertible_to<std::string>;
  { dist.id() } -> std::convertible_to<DistributionId>;
  { dist.prior_id() } -> std::convertible_to<DistributionId>;
  { dist.is_valid() } -> std::same_as<bool>;
  { dist.misc() } -> std::convertible_to<std::map<std::string, std::any>>;
};

template <class T, class UDT>
concept CPTCommonMODist = requires(T dist,
                                   Duration dt,
                                   DataSourceContainer<UDT> const& data_container,
                                   PredictionInformation const& prediction_info,
                                   std::vector<BernoulliState<UDT>> const& states) {
  { dist.predict(dt, prediction_info) } -> std::same_as<void>;
  { dist.is_valid() } -> std::same_as<bool>;
  { dist.get_estimate() } -> std::same_as<std::vector<BernoulliState<UDT>>>;
  { dist.add_tracks(states) } -> std::same_as<void>;
  { dist.meas_assignment_prob(data_container) } -> std::convertible_to<std::map<MeasurementId, UDT>>;
  { dist.all_tracks() } -> std::convertible_to<std::vector<StateId>>;
};

template <class T, class UDT>
concept CPTICUpdate = requires(T dist, DataSourceContainer<UDT> const& data_container) {
  { dist.update(data_container) } -> std::same_as<void>;
};

template <class T, class UDT>
concept CPTSingleObjectMeasurementModel = requires(T const& model,
                                                   DataSourceContainer<UDT> const& data_source_container,
                                                   State<UDT> const& state,
                                                   SensorInformation const& sensor_information,
                                                   Components const& comps,
                                                   Vector<UDT> const& state_vector,
                                                   std::optional<Region> const& fov) {
  { model.to_string() } -> std::convertible_to<std::string>;
  { model.id() } -> std::convertible_to<SourceId>;
  {
    model.calculate_innovation(data_source_container, state)
  } -> std::convertible_to<std::tuple<StateDetection<UDT>,
                                      std::vector<std::tuple<StateId, StateId, State<UDT>, UDT>>,
                                      std::vector<MeasClutter<UDT>>>>;
  { model.create_state(state) } -> std::convertible_to<std::optional<State<UDT>>>;
  { model.calculate_clutter_intensity(state, sensor_information) } -> std::convertible_to<UDT>;
  { model.meas_model_comps() } -> std::convertible_to<Components>;
  { model.manager() } -> std::convertible_to<TTBManager<UDT> const&>;
};

template <class UDT>
struct BirthTracks
{
  std::vector<BernoulliState<UDT>> birth_tracks;              /// the new birth track
  std::vector<std::optional<State<UDT>>> birth_measurements;  /// the measurement which is the origin of the new state
};

template <class T, class UDT>
concept CPTBirthModel = requires(T& model,
                                 DataSourceContainer<UDT> const& data_source_container,
                                 std::map<MeasurementId, UDT> const& meas_assign_probs,
                                 std::vector<State<UDT>> const& existing_tracks) {
  /// reset the birth model.
  { model.reset() } -> std::same_as<void>;
  { model.type() } -> std::convertible_to<BIRTH_MODEL_TYPE>;
  { model.manager() } -> std::convertible_to<TTBManager<UDT>*>;
  {
    model.getBirthTracks(data_source_container, meas_assign_probs, existing_tracks)
  } -> std::convertible_to<BirthTracks<UDT>>;
};

template <class T, class UDT>
concept CPTMSUpdate = requires(T dist, std::vector<DataSourceContainer<UDT>> const& data_containers) {
  { dist.update(data_containers) } -> std::same_as<void>;
};

template <class T, class UDT>
concept CPTHybridUpdate = CPTICUpdate<T, UDT> and CPTMSUpdate<T, UDT>;

template <class T, class UDT>
concept CPTSelfAssessableMODist = requires(T dist, DataSourceContainer<UDT> const& data_source_container) {
  { dist.clutter_distribution(data_source_container) } -> std::convertible_to<Vector<UDT>>;
  { dist.detection_distribution(data_source_container) } -> std::convertible_to<std::vector<Matrix<UDT>>>;
  {
    dist.bayes_factor_statistics_measurement_likelihood_without_detection(data_source_container)
  } -> std::convertible_to<std::map<std::pair<UDT, UDT>, UDT>>;
  {
    dist.bayes_factor_statistics_measurement_likelihood_without_clutter(data_source_container)
  } -> std::convertible_to<std::map<UDT, UDT>>;
  /// measurement likelihood
  { dist.bayes_factor_statistics_measurement_likelihood(data_source_container) } -> std::convertible_to<UDT>;
  { dist.surviving_distribution() } -> std::convertible_to<std::pair<std::vector<Matrix<UDT>>, Duration>>;
};

template <class T, class UDT>
concept CPTTracker = requires(T tracker, std::vector<DataSourceContainer<UDT>> const& data_containers) {
  { tracker.cycle(data_containers) } -> std::same_as<void>;
  { tracker.get_estimate() } -> std::same_as<std::vector<BernoulliState<UDT>>>;
  { tracker.all_tracks() } -> std::convertible_to<std::vector<StateId>>;
  { tracker.reset() } -> std::same_as<void>;
  { tracker.time() } -> std::same_as<Time>;
};

}  // namespace ttb
