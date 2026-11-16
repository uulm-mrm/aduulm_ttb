#pragma once

#include "tracking_lib/TTBTypes/Concepts.h"
#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/MeasurementModels/GaussianMeasurementModel.h"
#include "tracking_lib/TTBTypes/TTBTypes.h"
// #include "tracking_lib/Measurements/SensorInformation.h"
#include "tracking_lib/States/State.h"
#include "tracking_lib/TTBTypes/Params.h"

namespace ttb
{

template <class UDT>
class SingleObjectMeasurementModel
{
public:
  using AllModels = std::variant<GaussianMeasurementModel<UDT>>;

  std::string to_string() const
  {
    return std::visit([&](auto const& model) { return model.toString(); }, _meas_model);
  }
  SourceId id() const
  {
    return std::visit([&](auto const& model) { return model.id(); }, _meas_model);
  }
  [[nodiscard]] std::tuple<StateDetection<UDT>,
                           std::vector<std::tuple<StateId, StateId, State<UDT>, UDT>>,
                           std::vector<MeasClutter<UDT>>>
  calculate_innovation(DataSourceContainer<UDT> const& measurements, State<UDT> const& dist) const
  {
    return std::visit([&](auto const& model) { return model.calculateInnovation(measurements, dist); }, _meas_model);
  }
  [[nodiscard]] std::tuple<UDT, bool> detection_probability(DataSourceContainer<UDT> const& meas_container,
                                                            State<UDT> const& state) const
  {
    return std::visit([&](auto const& model) { return model.detection_probability(meas_container, state); },
                      _meas_model);
  }
  [[nodiscard]] TTBManager<UDT> const& manager() const
  {
    return _manager;
  }
  [[nodiscard]] Components const& meas_model_comps() const
  {
    return std::visit([&](auto const& model) -> Components const& { return model.meas_model_comps(); }, _meas_model);
  }
  [[nodiscard]] UDT calculate_detection_probability(Vector<UDT> const& predictedMeasurement,
                                                    Components const& comps,
                                                    std::optional<Region> const& fov) const
  {
    return std::visit(
        [&](auto const& model) { return model.getDetectionProbability(predictedMeasurement, comps, fov); },
        _meas_model);
  }
  /// Calculates the clutter intensity for a given Measurement. Needed for converting the clutter rate in case
  [[nodiscard]] UDT calculate_clutter_intensity(State<UDT> const& meas,
                                                SensorInformation const& sensorInformation) const
  {
    return std::visit([&](auto const& model) { return model.getClutterIntensity(meas, sensorInformation); },
                      _meas_model);
  }
  [[nodiscard]] std::optional<State<UDT>> create_state(State<UDT> const& meas, bool force = false) const
  {
    return std::visit([&](auto const& model) { return model.createState(meas, force); }, _meas_model);
  }
  TTBManager<UDT> const& _manager;
  AllModels _meas_model;
};

static_assert(CPTSingleObjectMeasurementModel<SingleObjectMeasurementModel<double>, double>);
static_assert(CPTSingleObjectMeasurementModel<SingleObjectMeasurementModel<AD>, AD>);

}  // namespace ttb
