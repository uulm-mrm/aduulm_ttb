#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/Measurements/SensorInformation.h"
#include "tracking_lib/States/State.h"
#include "tracking_lib/TTBTypes/Components.h"
#include "tracking_lib/TTBTypes/Params.h"

namespace ttb
{

class TTBManager;
class BaseStateModel;
class Measurement;
class MeasurementContainer;

/// This is the measurement model interface.
/// It provides the state update with the calcInnovation functionality, estimate the detectionProbability, estimate the
/// clutter and set upp new states based on measurements
class BaseMeasurementModel
{
public:
  virtual ~BaseMeasurementModel() = default;
  /// Access to the TTBManager.
  [[nodiscard]] virtual TTBManager* manager() const = 0;
  /// Calculates the innovation (kinematic state update/Kalman update) of the state with the measurements in the
  /// container and stores result in the InnovationMap of the dist
  [[nodiscard]] virtual Innovation calculateInnovation(MeasurementContainer const& measurements,
                                                       State const& dist) const = 0;
  /// String representation
  [[nodiscard]] virtual std::string toString() const = 0;
  /// The type of the measurement model.
  [[nodiscard]] virtual MEASUREMENT_MODEL_TYPE type() const = 0;
  /// The unique Measurement Model==Sensor/Data Source Id.
  /// Every sensor must have an unique id.
  [[nodiscard]] virtual MeasModelId id() const = 0;
  /// The maximal supported measurement components aka the measurement space.
  /// Note: The measurement model should also support measurements with only a subset of the components.
  [[nodiscard]] virtual Components const& meas_model_comps() const = 0;
  /// Create a new State based on the given measurement.
  /// If the transformation from the measurement space to the state space fails and force is not true, return nullopt.
  /// If force is set to true, a State is always created using the defaultValues of the measurement model if needed
  [[nodiscard]] virtual std::optional<State> createState(Measurement const& meas, bool force = false) const = 0;
  /// Creates a new State based on possible birth candidates from the previous time step
  /// and a measurement
  [[nodiscard]] virtual std::optional<State> initStage2(std::vector<Measurement> const& oldMeasurements,
                                                        Measurement const& measurement,
                                                        SensorInformation const& sensorInfo) const = 0;
  /// Return the detection probability of the given measurement vector.
  [[nodiscard]] virtual Probability getDetectionProbability(Vector const& predictedMeasurement,
                                                            Components const& comps,
                                                            std::optional<FieldOfView> const& fov) const = 0;
  /// Return the clutter intensity for a given measurement.
  [[nodiscard]] virtual double getClutterIntensity(Measurement const& meas,
                                                   SensorInformation const& sensorInformation) const = 0;

  struct DefaultVal
  {
    std::optional<double> mean;
    std::optional<double> var;
  };
  /// Default values for the measurement model.
  [[nodiscard]] virtual DefaultVal defaultVal(COMPONENT comp, CLASS type) const = 0;
};
/// Helper to read the default values from the params.
BaseMeasurementModel::DefaultVal default_val(std::vector<DefaultValuesParams> const& params,
                                             COMPONENT comp,
                                             CLASS clazz);
/// Helper to read the default values from the params.
BaseMeasurementModel::DefaultVal default_val(DefaultValuesParams const& params, COMPONENT comp);

}  // namespace ttb
