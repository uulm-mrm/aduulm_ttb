#pragma once

#include "tracking_lib/MeasurementModels/BaseMeasurementModel.h"
#include "tracking_lib/StateModels/BaseStateModel.h"
#include "tracking_lib/Measurements/Measurement.h"
#include "tracking_lib/Distributions/BaseDistribution.h"
#include "tracking_lib/OcclusionModels/BaseOcclusionModel.h"

namespace ttb
{

/// Information for the update of one! Gaussian state distribution with one! Gaussian measurement distribution
struct GaussianUpdateInfo
{
  Matrix R;         ///< measurement covariance
  Matrix S;         ///< R + R_pred
  Matrix iS;        ///< S^{-1}
  Vector residuum;  ///< the residuum of z_pred and the measurement vector
  double MHD2;      ///< the MHD distance between z_pred and the measurement vector
  bool isGated;     ///< is the measurement vector outside the gate?
};

/// The predicted measurement of one! Gaussian state distribution
struct PredictedMeasurement
{
  Vector z_pred;            ///< the predicted measurement mean
  Matrix R_pred;            ///< the predicted measurement cov
  std::optional<Matrix> H;  ///< the linear measurement matrix, transformation matrix from state to measurement space
  std::optional<Matrix> T;  ///< the cross covariance P_{xz} between state and measurement (only if not H.has_value())
};

/// GaussianMeasurementModel
/// -> State Distribution: (Mixture of) Gauss Distributions
/// -> Measurement Distribution: (Mixture of) Gauss Distributions
/// -> Linear and nonlinear state to measurement space transformations using the general transformations defined in
/// Transformations/Transformation.h
class GaussianMeasurementModel : public BaseMeasurementModel
{
public:
  GaussianMeasurementModel(TTBManager* manager, MeasModelId id);
  ~GaussianMeasurementModel() override = default;
  /// calculates the state update with a Kalman update in the linear and an Unscented update in the non-linear case for
  /// the given state and all measurements in the MeasurementContainer
  [[nodiscard]] Innovation calculateInnovation(MeasurementContainer const& measurements, State const& dist) const final;
  [[nodiscard]] std::tuple<Probability, bool> detection_probability(MeasurementContainer const& meas_container,
                                                                    State const& state) const;
  using PredMeasCache =
      std::map<std::tuple<std::vector<COMPONENT>, REFERENCE_POINT, DistributionId>, PredictedMeasurement>;
  [[nodiscard]] PredMeasCache create_predicted_measurement(MeasurementContainer const& meas_container,
                                                           State const& state) const;
  [[nodiscard]] MEASUREMENT_MODEL_TYPE type() const noexcept override;
  [[nodiscard]] std::string toString() const override;
  [[nodiscard]] MeasModelId id() const override;
  [[nodiscard]] TTBManager* manager() const override;
  [[nodiscard]] Components const& meas_model_comps() const override;
  [[nodiscard]] DefaultVal defaultVal(COMPONENT comp, CLASS type) const override;
  [[nodiscard]] double getDetectionProbability(Vector const& predictedMeasurement,
                                               Components const& comps,
                                               std::optional<FieldOfView> const& fov) const override;
  /// Calculates the clutter intensity for a given Measurement. Needed for converting the clutter rate in case
  [[nodiscard]] double getClutterIntensity(Measurement const& meas,
                                           SensorInformation const& sensorInformation) const override;
  [[nodiscard]] std::optional<State> createState(Measurement const& meas, bool force = false) const override;
  [[nodiscard]] std::optional<State> initStage2(std::vector<Measurement> const& oldMeasurements,
                                                Measurement const& measurement,
                                                SensorInformation const& sensorInfo) const override;
  /// Constraints for new states
  [[nodiscard]] std::optional<double> getConstraints(CLASS type, COMPONENT comp) const;
  /// The gating MHD distance threshold
  [[nodiscard]] double gatingMHD2Distance(std::size_t dimMeasSpace) const;
  struct Update
  {
    Vector mean;
    Matrix var;
  };
  /// Computes the Linear Kalman update
  /// This method implements the standard Kalman filter update in Joseph-Form and returns the updated state and
  /// covariance, see
  /// https://en.wikipedia.org/wiki/Kalman_filter
  [[nodiscard]] Update calculateUpdate(Vector const& x,
                                       Matrix const& P,
                                       Components const& stateComps,
                                       PredictedMeasurement const& pred_meas,
                                       GaussianUpdateInfo const& info) const;
  /// Computes the residuum z_pred-z
  [[nodiscard]] static Vector getResiduum(Vector const& z_pred,
                                          Vector const& z,
                                          std::optional<Index> rot_Ind = std::nullopt);
  /// Calculates the measurement specific values needed for the update.
  [[nodiscard]] GaussianUpdateInfo calcUpdateInfo(Measurement const& meas,
                                                  BaseDistribution const& meas_dist,
                                                  PredictedMeasurement const& pred_meas,
                                                  Components const& measComps) const;
  /// Calculates the measurement likelihood g(z|x)
  [[nodiscard]] static double getMeasurementLikelihood(double mhd2, double gatingConf, Matrix const& S);
  /// Calculates the observable reference points
  /// Calculation is based on the object state x and incorporates the mounting position of
  /// the sensor
  [[nodiscard]] std::vector<REFERENCE_POINT> observableReferencePoints(Vector const& state,
                                                                       Components const& comps,
                                                                       Measurement const& meas,
                                                                       BaseDistribution const& meas_dist,
                                                                       SensorInformation const& sensorInfo) const;
  TTBManager* _manager;
  MeasModelId _id;
  Components _model_comps;
  std::unique_ptr<BaseOcclusionModel> _occlusion_model;
};

}  // namespace ttb