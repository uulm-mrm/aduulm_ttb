#pragma once

#include "tracking_lib/StateModels/BaseStateModel.h"

#include <random>

namespace ttb
{

/// Information for the update of one! Gaussian state distribution with one! Gaussian measurement distribution
template <class UDT>
struct GaussianUpdateInfo
{
  Matrix<UDT> R;                                     ///< measurement covariance
  Matrix<UDT> S;                                     ///< R + R_pred
  Matrix<UDT> iS;                                    ///< S^{-1}
  Vector<UDT> residuum;                              ///< the residuum of z_pred and the measurement vector
  UDT MHD2 = std::numeric_limits<UDT>::quiet_NaN();  ///< the MHD distance between z_pred and the measurement vector
  bool isGated = false;                              ///< is the measurement vector outside the gate?
};

/// The predicted measurement of one! Gaussian state distribution
template <class UDT>
struct PredictedMeasurement
{
  Vector<UDT> z_pred;            ///< the predicted measurement mean
  Matrix<UDT> R_pred;            ///< the predicted measurement cov
  std::optional<Matrix<UDT>> H;  ///< the linear measurement matrix, transformation matrix from state to measurement
                                 ///< space
  std::optional<Matrix<UDT>> T;  ///< the cross covariance P_{xz} between state and measurement (only if not
                                 ///< H.has_value())
};

/// GaussianMeasurementModel
/// -> State Distribution: (Mixture of) Gauss Distributions
/// -> Measurement Distribution: (Mixture of) Gauss Distributions
/// -> Linear and nonlinear state to measurement space transformations using the general transformations defined in
/// Transformations/Transformation.h
template <class UDT>
class GaussianMeasurementModel
{
public:
  GaussianMeasurementModel(TTBManager<UDT> const* manager, SourceId id);
  /// calculates the state update with a Kalman update in the linear and an Unscented update in the non-linear case for
  /// the given state and all measurements in the MeasurementContainer
  [[nodiscard]] std::tuple<StateDetection<UDT>,
                           std::vector<std::tuple<StateId, StateId, State<UDT>, UDT>>,
                           std::vector<MeasClutter<UDT>>>
  calculateInnovation(DataSourceContainer<UDT> const& measurements, State<UDT> const& dist) const;
  [[nodiscard]] Vector<UDT> give_init_AD_detection_model_params() const;
  [[nodiscard]] std::tuple<Probability<UDT>, bool> detection_probability(DataSourceContainer<UDT> const& meas_container,
                                                                         State<UDT> const& state) const;
  using PredMeasCache =
      std::map<std::tuple<std::vector<COMPONENT>, REFERENCE_POINT, DistributionId>, PredictedMeasurement<UDT>>;
  [[nodiscard]] PredMeasCache create_predicted_measurement(DataSourceContainer<UDT> const& meas_container,
                                                           State<UDT> const& state) const;
  [[nodiscard]] MEASUREMENT_MODEL_TYPE type() const noexcept;
  [[nodiscard]] std::string toString() const;
  [[nodiscard]] SourceId id() const;
  [[nodiscard]] TTBManager<UDT> const* manager() const;
  [[nodiscard]] Components const& meas_model_comps() const;
  [[nodiscard]] UDT getDetectionProbability(Vector<UDT> const& predictedMeasurement,
                                            Components const& comps,
                                            std::optional<Region> const& fov) const;
  /// Calculates the clutter intensity for a given Measurement. Needed for converting the clutter rate in case
  [[nodiscard]] UDT getClutterIntensity(State<UDT> const& meas, SensorInformation const& sensorInformation) const;
  [[nodiscard]] std::optional<State<UDT>> createState(State<UDT> const& meas, bool force = false) const;
  /// Constraints for new states
  [[nodiscard]] std::optional<UDT> getConstraints(CLASS type, COMPONENT comp) const;
  /// The gating MHD distance threshold
  [[nodiscard]] UDT gatingMHD2Distance(std::size_t dimMeasSpace) const;
  struct Update
  {
    Vector<UDT> mean;
    Matrix<UDT> var;
  };
  /// Computes the Linear Kalman update
  /// This method implements the standard Kalman filter update in Joseph-Form and returns the updated state and
  /// covariance, see
  /// https://en.wikipedia.org/wiki/Kalman_filter
  [[nodiscard]] Update calculateUpdate(Vector<UDT> const& x,
                                       Matrix<UDT> const& P,
                                       Components const& stateComps,
                                       PredictedMeasurement<UDT> const& pred_meas,
                                       GaussianUpdateInfo<UDT> const& info) const;
  /// Computes the residuum z_pred-z
  [[nodiscard]] static Vector<UDT> getResiduum(Vector<UDT> const& z_pred,
                                               Vector<UDT> const& z,
                                               std::optional<Index> rot_Ind = std::nullopt);

  /// Calculates the measurement likelihood g(z|x)
  [[nodiscard]] static UDT getMeasurementLikelihood(UDT mhd2, UDT gatingConf, Matrix<UDT> const& S);
  /// Calculates the observable reference points
  /// Calculation is based on the object state x and incorporates the mounting position of
  /// the sensor
  [[nodiscard]] std::vector<REFERENCE_POINT> observableReferencePoints(Vector<UDT> const& state,
                                                                       Components const& comps,
                                                                       State<UDT> const& meas,
                                                                       REFERENCE_POINT meas_ref_point,
                                                                       SensorInformation const& sensorInfo) const;
  TTBManager<UDT> const* _manager;
  SourceId _id;
  Components _model_comps;
  Index n_elems = 64;
  Index n_center = n_elems * 2;
  Index n_log_b = n_elems;
  Index n_w = n_elems;
  Index n_b = 1;
  Index n_params = n_center + n_log_b + n_w + n_b;
  Eigen::Vector<UDT, -1> _detection_model_params;
};

}  // namespace ttb
