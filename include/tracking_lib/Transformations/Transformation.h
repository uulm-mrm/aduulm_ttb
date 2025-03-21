#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/TTBTypes/Components.h"
// #include "TrackingToolbox3/TTBHelpers/CommonFuncs.h"
#include "tracking_lib/Distributions/GaussianDistribution.h"
#include "tracking_lib/Measurements/Measurement.h"

namespace ttb
{

namespace SigmaPoints
{

/// create sigma points based on mean value and covariance matrix
[[nodiscard]] std::pair<Matrix, Vector> createSigmaPoints(Vector const& x, Matrix const& P);

/// Calculates the residuals between the state x and a vector of sigma points
[[nodiscard]] Matrix calcResiduum(Vector const& x, Matrix const& sigmaPoints);

/// Computes mean and covariance of some sigma Points
[[nodiscard]] std::pair<Vector, Matrix> SigmaPoints2Moments(Matrix const& sp,
                                                            Vector const& w,
                                                            std::optional<Index> rot_ind = std::nullopt);

/// Computes mean and covariance based on the sigma points
///        other than SigmaPoints2Moments, this method evaluates the covariance about the projected mean
///        see
///            Julier et. al, "A new method for the nonlinear transformation of means and covariances in filters and
///            estimators" https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=847726 doi: 10.1109/9.847726
[[nodiscard]] std::pair<Vector, Matrix> modifiedSigmaPoints2Moments(Matrix const& sp,
                                                                    Vector const& w,
                                                                    std::optional<Index> rot_ind = std::nullopt);

/// Computes cross-covariance between transformed and non-transformed sigma points
///@param sp vector of non-transformed sigma points
///@param w weights of the sigma points
///@param Xhat state vector
///@param res matrix of residuals between state vector and sigma points
[[nodiscard]] Matrix
SigmaPoints2CrossCovariance(Matrix const& sp, Vector const& weights, Vector const& Xhat, Matrix const& res);
}  // namespace SigmaPoints

namespace transformation
{

struct Transformed
{
  Vector mean;                        ///< mean of transformed gaussian
  Matrix cov;                         ///< var of transformed
  std::optional<Matrix> cross_cov{};  ///< cross_covariance if non-linear transform
  std::optional<Matrix> T{};          ///< transformation matrix if linear transform
};

/// Calculates the mean, variance and cross-covariance of a unscented transformed gaussian
///
/// this implementation follows the ideas of
///      Julier, Simon J., and Jeffrey K. Uhlmann. "New extension of the Kalman filter to nonlinear systems."
///      Signal processing, sensor fusion, and target recognition VI. Vol. 3068. International Society for Optics and
/// Photonics, 1997. https://doi.org/10.1117/12.280797
///
///      Wan, Eric A., and Rudolph Van Der Merwe. "The unscented Kalman filter for nonlinear estimation."
///      Proceedings of the IEEE 2000 Adaptive Systems for Signal Processing, Communications, and Control Symposium
///      (Cat.
/// No. 00EX373). Ieee, 2000. https://ieeexplore.ieee.org/abstract/document/882463
///
///      Wan, Eric A., and Rudolph Van Der Merwe. "The unscented Kalman filter." Kalman filtering and neural
/// networks 5.2007 (2001): 221-280. https://pdfs.semanticscholar.org/735c/aa8c394975ab26aeb9c911e6187f7bef4a24.pdf
template <class F>
[[nodiscard]] std::optional<Transformed> unscentTransform(Vector const& mean,
                                                          Matrix const& cov,
                                                          F transformation,
                                                          std::optional<Index> to_rot_ind = std::nullopt)
{
  LOG_DEB("unscent Trafo");
  auto [sigmaPoints, weights]{ SigmaPoints::createSigmaPoints(mean, cov) };
  Matrix const residuals = SigmaPoints::calcResiduum(mean, sigmaPoints);
  // transform each sigma point
  LOG_DEB("sigmaPoints" << sigmaPoints.col(0));
  auto transformed = transformation(std::move(sigmaPoints.col(0)));
  if (not transformed.has_value())
  {
    LOG_DEB("Can not unscent transform thtat");
    LOG_DEB("Return nullopt");
    return {};
  }
  Matrix transformedSigmaPoints(transformed.value().rows(), sigmaPoints.cols());
  {
    transformedSigmaPoints.col(0) = std::move(transformed.value());
    for (unsigned int i = 1; i < sigmaPoints.cols(); ++i)
    {
      auto trans = transformation(std::move(sigmaPoints.col(i)));
      if (trans.has_value())
      {
        transformedSigmaPoints.col(i) = std::move(trans.value());
      }
      else
      {
        return std::nullopt;
      }
    }
  }
  auto moments = [&] {
    if (weights(0) >= 0)
    {
      return SigmaPoints::SigmaPoints2Moments(transformedSigmaPoints, weights, to_rot_ind);
    }
    return SigmaPoints::modifiedSigmaPoints2Moments(transformedSigmaPoints, weights, to_rot_ind);
  }();
  Matrix Pcross = SigmaPoints::SigmaPoints2CrossCovariance(transformedSigmaPoints, weights, moments.first, residuals);
  return Transformed{ .mean = std::move(moments.first),
                      .cov = std::move(moments.second),
                      .cross_cov = std::move(Pcross) };
}

/// calculates the mean, variance and cross-covariance of a unscented transformed gaussian with explicit noise
/// matrix
///
/// this implementation follows the ideas of
///      Julier, Simon J., and Jeffrey K. Uhlmann. "New extension of the Kalman filter to nonlinear systems."
///      Signal processing, sensor fusion, and target recognition VI. Vol. 3068. International Society for Optics and
/// Photonics, 1997. https://doi.org/10.1117/12.280797
///
///      Wan, Eric A., and Rudolph Van Der Merwe. "The unscented Kalman filter for nonlinear estimation."
///      Proceedings of the IEEE 2000 Adaptive Systems for Signal Processing, Communications, and Control Symposium
///      (Cat.
/// No. 00EX373). Ieee, 2000. https://ieeexplore.ieee.org/abstract/document/882463
///
///      Wan, Eric A., and Rudolph Van Der Merwe. "The unscented Kalman filter." Kalman filtering and neural
/// networks 5.2007 (2001): 221-280. https://onlinelibrary.wiley.com/doi/10.1002/0471221546.ch7
template <class F>
[[nodiscard]] std::optional<Transformed> unscentTransform(Vector const& mean,
                                                          Matrix const& cov,
                                                          Matrix const& noise_cov,
                                                          F transformation,
                                                          std::optional<Index> to_rot_ind = std::nullopt)
{
  // for augmented state vector
  std::size_t const numNoiseVars = noise_cov.cols();
  std::size_t const augmentedSize = numNoiseVars + mean.rows();
  Vector augmentedState = Vector::Zero(augmentedSize);
  augmentedState(Eigen::seq(0, mean.rows() - 1)) = mean;
  // concatenate covariance matrices
  Matrix augmentedCov = Matrix::Zero(augmentedSize, augmentedSize);
  augmentedCov(Eigen::seq(0, mean.rows() - 1), Eigen::seq(0, mean.rows() - 1)) = cov;
  augmentedCov(Eigen::seq(mean.rows(), augmentedSize - 1), Eigen::seq(mean.rows(), augmentedSize - 1)) = noise_cov;

  auto [sigmaPoints, weights]{ SigmaPoints::createSigmaPoints(augmentedState, augmentedCov) };

  Matrix const residuals = SigmaPoints::calcResiduum(mean, sigmaPoints(Eigen::seq(0, mean.rows() - 1), Eigen::all));
  std::optional<Vector> transformed =
      transformation(std::move(sigmaPoints(Eigen::seq(0, mean.rows() - 1), 0)),
                     std::move(sigmaPoints(Eigen::seq(mean.rows(), augmentedSize - 1), 0)));
  if (not transformed.has_value())
  {
    return std::nullopt;
  }
  Matrix transformedSigmaPoints(transformed.value().rows(), sigmaPoints.cols());
  {
    transformedSigmaPoints.col(0) = std::move(transformed.value());
    for (std::size_t i = 0; i < sigmaPoints.cols(); ++i)
    {
      auto trans = transformation(std::move(sigmaPoints(Eigen::seq(0, mean.rows() - 1), i)),
                                  std::move(sigmaPoints(Eigen::seq(mean.rows(), augmentedSize - 1), i)));
      if (trans.has_value())
      {
        transformedSigmaPoints.col(i) = std::move(trans.value());
      }
      else
      {
        return std::nullopt;
      }
    }
  }
  auto moments{ [&] {
    if (weights(0) >= 0)
    {
      return SigmaPoints::SigmaPoints2Moments(transformedSigmaPoints, weights, to_rot_ind);
    }
    return SigmaPoints::modifiedSigmaPoints2Moments(transformedSigmaPoints, weights, to_rot_ind);
  }() };
  Matrix Pcross{ SigmaPoints::SigmaPoints2CrossCovariance(transformedSigmaPoints, weights, moments.first, residuals) };
  return Transformed{ .mean = std::move(moments.first),
                      .cov = std::move(moments.second),
                      .cross_cov = std::move(Pcross) };
}

/// (linear) transformation Matrix that transform the given Components if possible
[[nodiscard]] std::optional<Matrix> transformationMatrix(Components const& from, Components const& to);

/// nonlinear! (in general) transformation to the reference point
[[nodiscard]] Vector
transform(Vector const& state, Components const& state_comps, REFERENCE_POINT from, REFERENCE_POINT to);

/// most general (possible non-linear) Transformation of a Vector between the given Components
[[nodiscard]] std::optional<Vector> transform(Vector const& state,
                                              Components const& from,
                                              Components const& to,
                                              SE3Trafo const& sensor_pose = SE3Trafo::Identity());

struct TransformOptions
{
  bool assume_orient_in_velocity_direction = false;
};
/// Return all possible Components in which the given Components can be transformed into
/// first value are all Components which can linear transformed,
/// second value are all Components which can be transformed with a non linear transformation (Including the linear
/// ones)
struct TransformableComps
{
  Components linear;
  Components all;
};
[[nodiscard]] TransformableComps transformableComps(Components const& comps, TransformOptions const& options);

/// general (possible non-linear) transformation
[[nodiscard]] std::optional<Transformed> transform(Vector const& state,
                                                   Matrix const& cov,
                                                   Components const& from,
                                                   Components const& to,
                                                   REFERENCE_POINT from_rp,
                                                   REFERENCE_POINT to_rp,
                                                   SE3Trafo const& sensor_pose = SE3Trafo::Identity());

/// transform between components
[[nodiscard]] std::optional<Transformed> transform(Vector const& state,
                                                   Matrix const& cov,
                                                   Components const& from,
                                                   Components const& to,
                                                   SE3Trafo const& sensor_pose = SE3Trafo::Identity());

/// transform the reference point
[[nodiscard]] std::optional<Transformed> transform(Vector const& state,
                                                   Matrix const& cov,
                                                   Components const& from,
                                                   REFERENCE_POINT from_rp,
                                                   REFERENCE_POINT to_rp,
                                                   SE3Trafo const& sensor_pose = SE3Trafo::Identity());

/// transform to the given coordinate system
/// translation: state, cov frame -> new frame (xyz-pos)
/// rotation:    state, cov frame -> new frame
[[nodiscard]] std::optional<Transformed> transform(Vector const& state,
                                                   Matrix const& cov,
                                                   Components const& from,
                                                   Vector3 const& translation,
                                                   Matrix33 const& rotation);

/// flip the orientation of an Measurement by 180°
[[nodiscard]] Measurement flipMeasurementOrientation(Measurement const& meas);

/// flip the orientation of an Distribution by 180°
[[nodiscard]] std::unique_ptr<BaseDistribution> flipOrientation(BaseDistribution const& dist, Components const& comps);

/// transform a Distribution to another StateModel
[[nodiscard]] std::unique_ptr<BaseDistribution> transform(BaseDistribution const& dist,
                                                          BaseStateModel const& sourceStateModel,
                                                          BaseStateModel const& targetStateModel);

}  // namespace transformation

}  // namespace ttb
