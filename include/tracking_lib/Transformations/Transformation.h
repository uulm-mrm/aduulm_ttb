#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/TTBTypes/Components.h"
#include "tracking_lib/Distributions/GaussianDistribution.h"
#include "tracking_lib/Distributions/Distribution.h"

#include <aduulm_logger/aduulm_logger.hpp>
#include <boost/accumulators/framework/parameters/weights.hpp>
#include <tracy/tracy/Tracy.hpp>

namespace ttb
{

namespace transformation
{
constexpr auto tracy_color = tracy::Color::RoyalBlue;

namespace reference_point
{

/// rotation matrix for the yaw angle only
[[nodiscard]] inline Matrix<double> rotationMatrix(double yaw)
{
  Matrix<double> rot(2, 2);
  double c_y = cos(yaw);
  double s_y = sin(yaw);

  rot(0, 0) = c_y;
  rot(0, 1) = -s_y;
  rot(1, 0) = s_y;
  rot(1, 1) = c_y;
  return rot;
}

[[nodiscard]] inline Vector<double> leverarmRPtoCenterCenter(REFERENCE_POINT fromRP, double l, double w)
{
  Vector<double> leverarm = Vector<double>::Zero(2);
  switch (fromRP)
  {
    case REFERENCE_POINT::BACK:
    {
      leverarm(0) += l;
      break;
    }
    case REFERENCE_POINT::FRONT:
    {
      leverarm(0) -= l;
      break;
    }
    case REFERENCE_POINT::BACK_LEFT:
    {
      leverarm(0) += l;
      leverarm(1) -= w;
      break;
    }
    case REFERENCE_POINT::FRONT_RIGHT:
    {
      leverarm(0) -= l;
      leverarm(1) += w;
      break;
    }
    case REFERENCE_POINT::BACK_RIGHT:
    {
      leverarm(0) += l;
      leverarm(1) += w;
      break;
    }
    case REFERENCE_POINT::FRONT_LEFT:
    {
      leverarm(0) -= l;
      leverarm(1) -= w;
      break;
    }
    case REFERENCE_POINT::LEFT:
    {
      leverarm(1) -= w;
      break;
    }
    case REFERENCE_POINT::RIGHT:
    {
      leverarm(1) += w;
      break;
    }
    case REFERENCE_POINT::CENTER:
    {
      break;
    }
    default:
    {
      LOG_ERR("Unhandled referencepoint: " << to_string(fromRP));
      assert(false);
      break;
    }
  }
  return leverarm;
}

[[nodiscard]] inline Vector<double> calculateLeverarm(REFERENCE_POINT fromRP, REFERENCE_POINT toRP, double l, double w)
{
  Vector<double> leverarm = leverarmRPtoCenterCenter(fromRP, l / 2., w / 2.);
  leverarm -= leverarmRPtoCenterCenter(toRP, l / 2., w / 2.);
  return leverarm;
}

template <class UDT>
[[nodiscard]] Vector<UDT> transform(Vector<UDT> const& x_in, REFERENCE_POINT from, REFERENCE_POINT to)
{
  Vector<UDT> x_out = x_in;
  if (from != to)
  {
    Vector<double> leverarm = calculateLeverarm(from, to, to_val(x_in(3)), to_val(x_in(4)));

    Matrix<double> rot = rotationMatrix(to_val(x_in(2)));
    rot = rot * leverarm;
    x_out(0) += rot(0, 0);
    x_out(1) += rot(1, 0);
  }
  return x_out;
}

}  // namespace reference_point

/// create sigma points based on mean value and covariance matrix
template <class UDT>
struct SigmaPoints
{
  Matrix<UDT> sigma_points;
  Vector<UDT> weights;
};
template <class UDT>
[[nodiscard]] std::optional<SigmaPoints<UDT>> createSigmaPoints(Vector<UDT> const& x, Matrix<UDT> const& P);

/// Calculates the residuals between the state x and a vector of sigma points
template <class UDT>
[[nodiscard]] Matrix<UDT> calcResiduum(Vector<UDT> const& x, Matrix<UDT> const& sigmaPoints)
{
  Matrix<UDT> res_eigen = sigmaPoints.colwise() - x;
  return res_eigen;
}

/// Computes mean and covariance of some sigma Points
template <class UDT>
[[nodiscard]] std::pair<Vector<UDT>, Matrix<UDT>> SigmaPoints2Moments(Matrix<UDT> const& sp,
                                                                      Vector<UDT> const& weights,
                                                                      std::optional<Index> rot_ind = std::nullopt)
{
  // ZoneScopedNC("transformation::moments", tracy_color);
  Vector<UDT> mean = sp * weights;
  if (rot_ind.has_value())
  {
    Index const idx = rot_ind.value();
    UDT const center = sp(idx, 0);
    mean(idx) = center;
    for (Index i = 1; i < sp.cols(); ++i)
    {
      UDT delta = sp(idx, i) - center;
      angles::normalizeAngle(delta);
      mean(idx) += weights(i) * delta;
    }
    angles::normalizeAngle(mean(idx));
  }
  Matrix<UDT> P = Matrix<UDT>::Zero(sp.rows(), sp.rows());
  for (unsigned int i = 0; i < sp.cols(); i++)
  {
    Vector<UDT> res = sp.col(i) - mean;
    P += weights(i) * (res * res.transpose());
  }

  return { std::move(mean), std::move(P) };
}

/// Computes mean and covariance based on the sigma points
///        other than SigmaPoints2Moments, this method evaluates the covariance about the projected mean
///        see
///            Julier et. al, "A new method for the nonlinear transformation of means and covariances in filters and
///            estimators" https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=847726 doi: 10.1109/9.84772
template <class UDT>
[[nodiscard]] std::pair<Vector<UDT>, Matrix<UDT>>
modifiedSigmaPoints2Moments(Matrix<UDT> const& sp,
                            Vector<UDT> const& weights,
                            std::optional<Index> rot_ind = std::nullopt)
{
  // ZoneScopedNC("transformation::moments", tracy_color);
  Vector<UDT> mean = sp * weights;
  if (rot_ind.has_value())
  {
    mean(rot_ind.value()) = sp(rot_ind.value(), 0);
  }
  Matrix<UDT> P = Matrix<UDT>::Zero(sp.rows(), sp.rows());
  for (Index i = 1; i < sp.cols(); i++)
  {
    Vector<UDT> res = sp.col(i) - sp.col(0);
    if (rot_ind.has_value())
    {
      angles::normalizeAngle(res(rot_ind.value()));
      mean(rot_ind.value()) += weights(i) * res(rot_ind.value());
    }
    Vector<UDT> const non_normalized_res = sp.col(i) - sp.col(0);
    P += weights(i) * (non_normalized_res * non_normalized_res.transpose());
  }
  if (rot_ind.has_value())
  {
    angles::normalizeAngle(mean(rot_ind.value()));
  }
  return { std::move(mean), std::move(P) };
}

/// Computes cross-covariance between transformed and non-transformed sigma points
template <class UDT>
[[nodiscard]] Matrix<UDT> SigmaPoints2CrossCovariance(Matrix<UDT> const& sp,
                                                      Vector<UDT> const& weights,
                                                      Vector<UDT> const& Xhat,
                                                      Matrix<UDT> const& res,
                                                      std::optional<Index> rot_ind = std::nullopt)
{
  // ZoneScopedNC("transformation::crossCovariance", tracy_color);
  Matrix<UDT> cross_cov = Matrix<UDT>::Zero(res.rows(), sp.rows());
  for (unsigned int i = 1; i < sp.cols(); i++)
  {
    Vector<UDT> tmp = sp.col(i) - Xhat;
    cross_cov += weights(i) * (res.col(i) * tmp.transpose());
  }
  return cross_cov;
}

template <class UDT>
struct Transformed
{
  Vector<UDT> mean;                        ///< mean of transformed gaussian
  Matrix<UDT> cov;                         ///< var of transformed
  std::optional<Matrix<UDT>> cross_cov{};  ///< cross_covariance if non-linear transform
  std::optional<Matrix<UDT>> T{};          ///< transformation matrix if linear transform
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
template <class F, class UDT>
[[nodiscard]] std::optional<Transformed<UDT>> unscentTransform(Vector<UDT> const& mean,
                                                               Matrix<UDT> const& cov,
                                                               F transformation,
                                                               std::optional<Index> to_rot_ind = std::nullopt)
{
  assert(mean.rows() == cov.rows() and mean.rows() == cov.cols() and mean.rows() > 0);
  std::optional<SigmaPoints<UDT>> sigma_points = createSigmaPoints<UDT>(mean, cov);
  if (not sigma_points.has_value())
  {
    return {};
  }
  Matrix<UDT> const residuals = calcResiduum(mean, sigma_points.value().sigma_points);
  // transform each sigma point
  std::optional<Vector<UDT>> transformed = transformation(sigma_points.value().sigma_points.col(0));
  if (not transformed.has_value())
  {
    return {};
  }
  Matrix<UDT> transformedSigmaPoints(transformed.value().rows(), sigma_points.value().sigma_points.cols());
  {
    transformedSigmaPoints.col(0) = transformed.value();
    for (unsigned int i = 1; i < sigma_points.value().sigma_points.cols(); ++i)
    {
      std::optional<Vector<UDT>> trans = transformation(sigma_points.value().sigma_points.col(i));

      if (trans.has_value())
      {
        transformedSigmaPoints.col(i) = trans.value();
      }
      else
      {
        return {};
      }
    }
  }
  auto moments = [&] {
    if (sigma_points.value().weights(0) >= 0)
    {
      return SigmaPoints2Moments(transformedSigmaPoints, sigma_points.value().weights, to_rot_ind);
    }
    return modifiedSigmaPoints2Moments(transformedSigmaPoints, sigma_points.value().weights, to_rot_ind);
  }();
  Matrix<UDT> Pcross = SigmaPoints2CrossCovariance(
      transformedSigmaPoints, sigma_points.value().weights, moments.first, residuals, to_rot_ind);
  assert([&] {
    if (not moments.first.array().isFinite().all() or not moments.second.array().isFinite().all() or
        not Pcross.array().isFinite().all())
    {
      LOG_FATAL(moments.first);
      LOG_FATAL(moments.second);
      LOG_FATAL(Pcross);
      LOG_FATAL(mean);
      LOG_FATAL(cov);
      return false;
    }
    return true;
  }());
  return Transformed<UDT>{ .mean = std::move(moments.first),
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
template <class F, class UDT>
[[nodiscard]] std::optional<Transformed<UDT>> unscentTransform(Vector<UDT> const& mean,
                                                               Matrix<UDT> const& cov,
                                                               Matrix<UDT> const& noise_cov,
                                                               F transformation,
                                                               std::optional<Index> to_rot_ind = std::nullopt)
{
  // for augmented state vector
  std::size_t const numNoiseVars = noise_cov.cols();
  std::size_t const augmentedSize = numNoiseVars + mean.rows();
  Vector<UDT> augmentedState = Vector<UDT>::Zero(augmentedSize);
  augmentedState(Eigen::seq(0, mean.rows() - 1)) = mean;
  // concatenate covariance matrices
  Matrix<UDT> augmentedCov = Matrix<UDT>::Zero(augmentedSize, augmentedSize);
  augmentedCov(Eigen::seq(0, mean.rows() - 1), Eigen::seq(0, mean.rows() - 1)) = cov;
  augmentedCov(Eigen::seq(mean.rows(), augmentedSize - 1), Eigen::seq(mean.rows(), augmentedSize - 1)) = noise_cov;

  std::optional<SigmaPoints<UDT>> sigma_points = createSigmaPoints<UDT>(augmentedState, augmentedCov);
  if (not sigma_points.has_value())
  {
    return {};
  }

  Matrix<UDT> const residuals =
      calcResiduum<UDT>(mean, sigma_points.value().sigma_points(Eigen::seq(0, mean.rows() - 1), Eigen::all));
  std::optional<Vector<UDT>> transformed =
      transformation(sigma_points.value().sigma_points(Eigen::seq(0, mean.rows() - 1), 0),
                     sigma_points.value().sigma_points(Eigen::seq(mean.rows(), augmentedSize - 1), 0));
  if (not transformed.has_value())
  {
    return std::nullopt;
  }
  Matrix<UDT> transformedSigmaPoints(transformed.value().rows(), sigma_points.value().sigma_points.cols());
  {
    transformedSigmaPoints.col(0) = std::move(transformed.value());
    for (Index i = 0; i < sigma_points.value().sigma_points.cols(); ++i)
    {
      std::optional<Vector<UDT>> trans =
          transformation(sigma_points.value().sigma_points(Eigen::seq(0, mean.rows() - 1), i),
                         sigma_points.value().sigma_points(Eigen::seq(mean.rows(), augmentedSize - 1), i));
      if (trans.has_value())
      {
        transformedSigmaPoints.col(i) = trans.value();
      }
      else
      {
        return std::nullopt;
      }
    }
  }
  auto moments{ [&] {
    if (sigma_points.value().weights(0) >= 0)
    {
      return SigmaPoints2Moments(transformedSigmaPoints, sigma_points.value().weights, to_rot_ind);
    }
    return modifiedSigmaPoints2Moments(transformedSigmaPoints, sigma_points.value().weights, to_rot_ind);
  }() };
  Matrix<UDT> Pcross{ SigmaPoints2CrossCovariance(
      transformedSigmaPoints, sigma_points.value().weights, moments.first, residuals, to_rot_ind) };
  return Transformed<UDT>{ .mean = std::move(moments.first),
                           .cov = std::move(moments.second),
                           .cross_cov = std::move(Pcross) };
}

/// (linear) transformation Matrix that transform the given Components if possible
template <class UDT>
[[nodiscard]] std::optional<Matrix<UDT>> transformationMatrix(Components const& from, Components const& to)
{
  Matrix<UDT> H = Matrix<UDT>::Zero(std::ssize(to._comps), std::ssize(from._comps));
  for (auto [i, to_comp] : std::views::enumerate(to._comps))
  {
    assert(i < H.rows());
    std::optional<Index> const j = from.indexOf(to_comp);
    // from has the exact same component
    if (j.has_value())
    {
      assert(j.value() < H.cols());
      H(i, j.value()) = 1;
    }
    // length and width -> radius
    else if (auto lw = from.indexOf({ COMPONENT::LENGTH, COMPONENT::WIDTH });
             to_comp == COMPONENT::RADIUS and lw.has_value())
    {
      assert(lw.value()(0) < H.cols());
      assert(lw.value()(1) < H.cols());
      H(i, lw.value()).array() = 1.0 / 4;
    }
    // radius -> length or width
    else if (auto r = from.indexOf(COMPONENT::RADIUS);
             (to_comp == COMPONENT::LENGTH or to_comp == COMPONENT::WIDTH) and r.has_value())
    {
      assert(r.value() < H.cols());
      H(i, r.value()) = 2.0;
    }
    else
    {
      return {};
    }
  }
  return { std::move(H) };
}

/// nonlinear! (in general) transformation to the reference point
template <class UDT>
[[nodiscard]] Vector<UDT>
transform(Vector<UDT> const& state, Components const& state_comps, REFERENCE_POINT from, REFERENCE_POINT to)
{
  if (from == to)
  {
    return state;
  }
  if (from == REFERENCE_POINT::CENTER and (to == REFERENCE_POINT::TOP or to == REFERENCE_POINT::BOTTOM))
  {
    if (auto const inds = state_comps.indexOf({ COMPONENT::POS_Z, COMPONENT::HEIGHT }); inds.has_value())
    {
      Vector<UDT> out = state;
      if (to == REFERENCE_POINT::BOTTOM)
      {
        out(inds.value()(0)) -= out(inds.value()(1)) / 2;
      }
      else if (to == REFERENCE_POINT::TOP)
      {
        out(inds.value()(0)) += out(inds.value()(1)) / 2;
      }
      return out;
    }
    return state;
  }
  if (auto inds = state_comps.indexOf(
          { COMPONENT::POS_X, COMPONENT::POS_Y, COMPONENT::ROT_Z, COMPONENT::LENGTH, COMPONENT::WIDTH });
      inds.has_value())
  {
    Vector<UDT> x_in = state(inds.value());
    Vector<UDT> out = state;
    Vector<UDT> x_out = reference_point::transform(x_in, from, to);
    out(inds.value()) = x_out;
    return out;
  }
  if (auto inds = state_comps.indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y, COMPONENT::RADIUS }); inds.has_value())
  {
    // transform to Center
    UDT const r = state(inds.value()(2));
    UDT const r2 = r / std::sqrt(2);
    UDT delta_x = 0;
    UDT delta_y = 0;
    if (from == REFERENCE_POINT::FRONT)
    {
      delta_x -= r;
    }
    if (from == REFERENCE_POINT::BACK)
    {
      delta_x += r;
    }
    if (from == REFERENCE_POINT::LEFT)
    {
      delta_y += r;
    }
    if (from == REFERENCE_POINT::RIGHT)
    {
      delta_y -= r;
    }
    if (from == REFERENCE_POINT::BACK_LEFT)
    {
      delta_x += r2;
      delta_y -= r2;
    }
    if (from == REFERENCE_POINT::BACK_RIGHT)
    {
      delta_x += r2;
      delta_y += r2;
    }
    if (from == REFERENCE_POINT::FRONT_LEFT)
    {
      delta_x -= r2;
      delta_y -= r2;
    }
    if (from == REFERENCE_POINT::FRONT_RIGHT)
    {
      delta_x -= r2;
      delta_y += r2;
    }
    if (to == REFERENCE_POINT::FRONT)
    {
      delta_x += r;
    }
    if (to == REFERENCE_POINT::BACK)
    {
      delta_x -= r;
    }
    if (to == REFERENCE_POINT::LEFT)
    {
      delta_y -= r;
    }
    if (to == REFERENCE_POINT::RIGHT)
    {
      delta_y += r;
    }
    if (to == REFERENCE_POINT::BACK_LEFT)
    {
      delta_x -= r2;
      delta_y += r2;
    }
    if (to == REFERENCE_POINT::BACK_RIGHT)
    {
      delta_x -= r2;
      delta_y -= r2;
    }
    if (to == REFERENCE_POINT::FRONT_LEFT)
    {
      delta_x += r2;
      delta_y += r2;
    }
    if (to == REFERENCE_POINT::FRONT_RIGHT)
    {
      delta_x += r2;
      delta_y -= r2;
    }
    Vector<UDT> transformed_state = state;
    transformed_state(inds.value()(0)) += delta_x;
    transformed_state(inds.value()(1)) += delta_y;
    return transformed_state;
  }
  LOG_INF_THROTTLE(1,
                   "Can not transform state with Components " + state_comps.toString() + " from RP " + to_string(from) +
                       " to RP" + to_string(to));
  return state;
}

/// most general (possible non-linear) Transformation of a Vector between the given Components
template <class UDT>
[[nodiscard]] std::optional<Vector<UDT>> transform(Vector<UDT> const& state,
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
template <class UDT>
[[nodiscard]] std::optional<Transformed<UDT>> transform(Vector<UDT> const& state,
                                                        Matrix<UDT> const& cov,
                                                        Components const& from,
                                                        Components const& to,
                                                        REFERENCE_POINT from_rp,
                                                        REFERENCE_POINT to_rp,
                                                        SE3Trafo const& sensor_pose = SE3Trafo::Identity())
{
  // ZoneScopedNS("transformation::transform", 15);
  if (from._comps == to._comps and from_rp == to_rp)
  {
    return Transformed<UDT>{ .mean = state, .cov = cov, .T = Matrix<UDT>::Identity(state.rows(), state.rows()) };
  }
  if (from_rp == to_rp)  // try linear
  {
    auto T = transformationMatrix<UDT>(from, to);
    if (T.has_value())
    {
      assert(T.value().cols() == state.rows() and T.value().rows() == std::ssize(to._comps));
      Vector<UDT> transformed_mean = T.value() * state;
      Matrix<UDT> transformed_cov = T.value() * cov * T.value().transpose();
      return Transformed<UDT>{ .mean = std::move(transformed_mean),
                               .cov = std::move(transformed_cov),
                               .T = std::move(T) };
    }
    // use nonlinear
    return unscentTransform(
        state,
        cov,
        [&](Vector<UDT> const& x) { return transform(x, from, to, sensor_pose); },
        to.indexOf(COMPONENT::ROT_Z));
  }
  // treat the transformation of the reference point as linear transformation of the mean only!
  // otherwise the results strongly depend on the covariance of extent which results in very strange behaviour
  return unscentTransform(
      transform(state, from, from_rp, to_rp),
      cov,
      [&](Vector<UDT> const& x) { return transform(x, from, to, sensor_pose); },
      to.indexOf(COMPONENT::ROT_Z));
}

/// transform between components
template <class UDT>
[[nodiscard]] std::optional<Transformed<UDT>> transform(Vector<UDT> const& state,
                                                        Matrix<UDT> const& cov,
                                                        Components const& from,
                                                        Components const& to,
                                                        SE3Trafo const& sensor_pose = SE3Trafo::Identity())
{
  return transform(state, cov, from, to, REFERENCE_POINT::CENTER, REFERENCE_POINT::CENTER, sensor_pose);
}

/// transform the reference point
template <class UDT>
[[nodiscard]] std::optional<Transformed<UDT>> transform(Vector<UDT> const& state,
                                                        Matrix<UDT> const& cov,
                                                        Components const& from,
                                                        REFERENCE_POINT from_rp,
                                                        REFERENCE_POINT to_rp,
                                                        SE3Trafo const& sensor_pose = SE3Trafo::Identity())
{
  return transform(state, cov, from, from, from_rp, to_rp, sensor_pose);
}

/// transform to the given coordinate system
/// translation: state, cov frame -> new frame (xyz-pos)
/// rotation:    state, cov frame -> new frame
template <class UDT>
[[nodiscard]] Transformed<UDT>
transform(Vector<UDT> const& state, Matrix<UDT> const& cov, Components const& from, SE3Trafo const& trafo)
{
  // ZoneScopedN("transformation::CSTransform");
  Matrix<UDT> rot = Matrix<UDT>::Identity(state.rows(), state.rows());  ///< rotation matrix for the full state vector
  Vector<UDT> trans = Vector<UDT>::Zero(state.rows());  ///< translation vector for the full state vector
  if (auto const xyz_inds = from.indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y, COMPONENT::POS_Z });
      xyz_inds.has_value())
  {
    rot(xyz_inds.value(), xyz_inds.value()) = trafo.rotation().cast<UDT>();
    trans(xyz_inds.value()) = trafo.translation().cast<UDT>();
  }
  if (auto const xy_inds = from.indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y }); xy_inds.has_value())
  {
    rot(xy_inds.value(), xy_inds.value()) = trafo.rotation()({ 0, 1 }, { 0, 1 }).cast<UDT>();
    trans(xy_inds.value()) = trafo.translation()({ 0, 1 }).cast<UDT>();
  }
  if (auto const vel_inds = from.indexOf({ COMPONENT::VEL_X, COMPONENT::VEL_Y }); vel_inds.has_value())
  {
    rot(vel_inds.value(), vel_inds.value()) = trafo.rotation()({ 0, 1 }, { 0, 1 }).cast<UDT>();
  }
  /// Todo: do I have to transform the accelaration??
  if (auto const acc_inds = from.indexOf({ COMPONENT::ACC_X, COMPONENT::ACC_Y }); acc_inds.has_value())
  {
    rot(acc_inds.value(), acc_inds.value()) = trafo.rotation()({ 0, 1 }, { 0, 1 }).cast<UDT>();
  }
  auto const iROT_Z = from.indexOf(COMPONENT::ROT_Z);
  if (iROT_Z.has_value())
  {
    trans(iROT_Z.value()) = atan2(trafo.rotation().cast<UDT>()(1, 0), trafo.rotation().cast<UDT>()(0, 0));
  }
  Vector<UDT> mean = rot * state + trans;
  if (iROT_Z.has_value())
  {
    angles::normalizeAngle(mean(iROT_Z.value()));
  }
  Matrix<UDT> cov_transformed = rot * cov * rot.transpose();
  return Transformed<UDT>{ .mean = std::move(mean), .cov = std::move(cov_transformed), .T = std::move(rot) };
}

/// flip the orientation of an Measurement by 180°
template <class UDT>
[[nodiscard]] State<UDT> flip_orientation(State<UDT> const& meas);

/// flip the orientation of an Distribution by 180°
template <class UDT>
[[nodiscard]] Distribution<UDT> flipOrientation(Distribution<UDT> const& dist, Components const& comps);

template <class UDT>
[[nodiscard]] GaussianDistribution<UDT> flipOrientation(GaussianDistribution<UDT> const& dist, Components const& comps);

/// transform a Distribution to another StateModel
template <class UDT>
[[nodiscard]] GaussianDistribution<UDT> transform(GaussianDistribution<UDT> const& dist,
                                                  StateModel<UDT> const& sourceStateModel,
                                                  StateModel<UDT> const& targetStateModel);

}  // namespace transformation

}  // namespace ttb
