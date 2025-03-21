#include "tracking_lib/Transformations/Transformation.h"

#include "tracking_lib/Transformations/TransformReferencePoint.h"
#include "tracking_lib/Misc/AngleNormalization.h"
#include "tracking_lib/StateModels/BaseStateModel.h"
#include "tracking_lib/TTBManager/TTBManager.h"

#include <tracy/tracy/Tracy.hpp>
#include <algorithm>

namespace ttb
{

constexpr auto tracy_color = tracy::Color::RoyalBlue;

namespace SigmaPoints
{

std::pair<Matrix, Vector> createSigmaPoints(Vector const& x, Matrix const& P)
{
  ZoneScopedNC("transformation::createSigmaPoints", tracy_color);
  auto const L = static_cast<double>(x.rows());
  double const kappa = 3.0 - L;

  Eigen::VectorXd weights_eigen = 1.0 / (2 * (L + kappa)) * Eigen::VectorXd::Ones(2 * x.size() + 1);
  weights_eigen(0) = kappa / (L + kappa);

  Eigen::LLT<Eigen::MatrixXd> eigenChol((L + kappa) * P);
  Eigen::MatrixXd sigmaPoints_eigen = x.replicate(1, 2 * x.rows() + 1);
  sigmaPoints_eigen(Eigen::all, Eigen::seq(1, L)) += eigenChol.matrixL();
  sigmaPoints_eigen(Eigen::all, Eigen::seq(L + 1, 2 * L)) -= eigenChol.matrixL();
  return { std::move(sigmaPoints_eigen), std::move(weights_eigen) };
}

Matrix calcResiduum(Vector const& x, Matrix const& sigmaPoints)
{
  Eigen::MatrixXd res_eigen = sigmaPoints.colwise() - x;
  return res_eigen;
}

std::pair<Vector, Matrix> SigmaPoints2Moments(Matrix const& sp, Vector const& weights, std::optional<Index> rot_ind)
{
  ZoneScopedNC("transformation::moments", tracy_color);
  Eigen::VectorXd mean = sp * weights;
  Eigen::MatrixXd P_eigen = Eigen::MatrixXd::Zero(sp.rows(), sp.rows());
  for (unsigned int i = 0; i < sp.cols(); i++)
  {
    Eigen::VectorXd res = sp.col(i) - mean;
    P_eigen += weights(i) * (res * res.transpose());
  }
  if (rot_ind.has_value())
  {
    Vector angle_weights = weights(Eigen::seq(1, sp.cols() - 1));
    angle_weights /= angle_weights.sum();
    mean(rot_ind.value()) = angles::weightedMean(sp(rot_ind.value(), Eigen::seq(1, sp.cols() - 1)), angle_weights);
  }
  return { std::move(mean), std::move(P_eigen) };
}

std::pair<Vector, Matrix> modifiedSigmaPoints2Moments(Matrix const& sp,
                                                      Vector const& weights,
                                                      std::optional<Index> rot_ind)
{
  ZoneScopedNC("transformation::moments", tracy_color);
  Eigen::VectorXd mean = sp * weights;
  Eigen::MatrixXd P_eigen = Matrix::Zero(sp.rows(), sp.rows());
  for (unsigned int i = 1; i < sp.cols(); i++)
  {
    Eigen::VectorXd res = sp.col(i) - sp.col(0);
    P_eigen += weights(i) * (res * res.transpose());
  }
  if (rot_ind.has_value())
  {
    Vector angle_weights = weights(Eigen::seq(1, sp.cols() - 1));
    angle_weights /= angle_weights.sum();
    assert(std::abs(angle_weights.sum() - 1) < 1e-5);
    mean(rot_ind.value()) = angles::weightedMean(sp(rot_ind.value(), Eigen::seq(1, sp.cols() - 1)), angle_weights);
  }
  return { std::move(mean), std::move(P_eigen) };
}

Matrix SigmaPoints2CrossCovariance(Matrix const& sp, Vector const& weights, Vector const& Xhat, Matrix const& res)
{
  ZoneScopedNC("transformation::crossCovariance", tracy_color);
  Matrix cross_cov_eigen = Matrix::Zero(res.rows(), sp.rows());
  for (unsigned int i = 1; i < sp.cols(); i++)
  {
    Eigen::VectorXd tmp = sp.col(i) - Xhat;
    cross_cov_eigen += weights(i) * (res.col(i) * tmp.transpose());
  }
  return cross_cov_eigen;
}

}  // namespace SigmaPoints

namespace transformation
{

/// return the (linear) transformation Matrix that transform the given Components if possible
std::optional<Matrix> transformationMatrix(Components const& from, Components const& to)
{
  ZoneScopedNC("transformation::linearTransformationMatrix", tracy_color);
  Matrix H = Matrix::Zero(static_cast<Index>(to._comps.size()), static_cast<Index>(from._comps.size()));
  for (auto [i, to_comp] : std::views::enumerate(to._comps))
  {
    auto j = from.indexOf(to_comp);
    // from has the exact same component
    if (j.has_value())
    {
      H(i, j.value()) = 1;
    }
    // length and width -> radius
    else if (auto lw = from.indexOf({ COMPONENT::LENGTH, COMPONENT::WIDTH });
             to_comp == COMPONENT::RADIUS and lw.has_value())
    {
      H(i, lw.value()).array() = 1.0 / 4;
    }
    // radius -> length or width
    else if (auto r = from.indexOf(COMPONENT::RADIUS);
             (to_comp == COMPONENT::LENGTH or to_comp == COMPONENT::WIDTH) and r.has_value())
    {
      H(i, r.value()) = 2.0;
    }
    else
    {
      return std::nullopt;
    }
  }
  return H;
}

Vector transform(Vector const& state, Components const& state_comps, REFERENCE_POINT from, REFERENCE_POINT to)
{
  if (from == to)
  {
    return state;
  }
  if (auto inds = state_comps.indexOf(
          { COMPONENT::POS_X, COMPONENT::POS_Y, COMPONENT::ROT_Z, COMPONENT::LENGTH, COMPONENT::WIDTH });
      inds.has_value())
  {
    Vector x_in = state(inds.value());
    Vector out = state;
    Vector x_out = transformReferencePoint::transform(x_in, from, to);
    out(inds.value()) = x_out;
    return out;
  }
  if (auto inds = state_comps.indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y, COMPONENT::RADIUS }); inds.has_value())
  {
    // transform to Center
    double const r = state(inds.value()(2));
    double const r2 = r / std::sqrt(2);
    double delta_x = 0;
    double delta_y = 0;
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
    Vector transformed_state = state;
    transformed_state(inds.value()(0)) += delta_x;
    transformed_state(inds.value()(1)) += delta_y;
    return transformed_state;
  }
  LOG_INF_THROTTLE(1,
                   "Can not transform state with Components " + state_comps.toString() + " from RP " + to_string(from) +
                       " to RP" + to_string(to));
  return state;
}

std::optional<Vector>
transform(Vector const& state, Components const& from, Components const& to, SE3Trafo const& sensor_pose)
{
  ZoneScopedNC("transformation:: general vector transform", tracy_color);
  LOG_DEB("General vector transform");
  Vector trans(static_cast<Index>(to._comps.size()));
  for (std::size_t toInd = 0; toInd < to._comps.size(); ++toInd)
  {
    auto fromInd = from.indexOf(to._comps.at(toInd));  // The state model provide the required value
    std::optional<double> transfromedVal{ [&] -> std::optional<double> {
      if (fromInd.has_value())
      {
        return state(fromInd.value());
      }
      if (to._comps.at(toInd) == COMPONENT::VEL_ABS)  // try to convert cartesian to abs values
      {
        auto i_VEL_CART = from.indexOf({ COMPONENT::VEL_X, COMPONENT::VEL_Y });
        if (i_VEL_CART.has_value())
        {
          return state(i_VEL_CART.value()).norm();
        }
        return std::nullopt;
      }
      if (to._comps.at(toInd) == COMPONENT::ACC_ABS)  // try to convert cartesian to abs values
      {
        auto i_ACC_CART = from.indexOf({ COMPONENT::ACC_X, COMPONENT::ACC_Y });
        if (i_ACC_CART.has_value())
        {
          return state(i_ACC_CART.value()).norm();
        }
        return std::nullopt;
      }
      if (to._comps.at(toInd) == COMPONENT::JERK_ABS)  // try to convert cartesian to abs values
      {
        auto i_JERK_CART = from.indexOf({ COMPONENT::JERK_X, COMPONENT::JERK_Y });
        if (i_JERK_CART.has_value())
        {
          return state(i_JERK_CART.value()).norm();
        }
      }
      if (to._comps.at(toInd) == COMPONENT::ROT_Z)  // This assumes the object is moving forward
      {
        auto i_VEL_CART = from.indexOf({ COMPONENT::VEL_X, COMPONENT::VEL_Y });
        if (i_VEL_CART.has_value())
        {
          return std::atan2(state(i_VEL_CART.value()(1)), state(i_VEL_CART.value()(0)));
        }
        return std::nullopt;
      }
      if (to._comps.at(toInd) == COMPONENT::VEL_X)  // This assumes the object is looking forward
      {
        auto i_ROT_Z_VEL_ABS = from.indexOf({ COMPONENT::ROT_Z, COMPONENT::VEL_ABS });
        if (i_ROT_Z_VEL_ABS.has_value())
        {
          return std::cos(state(i_ROT_Z_VEL_ABS.value()(0))) * state(i_ROT_Z_VEL_ABS.value()(1));
        }
        return std::nullopt;
      }
      if (to._comps.at(toInd) == COMPONENT::VEL_Y)  // This assumes the object is looking forward
      {
        auto i_ROT_Z_VEL_ABS = from.indexOf({ COMPONENT::ROT_Z, COMPONENT::VEL_ABS });
        if (i_ROT_Z_VEL_ABS.has_value())
        {
          return std::sin(state(i_ROT_Z_VEL_ABS.value()(0))) * state(i_ROT_Z_VEL_ABS.value()(1));
        }
        return std::nullopt;
      }
      if (to._comps.at(toInd) == COMPONENT::AZIMUTH)
      {
        auto i_POS_XYZ = from.indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y, COMPONENT::POS_Z });
        if (i_POS_XYZ.has_value())
        {
          Vector const& z_tc = state(i_POS_XYZ.value());
          Vector z_sc = sensor_pose * z_tc.homogeneous();
          return std::atan2(z_sc(1), z_sc(0));
        }
        return std::nullopt;
      }
      if (to._comps.at(toInd) == COMPONENT::ELEVATION)
      {
        auto i_POS_XYZ = from.indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y, COMPONENT::POS_Z });
        if (i_POS_XYZ.has_value())
        {
          Vector const& z_tc = state(i_POS_XYZ.value());
          Vector z_sc = sensor_pose * z_tc.homogeneous();
          return std::atan2(z_sc(2), z_sc(0));
        }
        return std::nullopt;
      }
      if (to._comps.at(toInd) == COMPONENT::RADIUS)
      {
        auto i_lw = from.indexOf({ COMPONENT::LENGTH, COMPONENT::WIDTH });
        if (i_lw.has_value())
        {
          return state(i_lw.value()).sum() / 4;
        }
        return std::nullopt;
      }
      if (to._comps.at(toInd) == COMPONENT::LENGTH)
      {
        auto i_r = from.indexOf(COMPONENT::RADIUS);
        if (i_r.has_value())
        {
          return state(i_r.value()) * 2.0;
        }
        return std::nullopt;
      }
      if (to._comps.at(toInd) == COMPONENT::WIDTH)
      {
        auto i_r = from.indexOf(COMPONENT::RADIUS);
        if (i_r.has_value())
        {
          return state(i_r.value()) * 2.0;
        }
        return std::nullopt;
      }
      if (to._comps.at(toInd) == COMPONENT::X_CC_UPPER_RIGHT)
      {
        // find the upper right corner of the track described in camera coords at z=1
        auto i_POS_XYZ = from.indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y, COMPONENT::POS_Z });
        if (not i_POS_XYZ.has_value())
        {
          return std::nullopt;
        }
        double right_corner = -std::numeric_limits<double>::infinity();
        for (REFERENCE_POINT to_rp : { REFERENCE_POINT::FRONT_LEFT,
                                       REFERENCE_POINT::BACK_LEFT,
                                       REFERENCE_POINT::BACK_RIGHT,
                                       REFERENCE_POINT::FRONT_RIGHT })
        {
          Vector corner = transform(state, from, REFERENCE_POINT::CENTER, to_rp);
          if (i_POS_XYZ.has_value())
          {
            Vector const& z_tc = corner(i_POS_XYZ.value());
            Vector corner_sc = sensor_pose * z_tc.homogeneous();
            if (corner_sc(i_POS_XYZ.value()(2)) < 0)
            {
              return std::nullopt;
            }
            if (corner_sc(0) / corner_sc(2) > right_corner)
            {
              right_corner = corner_sc(0) / corner_sc(2);
            }
          }
        }
        return right_corner;
      }
      if (to._comps.at(toInd) == COMPONENT::X_CC_LOWER_LEFT)
      {
        // find the lower left corner of the track described in camera coords at z=1
        auto i_POS_XYZ = from.indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y, COMPONENT::POS_Z });
        if (not i_POS_XYZ.has_value())
        {
          return std::nullopt;
        }
        double left_corner = std::numeric_limits<double>::infinity();

        for (REFERENCE_POINT to_rp : { REFERENCE_POINT::FRONT_LEFT,
                                       REFERENCE_POINT::BACK_LEFT,
                                       REFERENCE_POINT::BACK_RIGHT,
                                       REFERENCE_POINT::FRONT_RIGHT })
        {
          Vector corner = transform(state, from, REFERENCE_POINT::CENTER, to_rp);
          if (i_POS_XYZ.has_value())
          {
            Vector const& z_tc = corner(i_POS_XYZ.value());
            Vector corner_sc = sensor_pose * z_tc.homogeneous();
            if (corner_sc(i_POS_XYZ.value()(2)) < 0)
            {
              return std::nullopt;
            }
            if (corner_sc(0) / corner_sc(2) < left_corner)
            {
              left_corner = corner_sc(0) / corner_sc(2);
            }
          }
        }
        return left_corner;
      }
      std::string const info_str = "Could not transform from " + from.toString() + " to " + to.toString() +
                                   ". Problem: " + to_string(to._comps.at(toInd));
      ZoneText(info_str.c_str(), info_str.size());
      return std::nullopt;
    }() };
    if (transfromedVal.has_value())
    {
      trans(static_cast<Index>(toInd)) = transfromedVal.value();
    }
    else
    {
      return std::nullopt;
    }
  }
  return trans;
}

std::optional<Transformed> transform(Vector const& state,
                                     Matrix const& cov,
                                     Components const& from,
                                     Components const& to,
                                     REFERENCE_POINT from_rp,
                                     REFERENCE_POINT to_rp,
                                     SE3Trafo const& sensor_pose)
{
  ZoneScopedNCS("transformation::transform", tracy_color, 15);
  LOG_DEB("General Transform for Distribution");
  if (from._comps == to._comps and from_rp == to_rp)
  {
    std::string info_str = "Identity Transformation";
    ZoneText(info_str.c_str(), info_str.size());
    return Transformed{ .mean = state, .cov = cov, .T = Matrix::Identity(state.rows(), state.rows()) };
  }
  if (from_rp == to_rp)  // try linear
  {
    std::optional<Matrix> T = transformationMatrix(from, to);
    if (T.has_value())
    {
      std::string info_str = "Same RP Linear Transformation";
      ZoneText(info_str.c_str(), info_str.size());
      return Transformed{ .mean = T.value() * state,
                          .cov = T.value() * cov * T.value().transpose(),
                          .T = std::move(T) };
    }
    // use nonlinear
    auto trafo = unscentTransform(
        state, cov, [&](Vector const& x) { return transform(x, from, to, sensor_pose); }, to.indexOf(COMPONENT::ROT_Z));
    if (not trafo.has_value())
    {
      std::string info_str = "Same RP Transformation not possible";
      ZoneText(info_str.c_str(), info_str.size());
      return {};
    }
    std::string info_str = "Same RP Nonlinear Transformation";
    ZoneText(info_str.c_str(), info_str.size());
    return Transformed{ .mean = std::move(trafo.value().mean),
                        .cov = std::move(trafo.value().cov),
                        .cross_cov = std::move(trafo.value().cross_cov) };
  }
  // treat the transformation of the reference point as linear transformation of the mean only!
  // otherwise the results strongly depend on the covariance of extent which results in very strange behaviour
  std::optional<Transformed> trafo = unscentTransform(
      transform(state, from, from_rp, to_rp),
      cov,
      [&](Vector const& x) {
        //        auto temp = transform(x, from, from_rp, to_rp);
        return transform(x, from, to, sensor_pose);
      },
      to.indexOf(COMPONENT::ROT_Z));
  if (not trafo.has_value())
  {
    std::string info_str = "Different RP Transformation not possible";
    ZoneText(info_str.c_str(), info_str.size());
    return {};
  }
  std::string info_str = "Different RP Nonlinear Transformation";
  ZoneText(info_str.c_str(), info_str.size());
  return Transformed{ .mean = std::move(trafo.value().mean),
                      .cov = std::move(trafo.value().cov),
                      .cross_cov = std::move(trafo.value().cross_cov) };
}

std::optional<Transformed> transform(Vector const& state,
                                     Matrix const& cov,
                                     Components const& from,
                                     Components const& to,
                                     SE3Trafo const& sensor_pose)
{
  return transform(state, cov, from, to, REFERENCE_POINT::CENTER, REFERENCE_POINT::CENTER, sensor_pose);
}

std::optional<Transformed> transform(Vector const& state,
                                     Matrix const& cov,
                                     Components const& from,
                                     REFERENCE_POINT from_rp,
                                     REFERENCE_POINT to_rp,
                                     SE3Trafo const& sensor_pose)
{
  return transform(state, cov, from, from, from_rp, to_rp, sensor_pose);
}

std::optional<Transformed> transform(Vector const& state,
                                     Matrix const& cov,
                                     Components const& from,
                                     Vector3 const& translation,
                                     Matrix33 const& rotation)
{
  ZoneScopedNC("transformation::CSTransform", tracy_color);
  Matrix rot = Matrix::Identity(state.rows(), state.rows());  ///< rotation matrix for the full state vector
  Vector trans = Vector::Zero(state.rows());                  ///< translation vector for the full state vector
  auto xyz_inds = from.indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y, COMPONENT::POS_Y });
  if (xyz_inds.has_value())
  {
    rot(xyz_inds.value(), xyz_inds.value()) = rotation;
    trans(xyz_inds.value()) = translation;
  }
  auto xy_inds = from.indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y });
  if (xy_inds.has_value())
  {
    rot(xy_inds.value(), xy_inds.value()) = rotation({ 0, 1 }, { 0, 1 });
    trans(xy_inds.value()) = translation({ 0, 1 });
  }
  auto vel_inds = from.indexOf({ COMPONENT::VEL_X, COMPONENT::VEL_Y });
  if (vel_inds.has_value())
  {
    rot(vel_inds.value(), vel_inds.value()) = rotation({ 0, 1 }, { 0, 1 });
  }
  auto acc_inds = from.indexOf({ COMPONENT::ACC_X, COMPONENT::ACC_Y });
  if (acc_inds.has_value())
  {
    rot(acc_inds.value(), acc_inds.value()) = rotation({ 0, 1 }, { 0, 1 });
  }
  auto iROT_Z = from.indexOf(COMPONENT::ROT_Z);
  if (iROT_Z.has_value())
  {
    trans(iROT_Z.value()) = std::atan2(rotation(1, 0), rotation(0, 0));
  }
  Vector mean = rot * state + trans;
  if (iROT_Z.has_value())
  {
    angles::normalizeAngle(mean(iROT_Z.value()));
  }
  Matrix cov_transformed = rot * cov * rot.transpose();
  return Transformed{ .mean = std::move(mean), .cov = std::move(cov_transformed), .T = std::move(rot) };
}

Measurement flipMeasurementOrientation(Measurement const& meas)
{
  ZoneScopedNC("transformation::flipMeasurement", tracy_color);
  Measurement flipped{ meas };
  flipped._dist = flipOrientation(*flipped._dist, flipped._meas_comps);
  return flipped;
}

std::unique_ptr<BaseDistribution> flipOrientation(BaseDistribution const& dist, Components const& comps)
{
  auto flipped = dist.clone();
  Vector const& mean = dist.mean();
  auto pos_ind{ [&] -> std::optional<Indices> {
    std::optional<Indices> tmp = comps.indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y, COMPONENT::POS_Z });
    if (tmp.has_value())
    {
      return tmp.value();
    }
    tmp = comps.indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y });
    if (tmp.has_value())
    {
      return tmp.value();
    }
    return {};
  }() };
  if (not pos_ind.has_value())
  {
    return dist.clone();
  }
  Vector const& pos = mean(pos_ind.value());
  Vector pos_zero = mean;
  pos_zero(pos_ind.value()).setZero();
  flipped->set(pos_zero);
  Matrix33 rot{ { -1, 0, 0 }, { 0, -1, 0 }, { 0, 0, 1 } };
  auto trafo = transform(flipped->mean(), flipped->covariance(), comps, Vector3::Zero(3), rot);
  trafo.value().mean(pos_ind.value()) = pos;
  flipped->set(std::move(trafo.value().mean), std::move(trafo.value().cov));
  return flipped;
}

std::unique_ptr<BaseDistribution> transform(BaseDistribution const& dist,
                                            BaseStateModel const& sourceStateModel,
                                            BaseStateModel const& targetStateModel)
{
  assert(dist.type() == DISTRIBUTION_TYPE::GAUSSIAN);
  std::optional<Transformed> transformed =
      transform(dist.mean(), dist.covariance(), sourceStateModel.state_comps(), targetStateModel.state_comps());
  if (transformed.has_value())
  {
    return std::make_unique<GaussianDistribution>(
        std::move(transformed.value().mean), std::move(transformed.value().cov), dist.sumWeights(), dist.refPoint());
  }
  LOG_INF_THROTTLE(10,
                   "Transformation from state model: " << sourceStateModel.toString()
                                                       << " to model: " << targetStateModel.toString()
                                                       << " not possible - Using default values");
  Components const transformable = transformableComps(sourceStateModel.state_comps(),
                                                      TransformOptions{ targetStateModel.manager()
                                                                            ->state_model_params(targetStateModel.id())
                                                                            .assume_orient_in_velocity_direction })
                                       .all.intersection(targetStateModel.state_comps());
  Vector mean(targetStateModel.state_comps()._comps.size());
  Matrix cov = Matrix::Identity(static_cast<Index>(targetStateModel.state_comps()._comps.size()),
                                static_cast<Index>(targetStateModel.state_comps()._comps.size()));
  std::optional<Transformed> transformedReduce =
      transform(dist.mean(), dist.covariance(), sourceStateModel.state_comps(), transformable);
  assert(transformedReduce.has_value());
  auto reducedInds = targetStateModel.state_comps().indexOf(transformable._comps).value();
  mean(reducedInds) = std::move(transformedReduce.value().mean);
  cov(reducedInds, reducedInds) = std::move(transformedReduce.value().cov);
  Components const remainingComps = targetStateModel.state_comps().diff(transformable);
  for (COMPONENT comp : remainingComps._comps)
  {
    if (not targetStateModel.manager()
                ->state_model_params(targetStateModel.id())
                .default_mean.contains(to_string(comp)))
    {
      LOG_FATAL("State Model: " << targetStateModel.toString()
                                << " does not contain default mean for component: " << to_string(comp));
      throw std::runtime_error("missing default mean component");
    }
    mean(targetStateModel.state_comps().indexOf(comp).value()) =
        targetStateModel.manager()->state_model_params(targetStateModel.id()).default_mean.at(to_string(comp));
    if (not targetStateModel.manager()->state_model_params(targetStateModel.id()).default_var.contains(to_string(comp)))
    {
      LOG_FATAL("State Model: " << targetStateModel.toString()
                                << " does not contain default var for component: " << to_string(comp));
      throw std::runtime_error("missing default var component");
    }
    cov(targetStateModel.state_comps().indexOf(comp).value(), targetStateModel.state_comps().indexOf(comp).value()) =
        targetStateModel.manager()->state_model_params(targetStateModel.id()).default_var.at(to_string(comp));
  }
  return std::make_unique<GaussianDistribution>(std::move(mean), std::move(cov), dist.sumWeights(), dist.refPoint());
}

TransformableComps transformableComps(Components const& comps, TransformOptions const& options)
{
  std::vector<COMPONENT> linear;
  std::vector<COMPONENT> nonLinear;
  for (COMPONENT comp : comps._comps)
  {
    linear.push_back(comp);     // Can predict myself linear
    nonLinear.push_back(comp);  // Can predict myself nonlinear
  }
  if (comps.indexOf({ COMPONENT::VEL_X, COMPONENT::VEL_Y }).has_value())
  {
    nonLinear.push_back(COMPONENT::VEL_ABS);  // can convert cartesian velocity to abs velocity
    if (options.assume_orient_in_velocity_direction)
    {
      nonLinear.push_back(COMPONENT::ROT_Z);  // can convert cartesian velocity to orientation
    }
  }
  if (options.assume_orient_in_velocity_direction and
      comps.indexOf({ COMPONENT::VEL_ABS, COMPONENT::ROT_Z }).has_value())
  {
    nonLinear.push_back(COMPONENT::VEL_X);  // can convert polar velocity to cartesian
    nonLinear.push_back(COMPONENT::VEL_Y);  // can convert polar velocity to cartesian
  }
  if (comps.indexOf({ COMPONENT::ACC_ABS, COMPONENT::ACC_Y }).has_value())
  {
    nonLinear.push_back(COMPONENT::ACC_ABS);  // can convert cartesian velocity to abs velocity
  }
  if (comps.indexOf({ COMPONENT::JERK_X, COMPONENT::JERK_Y }).has_value())
  {
    nonLinear.push_back(COMPONENT::JERK_ABS);  // can convert cartesian velocity to abs velocity
  }
  if (comps.indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y }).has_value())
  {
    nonLinear.push_back(COMPONENT::AZIMUTH);  // azimuth angle
  }
  if (comps.indexOf({ COMPONENT::POS_X, COMPONENT::POS_Z }).has_value())
  {
    nonLinear.push_back(COMPONENT::ELEVATION);  // elevation angle
  }
  if (comps.indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y, COMPONENT::POS_Z }).has_value())
  {
    nonLinear.push_back(COMPONENT::X_CC_LOWER_LEFT);   // camera coords
    nonLinear.push_back(COMPONENT::X_CC_UPPER_RIGHT);  // camera cords
  }
  if (comps.indexOf({ COMPONENT::LENGTH, COMPONENT::WIDTH }))
  {
    nonLinear.push_back(COMPONENT::RADIUS);
  }
  if (comps.indexOf(COMPONENT::RADIUS))
  {
    nonLinear.push_back(COMPONENT::LENGTH);
    nonLinear.push_back(COMPONENT::WIDTH);
  }
  return { .linear = Components(linear), .all = Components(nonLinear) };
}

}  // namespace transformation

}  // namespace ttb
