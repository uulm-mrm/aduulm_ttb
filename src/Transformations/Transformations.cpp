#include "tracking_lib/Transformations/Transformation.h"

#include "tracking_lib/StateModels/BaseStateModel.h"
#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/StateModels/BaseStateModel.h"

#include <aduulm_logger/aduulm_logger.hpp>
#include <algorithm>

namespace ttb
{

namespace transformation
{

template State<double> flip_orientation(State<double> const& meas);
template State<AD> flip_orientation(State<AD> const& meas);
template <class UDT>
State<UDT> flip_orientation(State<UDT> const& meas)
{
  State flipped{ meas };
  for (auto& [model_id, dist] : flipped._state_dist)
  {
    dist.second = flipOrientation(dist.second, meas._manager->state_comps(model_id));
  }
  return flipped;
}

template GaussianDistribution<double> flipOrientation(GaussianDistribution<double> const& dist,
                                                      Components const& comps);
template GaussianDistribution<AD> flipOrientation(GaussianDistribution<AD> const& dist, Components const& comps);
template <class UDT>
GaussianDistribution<UDT> flipOrientation(GaussianDistribution<UDT> const& dist, Components const& comps)
{
  GaussianDistribution flipped = dist;

  Vector<UDT> const& mean = dist.mean();
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
    return dist;
  }
  Vector<UDT> const& pos = mean(pos_ind.value());
  Vector<UDT> pos_zero = mean;
  pos_zero(pos_ind.value()).setZero();
  flipped.mean() = pos_zero;
  Matrix33 rot{ { -1, 0, 0 }, { 0, -1, 0 }, { 0, 0, 1 } };
  SE3Trafo t = SE3Trafo::Identity();
  t.matrix()({ 0, 1, 2 }, { 0, 1, 2 }) = rot;
  auto trafo = transform(flipped.mean(), flipped.covariance(), comps, t);
  trafo.mean(pos_ind.value()) = pos;
  flipped.mean() = std::move(trafo.mean);
  flipped.covariance() = std::move(trafo.cov);
  return flipped;
}

template Distribution<double> flipOrientation(Distribution<double> const& dist, Components const& comps);
template Distribution<AD> flipOrientation(Distribution<AD> const& dist, Components const& comps);
template <class UDT>
Distribution<UDT> flipOrientation(Distribution<UDT> const& dist, Components const& comps)
{
  Distribution flipped = dist;
  for (auto const& [flipped_comps, dist_comps] :
       std::views::zip(flipped.to_gaussian_components(), dist.to_gaussian_components()))
  {
    flipped_comps.second = flipOrientation(dist_comps.second, comps);
  }
  return flipped;
}

template std::optional<SigmaPoints<double>> createSigmaPoints(Vector<double> const& x, Matrix<double> const& P);
template std::optional<SigmaPoints<AD>> createSigmaPoints(Vector<AD> const& x, Matrix<AD> const& P);
template <class UDT>
[[nodiscard]] std::optional<SigmaPoints<UDT>> createSigmaPoints(Vector<UDT> const& x, Matrix<UDT> const& P)
{
  auto const L = static_cast<double>(x.rows());
  double const kappa = 3.0 - L;
  Vector<UDT> weights_eigen = 1.0 / (2 * (L + kappa)) * Vector<UDT>::Ones(2 * x.rows() + 1);
  weights_eigen(0) = kappa / (L + kappa);
  Matrix<UDT> chol = (L + kappa) * P;
  auto const adjoint = chol.template selfadjointView<Eigen::Lower>();
  Eigen::LDLT<Matrix<UDT>> eigenChol(adjoint.ldlt());
  if (eigenChol.info() != Eigen::Success)
  {
    LOG_WARN("Can not create Sigma points for matrix " << P);
    LOG_WARN("Matrix for performing the choleky decomposition: " << chol);
    if (((chol - chol.transpose()).array().abs() > 1e-5).any())
    {
      LOG_FATAL("chol matrix is not symmetric");
    }
    if (eigenChol.info() == Eigen::NumericalIssue)
    {
      LOG_FATAL("eigen numerical issues");
    }
    return {};
  }
  if ((eigenChol.vectorD().array() < -1e-7).any())
  {
    LOG_FATAL("negative eigenvalue: " << eigenChol.vectorD());
    LOG_FATAL(std::stacktrace::current());
    return {};
  }
  Eigen::Transpositions const transpositions = eigenChol.transpositionsP();
  Matrix<UDT> lower = eigenChol.matrixL();
  Vector<UDT> diag = eigenChol.vectorD();
  UDT const min_val = eigenChol.vectorD().minCoeff();
  if (min_val >= -1e-7)
  {
    diag = diag.cwiseMax(0.0).array().sqrt();
  }
  assert(diag.array().isFinite().all());
  lower = transpositions.transpose() * lower * diag.asDiagonal();
  assert(((lower * lower.transpose()) - chol).norm() < 1e-6 * chol.norm());
  Matrix<UDT> sigmaPoints_eigen = x.replicate(1, 2 * x.rows() + 1);
  sigmaPoints_eigen(Eigen::all, Eigen::seq(1, x.rows())) += lower;
  sigmaPoints_eigen(Eigen::all, Eigen::seq(1 + x.rows(), 2 * x.rows())) -= lower;
  SigmaPoints<UDT> sp;
  sp.sigma_points = std::move(sigmaPoints_eigen);
  sp.weights = std::move(weights_eigen);
  assert([&] {
    for (Index i = 0; i < sp.sigma_points.cols(); ++i)
    {
      if (not sp.sigma_points.col(i).array().isFinite().all())
      {
        LOG_FATAL(x);
        LOG_FATAL(P);
        LOG_FATAL(lower);
        LOG_FATAL(Matrix<UDT>(eigenChol.matrixL()));
        LOG_FATAL(eigenChol.vectorD());
        return false;
      }
    }
    return true;
  }());
  return { std::move(sp) };
}

template std::optional<Vector<double>>
transform(Vector<double> const& state, Components const& from, Components const& to, SE3Trafo const& sensor_pose);
template std::optional<Vector<AD>>
transform(Vector<AD> const& state, Components const& from, Components const& to, SE3Trafo const& sensor_pose);
template <class UDT>
[[nodiscard]] std::optional<Vector<UDT>>
transform(Vector<UDT> const& state, Components const& from, Components const& to, SE3Trafo const& sensor_pose)
{
  ZoneScopedN("transformation::transform::vector");
  assert(state.rows() == std::ssize(from._comps));
  Vector<UDT> trans = Vector<UDT>::Zero(static_cast<Index>(to._comps.size()));

  if (to == Components({ COMPONENT::HULL_POS_X_MIN,
                         COMPONENT::HULL_POS_Y_MIN,
                         COMPONENT::HULL_POS_X_MAX,
                         COMPONENT::HULL_POS_Y_MAX }))
  {
    SE3Trafo position = sensor_pose;
    position.translation() = sensor_pose.translation();
    auto const [xs, ys] = to_box(state, from);
    if (xs.empty())
    {
      return std::nullopt;
    }
    std::pair<double, double> max_angle = { -std::numbers::pi, 0 };
    std::pair<double, double> min_angle = { std::numbers::pi, 0 };
    double min_dist = std::numeric_limits<double>::infinity();

    for (auto const [x, y] : std::views::zip(xs, ys))
    {
      Vector<double> corner{ { x, y, 0 } };
      corner = position * corner.homogeneous();
      double const dist = std::hypot(corner(0), corner(1));
      double const angle = std::atan2(corner(1), corner(0));
      if (angle > max_angle.first)
      {
        max_angle = { angle, dist };
      }
      if (angle < min_angle.first)
      {
        min_angle = { angle, dist };
      }
      if (dist < min_dist)
      {
        min_dist = dist;
      }
    }
    constexpr bool use_min_trafo = true;
    Vector<double> min, max;
    if constexpr (use_min_trafo)
    {
      max = Vector<double>{ { std::cos(max_angle.first) * min_dist, std::sin(max_angle.first) * min_dist, 0 } };
      max = position.inverse() * max.homogeneous();
      min = Vector<double>{ { std::cos(min_angle.first) * min_dist, std::sin(min_angle.first) * min_dist, 0 } };
      min = position.inverse() * min.homogeneous();
    }
    else
    {
      max = Vector<UDT>{
        { std::cos(max_angle.first) * max_angle.second, std::sin(max_angle.first) * max_angle.second, 0 }
      };
      max = position.inverse() * max.homogeneous();
      min = Vector<UDT>{
        { std::cos(min_angle.first) * min_angle.second, std::sin(min_angle.first) * min_angle.second, 0 }
      };
      min = position.inverse() * min.homogeneous();
    }
    auto const [nom_min, nom_max] = [&] {
      if (min(0) < max(0))
      {
        return std::pair{ min, max };
      }
      return std::pair{ max, min };
    }();
    trans(to.indexOf(COMPONENT::HULL_POS_X_MIN).value()) = nom_min(0);
    trans(to.indexOf(COMPONENT::HULL_POS_Y_MIN).value()) = nom_min(1);
    trans(to.indexOf(COMPONENT::HULL_POS_X_MAX).value()) = nom_max(0);
    trans(to.indexOf(COMPONENT::HULL_POS_Y_MAX).value()) = nom_max(1);
    return trans;
  }
  else
  {
    for (std::size_t toInd = 0; toInd < to._comps.size(); ++toInd)
    {
      auto fromInd = from.indexOf(to._comps.at(toInd));  // The state model provide the required value
      std::optional<UDT> transfromedVal{ [&] -> std::optional<UDT> {
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
            return atan2(state(i_VEL_CART.value()(1)), state(i_VEL_CART.value()(0)));
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
            Vector<UDT> z_tc = state(i_POS_XYZ.value());
            Vector<UDT> z_sc = sensor_pose.template cast<UDT>() * z_tc.homogeneous();
            return std::atan2(z_sc(1), z_sc(0));
          }
          return std::nullopt;
        }
        if (to._comps.at(toInd) == COMPONENT::ELEVATION)
        {
          auto i_POS_XYZ = from.indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y, COMPONENT::POS_Z });
          if (i_POS_XYZ.has_value())
          {
            Vector<UDT> const& z_tc = state(i_POS_XYZ.value());
            Vector<UDT> z_sc = sensor_pose.cast<UDT>() * z_tc.homogeneous();
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
        if (to._comps.at(toInd) == COMPONENT::POS_X and
            from.indexOf({ COMPONENT::HULL_POS_X_MIN, COMPONENT::HULL_POS_X_MAX }).has_value())
        {
          Indices ind = from.indexOf({ COMPONENT::HULL_POS_X_MIN, COMPONENT::HULL_POS_X_MAX }).value();
          return state(ind).mean();
        }
        if (to._comps.at(toInd) == COMPONENT::POS_Y and
            from.indexOf({ COMPONENT::HULL_POS_Y_MIN, COMPONENT::HULL_POS_Y_MAX }).has_value())
        {
          Indices ind = from.indexOf({ COMPONENT::HULL_POS_Y_MIN, COMPONENT::HULL_POS_Y_MAX }).value();
          return state(ind).mean();
        }
        if (to._comps.at(toInd) == COMPONENT::X_CC_UPPER_RIGHT)
        {
          // find the upper right corner of the track described in camera coords at z=1
          auto i_POS_XYZ = from.indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y, COMPONENT::POS_Z });
          if (not i_POS_XYZ.has_value())
          {
            return std::nullopt;
          }
          UDT right_corner = UDT{ -std::numeric_limits<double>::infinity() };
          for (REFERENCE_POINT to_rp : { REFERENCE_POINT::FRONT_LEFT,
                                         REFERENCE_POINT::BACK_LEFT,
                                         REFERENCE_POINT::BACK_RIGHT,
                                         REFERENCE_POINT::FRONT_RIGHT })
          {
            Vector<UDT> corner = transform(state, from, REFERENCE_POINT::CENTER, to_rp);
            Vector<UDT> const& z_tc = corner(i_POS_XYZ.value());
            Vector<UDT> corner_sc = sensor_pose.cast<UDT>() * z_tc.homogeneous();
            if (corner_sc(i_POS_XYZ.value()(2)) < 0)  // object is behind camera
            {
              return std::nullopt;
            }
            if (corner_sc(0) / corner_sc(2) > right_corner)
            {
              right_corner = corner_sc(0) / corner_sc(2);
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
          UDT left_corner = UDT{ std::numeric_limits<double>::infinity() };

          for (REFERENCE_POINT to_rp : { REFERENCE_POINT::FRONT_LEFT,
                                         REFERENCE_POINT::BACK_LEFT,
                                         REFERENCE_POINT::BACK_RIGHT,
                                         REFERENCE_POINT::FRONT_RIGHT })
          {
            Vector<UDT> corner = transform(state, from, REFERENCE_POINT::CENTER, to_rp);
            Vector<UDT> const& z_tc = corner(i_POS_XYZ.value());
            Vector<UDT> corner_sc = sensor_pose.cast<UDT>() * z_tc.homogeneous();
            if (corner_sc(i_POS_XYZ.value()(2)) < 0)
            {
              return std::nullopt;
            }
            if (corner_sc(0) / corner_sc(2) < left_corner)
            {
              left_corner = corner_sc(0) / corner_sc(2);
            }
          }
          return left_corner;
        }
        if (to._comps.at(toInd) == COMPONENT::Y_CC_UPPER_RIGHT)
        {
          // find the upper right corner of the track described in camera coords at z=1
          auto i_POS_XYZ = from.indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y, COMPONENT::POS_Z });
          if (not i_POS_XYZ.has_value())
          {
            return std::nullopt;
          }
          Vector<UDT> corner = transform(state, from, REFERENCE_POINT::CENTER, REFERENCE_POINT::TOP);
          Vector<UDT> const& z_tc = corner(i_POS_XYZ.value());
          Vector<UDT> corner_sc = sensor_pose.cast<UDT>() * z_tc.homogeneous();
          if (corner_sc(i_POS_XYZ.value()(2)) < 0)
          {
            return std::nullopt;
          }
          return corner_sc(1) / corner_sc(2);
        }
        if (to._comps.at(toInd) == COMPONENT::Y_CC_LOWER_LEFT)
        {
          // find the lower left corner of the track described in camera coords at z=1
          auto i_POS_XYZ = from.indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y, COMPONENT::POS_Z });
          if (not i_POS_XYZ.has_value())
          {
            return std::nullopt;
          }
          Vector<UDT> corner = transform(state, from, REFERENCE_POINT::CENTER, REFERENCE_POINT::BOTTOM);
          Vector<UDT> const& z_tc = corner(i_POS_XYZ.value());
          Vector<UDT> corner_sc = sensor_pose.cast<UDT>() * z_tc.homogeneous();
          if (corner_sc(i_POS_XYZ.value()(2)) < 0)
          {
            return std::nullopt;
          }
          return corner_sc(1) / corner_sc(2);
        }
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
  assert(false);
  return {};
}

template GaussianDistribution<double> transform(GaussianDistribution<double> const& dist,
                                                StateModel<double> const& sourceStateModel,
                                                StateModel<double> const& targetStateModel);
template GaussianDistribution<AD> transform(GaussianDistribution<AD> const& dist,
                                            StateModel<AD> const& sourceStateModel,
                                            StateModel<AD> const& targetStateModel);
template <class UDT>
GaussianDistribution<UDT> transform(GaussianDistribution<UDT> const& dist,
                                    StateModel<UDT> const& sourceStateModel,
                                    StateModel<UDT> const& targetStateModel)
{
  if (sourceStateModel.id() == targetStateModel.id())
  {
    return dist;
  }
  std::optional<Transformed<UDT>> transformed =
      transform(dist.mean(), dist.covariance(), sourceStateModel.state_comps(), targetStateModel.state_comps());
  if (transformed.has_value())
  {
    return { std::move(transformed.value().mean), std::move(transformed.value().cov), dist.ref_point() };
  }
  LOG_INF_THROTTLE(10,
                   "Transformation from state model: " << sourceStateModel.id()
                                                       << " to model: " << targetStateModel.id()
                                                       << " not possible - Using default values");
  Components const transformable = transformableComps(sourceStateModel.state_comps(),
                                                      TransformOptions{ targetStateModel.manager()
                                                                            .params()
                                                                            ->state_model_params(targetStateModel.id())
                                                                            .assume_orient_in_velocity_direction })
                                       .all.intersection(targetStateModel.state_comps());
  Vector<UDT> mean(targetStateModel.state_comps()._comps.size());
  Matrix<UDT> cov = Matrix<UDT>::Identity(static_cast<Index>(targetStateModel.state_comps()._comps.size()),
                                          static_cast<Index>(targetStateModel.state_comps()._comps.size()));
  std::optional<Transformed<UDT>> transformedReduce =
      transform(dist.mean(), dist.covariance(), sourceStateModel.state_comps(), transformable);
  assert([&] {
    if (not transformedReduce.has_value())
    {
      LOG_FATAL(std::format("Problem to transform from {} to {}", sourceStateModel.state_comps(), transformable));
      return false;
    }
    return true;
  }());
  Indices const reducedInds = targetStateModel.state_comps().indexOf(transformable._comps).value();
  mean(reducedInds) = std::move(transformedReduce.value().mean);
  cov(reducedInds, reducedInds) = std::move(transformedReduce.value().cov);
  Components const remainingComps = targetStateModel.state_comps().diff(transformable);
  for (COMPONENT comp : remainingComps._comps)
  {
    if (not targetStateModel.manager()
                .params()
                ->state_model_params(targetStateModel.id())
                .default_mean.contains(to_string(comp)))
    {
      LOG_FATAL("State Model: " << targetStateModel.id()
                                << " does not contain default mean for component: " << to_string(comp));
      throw std::runtime_error("missing default mean component");
    }
    mean(targetStateModel.state_comps().indexOf(comp).value()) =
        targetStateModel.manager().params()->state_model_params(targetStateModel.id()).default_mean.at(to_string(comp));
    if (not targetStateModel.manager()
                .params()
                ->state_model_params(targetStateModel.id())
                .default_var.contains(to_string(comp)))
    {
      LOG_FATAL("State Model: " << targetStateModel.id()
                                << " does not contain default var for component: " << to_string(comp));
      throw std::runtime_error("missing default var component");
    }
    cov(targetStateModel.state_comps().indexOf(comp).value(), targetStateModel.state_comps().indexOf(comp).value()) =
        targetStateModel.manager().params()->state_model_params(targetStateModel.id()).default_var.at(to_string(comp));
  }
  return { std::move(mean), std::move(cov), dist.ref_point() };
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
    // this is slightly incorrect, if the object has no extent, the transformation would be, in fact, linear
    nonLinear.push_back(COMPONENT::X_CC_LOWER_LEFT);
    nonLinear.push_back(COMPONENT::X_CC_UPPER_RIGHT);
    nonLinear.push_back(COMPONENT::Y_CC_LOWER_LEFT);
    nonLinear.push_back(COMPONENT::Y_CC_UPPER_RIGHT);
  }
  if (comps.indexOf({ COMPONENT::HULL_POS_X_MAX, COMPONENT::HULL_POS_X_MIN }).has_value())
  {
    nonLinear.push_back(COMPONENT::POS_X);
  }
  if (comps.indexOf({ COMPONENT::HULL_POS_Y_MAX, COMPONENT::HULL_POS_Y_MIN }).has_value())
  {
    nonLinear.push_back(COMPONENT::POS_Y);
  }
  if (comps.indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y }).has_value())
  {
    // this is slightly incorrect, if the object has no extent, the transformation would be, in fact, linear
    nonLinear.push_back(COMPONENT::HULL_POS_X_MIN);
    nonLinear.push_back(COMPONENT::HULL_POS_X_MAX);
    nonLinear.push_back(COMPONENT::HULL_POS_Y_MIN);
    nonLinear.push_back(COMPONENT::HULL_POS_Y_MAX);
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
