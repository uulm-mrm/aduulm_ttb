#pragma once
#include "tracking_lib/StateModels/CPStateModel.h"
#include "tracking_lib/StateModels/CVStateModel.h"
#include "tracking_lib/StateModels/CAStateModel.h"
#include "tracking_lib/StateModels/ISCATRStateModel.h"
#include "tracking_lib/StateModels/CTP.h"
#include "tracking_lib/StateModels/CTRVStateModel.h"
#include "tracking_lib/StateModels/CTRAStateModel.h"

#include "tracking_lib/Distributions/GaussianDistribution.h"
#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/Transformations/Transformation.h"

namespace ttb
{

class EgoMotionDistribution;
class BaseDistribution;

template <class UDT>
struct EgoCompensated
{
  Vector<UDT> state;  ///< the egoMotion compensated State
  Matrix<UDT> cov;    ///< the egoMotion compensated covariance
};

/// Compensate the EgoMotion for the given state
/// Assumes:    EgoMotion in TrackingPlatform/Frame Coords, i.e., X-Axis to the front, Y-Axis to the left, Z-Asis to the
/// Top
template <class UDT>
[[nodiscard]] EgoCompensated<UDT> compensateEgoMotion(Vector<UDT> const& state,
                                                      Matrix<UDT> const& cov,
                                                      Components const& comps,
                                                      Duration dt,
                                                      EgoMotionDistribution const& egoMotion);

/// State Model Interface
template <class UDT>
class StateModel
{
public:
  using AllModels = std::variant<CPStateModel<UDT>,
                                 CVStateModel<UDT>,
                                 CAStateModel<UDT>,
                                 CTP<UDT>,
                                 ISCATRStateModel<UDT>,
                                 CTRVStateModel<UDT>,
                                 CTRAStateModel<UDT>>;

  /// check if required parameters are set for this model
  [[nodiscard]] bool is_valid() const;

  [[nodiscard]] STATE_MODEL_TYPE type() const
  {
    return std::visit([&](auto const& model) { return model.type(); }, _model);
  }
  /// predict a single vector
  [[nodiscard]] Vector<UDT> predict(Duration dt, Vector<UDT> const& state_vector) const
  {
    EgoMotionDistribution zero = EgoMotionDistribution::zero();
    GaussianDistribution<UDT> dist(state_vector, Matrix<UDT>::Identity(state_vector.rows(), state_vector.rows()));
    return std::visit([&](auto const& model) { return model.predict(dt, dist, zero).mean(); }, _model);
  }
  /// predict the distribution + compensate EgoMotion
  [[nodiscard]] GaussianDistribution<UDT> predict(Duration dt,
                                                  GaussianDistribution<UDT> const& dist,
                                                  EgoMotionDistribution const& ego) const
  {
    return std::visit([&](auto const& model) { return model.predict(dt, dist, ego); }, _model);
  }
  /// compensate EgoMotion only
  [[nodiscard]] GaussianDistribution<UDT> compensateEgoMotion(Duration dt,
                                                              GaussianDistribution<UDT> const& dist,
                                                              EgoMotionDistribution const& egoMotion) const;
  /// the state components
  [[nodiscard]] Components const& state_comps() const;
  /// the noise components
  [[nodiscard]] Components const& noise_comps() const;
  /// use this noise value for this component
  [[nodiscard]] double std_noise(COMPONENT comp) const;
  /// the unique id of this state model
  [[nodiscard]] StateModelId id() const
  {
    return _id;
  }
  /// access to the manager
  [[nodiscard]] TTBManager<UDT> const& manager() const
  {
    return _manager;
  }

  TTBManager<UDT> const& _manager;
  StateModelId _id;
  AllModels _model;
};

template <class UDT>
[[nodiscard]] double std_noise(TTBManager<UDT> const& manager, StateModelId id, COMPONENT comp);

template <class UDT>
[[nodiscard]] GaussianDistribution<UDT> predict_linear(Duration dt,
                                                       GaussianDistribution<UDT> const& dist,
                                                       EgoMotionDistribution const& egoDist,
                                                       Components const& state_comps,
                                                       ProcessMatrices<UDT> const& kinematic);

template <class F, class UDT>
[[nodiscard]] GaussianDistribution<UDT> predict_non_linear(TTBManager<UDT> const& manager,
                                                           StateModelId id,
                                                           Duration dt,
                                                           GaussianDistribution<UDT> const& dist,
                                                           EgoMotionDistribution const& egoDist,
                                                           Components const& state_comps,
                                                           Components const& noise_comps,
                                                           F&& kinematic)
{
  // hpx::tracy::region("NonLinearStateModel::predict", 0, 0);
  LOG_DEB("NonlinearModel predict");
  if (dt < std::chrono::microseconds(1))
  {
    return dist;
  }
  Vector<UDT> const x = dist.mean();
  Matrix<UDT> const P = dist.covariance();
  Vector<UDT> sigma(noise_comps._comps.size());
  for (auto const& [i, comp] : std::views::enumerate(noise_comps._comps))
  {
    double const std = std_noise(manager, id, comp);
    sigma(i) = std * std;
    assert([&] {
      if (sigma(i) == 0)
      {
        LOG_FATAL(
            std::format("invalid error variance {} noise comp {} dt {}", sigma(i), std_noise(manager, id, comp), dt));
        LOG_FATAL(std::format("State model id {}", id));
        LOG_FATAL(to_string(comp));
        return false;
      }
      return true;
    }());
  }
  Matrix<UDT> Q = sigma.asDiagonal();
  std::optional<transformation::Transformed<UDT>> trafo = transformation::unscentTransform(
      x,
      P,
      Q,
      [&](Vector<UDT> const& state, Vector<UDT> const& noise) -> std::optional<Vector<UDT>> {
        Vector<UDT> predicted_state = state;
        predicted_state =
            compensateEgoMotion<UDT>(state, Matrix<UDT>::Identity(state.rows(), state.rows()), state_comps, dt, egoDist)
                .state;
        predicted_state = kinematic(dt, predicted_state, noise);
        return predicted_state;
      },
      state_comps.indexOf(COMPONENT::ROT_Z));
  return { std::move(trafo.value().mean), std::move(trafo.value().cov), dist.ref_point() };
}

}  // namespace ttb
