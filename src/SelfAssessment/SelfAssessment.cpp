#include "tracking_lib/SelfAssessment/SelfAssessment.h"
#include <Eigen/src/Core/ArithmeticSequence.h>

#include "tracking_lib/Misc/Region.h"
#include "tracking_lib/States/State.h"
#include "tracking_lib/Misc/Numeric.h"
#include "tracking_lib/TTBManager/TTBManager.h"
#include <aduulm_logger/aduulm_logger.hpp>
#include <ranges>
#include <stacktrace>
#include <boost/math/distributions/beta.hpp>
#include <boost/math/distributions/gamma.hpp>
#include <subjective_logic_lib/multi_source/trusted_fusion_operators.hpp>
#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/beta.hpp>
#include "tracking_lib/Misc/SeparatingAxisTest.h"
#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/Transformations/Transformation.h"

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/geometry/index/rtree.hpp>

namespace ttb
{

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;
using point = bg::model::point<double, 2, bg::cs::cartesian>;
using Box = bg::model::box<point>;

constexpr bool use_approximated_estimations = true;

std::string to_string(BayesFactorProfilerData const& data)
{
  return std::format("{}, {}, {}\n", data.meas_id, to_nanoseconds(data.time.time_since_epoch()), data.bf);
}

std::string to_string(ClutterEstimate_ProfilerData const& data)
{
  std::string out;
  for (auto const& [id, est] : data._estimation)
  {
    out += id.value_ + ", " + std::to_string(to_nanoseconds(data._time.time_since_epoch())) + ", " +
           std::to_string(std::get<0>(est)) + ", " + std::to_string(std::get<1>(est)) + ", " +
           std::to_string(std::get<2>(est)) + "\n";
  }
  return out;
}

std::string to_string(DetectionEstimate_ProfilerData const& data)
{
  std::string out;
  for (auto const& [id, est] : data._estimation)
  {
    out += id.value_ + ", " + std::to_string(to_nanoseconds(data._time.time_since_epoch())) + ", " +
           std::to_string(std::get<0>(est)) + ", " + std::to_string(std::get<1>(est)) + ", " +
           std::to_string(std::get<2>(est)) + "\n";
  }
  return out;
}

double integrate(Vector<double> const& steps, Vector<double> const& vals)
{
  assert(steps.rows() >= 2);
  assert(steps.rows() == vals.rows());
  double const dx = steps(1) - steps(0);
  return dx * (vals.sum());
}

template class PersistenceModelAssessment<double>;
template class PersistenceModelAssessment<AD>;

template <class UDT>
PersistenceModelAssessment<UDT>::PersistenceModelAssessment(TTBManager<UDT>* manager)
  : _manager{ manager }, _data{ [&] {
    Vector<double> steps(1000);
    double const step_size = (1.0) / steps.rows();

    for (Index ps_ind = 0; ps_ind < steps.rows(); ++ps_ind)
    {
      double const ps = ps_ind * step_size + step_size / 2;
      steps(ps_ind) = ps;
    }

    BetaDistribution::Params const params =
        BetaDistribution::from_mean_var(_manager->params()->self_assessment.persistence_prob_estimation.prior_mean,
                                        _manager->params()->self_assessment.persistence_prob_estimation.prior_var);
    boost::math::beta_distribution const prior(params.p, params.q);

    Vector<double> prior_v(1000);
    for (Index ps_ind = 0; ps_ind < 1000; ++ps_ind)
    {
      double const ps = ps_ind * step_size + step_size / 2;
      prior_v(ps_ind) = boost::math::pdf(prior, ps);
    }
    prior_v /= integrate(steps, prior_v);
    return GridData{ steps, prior_v, prior_v };
  }() }
{
}

template <class UDT>
void PersistenceModelAssessment<UDT>::reset()
{
  *this = PersistenceModelAssessment(_manager);
}

template <class UDT>
void PersistenceModelAssessment<UDT>::update(std::pair<std::vector<Matrix<UDT>>, Duration> const& surviving_info)
{
  LOG_DEB("persistence prob update");
  // ZoneScopedN("PersistenceModelAssessment::update");
  if (surviving_info.second < 1ms or surviving_info.second > 10s)
  {
    return;
  }
  assert(std::abs(integrate(_data._steps, _data._prior) - 1) < 1e-7);
  Vector<double> post = _data._prior;
  for (auto const& surv_info : surviving_info.first)
  {
    Vector<double> new_post = Vector<double>::Zero(_data._steps.rows());
    for (Index p = 0; p < surv_info.rows(); ++p)
    {
      for (Index v = 0; v < surv_info.cols(); ++v)
      {
        Vector<double> likelihood = post;
        for (Index ind = 0; ind < _data._steps.rows(); ++ind)
        {
          double const ps = std::pow(_data._steps(ind), to_seconds(surviving_info.second));
          likelihood(ind) *= std::pow(ps, p) * std::pow(1 - ps, v);
          assert(likelihood(ind) >= 0);
        }
        likelihood /= integrate(_data._steps, likelihood);
        new_post += likelihood * std::exp(to_val(surv_info(p, v)));
      }
    }
    new_post /= integrate(_data._steps, new_post);
    post = std::move(new_post);
  }
  _data._posterior = std::move(post);

  if (_manager->params()->show_gui)
  {
    _manager->vizu()._cycle_data.wlock().data.persistence_prob_estimation.emplace_back(
        _manager->filter_time(), mean(), quantile(0.025), quantile(1 - 0.025));
  }
  auto [it, _] = _manager->_profilers.try_emplace("persistence_estimation", "/tmp/persistence");
  it->second.addData(to_string(PersistenceEstimate_ProfilerData{
      .mean = mean(), .lower = quantile(0.025), .upper = quantile(1 - 0.025), ._time = _manager->filter_time() }));
}

template <class UDT>
double PersistenceModelAssessment<UDT>::quantile(double alpha) const
{
  return _data.quantile(alpha);
}

template <class UDT>
double PersistenceModelAssessment<UDT>::mean() const
{
  return _data.mean();
}

std::string to_string(PersistenceEstimate_ProfilerData const& data)
{
  return std::format(
      "{}, {}, {}, {}\n", to_nanoseconds(data._time.time_since_epoch()), data.mean, data.lower, data.upper);
}

template class TrackingAssessment<double>;
template class TrackingAssessment<AD>;

template <class UDT>
TrackingAssessment<UDT>::TrackingAssessment(TTBManager<UDT>* manager)
  : _manager{ manager }, _model_assessment{ manager }, _unmodeled_aspects{ manager }
{
}

template class ClutterAssessmentMethodBayesFactor<double>;
template class ClutterAssessmentMethodBayesFactor<AD>;

template <class UDT>
void ClutterAssessmentMethodBayesFactor<UDT>::reset()
{
  _old_meas_likelihood.clear();
  _old_prior_weight_clutter.clear();
}

template <class UDT>
void ClutterAssessmentMethodBayesFactor<UDT>::update_clutter_bf(DataSourceContainer<UDT> const& measurement_container,
                                                                std::map<UDT, UDT> const& num_detected_and_weights,
                                                                UDT reference_measurement_likelihood)
{
  double const fov_area =
      measurement_container._sensorInfo._sensor_fov.value().area(Components({ COMPONENT::POS_X, COMPONENT::POS_Y }));

  std::vector<double> alternative_log_likelihoods;
  for (auto const& [num_detected, weight] : num_detected_and_weights)
  {
    GammaDistributionParams params = GammaDistribution::from_mean_var(
        _manager->params()->self_assessment.bayes_factor.clutter.comparison.mean / fov_area,
        _manager->params()->self_assessment.bayes_factor.clutter.comparison.var / (fov_area * fov_area));
    alternative_log_likelihoods.push_back(
        to_val(weight) +
        boost::math::lgamma(measurement_container._measurements.size() + params.alpha - to_val(num_detected)) -
        boost::math::lgamma(params.alpha) + params.alpha * std::log(params.beta) -
        (measurement_container._measurements.size() + params.alpha - to_val(num_detected)) *
            std::log(fov_area + params.beta));
  }
  double const alternative_log_likelihood = numeric::logsumexp(alternative_log_likelihoods);
  double const clutter_bf = std::exp(to_val(reference_measurement_likelihood) - alternative_log_likelihood);
  auto [profiler, _] = _manager->_profilers.try_emplace("clutter_bf", "/tmp/clutter_bf");
  profiler->second.addData(to_string(BayesFactorProfilerData{
      .meas_id = measurement_container._source_id.value_, .time = measurement_container._time, .bf = clutter_bf }));
  _bayes_factor[measurement_container._source_id] = clutter_bf;
  _cum_bayes_factor[measurement_container._source_id] =
      std::min(std::log(1e3),
               std::max(std::log(1e-3),
                        _cum_bayes_factor[measurement_container._source_id] + to_val(reference_measurement_likelihood) -
                            alternative_log_likelihood));
  if (_manager->params()->show_gui)
  {
    _manager->vizu()._meas_model_data.wlock().data[measurement_container._source_id]._clutter_bayes_factor.emplace_back(
        measurement_container._time, clutter_bf);
    _manager->vizu()
        ._meas_model_data.wlock()
        .data[measurement_container._source_id]
        ._clutter_cum_bayes_factor.emplace_back(measurement_container._time,
                                                std::exp(_cum_bayes_factor.at(measurement_container._source_id)));
  }
}

template class ClutterAssessmentMethodEstimation<double>;
template class ClutterAssessmentMethodEstimation<AD>;

template <class UDT>
void ClutterAssessmentMethodEstimation<UDT>::reset()
{
  _grid_approx_clutter_estimate.clear();
}

template <class UDT>
void ClutterModelAssessment<UDT>::reset()
{
  _bf_method.reset();
  _estimation_method.reset();
}

template <class UDT>
double ClutterAssessmentMethodEstimation<UDT>::quantile(SourceId id, double alpha) const
{
  double sum = 0;
  assert(_grid_approx_clutter_estimate.contains(id));
  GridData const& data = _grid_approx_clutter_estimate.at(id);
  assert(data._steps.rows() >= 2);
  for (auto const [step, prob] : std::views::zip(data._steps, data._posterior))
  {
    if (sum >= alpha)
    {
      return step;
    }
    sum += prob * (data._steps(1) - data._steps(0));
  }
  return 1.0;
}

double GridData::mean() const
{
  return mean(_steps, _posterior);
}

double GridData::mean(Vector<double> const& steps, Vector<double> const& posterior)
{
  assert(steps.rows() == posterior.rows());
  assert(std::abs(integrate(steps, posterior) - 1) < 1e-7);
  return steps.dot(posterior) * (steps(1) - steps(0));
}

double GridData::quantile(double alpha) const
{
  double sum = 0;
  assert(_steps.rows() >= 2);
  double const dx = _steps(1) - _steps(0);
  for (auto const [step, prob] : std::views::zip(_steps, _posterior))
  {
    if (sum >= alpha)
    {
      return step;
    }
    sum += prob * dx;
  }
  return _steps(_steps.rows() - 1);
}

template <class UDT>
double ClutterAssessmentMethodEstimation<UDT>::mean(SourceId id) const
{
  assert(_grid_approx_clutter_estimate.contains(id));
  return _grid_approx_clutter_estimate.at(id).mean();
}

template class DetectionModelAssessmentEstimationMethod<double>;
template class DetectionModelAssessmentEstimationMethod<AD>;

template <class UDT>
double DetectionModelAssessmentEstimationMethod<UDT>::quantile(SourceId id, double alpha) const
{
  assert(_grid_based_detection_estimate.contains(id));
  return _grid_based_detection_estimate.at(id).quantile(alpha);
}

template <class UDT>
double DetectionModelAssessmentEstimationMethod<UDT>::mean(SourceId id) const
{
  assert(_grid_based_detection_estimate.contains(id));
  return _grid_based_detection_estimate.at(id).mean();
}

template <class UDT>
Vector<double> update_grid_clutter(Vector<double> const& steps,
                                   Vector<double> const& prior,
                                   Vector<UDT> const& clutter_dist)
{
  Vector<double> new_post = Vector<double>::Zero(prior.rows());
  for (Index c = 0; c < clutter_dist.rows(); ++c)
  {
    Vector<double> likelihood = prior;
    for (Index ind = 0; ind < steps.rows(); ++ind)
    {
      double const rate = steps(ind);
      likelihood(ind) *= std::exp(-rate + c * std::log(rate) - boost::math::lgamma(c + 1));
    }
    likelihood /= integrate(steps, likelihood);
    new_post += likelihood * to_val(clutter_dist(c));
  }
  new_post /= integrate(steps, new_post);
  return new_post;
}

template <class UDT>
Vector<double> update_grid_clutter(Vector<double> const& steps,
                                   Vector<double> const& prior,
                                   SlidingWindow<Vector<UDT>> const& clutter_dists)
{
  Vector<double> post = prior;
  for (auto const& clutter_dist : clutter_dists.data_buffer)
  {
    post = update_grid_clutter(steps, post, clutter_dist.data);
  }
  return post;
}

template <class UDT>
ClutterAssessmentMethodEstimation<UDT>::ClutterAssessmentMethodEstimation(TTBManager<UDT>* manager)
  : _manager{ manager }
{
}

template <class UDT>
void ClutterAssessmentMethodEstimation<UDT>::update_clutter_estimate(
    DataSourceContainer<UDT> const& measurement_container,
    Vector<UDT> const& clutter_distribution)
{
  // ZoneScopedN("ClutterAssessmentMethodEstimation::update_clutter_estimate");
  if (not _manager->params()->meas_model_params(measurement_container._source_id).clutter.rate_estimation.enable)
  {
    return;
  }
  if (not _grid_approx_clutter_estimate.contains(measurement_container._source_id))
  {
    Vector<double> steps = [&] {
      Vector<double> steps(100);
      double const step_size = (100.0) / steps.rows();

      for (Index ps_ind = 0; ps_ind < steps.rows(); ++ps_ind)
      {
        double const ps = ps_ind * step_size + step_size / 2;
        steps(ps_ind) = ps;
      }
      return steps;
    }();
    Vector<double> prior = [&] {
      Vector<double> prior_v(steps.rows());
      GammaDistributionParams params = GammaDistribution::from_mean_var(
          _manager->params()->meas_model_params(measurement_container._source_id).clutter.rate_estimation.prior_mean,
          _manager->params()->meas_model_params(measurement_container._source_id).clutter.rate_estimation.prior_var);
      boost::math::gamma_distribution const prior_dist(params.alpha, 1 / params.beta);
      for (Index ps_ind = 0; ps_ind < steps.rows(); ++ps_ind)
      {
        prior_v(ps_ind) = boost::math::pdf(prior_dist, steps(ps_ind));
      }
      prior_v /= integrate(steps, prior_v);
      return prior_v;
    }();
    _grid_approx_clutter_estimate.emplace(measurement_container._source_id,
                                          GridData{ ._steps = steps, ._prior = prior, ._posterior = prior });
  }

  GridData& data = _grid_approx_clutter_estimate.at(measurement_container._source_id);
  double const lambda = 0.99;
  data._posterior = data._posterior.array().pow(lambda) * data._prior.array().pow(1 - lambda);
  data._posterior /= integrate(data._steps, data._posterior);
  data._posterior = update_grid_clutter(data._steps, data._posterior, clutter_distribution);

  if (_manager->params()->meas_model_params(measurement_container._source_id).clutter.rate_estimation.use_in_filter)
  {
    Params next_params = *_manager->params();
    next_params.meas_model_params(measurement_container._source_id).clutter.rate =
        mean(measurement_container._source_id);
    _manager->update_params(std::move(next_params));
  }
  constexpr double bayes_factor = 0;
  if (_manager->params()->show_gui)
  {
    _manager->vizu()
        ._meas_model_data.wlock()
        .data[measurement_container._source_id]
        ._clutter_estimation_bayes_approx.emplace_back(
            measurement_container._time,
            mean(measurement_container._source_id),
            quantile(measurement_container._source_id,
                     (1 - _manager->params()->self_assessment.clutter_rate_estimation.quantile) / 2),
            quantile(measurement_container._source_id,
                     (1 - (1 - _manager->params()->self_assessment.clutter_rate_estimation.quantile) / 2)));
    _manager->vizu()._meas_model_data.wlock().data[measurement_container._source_id]._clutter_rate.emplace_back(
        measurement_container._time,
        _manager->params()->meas_model_params(measurement_container._source_id).clutter.rate.value_or(0.0));
  }

  auto [profiler, _] = _manager->_profilers.try_emplace("clutter_estimation", "/tmp/clutter_rate_estimation");
  profiler->second.addData(to_string(
      ClutterEstimate_ProfilerData{ ._estimation = { { measurement_container._source_id, { 0, 0, bayes_factor } } },
                                    ._time = measurement_container._time }));
}

[[nodiscard]] Opinion trust_revised_fuse_opinions(auto const it_first, auto const it_last)
{
  subjective_logic::Trust<double> default_trust{ 1.0, 0. };
  using TrustedOp = subjective_logic::TrustedOpinion<Opinion>;
  std::vector<TrustedOp> trusted_ops;
  for (auto it = it_first; it != it_last; ++it)
  {
    trusted_ops.emplace_back(default_trust, *it);
  }
  std::vector<subjective_logic::multisource::TrustedFusion::WeightedTypes> params{
    { subjective_logic::RelationType::CONFLICT,
      subjective_logic::TrustRevisionType::SHARES,
      subjective_logic::ConflictType::AVERAGE,
      10.0 }
  };
  return subjective_logic::multisource::TrustedFusion::fuse_opinions(
      subjective_logic::FusionType::AVERAGE, params, trusted_ops);
}

Opinion nis_to_opinion(double nis_value, std::size_t dof)
{
  boost::math::chi_squared_distribution chi2(dof);
  double const quan = 0.95;
  double const lower = [&] {
    try
    {
      return boost::math::quantile(chi2, (1 - quan) / 2);
    }
    catch (...)
    {
      return -std::numeric_limits<double>::infinity();
    }
  }();
  double const upper = [&] {
    try
    {
      return boost::math::quantile(chi2, quan + (1 - quan) / 2);
    }
    catch (...)
    {
      return std::numeric_limits<double>::infinity();
    }
  }();
  double const dist = [&] {
    if (lower <= nis_value and nis_value <= upper)
    {
      return 0.0;
    }
    if (lower > nis_value)
    {
      return lower - nis_value;
    }
    if (upper < nis_value)
    {
      return nis_value - upper;
    }
    assert(false);
    std::unreachable();
  }();
  double u = 1 - quan;
  if (dist == 0)
  {
    return Opinion{ Opinion::BeliefType{ 1 - u, 0.0 }, Opinion::BeliefType{ std::array{ 0.5, 0.5 } } };
  }
  return Opinion{ Opinion::BeliefType{ 0.0, 1 - u }, Opinion::BeliefType{ std::array{ 0.5, 0.5 } } };
}

template <class UDT>
Vector<double> update_grid_detection(Vector<double> const& steps,
                                     Vector<double> const& prior,
                                     std::vector<Matrix<UDT>> const& detection_distributions)
{
  Vector<double> post = prior;
  for (Matrix<UDT> const& detection_dist : detection_distributions)
  {
    Vector<double> new_post = Vector<double>::Zero(steps.rows());
    assert(std::abs(detection_dist.sum() - 1) < 1e-7);
    for (Index d = 0; d < detection_dist.rows(); ++d)
    {
      for (Index m = 0; m < detection_dist.cols(); ++m)
      {
        Vector<double> likelihood = post;
        for (Index ind = 0; ind < steps.rows(); ++ind)
        {
          assert(steps.rows() == likelihood.rows());
          double const pd = steps(ind);
          likelihood(ind) *= std::pow(pd, d) * std::pow(1 - pd, m);
          assert(likelihood(ind) >= 0);
        }
        likelihood /= integrate(steps, likelihood);
        new_post += likelihood * to_val(detection_dist(d, m));
      }
    }
    new_post /= integrate(steps, new_post);
    post = std::move(new_post);
  }
  return post;
}
template Vector<double> update_grid_detection(Vector<double> const& steps,
                                              Vector<double> const& prior,
                                              std::vector<Matrix<double>> const& detection_distributions);
template Vector<double> update_grid_detection(Vector<double> const& steps,
                                              Vector<double> const& prior,
                                              std::vector<Matrix<AD>> const& detection_distributions);

template <class UDT>
DetectionModelAssessmentEstimationMethod<UDT>::DetectionModelAssessmentEstimationMethod(TTBManager<UDT>* manager)
  : _manager{ manager }
  , short_term_detection_distributions(std::chrono::milliseconds(
        _manager->params()->self_assessment.detection_prob_estimation.short_term_window_length_ms))
  , long_term_detection_distributions(std::chrono::milliseconds(
        _manager->params()->self_assessment.detection_prob_estimation.long_term_window_length_ms))
{
}

Vector<double> update_grid_detections(Vector<double> const& steps,
                                      Vector<double> const& prior,
                                      SlidingWindow<std::vector<Matrix<double>>> const& detections)
{
  Vector<double> post = prior;
  for (auto const& vals : detections.data_buffer)
  {
    post = update_grid_detection(steps, post, vals.data);
  }
  return post;
}

template <class UDT>
void DetectionModelAssessmentEstimationMethod<UDT>::update_detection_estimate(
    DataSourceContainer<UDT> const& measurement_container,
    std::vector<Matrix<UDT>> const& detection_distributions)
{
  // ZoneScopedN("DetectionModelAssessmentEstimationMethod::update_detection_estimate");
  if (not _manager->params()->meas_model_params(measurement_container._source_id).detection.prob_estimation.enable)
  {
    return;
  }
  if (not _grid_based_detection_estimate.contains(measurement_container._source_id))
  {
    Vector<double> steps = [&] {
      Vector<double> steps(1000);
      double const step_size = 1.0 / steps.rows();
      for (Index ps_ind = 0; ps_ind < steps.rows(); ++ps_ind)
      {
        double const ps = ps_ind * step_size + step_size / 2;
        steps(ps_ind) = ps;
      }
      return steps;
    }();
    Vector<double> prior = [&] {
      BetaDistribution::Params params = BetaDistribution::from_mean_var(
          _manager->params()->meas_model_params(measurement_container._source_id).detection.prob_estimation.prior_mean,
          _manager->params()->meas_model_params(measurement_container._source_id).detection.prob_estimation.prior_var);
      boost::math::beta_distribution const prior_dist(params.p, params.q);

      Vector<double> prior_v(steps.rows());
      for (Index ps_ind = 0; ps_ind < steps.rows(); ++ps_ind)
      {
        prior_v(ps_ind) = boost::math::pdf(prior_dist, steps(ps_ind));
      }
      return prior_v;
    }();
    prior /= integrate(steps, prior);
    _grid_based_detection_estimate.emplace(measurement_container._source_id, GridData{ steps, prior, prior });
  }

  GridData& data = _grid_based_detection_estimate.at(measurement_container._source_id);
  double const lambda = 0.99;
  data._posterior = data._posterior.array().pow(lambda) * data._prior.array().pow(1 - lambda);
  data._posterior /= integrate(data._steps, data._posterior);
  data._posterior = update_grid_detection(data._steps, data._posterior, detection_distributions);
  if (_manager->params()->meas_model_params(measurement_container._source_id).detection.prob_estimation.use_in_filter)
  {
    Params next_params = *_manager->params();
    next_params.meas_model_params(measurement_container._source_id).detection.prob =
        std::min(mean(measurement_container._source_id), 0.99999);
    _manager->update_params(std::move(next_params));
  }
  constexpr double bayes_factor = 0;
  if (_manager->params()->show_gui)
  {
    _manager->vizu()
        ._meas_model_data.wlock()
        .data[measurement_container._source_id]
        ._detection_estimation_bayes_approx.emplace_back(
            measurement_container._time,
            mean(measurement_container._source_id),
            quantile(measurement_container._source_id,
                     (1 - _manager->params()->self_assessment.detection_prob_estimation.quantile) / 2),
            quantile(measurement_container._source_id,
                     1 - (1 - _manager->params()->self_assessment.detection_prob_estimation.quantile) / 2));
    _manager->vizu()
        ._meas_model_data.wlock()
        .data[measurement_container._source_id]
        ._detection_probability.emplace_back(
            measurement_container._time,
            _manager->params()->meas_model_params(measurement_container._source_id).detection.prob);
  }
  auto [profiler, _] =
      _manager->_profilers.try_emplace("detection_probability_estimation", "/tmp/detection_probability_estimation");

  profiler->second.addData(to_string(
      DetectionEstimate_ProfilerData{ ._estimation = { { measurement_container._source_id, { 0, 0, bayes_factor } } },
                                      ._time = measurement_container._time }));
}

template class DetectionModelAssessmentBayesFactor<double>;
template class DetectionModelAssessmentBayesFactor<AD>;

template <class UDT>
void DetectionModelAssessmentBayesFactor<UDT>::update_detection_bf(
    DataSourceContainer<UDT> const& measurement_container,
    std::map<std::pair<UDT, UDT>, UDT> const& num_detected_and_weights,
    UDT reference_measurement_likelihood)
{
  std::vector<double> alternative_log_likelihoods;
  for (auto const& [dm, weight] : num_detected_and_weights)
  {
    BetaDistribution::Params params = BetaDistribution::from_mean_var(
        _manager->params()->self_assessment.bayes_factor.detection.comparison.front().mean,
        _manager->params()->self_assessment.bayes_factor.detection.comparison.front().var);
    alternative_log_likelihoods.push_back(to_val(weight) + boost::math::lgamma(params.p + to_val(dm.first)) +
                                          boost::math::lgamma(params.q + to_val(dm.second)) -
                                          boost::math::lgamma(params.p + params.q + to_val(dm.first + dm.second)) -
                                          boost::math::lgamma(params.p) - boost::math::lgamma(params.q) +
                                          boost::math::lgamma(params.p + params.q));
  }
  double const alternative_log_likelihood = numeric::logsumexp(alternative_log_likelihoods);
  double const bayes_factor = std::exp(to_val(reference_measurement_likelihood) - alternative_log_likelihood);
  auto [profiler, _] = _manager->_profilers.try_emplace("detection_bf", "/tmp/detection_bf");
  profiler->second.addData(to_string(BayesFactorProfilerData{
      .meas_id = measurement_container._source_id.value_, .time = measurement_container._time, .bf = bayes_factor }));
  _bayes_factor[measurement_container._source_id] = bayes_factor;
  _cum_bayes_factor[measurement_container._source_id] =
      std::min(std::log(1e3),
               std::max(std::log(1e-3),
                        _cum_bayes_factor[measurement_container._source_id] + to_val(reference_measurement_likelihood) -
                            alternative_log_likelihood));

  if (_manager->params()->show_gui)
  {
    _manager->vizu()
        ._meas_model_data.wlock()
        .data[measurement_container._source_id]
        ._detection_bayes_factor.emplace_back(measurement_container._time, bayes_factor);
    _manager->vizu()
        ._meas_model_data.wlock()
        .data[measurement_container._source_id]
        ._detection_cum_bayes_factor.emplace_back(measurement_container._time,
                                                  std::exp(_cum_bayes_factor[measurement_container._source_id]));
  }
}

template <class UDT>
void StoachsticModelAssessment<UDT>::update_assessment_opinions(DataSourceContainer<UDT> const& measurement_container,
                                                                std::vector<State<UDT>> const& estimated_tracks)
{
  // ZoneScopedN("StochasticModelAssessment::update_assessment_opinions");

  Opinion const clutter_estimation_opinion = [&] -> Opinion {
    if (auto const it =
            _clutter_model._estimation_method._grid_approx_clutter_estimate.find(measurement_container._source_id);
        it != _clutter_model._estimation_method._grid_approx_clutter_estimate.end() and
        _manager->params()->meas_model_params(measurement_container._source_id).clutter.rate.has_value())
    {
      return _clutter_model._estimation_method.create_clutter_opinion(
          _clutter_model._estimation_method.quantile(
              measurement_container._source_id,
              (1 - _manager->params()->self_assessment.clutter_rate_estimation.quantile) / 2),
          _clutter_model._estimation_method.quantile(
              measurement_container._source_id,
              1 - (1 - _manager->params()->self_assessment.clutter_rate_estimation.quantile) / 2),
          _manager->params()->meas_model_params(measurement_container._source_id).clutter.rate.value());
    }
    return {};
  }();
  Opinion const clutter_bf_opinion = [&] -> Opinion {
    if (auto const it = _clutter_model._bf_method._bayes_factor.find(measurement_container._source_id);
        it != _clutter_model._bf_method._bayes_factor.end())
    {
      return _clutter_model._bf_method.to_opinion(it->second);
    }
    return {};
  }();
  Opinion const clutter_opinion = [&] { return clutter_estimation_opinion.average_fuse(clutter_bf_opinion); }();
  if (not _old_clutter_opinions.contains(measurement_container._source_id))
  {
    _old_clutter_opinions.emplace(
        measurement_container._source_id,
        boost::circular_buffer<Opinion>(_manager->params()->self_assessment.opinion_filter_size));
  }
  _old_clutter_opinions.at(measurement_container._source_id).push_back(clutter_opinion);
  _clutter_opinion[measurement_container._source_id] =
      trust_revised_fuse_opinions(_old_clutter_opinions.at(measurement_container._source_id).cbegin(),
                                  _old_clutter_opinions.at(measurement_container._source_id).cend());
  ;
  Opinion const detection_estimation_opinion = [&] -> Opinion {
    if (auto const it =
            _detection_model._estimation_method._grid_based_detection_estimate.find(measurement_container._source_id);
        it != _detection_model._estimation_method._grid_based_detection_estimate.end())
    {
      return {};
    }
    return {};
  }();
  Opinion const detection_bf_opinion = [&] -> Opinion {
    if (auto const it = _detection_model._bf_method._bayes_factor.find(measurement_container._source_id);
        it != _detection_model._bf_method._bayes_factor.end())
    {
      return _detection_model._bf_method.to_opinion(it->second);
    }
    return {};
  }();

  Opinion const detection_opinion = [&] { return detection_estimation_opinion.average_fuse(detection_bf_opinion); }();
  if (not _old_detection_opinions.contains(measurement_container._source_id))
  {
    _old_detection_opinions.emplace(
        measurement_container._source_id,
        boost::circular_buffer<Opinion>(_manager->params()->self_assessment.opinion_filter_size));
  }
  _old_detection_opinions.at(measurement_container._source_id).push_back(detection_opinion);
  _detection_opinion[measurement_container._source_id] =
      trust_revised_fuse_opinions(_old_detection_opinions.at(measurement_container._source_id).cbegin(),
                                  _old_detection_opinions.at(measurement_container._source_id).cend());

  /// nis values
  std::map<COMPONENT, std::pair<double, std::size_t>> nis_values;
  for (State<UDT> const& track : estimated_tracks)
  {
    if (not track._meta_data._nis.contains(measurement_container._source_id))
    {
      continue;
    }
    double best_weight = 0;
    std::map<COMPONENT, UDT> track_nis;
    for (auto const& [state_model, data] : track._meta_data._nis.at(measurement_container._source_id))
    {
      for (auto const& [weight, values] : data)
      {
        if (weight > best_weight)
        {
          best_weight = to_val(weight);
          track_nis = values;
        }
      }
    }
    for (auto const& [comp, nis] : track_nis)
    {
      nis_values[comp] = { nis_values[comp].first + to_val(nis), nis_values[comp].second + 1 };
    }
  }
  if (not _old_nis_values.contains(measurement_container._source_id))
  {
    _old_nis_values.emplace(measurement_container._source_id,
                            std::map<COMPONENT, boost::circular_buffer<std::pair<double, std::size_t>>>{});
  }
  if (not _nis_filtered_opinions.contains(measurement_container._source_id))
  {
    _nis_filtered_opinions.emplace(measurement_container._source_id, std::map<COMPONENT, Opinion>{});
  }
  if (not _old_nis_opinions.contains(measurement_container._source_id))
  {
    _old_nis_opinions.emplace(measurement_container._source_id, std::map<COMPONENT, boost::circular_buffer<Opinion>>{});
  }
  for (auto const& [comp, nis_dof] : nis_values)
  {
    if (not _old_nis_opinions.at(measurement_container._source_id).contains(comp))
    {
      _old_nis_opinions.at(measurement_container._source_id)
          .emplace(comp, boost::circular_buffer<Opinion>(_manager->params()->self_assessment.opinion_filter_size));
    }
    if (not _old_nis_values.at(measurement_container._source_id).contains(comp))
    {
      _old_nis_values.at(measurement_container._source_id)
          .emplace(comp,
                   boost::circular_buffer<std::pair<double, std::size_t>>(
                       _manager->params()->self_assessment.opinion_filter_size));
    }
    _old_nis_values.at(measurement_container._source_id).at(comp).push_back(nis_dof);

    _old_nis_opinions.at(measurement_container._source_id)
        .at(comp)
        .push_back(nis_to_opinion(
            std::accumulate(_old_nis_values.at(measurement_container._source_id).at(comp).begin(),
                            _old_nis_values.at(measurement_container._source_id).at(comp).end(),
                            0.0,
                            [](double old, std::pair<double, std::size_t> const& nd) { return old + nd.first; }),
            std::accumulate(_old_nis_values.at(measurement_container._source_id).at(comp).begin(),
                            _old_nis_values.at(measurement_container._source_id).at(comp).end(),
                            0,
                            [](double old, std::pair<double, std::size_t> const& nd) { return old + nd.second; })));
    _nis_filtered_opinions.at(measurement_container._source_id)[comp] =
        trust_revised_fuse_opinions(_old_nis_opinions.at(measurement_container._source_id).at(comp).cbegin(),
                                    _old_nis_opinions.at(measurement_container._source_id).at(comp).cend());
  }
  if (_manager->params()->show_gui)
  {
    for (auto const& [comp, nis_dof] : nis_values)
    {
      _manager->vizu()._meas_model_data.wlock().data[measurement_container._source_id]._nis_dof[comp].emplace_back(
          measurement_container._time, nis_dof.first, nis_dof.second);
    }
    _manager->vizu()._meas_model_data.wlock().data[measurement_container._source_id]._assessment.emplace_back(
        measurement_container._time,
        Assessment{ ._clutter_estimation = clutter_estimation_opinion,
                    ._clutter_bf = clutter_bf_opinion,
                    ._clutter = _clutter_opinion.at(measurement_container._source_id),
                    ._detection_estimation = detection_estimation_opinion,
                    ._detection_bf = detection_bf_opinion,
                    ._detection = _detection_opinion.at(measurement_container._source_id),
                    ._nis = _nis_filtered_opinions.at(measurement_container._source_id) });
  }
}

template <class UDT>
Opinion ClutterAssessmentMethodEstimation<UDT>::create_clutter_opinion(double estimation_lower_quantile,
                                                                       double estimation_upper_quantile,
                                                                       double parameter_used)
{
  double const lower = estimation_lower_quantile;
  double const upper = estimation_upper_quantile;
  double const dist = [&] {
    if (lower <= parameter_used and parameter_used <= upper)
    {
      return 0.0;
    }
    if (lower > parameter_used)
    {
      return lower - parameter_used;
    }
    if (upper < parameter_used)
    {
      return parameter_used - upper;
    }
    assert(false);
    std::unreachable();
  }();
  double u = (upper - lower) / 20;
  if (not std::isfinite(u))
  {
    u = 1;
  }
  double const max_dist = 5;
  return Opinion{ Opinion::BeliefType{ std::array{ std::clamp(1 - dist / max_dist - u, 0.0, 1.0),
                                                   std::clamp(dist / max_dist - u, 0.0, 1.0) } },
                  Opinion::BeliefType{ std::array{ 0.5, 0.5 } } };
}

template <class UDT>
Opinion DetectionModelAssessmentEstimationMethod<UDT>::create_detection_opinion(
    DirichletDistribution const& parameter_estimation,
    double parameter_used)
{
  boost::math::beta_distribution const beta(parameter_estimation._alpha(0), parameter_estimation._alpha(1));
  double const quan = 0.95;
  double const lower = [&] {
    try
    {
      return boost::math::quantile(beta, (1 - quan) / 2);
    }
    catch (...)
    {
      return -std::numeric_limits<double>::infinity();
    }
  }();
  double const upper = [&] {
    try
    {
      return boost::math::quantile(beta, quan + (1 - quan) / 2);
    }
    catch (...)
    {
      return std::numeric_limits<double>::infinity();
    }
  }();
  double const dist = [&] {
    if (lower <= parameter_used and parameter_used <= upper)
    {
      return 0.0;
    }
    if (lower > parameter_used)
    {
      return lower - parameter_used;
    }
    if (upper < parameter_used)
    {
      return parameter_used - upper;
    }
    assert(false);
    std::unreachable();
  }();
  double u = (upper - lower) / 0.3;
  if (not std::isfinite(u))
  {
    u = 1;
  }
  double const max_dist = 0.1;
  return Opinion{ Opinion::BeliefType{ std::array{ std::clamp(1 - dist / max_dist - u, 0.0, 1.0),
                                                   std::clamp(dist / max_dist - u, 0.0, 1.0) } },
                  Opinion::BeliefType{ std::array{ 0.5, 0.5 } } };
}

template <class UDT>
Opinion DetectionModelAssessmentBayesFactor<UDT>::to_opinion(double bayes_factor)
{
  Opinion::BeliefType const prior{ std::array{ 0.5, 0.5 } };
  if (bayes_factor > 10.0)
  {
    return Opinion{ Opinion::BeliefType{ std::array{ 0.99, 0.0 } }, prior };
  }
  if (bayes_factor > 3.2)
  {
    return Opinion{ Opinion::BeliefType{ std::array{ 0.7, 0.0 } }, prior };
  }
  if (bayes_factor > 1)
  {
    return Opinion{ Opinion::BeliefType{ std::array{ 0.1, 0.0 } }, prior };
  }
  if (bayes_factor > 1.0 / 3.2)
  {
    return Opinion{ Opinion::BeliefType{ std::array{ 0.0, 0.1 } }, prior };
  }
  if (bayes_factor > 1.0 / 10.0)
  {
    return Opinion{ Opinion::BeliefType{ std::array{ 0.0, 0.5 } }, prior };
  }
  return Opinion{ Opinion::BeliefType{ std::array{ 0.0, 0.8 } }, prior };
}

template <class UDT>
Opinion ClutterAssessmentMethodBayesFactor<UDT>::to_opinion(double bayes_factor)
{
  Opinion::BeliefType const prior{ std::array{ 0.5, 0.5 } };
  if (bayes_factor > 10.0)
  {
    return Opinion{ Opinion::BeliefType{ std::array{ 0.99, 0.0 } }, prior };
  }
  if (bayes_factor > 3.2)
  {
    return Opinion{ Opinion::BeliefType{ std::array{ 0.7, 0.0 } }, prior };
  }
  if (bayes_factor > 1)
  {
    return Opinion{ Opinion::BeliefType{ std::array{ 0.1, 0.0 } }, prior };
  }
  if (bayes_factor > 1.0 / 3.2)
  {
    return Opinion{ Opinion::BeliefType{ std::array{ 0.0, 0.1 } }, prior };
  }
  if (bayes_factor > 1.0 / 10.0)
  {
    return Opinion{ Opinion::BeliefType{ std::array{ 0.0, 0.5 } }, prior };
  }
  return Opinion{ Opinion::BeliefType{ std::array{ 0.0, 0.8 } }, prior };
}

template <class UDT>
Opinion create_track_trajectory_opinion(std::vector<StateId> trajectory, std::optional<Map> const& map)
{
  return Opinion::VacuousBeliefOpinion();
}

template <class UDT>
std::optional<std::reference_wrapper<Map const>> get_map(State<UDT> const& state)
{
  TTBManager<UDT> const* const manager = state._manager;
  auto const [first, last] = manager->database().data_containers.template get<Time>().equal_range(state._time);
  for (auto it = first; it != last; ++it)
  {
    if (it->_map.has_value())
    {
      return std::reference_wrapper{ it->_map.value() };
    }
  }
  return {};
}

struct MapInfo
{
  bool on_usable_way;
  bool inside_building;
};

template <class UDT>
std::optional<MapInfo> get_map_info(Map const& map, State<UDT> const& track)
{
  auto const [model_id, dist] = track.getEstimate();
  std::optional<Indices> xy_inds =
      track._manager->state_comps(model_id).indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y });
  if (not xy_inds.has_value())
  {
    return {};
  }
  CLASS const estimated_class = track._classification.getEstimate();
  bool on_way = false;
  for (Way const& way : map.ways)
  {
    if (way.overlaps(track) and std::ranges::find(way.usable_by, estimated_class) != way.usable_by.end())
    {
      on_way = true;
      break;
    }
  }
  bool inside_building = false;
  for (Building const& building : map.buildings)
  {
    if (building.overlaps(track))
    {
      inside_building = true;
      break;
    }
  }
  return MapInfo{ .on_usable_way = on_way, .inside_building = inside_building };
}

template <class UDT>
[[nodiscard]] std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
bounding_points(State<UDT> const& track, Duration prediction_horizon, double min_distance)
{
  auto const& [model_id, dist] = track.getEstimate();
  std::vector<std::vector<double>> xs, ys;
  Components const& comps = track._manager->state_comps(model_id);
  constexpr std::size_t num_steps = 5;
  for (std::size_t step = 0; step < num_steps; ++step)
  {
    xs.emplace_back(std::vector<double>{});
    ys.emplace_back(std::vector<double>{});
    Duration const dt = std::chrono::milliseconds(static_cast<std::size_t>(to_milliseconds(prediction_horizon) / step));
    Vector<double> predicted_mean = to_val(track._manager->getStateModel(model_id).predict(dt, dist.mean()));
    if (auto const ind = comps.indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y, COMPONENT::RADIUS }); ind.has_value())
    {
      Vector2 const m = predicted_mean(ind.value()({ 0, 1 }));
      double const r = predicted_mean(ind.value()(2)) + min_distance / 2;
      for (double ts = 0; ts < 2 * std::numbers::pi; ts += 0.05)
      {
        xs.at(step).push_back(m(0) + std::cos(ts) * r);
        ys.at(step).push_back(m(1) + std::sin(ts) * r);
      }
    }
    else if (comps
                 .indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y, COMPONENT::ROT_Z, COMPONENT::LENGTH, COMPONENT::WIDTH })
                 .has_value())
    {
      predicted_mean(comps.indexOf({ COMPONENT::LENGTH, COMPONENT::WIDTH }).value()).array() += min_distance / 2;
      for (REFERENCE_POINT const rp : { REFERENCE_POINT::FRONT_RIGHT,
                                        REFERENCE_POINT::FRONT_LEFT,
                                        REFERENCE_POINT::BACK_LEFT,
                                        REFERENCE_POINT::BACK_RIGHT })
      {
        auto ind_xy = comps.indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y });
        if (not ind_xy.has_value())
        {
          return {};
        }
        Vector2 const& corner =
            transformation::transform(predicted_mean, comps, REFERENCE_POINT::CENTER, rp)(ind_xy.value());
        xs.at(step).emplace_back(corner(0));
        ys.at(step).emplace_back(corner(1));
      }
      xs.at(step).emplace_back(xs.at(step).front());
      ys.at(step).emplace_back(ys.at(step).front());
    }
    else if (auto const xy = comps.indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y }); xy.has_value())
    {
      Vector2 const m = predicted_mean(xy.value());
      double const r = min_distance / 2;
      for (double ts = 0; ts < 2 * std::numbers::pi; ts += 0.05)
      {
        xs.at(step).push_back(m(0) + std::cos(ts) * r);
        ys.at(step).push_back(m(1) + std::sin(ts) * r);
      }
    }
    else
    {
      LOG_FATAL("Do not know how to plot distribuion with comsp: " << comps.toString());
      assert(false);
    }
  }
  return { std::move(xs), std::move(ys) };
}

template <class UDT>
[[nodiscard]] Box axis_aligned_box(State<UDT> const& track, Duration prediction_horizon, double min_distance)
{
  auto const [xss, yss] = bounding_points(track, prediction_horizon, min_distance);
  double tminx = std::numeric_limits<double>::infinity();
  double tmaxx = -std::numeric_limits<double>::infinity();
  double tminy = std::numeric_limits<double>::infinity();
  double tmaxy = -std::numeric_limits<double>::infinity();
  for (auto const& [xs, ys] : std::views::zip(xss, yss))
  {
    auto const [minx, maxx] = std::ranges::minmax(xs);
    auto const [miny, maxy] = std::ranges::minmax(ys);
    tminx = std::min(minx, tminx);
    tminy = std::min(miny, tminy);
    tmaxx = std::max(maxx, tmaxx);
    tmaxy = std::max(maxy, tmaxy);
  }
  return Box(point(tminx, tminy), point(tmaxx, tmaxy));
}

template <class UDT>
[[nodiscard]] bool
check_overlapping(State<UDT> const& first, State<UDT> const& second, Duration prediction_horizon, double min_distance)
{
  auto const [fxs, fys] = bounding_points(first, prediction_horizon, min_distance);
  auto const [sxs, sys] = bounding_points(second, prediction_horizon, min_distance);
  assert(fxs.size() == fys.size());
  assert(sxs.size() == sys.size());
  assert(fxs.size() == sys.size());
  for (std::size_t i = 0; i < fxs.size(); ++i)
  {
    Polygon::SinglePolygon first_poly;
    for (auto const& [x, y] : std::views::zip(fxs.at(i), fys.at(i)))
    {
      boost::geometry::append(first_poly, Point(x, y));
    }
    Polygon::SinglePolygon first_hull;
    boost::geometry::convex_hull(first_poly, first_hull);
    Polygon::SinglePolygon second_poly;
    for (auto const& [x, y] : std::views::zip(sxs.at(i), sys.at(i)))
    {
      boost::geometry::append(second_poly, Point(x, y));
    }
    Polygon::SinglePolygon second_hull;
    boost::geometry::convex_hull(second_poly, second_hull);
    if (bg::intersects(first_hull, second_hull))
    {
      return true;
    }
  }
  return false;
}

template class UnmodeledAspectsAssessment<double>;
template class UnmodeledAspectsAssessment<AD>;

template <class UDT>
void UnmodeledAspectsAssessment<UDT>::evaluate_overlapping(std::vector<State<UDT>> const& current_tracks)
{
  // ZoneScopedN("UnmodeledAspectsAssessment::evaluate_overlapping");
  if (current_tracks.empty())
  {
    return;
  }
  std::vector<StateId> ids;
  for (State<UDT> const& state : current_tracks)
  {
    ids.push_back(state._id);
  }
  std::vector<std::tuple<StateId, StateId, Probability<double>>> overlapping =
      overlapping_tracks(*_manager, ids, 0ms, 0.0);
  Time time = current_tracks.front()._time;
  std::vector<State<UDT>> overlapping_states;
  for (auto const& [first, second, prob] : overlapping)
  {
    for (StateId state_id : { first, second })
    {
      State<UDT> const& state = _manager->get_state(state_id);
      if (not std::ranges::contains(overlapping_states, state_id, &State<UDT>::_id))
      {
        overlapping_states.push_back(state);
      }
    }
  }
  if (_manager->params()->show_gui)
  {
    _manager->vizu()._cycle_data.wlock().data._overlapping_tracks.emplace_back(time, std::move(overlapping_states));
  }
}

template <class UDT>
std::vector<std::tuple<StateId, StateId, Probability<double>>>
overlapping_tracks(TTBManager<UDT> const& manager,
                   std::vector<StateId> const& current_tracks,
                   Duration prediction_horizon,
                   double min_distance)
{
  // ZoneScopedN("overlapping_tracks");
  bgi::rtree<std::pair<Box, StateId>, bgi::quadratic<16>> rtree;
  for (StateId state : current_tracks)
  {
    rtree.insert(std::pair{ axis_aligned_box(manager.get_state(state), prediction_horizon, min_distance),
                            manager.get_state(state)._id });
  }
  std::vector<std::tuple<StateId, StateId, Probability<double>>> intersecting_states;
  for (StateId state : current_tracks)
  {
    std::vector<std::pair<Box, StateId>> intersecting_res;
    rtree.query(bgi::intersects(axis_aligned_box(manager.get_state(state), prediction_horizon, min_distance)),
                std::back_inserter(intersecting_res));
    for (auto const& [box, id] : intersecting_res)
    {
      if (id < state)
      {
        if (check_overlapping(manager.get_state(state), manager.get_state(id), prediction_horizon, min_distance))
        {
          intersecting_states.emplace_back(id, state, 1.0);
        }
      }
    }
  }
  return intersecting_states;
}

template <class UDT>
Opinion create_track_map_opinion(State<UDT> const& track)
{
  std::optional<std::reference_wrapper<Map const>> map_ = get_map(track);
  if (not map_.has_value())
  {
    return Opinion::DogmaticTrust();
  }
  Map const& map = map_.value().get();
  CLASS estimated_class = track._classification.getEstimate();
  std::optional<MapInfo> const map_info = get_map_info(map, track);
  if (not map_info.has_value())
  {
    return Opinion::DogmaticTrust();
  }
  if (map_info->inside_building)
  {
    return Opinion{ Opinion::BeliefType{ std::array{ 0.0, 0.9 } } };
  }
  if (not map_info->on_usable_way and estimated_class != CLASS::PEDESTRIAN)
  {
    return Opinion{ Opinion::BeliefType{ std::array{ 0.0, 0.9 } } };
  }

  return Opinion::DogmaticTrust();
}

template <class UDT>
Opinion create_track_class_opinion(State<UDT> const& track)
{
  // ZoneScopedN("Assessment::track_class_opinion");
  CLASS class_estimation = track._classification.getEstimate();
  auto [model, dist] = track.getEstimate();
  Vector<UDT> const& mean = dist.mean();
  Components const& comps = track._manager->state_comps(model);
  Opinion const extent_op = [&] {
    std::optional<Indices> lwh_inds = comps.indexOf({ COMPONENT::LENGTH, COMPONENT::WIDTH, COMPONENT::HEIGHT });

    auto const to_op = [&](CLASS est,
                           double min_length,
                           double max_length,
                           double min_width,
                           double max_width,
                           double min_height,
                           double max_height) {
      if (not lwh_inds.has_value())
      {
        return Opinion::DogmaticTrust();
      }
      if (double const length = mean(lwh_inds.value()(0)); length < min_length or length > max_length)
      {
        LOG_WARN(std::format("{} {} with unusual length {}", est, track._id, length));
        return Opinion{ Opinion::BeliefType{ std::array{ 0.0, 0.9 } } };
      }
      if (double const width = mean(lwh_inds.value()(1)); width < min_width or width > max_width)
      {
        LOG_WARN(std::format("{} {} with unusual width {}", est, track._id, width));
        return Opinion{ Opinion::BeliefType{ std::array{ 0.0, 0.9 } } };
      }
      if (double const height = mean(lwh_inds.value()(2)); height < min_height or height > max_height)
      {
        LOG_WARN(std::format("{} {} with unusual height {}", est, track._id, height));
        return Opinion{ Opinion::BeliefType{ std::array{ 0.0, 0.9 } } };
      }
      return Opinion{ Opinion::BeliefType{ std::array{ 1.0, 0.0 } } };
    };
    if (class_estimation == CLASS::CAR or class_estimation == CLASS::CAR_UNION)
    {
      return to_op(class_estimation, 2, 7, 1, 3, 1, 3);
    }
    if (class_estimation == CLASS::PEDESTRIAN)
    {
      return to_op(class_estimation, 0.2, 1.5, 0.2, 1.5, 0.5, 2.5);
    }
    if (class_estimation == CLASS::TRUCK_UNION)
    {
      return to_op(class_estimation, 2, 10, 1.5, 3, 1.5, 4);
    }
    return Opinion::DogmaticTrust();
  }();
  Opinion const speed_op = [&] {
    auto const vel_abs_v = transformation::transform(mean, comps, Components({ COMPONENT::VEL_ABS }));
    if (not vel_abs_v.has_value())
    {
      return Opinion::VacuousBeliefOpinion();
    }

    double const vel_abs = std::abs(vel_abs_v.value()(0));

    auto const to_op = [&](CLASS est, double min_speed, double max_speed) {
      if (vel_abs < min_speed or vel_abs > max_speed)
      {
        LOG_WARN(std::format("{} {} with unusual speed {}", est, track._id, vel_abs));
        return Opinion{ Opinion::BeliefType{ std::array{ 0.0, 0.9 } } };
      }
      return Opinion{ Opinion::BeliefType{ std::array{ 1.0, 0.0 } } };
    };
    if (class_estimation == CLASS::CAR or class_estimation == CLASS::CAR_UNION)
    {
      return to_op(class_estimation, 0, 100.0 / 3.6);
    }
    if (class_estimation == CLASS::PEDESTRIAN)
    {
      return to_op(class_estimation, 0, 12 / 3.6);
    }
    if (class_estimation == CLASS::TRUCK_UNION)
    {
      return to_op(class_estimation, 0, 100 / 3.6);
    }
    return Opinion::DogmaticTrust();
  }();
  return extent_op.multiply(speed_op);
}

template <class UDT>
Opinion create_measurement_opinion(State<UDT> const& measurement)
{
  std::optional<std::reference_wrapper<Map const>> const map_ = get_map(measurement);
  if (not map_.has_value())
  {
    return Opinion::DogmaticTrust();
  }
  Map const& map = map_.value().get();
  CLASS const estimated_class = measurement._classification.getEstimate();
  std::optional<MapInfo> const map_info = get_map_info(map, measurement);
  if (not map_info.has_value())
  {
    return Opinion::DogmaticTrust();
  }
  if (map_info->on_usable_way)
  {
    return Opinion{ Opinion::BeliefType{ std::array{ 0.9, 0.0 } } };
  }
  if (map_info->inside_building and estimated_class != CLASS::PEDESTRIAN)
  {
    return Opinion{ Opinion::BeliefType{ std::array{ 0.0, 0.9 } } };
  }
  return Opinion::DogmaticTrust();
}

template class StoachsticModelAssessment<double>;
template class StoachsticModelAssessment<AD>;

template <class UDT>
void StoachsticModelAssessment<UDT>::reset()
{
  *this = StoachsticModelAssessment(_manager);
}

template <class UDT>
void TrackingAssessment<UDT>::compute_implementation_opinion()
{
  for (auto const& [type, opinions] : implementation_opinions.rlock().data)
  {
    Opinion worst = Opinion::DogmaticTrust();
    for (TimedData<subjective_logic::Opinion<2, double>> const& op : opinions.data_buffer)
    {
      if (op.data.getProbability() < worst.getProbability())
      {
        worst = op.data;
      }
    }
    _manager->vizu()._cycle_data.wlock().data.implementation_op[type].emplace_back(_manager->filter_time(), worst);
  }
}

template <class UDT>
void TrackingAssessment<UDT>::reset()
{
  _model_assessment.reset();
  _track_assessment.clear();
  _meas_influences.clear();
  _meas_opinions.clear();
  _track_map_opinions.clear();
  _track_class_opinions.clear();
  implementation_opinions.wlock().data.clear();
}

template <class UDT>
void TrackingAssessment<UDT>::compute_track_assessment(std::vector<StateId> const& track_ids, Duration window_length)
{
}

}  // namespace ttb
