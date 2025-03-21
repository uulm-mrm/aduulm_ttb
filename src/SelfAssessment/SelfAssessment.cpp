#include "tracking_lib/SelfAssessment/SelfAssessment.h"

#include "tracking_lib/States/State.h"
#include "tracking_lib/Misc/Numeric.h"
#include "tracking_lib/TTBManager/TTBManager.h"
#include <ranges>
#include <boost/math/distributions/beta.hpp>
#include <boost/math/distributions/gamma.hpp>
#include <tracy/tracy/Tracy.hpp>

namespace ttb::sa
{

constexpr auto tracy_color = tracy::Color::Orchid;

Vector merge_clutter_dist(Vector const& first, Vector const& second)
{
  if (first.size() == 0 || second.size() == 0)
  {
    Vector empty_sol;
    return empty_sol;
  }
  return numeric::fold(first, second);
}

Matrix merge_detection_dist(Matrix const& first, Matrix const& second)
{
  if (first.size() == 0 || second.size() == 0)
  {
    Matrix empty_sol;
    return empty_sol;
  }
  return numeric::fold(first, second);
}

ClutterDetectionDistribution compute_dists(std::vector<State> const& tracks, Index num_meas)
{
  ZoneScopedNC("Estimation::compute_stats", tracy_color);
  Matrix detection_dist{ { 1 } };
  for (State const& track : tracks)
  {
    if (not track._detectable or track._meta_data._numUpdates <= 1)
    {
      continue;
    }
    double ed = track._existenceProbability * (1 - track._weight_mis_detection);  ///< track exists + detected
    double em = track._existenceProbability * track._weight_mis_detection;        ///< track exists + missed
    double n = 1 - track._existenceProbability;                                   ///< track does not exists
    Matrix second{ { n, 0 }, { em, ed } };
    detection_dist = merge_detection_dist(detection_dist, second);
  }
  // you can not detect more tracks than #measurements
  if (detection_dist.rows() - 1 > num_meas)
  {
    for (Index num_tracks = 0; num_tracks <= detection_dist.rows() - 1; ++num_tracks)
    {
      auto impossible_cols = Eigen::seq(std::min(num_tracks, num_meas) + 1, detection_dist.cols() - 1);
      double const impossible_detected = detection_dist(num_tracks, impossible_cols).sum();
      detection_dist(num_tracks, std::min(num_tracks, num_meas)) += impossible_detected;
      detection_dist(num_tracks, impossible_cols).array() = 0;
    }
  }
  assert([&] {  // NOLINT
    if (std::abs(detection_dist.sum() - 1) > 1e-5)
    {
      LOG_FATAL("Invalid Detection dist: " << detection_dist);
      return false;
    }
    for (Index num_tracks = 0; num_tracks <= detection_dist.rows() - 1; ++num_tracks)
    {
      for (Index detected = 0; detected <= detection_dist.rows() - 1; ++detected)
      {
        if ((detected > num_tracks or detected > num_meas) and detection_dist(num_tracks, detected) > 0)
        {
          LOG_FATAL("more detected as tracks or #measurements: " << detection_dist);
          LOG_FATAL("num tracks: " << detection_dist.rows() - 1);
          LOG_FATAL("#measurements: " << num_meas);
          LOG_FATAL("#detected: " << detected);
          LOG_FATAL("prob: " << detection_dist(num_tracks, detected));
          return false;
        }
      }
    }
    return true;
  }());
  Vector clutter_dist = Vector::Zero(num_meas + 1);
  for (Index num_tracks = 0; num_tracks <= detection_dist.rows() - 1; ++num_tracks)
  {
    for (Index detected = 0; detected <= std::min(num_tracks, num_meas); ++detected)
    {
      clutter_dist(num_meas - detected) += detection_dist(num_tracks, detected);
    }
  }
  assert([&] {  // NOLINT
    if (std::abs(clutter_dist.sum() - 1) > 1e-5)
    {
      LOG_FATAL("Invalid clutter dist: " << clutter_dist << " with sum: " << clutter_dist.sum()
                                         << " detection dist sum: " << detection_dist.sum());
      return false;
    }
    return true;
  }());
  return { .clutter_dist = std::move(clutter_dist), .detection_dist = std::move(detection_dist) };
}

GammaDistribution updateClutterEstimate(GammaDistribution const& prior,
                                        Vector const& clutter_dist,
                                        double static_discount,
                                        double dynamic_discount_alpha,
                                        double dynamic_discount_min)
{
  ZoneScopedNC("Estimation::updateClutterRate", tracy_color);
  LOG_DEB("UpdateClutter");
  LOG_DEB("updateClutterEstimate");
  GammaDistribution discounted_prior = prior;
  discounted_prior._alpha *= static_discount;
  discounted_prior._beta *= static_discount;
  std::vector<std::pair<GammaDistribution, Probability>> posterior;
  for (auto const& [numClutter, weight] : std::views::enumerate(clutter_dist))
  {
    posterior.emplace_back(GammaDistribution{ discounted_prior._alpha + numClutter, discounted_prior._beta + 1 },
                           weight);
  }
  double mean = std::accumulate(posterior.begin(), posterior.end(), 0.0, [](double old, auto const& comp) -> double {
    return old + comp.first.mean() * comp.second;
  });
  double var = std::accumulate(posterior.begin(), posterior.end(), 0.0, [](double old, auto const& comp) -> double {
    return old + comp.first.variance() * comp.second;
  });
  var += std::accumulate(posterior.begin(), posterior.end(), 0.0, [mean](double old, auto const& comp) -> double {
    double add = (comp.first.mean() - mean) * (comp.first.mean() - mean) * comp.second;
    return old + add;
  });
  double alpha = mean * mean / var;
  double beta = mean / var;
  assert(alpha > 0);
  assert(beta > 0);
  GammaDistribution post(alpha, beta);
  double dynamic_discount =
      std::max(1 - dynamic_discount_alpha * std::abs(prior.mean() - post.mean()), dynamic_discount_min);
  return { dynamic_discount * alpha, std::max(dynamic_discount * beta, 1e-5) };
}

DirichletDistribution updateDetectionEstimate(DirichletDistribution const& prior,
                                              Matrix const& detection_dist,
                                              double static_discount,
                                              double dynamic_discount_alpha,
                                              double dynamic_discount_min)
{
  ZoneScopedNC("Estimation::updateDetectionProbability", tracy_color);
  double prior_mean = prior.mean()(0);
  DirichletDistribution discountedPrior = prior;
  discountedPrior._alpha *= static_discount;
  std::vector<std::pair<DirichletDistribution, Probability>> posterior;
  for (Index num_tracks = 0; num_tracks < detection_dist.rows(); ++num_tracks)
  {
    for (Index num_detections = 0; num_detections < detection_dist.cols(); ++num_detections)
    {
      if (detection_dist(num_tracks, num_detections) > 0)
      {
        posterior.emplace_back(
            DirichletDistribution(discountedPrior._alpha +
                                  Vector{ { static_cast<double>(num_detections),
                                            static_cast<double>(num_tracks) - static_cast<double>(num_detections) } }),
            detection_dist(num_tracks, num_detections));
      }
    }
  }
  Vector const mean_init = Vector::Zero(posterior.front().first.mean().rows());
  Vector const mean =
      std::accumulate(posterior.begin(),
                      posterior.end(),
                      mean_init,
                      [](Vector const& old, std::pair<DirichletDistribution, Probability> const& comp) -> Vector {
                        return old + comp.first.mean() * comp.second;
                      });

  Matrix var_init = Matrix::Zero(posterior.front().first.variance().rows(), posterior.front().first.variance().rows());
  Matrix var =
      std::accumulate(posterior.begin(), posterior.end(), var_init, [](Matrix const& old, auto const& comp) -> Matrix {
        return old + comp.first.variance() * comp.second;
      });

  var = std::accumulate(posterior.begin(), posterior.end(), var, [mean](Matrix const& old, auto const& comp) -> Matrix {
    return old + (comp.first.mean() - mean) * (comp.first.mean() - mean).transpose() * comp.second;
  });

  double const vmax = var.diagonal().array().maxCoeff();
  Vector const lambda = ((mean.array() * (1 - mean.array())) / vmax - 1) / mean.sum();
  Vector const alpha = std::max(lambda.minCoeff(), 1e-5) * mean;
  DirichletDistribution const post(alpha);
  double dynamic_discount =
      std::max(1 - dynamic_discount_alpha * std::abs(prior_mean - post.mean()(0)), dynamic_discount_min);
  return DirichletDistribution(alpha * dynamic_discount);
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
std::string to_stringStatistics(std::vector<ClutterEstimate_ProfilerData> const& datas)
{
  return {};
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
std::string to_stringStatistics(std::vector<DetectionEstimate_ProfilerData> const& datas)
{
  return {};
}

ParameterEstimation::ParameterEstimation(TTBManager* manager) : _manager{ manager }
{
}

std::optional<double> ParameterEstimation::update_clutter(MeasurementContainer const& measurement_container,
                                                          Vector const& clutter_distribution)
{
  ZoneScopedNC("ParameterEstimation::update_clutter", tracy_color);
  static profiling::GeneralDataProfiler<ClutterEstimate_ProfilerData> clutterProfiler("/tmp/clutter_rate_estimation");
  if (not _manager->meas_model_params(measurement_container._id).clutter.rate_estimation.enable)
  {
    return std::nullopt;
  }
  auto estimate_it = _clutterEstimate.find(measurement_container._id);
  if (estimate_it == _clutterEstimate.end())
  {
    double const mu = _manager->meas_model_params(measurement_container._id).clutter.rate_estimation.prior_mean;
    double const s2 = _manager->meas_model_params(measurement_container._id).clutter.rate_estimation.prior_var;
    std::tie(estimate_it, std::ignore) =
        _clutterEstimate.emplace(measurement_container._id, GammaDistribution(mu * mu / s2, mu / s2));
  }
  estimate_it->second =
      updateClutterEstimate(estimate_it->second,
                            clutter_distribution,
                            _manager->params().self_assessment.clutter_rate_estimation.static_discount,
                            _manager->params().self_assessment.clutter_rate_estimation.dynamic_discount_alpha,
                            _manager->params().self_assessment.clutter_rate_estimation.dynamic_discount_min);
  if (_manager->meas_model_params(measurement_container._id).clutter.rate_estimation.use_in_filter)
  {
    _manager->meas_model_next_params(measurement_container._id).clutter.rate = estimate_it->second.mean();
  }

  double const bayes_factor = [&] {
    if (GaussianMeasurementModelParams const& model_params = *std::ranges::find_if(
            _manager->original_params().meas_models.gaussian_models,
            [&](GaussianMeasurementModelParams const& params) { return params.id == measurement_container._id; });
        model_params.clutter.rate.has_value())
    {
      double const clutter = model_params.clutter.rate.value();
      GammaDistribution const& est = estimate_it->second;
      assert(est._alpha > 0);
      assert(1 / est._beta > 0);
      boost::math::gamma_distribution const posterior(est._alpha, 1 / est._beta);
      boost::math::gamma_distribution const gamma_aprior{ [&] {
        double const mu = model_params.clutter.rate_estimation.prior_mean;
        double const s2 = model_params.clutter.rate_estimation.prior_var;
        double const shape = mu * mu / s2;
        double const scale = mu / s2;
        assert(shape > 0);
        assert(scale > 0);
        return boost::math::gamma_distribution(shape, scale);
      }() };
      return boost::math::pdf(posterior, clutter) / boost::math::pdf(gamma_aprior, clutter);
    }
    return std::numeric_limits<double>::quiet_NaN();
  }();
  if (_manager->params().show_gui)
  {
    std::lock_guard lock(_manager->vizu().add_data_mutex);
    _manager->vizu()._meas_model_data[measurement_container._id]._clutter_estimation.emplace_back(
        measurement_container._time, estimate_it->second);

    _manager->vizu()._meas_model_data[measurement_container._id]._clutter_bayes_factor.emplace_back(
        measurement_container._time, bayes_factor);
  }
  clutterProfiler.addData(ClutterEstimate_ProfilerData{
      ._estimation = { { measurement_container._id,
                         { estimate_it->second._alpha, estimate_it->second._beta, bayes_factor } } },
      ._time = measurement_container._time });
  return estimate_it->second.mean();
}

std::optional<Probability> ParameterEstimation::update_detection(MeasurementContainer const& measurement_container,
                                                                 Matrix const& detection_distribution)
{
  ZoneScopedNC("ParameterEstimation::update_detection", tracy_color);
  LOG_DEB("Update Detection Estimate");
  static profiling::GeneralDataProfiler<DetectionEstimate_ProfilerData> detectionProfiler("/tmp/detection_probability_"
                                                                                          "estimation");
  if (not _manager->meas_model_params(measurement_container._id).detection.prob_estimation.enable)
  {
    return std::nullopt;
  }
  auto estimation_it = _detectionEstimate.find(measurement_container._id);
  if (estimation_it == _detectionEstimate.end())
  {
    double const mean = _manager->meas_model_params(measurement_container._id).detection.prob_estimation.prior_mean;
    double const var = _manager->meas_model_params(measurement_container._id).detection.prob_estimation.prior_var;
    double p = mean * (mean - var - mean * mean) / var;
    double q = (mean - 1) * (var + mean * mean - mean) / var;
    std::tie(estimation_it, std::ignore) =
        _detectionEstimate.emplace(measurement_container._id, DirichletDistribution(Vector{ { p, q } }));
  }
  estimation_it->second =
      updateDetectionEstimate(estimation_it->second,
                              detection_distribution,
                              _manager->params().self_assessment.detection_prob_estimation.static_discount,
                              _manager->params().self_assessment.detection_prob_estimation.dynamic_discount_alpha,
                              _manager->params().self_assessment.detection_prob_estimation.dynamic_discount_min);
  if (_manager->meas_model_params(measurement_container._id).detection.prob_estimation.use_in_filter)
  {
    _manager->meas_model_next_params(measurement_container._id).detection.prob =
        std::min(estimation_it->second.mean()(0), 0.99999);
  }
  double const bayes_factor = [&] {
    GaussianMeasurementModelParams const& model_params = *std::ranges::find_if(
        _manager->original_params().meas_models.gaussian_models,
        [&](GaussianMeasurementModelParams const& params) { return params.id == measurement_container._id; });
    DirichletDistribution const& est = estimation_it->second;
    boost::math::beta_distribution const posterior(est._alpha(0), est._alpha(1));
    boost::math::beta_distribution const beta_prior{ [&] {
      double const mean = model_params.detection.prob_estimation.prior_mean;
      double const var = model_params.detection.prob_estimation.prior_var;
      double const p = mean * (mean - var - mean * mean) / var;
      double const q = (mean - 1) * (var + mean * mean - mean) / var;
      return boost::math::beta_distribution(p, q);
    }() };
    return boost::math::pdf(posterior, model_params.detection.prob) /
           boost::math::pdf(beta_prior, model_params.detection.prob);
  }();
  if (_manager->params().show_gui)
  {
    std::lock_guard lock(_manager->vizu().add_data_mutex);
    _manager->vizu()._meas_model_data[measurement_container._id]._detection_estimation.emplace_back(
        measurement_container._time, estimation_it->second);
    _manager->vizu()._meas_model_data[measurement_container._id]._detection_bayes_factor.emplace_back(
        measurement_container._time, bayes_factor);
  }
  detectionProfiler.addData(DetectionEstimate_ProfilerData{
      ._estimation = { { measurement_container._id,
                         { estimation_it->second._alpha(0), estimation_it->second._alpha(1), bayes_factor } } },
      ._time = measurement_container._time });
  return estimation_it->second.mean()(0);
}

ParameterEstimation::Estimation ParameterEstimation::update(MeasurementContainer const& measurement_container,
                                                            Vector const& clutter_distribution,
                                                            Matrix const& detection_distribution)
{
  return { .clutter_rate = update_clutter(measurement_container, clutter_distribution),
           .detection_probability = update_detection(measurement_container, detection_distribution) };
}

}  // namespace ttb::sa