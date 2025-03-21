#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"

#include "tracking_lib/Distributions/GammaDistribution.h"
#include "tracking_lib/Distributions/DirichletDistribution.h"

#include "tracking_lib/Measurements/MeasurementContainer.h"

namespace ttb::sa
{

Vector merge_clutter_dist(Vector const& first, Vector const& second);

Matrix merge_detection_dist(Matrix const& first, Matrix const& second);

struct ClutterDetectionDistribution
{
  Vector clutter_dist;    ///< P(#Clutter=i) = clutter_dist(i)
  Matrix detection_dist;  ///< i=#tracks, j=#detections-> prob
};

/// compute the #clutter and
/// #tracks/#detections distribution
/// given the tracks
[[nodiscard]] ClutterDetectionDistribution compute_dists(std::vector<State> const& tracks, Index num_meas);

/// update the clutter rate estimation by the assignment map computed by some update method
[[nodiscard]] GammaDistribution updateClutterEstimate(GammaDistribution const& prior,
                                                      Vector const& clutter_dist,
                                                      double static_discount,
                                                      double dynamic_discount_alpha,
                                                      double dynamic_discount_min);

/// update the detection probability estimation with information of the updated tracks
[[nodiscard]] DirichletDistribution updateDetectionEstimate(DirichletDistribution const& prior,
                                                            Matrix const& detection_dist,
                                                            double static_discount,
                                                            double dynamic_discount_alpha,
                                                            double dynamic_discount_min);

struct ClutterEstimate_ProfilerData
{
  std::map<MeasModelId, std::tuple<double, double, double>> _estimation;  ///< alpha, beta, bayes factor
  Time _time;
};
std::string to_string(ClutterEstimate_ProfilerData const& data);
std::string to_stringStatistics(std::vector<ClutterEstimate_ProfilerData> const& datas);

struct DetectionEstimate_ProfilerData
{
  std::map<MeasModelId, std::tuple<double, double, double>> _estimation;  /// alpha(0), alpha(1), bayes factor
  Time _time;
};
std::string to_string(DetectionEstimate_ProfilerData const& data);
std::string to_stringStatistics(std::vector<DetectionEstimate_ProfilerData> const& datas);

class ParameterEstimation
{
public:
  explicit ParameterEstimation(TTBManager* manager);
  /// update the current clutter and detection estimation and return the current estimation
  struct Estimation
  {
    std::optional<double> clutter_rate;
    std::optional<Probability> detection_probability;
  };
  Estimation update(MeasurementContainer const& measurement_container,
                    Vector const& clutter_distribution,
                    Matrix const& detection_distribution);
  /// update only the clutter rate estimation
  std::optional<double> update_clutter(MeasurementContainer const& measurement_container,
                                       Vector const& clutter_distribution);
  /// update only the detection probability estimation
  std::optional<Probability> update_detection(MeasurementContainer const& measurement_container,
                                              Matrix const& detection_distribution);

  TTBManager* _manager;
  /// estimation of the clutter rate
  std::map<MeasModelId, GammaDistribution> _clutterEstimate;
  /// estimation of the detection probability
  std::map<MeasModelId, DirichletDistribution> _detectionEstimate;
};

}  // namespace ttb::sa