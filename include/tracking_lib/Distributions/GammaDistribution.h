#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"

namespace ttb
{

struct GammaDistributionParams {
  double alpha;
  double beta;
};

/// This is a simple representation of a Gamma Distribution, see
/// https://en.wikipedia.org/wiki/Gamma_distribution
class GammaDistribution
{
public:
  GammaDistribution(double alpha, double beta);
  [[nodiscard]] double pdf(double x) const;
  [[nodiscard]] double variance() const;
  [[nodiscard]] double mean() const;
  [[nodiscard]] std::string toString() const;
  double _alpha;
  double _beta;
  [[nodiscard]] static GammaDistributionParams from_mean_var(double mean, double var);
};





}  // namespace ttb