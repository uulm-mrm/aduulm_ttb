#include "tracking_lib/Misc/AngleNormalization.h"

#include <ranges>

namespace ttb::angles
{

void normalizeAngle(double& phi_in)
{
  phi_in = std::remainder(phi_in, 2 * std::numbers::pi);
}

double weightedMean(Eigen::RowVectorXd const& angles, Eigen::VectorXd const& weights)
{
  assert([&] {//NOLINT
    if (std::abs(weights.sum() - 1) > 1e-5)
    {
      LOG_FATAL("Weights do not sum to 1: " << weights);
      return false;
    }
    return true;
  }());
  return std::atan2(Eigen::sin(angles.array()).matrix() * weights, Eigen::cos(angles.array()).matrix() * weights);
}

double weightedMean(std::span<double const> angles, std::span<double const> weights)
{
  double x = 0;
  double y = 0;
  double w = 0;
  for (auto [angle, weight] : std::views::zip(angles, weights))
  {
    x += std::cos(angle) * weight;
    y += std::sin(angle) * weight;
    w += std::abs(weight);
  }
  assert(std::abs(w - 1) < 1e-5);
  return std::atan2(y, x);
}

std::vector<double> lerp_angles(double from, double to, std::size_t num_steps)
{
  from += std::numbers::pi;
  to += std::numbers::pi;
  if (from > to)
  {
    to += 2 * std::numbers::pi;
  }
  std::vector<double> out;
  for (std::size_t i = 1; i < num_steps; ++i)
  {
    double norm = from + -std::numbers::pi + static_cast<double>(i) / static_cast<double>(num_steps) * (to - from);
    normalizeAngle(norm);
    out.push_back(norm);
  }
  return out;
}

double smaller_diff(double angle1, double angle2)
{
  return std::acos(std::cos(angle1) * std::cos(angle2) + std::sin(angle1) * std::sin(angle2));
}

}  // namespace ttb::angles
