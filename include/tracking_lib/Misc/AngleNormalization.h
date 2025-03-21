#pragma once
#include "tracking_lib/TTBTypes/TTBTypes.h"

namespace ttb::angles
{

/// normalize angle to the interval (-pi, pi]
void normalizeAngle(double& phi_in);
/// compute a weighted mean of angles
[[nodiscard]] double weightedMean(Eigen::RowVectorXd const& angles, Eigen::VectorXd const& weights);
/// compute a weighted mean of angles
[[nodiscard]] double weightedMean(std::span<double const> angles, std::span<double const> weights);
/// linear interpolate a pair of angles
[[nodiscard]] std::vector<double> lerp_angles(double from, double to, std::size_t num_steps);
/// smaller difference between to angles
[[nodiscard]] double smaller_diff(double angle1, double angle2);

}  // namespace ttb::angles