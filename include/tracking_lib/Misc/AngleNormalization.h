#pragma once
#include "tracking_lib/TTBTypes/TTBTypes.h"

namespace ttb::angles
{

/// angle between to vectors
[[nodiscard]] double angle_between(Vector3 const& from, Vector3 const& to);
/// normalize angle to the interval (-pi, pi]
template <class UDT>
void normalizeAngle(UDT& phi_in);
/// compute a weighted mean of angles
template <class UDT>
[[nodiscard]] UDT weighted_mean(std::vector<std::pair<UDT, UDT>> const& weight_angle);
/// compute a weighted mean of angles
template <class UDT>
[[nodiscard]] UDT weighted_mean(std::span<UDT const> angles, std::span<UDT const> weights);
/// linear interpolate a pair of angles
template <class UDT>
[[nodiscard]] std::vector<UDT> lerp_angles(UDT from, UDT to, std::size_t num_steps);
/// smaller difference between to angles
template <class UDT>
[[nodiscard]] UDT smaller_diff(UDT angle1, UDT angle2);

}  // namespace ttb::angles