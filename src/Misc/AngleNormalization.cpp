#include "tracking_lib/Misc/AngleNormalization.h"

#include <aduulm_logger/aduulm_logger.hpp>
#include <ranges>

namespace ttb::angles
{

double angle_between(Vector3 const& from, Vector3 const& to)
{
  return std::atan2(from.cross(to).norm(), from.dot(to));
}

template void normalizeAngle(double& phi_in);
template void normalizeAngle(AD& phi_in);
template <class UDT>
void normalizeAngle(UDT& phi_in)
{
  phi_in = remainder(phi_in, 2 * std::numbers::pi);
}

template double weighted_mean(std::vector<std::pair<double, double>> const& weight_angle);
template AD weighted_mean(std::vector<std::pair<AD, AD>> const& weight_angle);
template <class UDT>
UDT weighted_mean(std::vector<std::pair<UDT, UDT>> const& weight_angle)
{
  UDT fsum = 0;
  UDT ssum = 0;
  UDT w = 0;
  for (auto [weight, angle] : weight_angle)
  {
    fsum += std::sin(angle) * weight;
    ssum += std::cos(angle) * weight;
    w += std::abs(weight);
  }
  assert(std::abs(w - 1) < 1e-5);
  return atan2(fsum, ssum);
}

template double weighted_mean(std::span<double const> angles, std::span<double const> weights);
template AD weighted_mean(std::span<AD const> angles, std::span<AD const> weights);
template <class UDT>
UDT weighted_mean(std::span<UDT const> angles, std::span<UDT const> weights)
{
  UDT x = 0;
  UDT y = 0;
  UDT w = 0;
  for (auto [angle, weight] : std::views::zip(angles, weights))
  {
    x += std::cos(angle) * weight;
    y += std::sin(angle) * weight;
    w += std::abs(weight);
  }
  assert(std::abs(w - 1) < 1e-5);
  return atan2(y, x);
}

template std::vector<double> lerp_angles(double from, double to, std::size_t num_steps);
template std::vector<AD> lerp_angles(AD from, AD to, std::size_t num_steps);
template <class UDT>
std::vector<UDT> lerp_angles(UDT from, UDT to, std::size_t num_steps)
{
  from += std::numbers::pi;
  to += std::numbers::pi;
  if (from > to)
  {
    to += 2 * std::numbers::pi;
  }
  std::vector<UDT> out;
  for (std::size_t i = 1; i < num_steps; ++i)
  {
    UDT norm = from + -std::numbers::pi + static_cast<UDT>(i) / static_cast<UDT>(num_steps) * (to - from);
    normalizeAngle(norm);
    out.push_back(norm);
  }
  return out;
}

template double smaller_diff(double angle1, double angle2);
template AD smaller_diff(AD angle1, AD angle2);
template <class UDT>
UDT smaller_diff(UDT angle1, UDT angle2)
{
  return acos(cos(angle1) * cos(angle2) + sin(angle1) * sin(angle2));
}

}  // namespace ttb::angles
