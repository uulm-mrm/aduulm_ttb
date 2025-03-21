#include "tracking_lib/Misc/SeparatingAxisTest.h"

#include "tracking_lib/Transformations/Transformation.h"

namespace ttb::sat
{

// Attention: this asserts the same sizes for the two polygons and works only for convex shapes!!

Vector2 projection_axis(Vector2 const& current_point, Vector2 const& next_point)
{
  return Vector2{ { -(next_point.y() - current_point.y()), next_point.x() - current_point.x() } };
}

// Project the vertices of each polygon onto a axis
std::pair<std::vector<double>, std::vector<double>> compute_projections(std::vector<Vector2> const& bounds_a,
                                                                        std::vector<Vector2> const& bounds_b,
                                                                        Vector2 const& axis_normalised)
{
  std::vector<double> projections_a;
  std::vector<double> projections_b;
  projections_a.reserve(bounds_a.size());
  projections_b.reserve(bounds_b.size());
  for (size_t i = 0; i < bounds_a.size(); i++)
  {
    const double projection_a = axis_normalised.dot(bounds_a[i]);
    const double projection_b = axis_normalised.dot(bounds_b[i]);
    projections_a.push_back(projection_a);
    projections_b.push_back(projection_b);
  }
  return { std::move(projections_a), std::move(projections_b) };
}

// Check if the projections of two polygons overlap
bool is_overlapping(std::vector<double> const& projections_a, std::vector<double> const& projections_b)
{
  const double max_projection_a = *std::ranges::max_element(projections_a);
  const double min_projection_a = *std::ranges::min_element(projections_a);
  const double max_projection_b = *std::ranges::max_element(projections_b);
  const double min_projection_b = *std::ranges::min_element(projections_b);
  // True if projection overlaps but does not necessarily mean the polygons are intersecting yet
  return not(max_projection_a < min_projection_b or max_projection_b < min_projection_a);
}

// Check if two convex polygons intersect
bool separating_axis_intersect(std::vector<Vector2> const& bounds_a, std::vector<Vector2> const& bounds_b)
{
  assert(bounds_a.size() == bounds_b.size());
  for (size_t i = 0; i < bounds_a.size(); ++i)
  {
    Vector2 const axis_normalised = projection_axis(bounds_a[i], bounds_a[(i + 1) % bounds_a.size()]);
    auto [projections_a, projections_b] = compute_projections(bounds_a, bounds_b, axis_normalised);
    if (not is_overlapping(projections_a, projections_b))
      return false;
  }
  for (size_t i = 0; i < bounds_b.size(); ++i)
  {
    Vector2 const axis_normalised = projection_axis(bounds_b[i], bounds_b[(i + 1) % bounds_b.size()]);
    auto [projections_a, projections_b] = compute_projections(bounds_a, bounds_b, axis_normalised);
    if (not is_overlapping(projections_a, projections_b))
      return false;
  }
  // Intersects if all projections overlap
  return true;
}

[[nodiscard]] bool
is_overlapping(Vector const& first, Components const& first_comps, Vector const& second, Components const& second_comps)
{
  auto first_inds = first_comps.indexOf(
      { COMPONENT::POS_X, COMPONENT::POS_Y, COMPONENT::LENGTH, COMPONENT::WIDTH, COMPONENT::ROT_Z });
  auto second_inds = second_comps.indexOf(
      { COMPONENT::POS_X, COMPONENT::POS_Y, COMPONENT::LENGTH, COMPONENT::WIDTH, COMPONENT::ROT_Z });
  if (not first_inds.has_value() or not second_inds.has_value())
  {
    return false;
  }
  std::vector<Vector2> first_rectangle;
  std::vector<Vector2> second_rectangle;
  for (REFERENCE_POINT rp : { REFERENCE_POINT::FRONT_LEFT,
                              REFERENCE_POINT::FRONT_RIGHT,
                              REFERENCE_POINT::BACK_RIGHT,
                              REFERENCE_POINT::BACK_LEFT })
  {
    Vector2 xy =
        transformation::transform(first, first_comps, REFERENCE_POINT::CENTER, rp)(first_inds.value()({ 0, 1 }));
    first_rectangle.emplace_back(std::move(xy));
    Vector2 xy_second =
        transformation::transform(second, second_comps, REFERENCE_POINT::CENTER, rp)(second_inds.value()({ 0, 1 }));
    second_rectangle.emplace_back(std::move(xy_second));
  }
  return separating_axis_intersect(first_rectangle, second_rectangle);
}

}  // namespace ttb::sat