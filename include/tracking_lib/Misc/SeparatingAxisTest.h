#pragma once
#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/TTBTypes/Components.h"

namespace ttb::sat
{

/// compute the vector between the two closest points of the two objects
template <class UDT>
[[nodiscard]] Vector2 distance_vector(Vector<UDT> const& first,
                                      Components const& first_comps,
                                      Vector<UDT> const& second,
                                      Components const& second_comps);

Components const
    requiredComponents({ COMPONENT::POS_X, COMPONENT::POS_Y, COMPONENT::ROT_Z, COMPONENT::WIDTH, COMPONENT::LENGTH });

/// check whether two states overlap
template <class UDT>
[[nodiscard]] bool is_overlapping(Vector<UDT> const& first,
                                  Components const& first_comps,
                                  Vector<UDT> const& second,
                                  Components const& second_comps);

}  // namespace ttb::sat