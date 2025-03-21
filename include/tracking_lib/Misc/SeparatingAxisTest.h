#pragma once
#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/TTBTypes/Components.h"

namespace ttb::sat
{

std::vector const requiredComponents = { COMPONENT::POS_X,
                                         COMPONENT::POS_Y,
                                         COMPONENT::ROT_Z,
                                         COMPONENT::WIDTH,
                                         COMPONENT::LENGTH };

/// check whether two states overlap
[[nodiscard]] bool is_overlapping(Vector const& first,
                                  Components const& first_comps,
                                  Vector const& second,
                                  Components const& second_comps);

}  // namespace ttb::sat