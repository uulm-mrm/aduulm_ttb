#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"

namespace ttb
{

/// Allocates T slots proportionally to the weights
/// Share in (0, 1) sets the share of the T slots that are given to T*share best weights (every one of them get 1 slot)
[[nodiscard]] Indices propAllocation(size_t T, Vector const& weights, double share);

}  // namespace ttb