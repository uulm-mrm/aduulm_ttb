#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"

namespace ttb::transformReferencePoint
{

/// transformation of a vector with given reference point to another reference point without regarding covariances
/// @param fromRP reference point of the x_in parameter
/// @param toRP   the reference point the state shall be transformed to
/// @param x_in   state to transform x = [x y yaw length width]'
/// @param[out] x_out transformed state
Vector transform(Vector const& x, REFERENCE_POINT fromRP, REFERENCE_POINT toRP);

}  // namespace ttb::transformReferencePoint
