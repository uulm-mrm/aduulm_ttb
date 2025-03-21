#include "tracking_lib/Transformations/TransformReferencePoint.h"

namespace ttb::transformReferencePoint
{

/// simple transformation of a state with given reference point to another reference point without regarding covariances
/// @param fromRP reference point of the x_in parameter
/// @param toRP   the reference point the state shall be transformed to
/// @param x_in   state to transform x = [x y yaw length width]'
/// @param[out] x_out transformed state
/// @param[out] leverarm leverarm
void SimpleTransform(REFERENCE_POINT from, REFERENCE_POINT to, Vector const& x_in, Vector& x_out, Vector& leverarm);

/// calculate the leverarm for the the reference point
/// @param fromRP
/// @param toRP
/// @param l length of the object
/// @param w width of the object
/// @param[out] leverarm
void calculateLeverarm(REFERENCE_POINT from, REFERENCE_POINT to, double l, double w, Vector& leverarm);

/// calculate the leverarm to the CenterCenter RP
/// @param fromRP
/// @param l2
/// @param w2
/// @param[out] leverarm
void leverarmRPtoCenterCenter(REFERENCE_POINT fromRP, double l, double w, Vector& leverarm);

/// rotation matrix for the yaw angle only
/// @param yaw
/// @param[out] rot
void rotationMatrix(double yaw, Matrix& rot);

Vector transform(Vector const& x_in, REFERENCE_POINT fromRP, REFERENCE_POINT toRP)
{
  Vector leverarm;
  Vector x_out;
  SimpleTransform(fromRP, toRP, x_in, x_out, leverarm);
  return x_out;
}

void SimpleTransform(REFERENCE_POINT fromRP, REFERENCE_POINT toRP, Vector const& x_in, Vector& x_out, Vector& leverarm)
{
  x_out = x_in;
  if (fromRP != toRP)
  {
    calculateLeverarm(fromRP, toRP, x_in(3), x_in(4), leverarm);

    Matrix rot(2, 2);
    rotationMatrix(x_in(2), rot);
    rot = rot * leverarm;
    x_out(0) += rot(0, 0);
    x_out(1) += rot(1, 0);
  }
}

void calculateLeverarm(REFERENCE_POINT fromRP, REFERENCE_POINT toRP, double l, double w, Vector& leverarm)
{
  leverarm.setZero(2);
  Vector temp(2);
  leverarmRPtoCenterCenter(fromRP, l / 2., w / 2., temp);
  leverarm += temp;
  leverarmRPtoCenterCenter(toRP, l / 2., w / 2., temp);
  leverarm -= temp;
}

void leverarmRPtoCenterCenter(REFERENCE_POINT fromRP, double l, double w, Vector& leverarm)
{
  // TODO: check matrix size (2,1)??
  leverarm.setZero();
  switch (fromRP)
  {
    case REFERENCE_POINT::BACK:
    {
      leverarm(0) += l;
      break;
    }
    case REFERENCE_POINT::FRONT:
    {
      leverarm(0) -= l;
      break;
    }
    case REFERENCE_POINT::BACK_LEFT:
    {
      leverarm(0) += l;
      leverarm(1) -= w;
      break;
    }
    case REFERENCE_POINT::FRONT_RIGHT:
    {
      leverarm(0) -= l;
      leverarm(1) += w;
      break;
    }
    case REFERENCE_POINT::BACK_RIGHT:
    {
      leverarm(0) += l;
      leverarm(1) += w;
      break;
    }
    case REFERENCE_POINT::FRONT_LEFT:
    {
      leverarm(0) -= l;
      leverarm(1) -= w;
      break;
    }
    case REFERENCE_POINT::LEFT:
    {
      leverarm(1) -= w;
      break;
    }
    case REFERENCE_POINT::RIGHT:
    {
      leverarm(1) += w;
      break;
    }
    case REFERENCE_POINT::CENTER:
    {
      break;
    }
    default:
    {
      LOG_ERR("Unhandled referencepoint: " << to_string(fromRP));
      assert(false);
      break;
    }
  }
}

void rotationMatrix(double yaw, Matrix& rot)
{
  double c_y = cos(yaw);
  double s_y = sin(yaw);

  rot(0, 0) = c_y;
  rot(0, 1) = -s_y;
  rot(1, 0) = s_y;
  rot(1, 1) = c_y;
}

}  // namespace ttb::transformReferencePoint
