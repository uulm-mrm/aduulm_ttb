#include "tracking_lib/Misc/Numeric.h"
#include <tracy/tracy/Tracy.hpp>

namespace ttb::numeric
{

template Vector<double> fold(Vector<double> const& first, Vector<double> const& second);
template Vector<AD> fold(Vector<AD> const& first, Vector<AD> const& second);
template<class UDT>
Vector<UDT> fold(Vector<UDT> const& first, Vector<UDT> const& second)
{
  ZoneScopedN("Numeric::fold_vector");
  assert([&] {  // NOLINT
    if (first.size() == 0 or second.size() == 0)
    {
      LOG_FATAL("empty input");
      return false;
    }
    return true;
  }());
  Vector<UDT> merged = Vector<UDT>::Zero(first.rows() + second.rows() - 1);
  for (Index i = 0; i < first.rows(); ++i)
  {
    merged(Eigen::seq(i, i - 1 + second.rows())) += first(i) * second;
  }
  Index max = 0;
  for (Index i = 0; i < merged.rows(); ++i)
  {
    if (merged(i) > 0)
    {
      max = std::max(max, i);
    }
  }
  Vector<UDT> reduced = merged(Eigen::seq(0, max));
  return reduced;
}

}  // namespace ttb::numeric
