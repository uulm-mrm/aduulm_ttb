#include "tracking_lib/Misc/ProportionalAllocation.h"

namespace ttb
{

Indices propAllocation(size_t T, Vector const& weights, double share)
{
  assert(share >= 0 and share <= 1);
  Indices alloc_slots = Indices::Zero(weights.rows());
  if (weights.rows() == 0)
  {
    return alloc_slots;
  }
  assert(weights.sum() > 0. && "linear weights are required");
  Index const equal_share_slots = std::min(static_cast<Index>(static_cast<double>(T) * share), weights.rows());
  size_t const Tp = T - equal_share_slots;

  // Allocate equally one slot to the best hyps
  if (equal_share_slots > 0)
  {
    std::vector<std::size_t> inds(weights.size());
    std::iota(inds.begin(), inds.end(), 0);
    std::ranges::nth_element(inds,
                             std::next(inds.begin(), std::min(equal_share_slots, static_cast<Index>(inds.size()))),
                             [&](std::size_t a, std::size_t b) { return weights(a) > weights(b); });
    for (Index i = 0; i < equal_share_slots; ++i)
    {
      alloc_slots(i)++;
    }
  }
  assert([&] {  // NOLINT
    if (alloc_slots.sum() > static_cast<Index>(T))
    {
      LOG_FATAL("Requested " << T << " slots but already allocated " << alloc_slots.sum());
      return false;
    }
    return true;
  }());

  // Allocate the others by weight
  alloc_slots = alloc_slots + (weights * static_cast<double>(Tp)).array().cast<Index>();

  assert([&] {  // NOLINT
    if (alloc_slots.sum() > static_cast<Index>(T))
    {
      LOG_FATAL("Requested " << T << " slots but already allocated " << alloc_slots.sum());
      return false;
    }
    return true;
  }());

  // Distribute rest
  Index const n_rest = std::max(Index{ 0 }, static_cast<Index>(T) - alloc_slots.sum());
  assert(n_rest <= alloc_slots.size());
  for (Index i = 0; i < n_rest and i < alloc_slots.size(); ++i)
  {
    alloc_slots(i)++;
  }

  assert([&] {  // NOLINT
    if (alloc_slots.sum() != static_cast<Index>(T))
    {
      LOG_FATAL("Requested " << T << " slots but allocated " << alloc_slots.sum());
      return false;
    }
    return true;
  }());

  return alloc_slots;
}

}  // namespace ttb