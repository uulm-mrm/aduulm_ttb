#include "tracking_lib/TTBTypes/Components.h"
#include <ranges>
#include <set>

namespace ttb
{

Components::Components(std::vector<COMPONENT> comps)
  : _comps{ [&] {
    std::set<COMPONENT> set_comps;
    for (COMPONENT comp : comps)
    {
      set_comps.insert(comp);
    }
    std::vector<COMPONENT> sorted_comps(set_comps.begin(), set_comps.end());
    assert(std::ranges::is_sorted(sorted_comps));
    return sorted_comps;
  }() }
{
}

std::string Components::toString(std::string const& prefix) const
{
  std::string out = prefix + "Components: ";
  for (COMPONENT comp : _comps)
  {
    out += to_string(comp) + " ";
  }
  out += '\n';
  return out;
}

std::optional<Index> Components::indexOf(COMPONENT comp) const
{
  if (auto const it = std::ranges::find(_comps, comp); it != _comps.end())
  {
    return std::distance(_comps.begin(), it);
  }
  return {};
}

std::optional<Indices> Components::indexOf(std::vector<COMPONENT> const& comps) const
{
  Indices indices(comps.size());
  for (auto [ctr, sc] : std::views::enumerate(comps))
  {
    if (std::optional<Index> ind = indexOf(sc); ind.has_value())
    {
      indices(ctr) = ind.value();
    }
    else
    {
      return {};
    }
    ctr++;
  }
  return indices;
}

Components Components::intersection(Components const& otherComps) const
{
  std::vector<COMPONENT> inter;
  std::ranges::set_intersection(_comps, otherComps._comps, std::back_inserter(inter));
  return Components(inter);
}

bool Components::contains(Components const& other) const
{
  return std::ranges::all_of(other._comps, [&](COMPONENT other_comp) { return indexOf(other_comp).has_value(); });
}

Components Components::diff(Components const& other) const
{
  std::vector<COMPONENT> diff;
  std::ranges::set_difference(_comps, other._comps, std::back_inserter(diff));
  return Components(diff);
}

Components Components::merge(Components const& others) const
{
  std::vector<COMPONENT> all_comps;
  all_comps.insert(all_comps.end(), _comps.begin(), _comps.end());
  all_comps.insert(all_comps.end(), others._comps.begin(), others._comps.end());
  return Components(all_comps);
}

}  // namespace ttb