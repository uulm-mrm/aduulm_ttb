#include "tracking_lib/TTBTypes/Components.h"
#include <ranges>

namespace ttb
{

Components::Components(std::vector<COMPONENT> comps)
  : _comps{ [&] {
    std::ranges::sort(comps);
    auto [first, last] = std::ranges::unique(comps);
    comps.erase(first, last);
    assert(std::ranges::is_sorted(comps));
    return comps;
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
  for (auto const [ind, c] : std::views::enumerate(_comps))
  {
    if (c == comp)
    {
      return ind;
    }
  }
  return {};
}

Components Components::intersection(Components const& otherComps) const
{
  std::vector<COMPONENT> inter;
  std::ranges::set_intersection(_comps, otherComps._comps, std::back_inserter(inter));
  return Components(std::move(inter));
}

bool Components::contains(Components const& other) const
{
  return std::ranges::all_of(other._comps, [&](COMPONENT other_comp) { return indexOf(other_comp).has_value(); });
}

Components Components::diff(Components const& other) const
{
  std::vector<COMPONENT> diff;
  std::ranges::set_difference(_comps, other._comps, std::back_inserter(diff));
  return Components(std::move(diff));
}

Components Components::merge(Components const& others) const
{
  std::vector<COMPONENT> all_comps;
  all_comps.insert(all_comps.end(), _comps.begin(), _comps.end());
  all_comps.insert(all_comps.end(), others._comps.begin(), others._comps.end());
  return Components(std::move(all_comps));
}

std::string to_string(Components const& comps)
{
  return comps.toString();
}

std::pair<Components, Components> state_and_noise_comps(STATE_MODEL_TYPE model, EXTENT extent)
{
  Components base_comps = [&] {
    if (model == STATE_MODEL_TYPE::CA)
    {
      return CA_state_comps;
    }
    if (model == STATE_MODEL_TYPE::CP)
    {
      return CP_state_comps;
    }
    if (model == STATE_MODEL_TYPE::CTP)
    {
      return CTP_state_comps;
    }
    if (model == STATE_MODEL_TYPE::CTRA)
    {
      return CTRA_state_comps;
    }
    if (model == STATE_MODEL_TYPE::CTRV)
    {
      return CTRV_state_comps;
    }
    if (model == STATE_MODEL_TYPE::CV)
    {
      return CV_state_comps;
    }
    if (model == STATE_MODEL_TYPE::ISCATR)
    {
      return ISCATR_state_comps;
    }
    LOG_FATAL("unknown state mode");
    assert(false);
    std::unreachable();
  }();
  Components base_noise = [&] {
    if (model == STATE_MODEL_TYPE::CA)
    {
      return CA_noise_comps;
    }
    if (model == STATE_MODEL_TYPE::CP)
    {
      return CP_noise_comps;
    }
    if (model == STATE_MODEL_TYPE::CTP)
    {
      return CTP_noise_comps;
    }
    if (model == STATE_MODEL_TYPE::CTRA)
    {
      return CTRA_noise_comps;
    }
    if (model == STATE_MODEL_TYPE::CTRV)
    {
      return CTRV_noise_comps;
    }
    if (model == STATE_MODEL_TYPE::CV)
    {
      return CV_noise_comps;
    }
    if (model == STATE_MODEL_TYPE::ISCATR)
    {
      return ISCATR_noise_comps;
    }
    LOG_FATAL("unknown state mode");
    assert(false);
    std::unreachable();
  }();
  Components extent_comps = [&] -> Components {
    if (extent == EXTENT::NONE)
    {
      return Components({});
    }
    if (extent == EXTENT::RECTANGULAR)
    {
      return EXTENT_RECTANGULAR;
    }
    if (extent == EXTENT::CIRCULAR)
    {
      return EXTENT_CIRCULAR;
    }
    LOG_FATAL("unknown extent");
    assert(false);
    std::unreachable();
  }();

  Components extent_noise = [&] -> Components {
    if (extent == EXTENT::NONE)
    {
      return Components({});
    }
    if (extent == EXTENT::RECTANGULAR)
    {
      return EXTENT_RECTANGULAR_NOISE;
    }
    if (extent == EXTENT::CIRCULAR)
    {
      return EXTENT_CIRCULAR_NOISE;
    }
    LOG_FATAL("unknown extent");
    assert(false);
    std::unreachable();
  }();
  return { base_comps.merge(extent_comps), base_noise.merge(extent_noise) };
}

}  // namespace ttb
