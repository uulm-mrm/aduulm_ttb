#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"

#include <ranges>

namespace ttb
{

/// This stores a collection of unique! and sorted! Components
class Components
{
public:
  /// Create Components out of comps, the comps get SORTED and only unique Components are kept
  explicit Components(std::vector<COMPONENT> comps);
  /// string representation
  [[nodiscard]] std::string toString(std::string const& prefix = "") const;
  /// return the index of a comp
  [[nodiscard]] std::optional<Index> indexOf(COMPONENT comp) const;
  [[nodiscard]] static std::optional<Indices> find_indices(Components const& comps, std::vector<COMPONENT> const& data)
  {
    Indices indices(data.size());
    for (auto [ctr, sc] : std::views::enumerate(data))
    {
      if (std::optional<Index> ind = comps.indexOf(sc); ind.has_value())
      {
        indices(ctr) = ind.value();
      }
      else
      {
        return {};
      }
    }
    return indices;
  }

  [[nodiscard]] std::optional<Indices> indexOf(Components const& comps) const
  {
    return find_indices(*this, comps._comps);
  }

  [[nodiscard]] std::optional<Indices> indexOf(std::vector<COMPONENT> const& comps) const
  {
    return find_indices(*this, comps);
  }
  /// Return the intersection of my Components and the other Components
  [[nodiscard]] Components intersection(Components const& otherComps) const;
  /// Check, whether I contain all the other Components
  [[nodiscard]] bool contains(Components const& other) const;
  /// Set Difference between this and other; return = this\other
  [[nodiscard]] Components diff(Components const& other) const;
  /// Merge all other Components with myself
  [[nodiscard]] Components merge(Components const& others) const;
  /// data
  std::vector<COMPONENT> _comps;
  std::strong_ordering operator<=>(const Components&) const = default;
};

std::string to_string(Components const& comps);

Components const CA_state_comps({ COMPONENT::POS_X, COMPONENT::POS_Y, COMPONENT::POS_Z, COMPONENT::VEL_X,
                                 COMPONENT::VEL_Y, COMPONENT::ACC_X, COMPONENT::ACC_Y });
Components const CA_noise_comps({ COMPONENT::JERK_X, COMPONENT::JERK_Y, COMPONENT::VEL_Z });

Components const CP_state_comps({ COMPONENT::POS_X, COMPONENT::POS_Y, COMPONENT::POS_Z });
Components const CP_noise_comps({ COMPONENT::VEL_X, COMPONENT::VEL_Y, COMPONENT::VEL_Z });

Components const CTP_state_comps({ COMPONENT::POS_X, COMPONENT::POS_Y, COMPONENT::POS_Z, COMPONENT::ROT_Z });
Components const CTP_noise_comps({ COMPONENT::VEL_X, COMPONENT::VEL_Y, COMPONENT::VEL_Z, COMPONENT::VEL_ROT_Z });

Components const CTRA_state_comps({ COMPONENT::POS_X,   COMPONENT::POS_Y, COMPONENT::POS_Z,    COMPONENT::VEL_ABS,
                                   COMPONENT::ACC_ABS, COMPONENT::ROT_Z, COMPONENT::VEL_ROT_Z });
Components const CTRA_noise_comps({ COMPONENT::JERK_ABS, COMPONENT::ACC_ROT_Z });

Components const CTRV_state_comps({ COMPONENT::POS_X,   COMPONENT::POS_Y, COMPONENT::POS_Z,
                                   COMPONENT::VEL_ABS, COMPONENT::ROT_Z, COMPONENT::VEL_ROT_Z });
Components const CTRV_noise_comps({ COMPONENT::ACC_ABS, COMPONENT::ACC_ROT_Z });

Components const CV_state_comps({ COMPONENT::POS_X,
                                 COMPONENT::POS_Y,
                                 COMPONENT::POS_Z,
                                 COMPONENT::VEL_X,
                                 COMPONENT::VEL_Y });
Components const CV_noise_comps({ COMPONENT::ACC_X, COMPONENT::ACC_Y, COMPONENT::VEL_Z });

Components const ISCATR_state_comps({ COMPONENT::POS_X, COMPONENT::POS_Y, COMPONENT::POS_Z,
                                     COMPONENT::VEL_X, COMPONENT::VEL_Y, COMPONENT::ACC_X,
                                     COMPONENT::ACC_Y, COMPONENT::ROT_Z, COMPONENT::VEL_ROT_Z });
Components const ISCATR_noise_comps({ COMPONENT::JERK_X, COMPONENT::JERK_Y, COMPONENT::VEL_Z, COMPONENT::ACC_ROT_Z });

Components const EXTENT_RECTANGULAR({ COMPONENT::LENGTH, COMPONENT::WIDTH, COMPONENT::HEIGHT });
Components const EXTENT_RECTANGULAR_NOISE({ COMPONENT::LENGTH_CHANGE,
                                           COMPONENT::WIDTH_CHANGE,
                                           COMPONENT::HEIGHT_CHANGE });
Components const EXTENT_CIRCULAR({ COMPONENT::RADIUS });
Components const EXTENT_CIRCULAR_NOISE({ COMPONENT::RADIUS_CHANGE });

template <STATE_MODEL_TYPE model, EXTENT extent>
std::pair<Components, Components> state_and_noise_comps()
{
  Components base_comps = [] {
    if constexpr (model == STATE_MODEL_TYPE::CA)
    {
      return CA_state_comps;
    }
    if constexpr (model == STATE_MODEL_TYPE::CP)
    {
      return CP_state_comps;
    }
    if constexpr (model == STATE_MODEL_TYPE::CTP)
    {
      return CTP_state_comps;
    }
    if constexpr (model == STATE_MODEL_TYPE::CTRA)
    {
      return CTRA_state_comps;
    }
    if constexpr (model == STATE_MODEL_TYPE::CTRV)
    {
      return CTRV_state_comps;
    }
    if constexpr (model == STATE_MODEL_TYPE::CV)
    {
      return CV_state_comps;
    }
    if constexpr (model == STATE_MODEL_TYPE::ISCATR)
    {
      return ISCATR_state_comps;
    }
    LOG_FATAL("unknown state mode");
    assert(false);
    std::unreachable();
  }();
  Components base_noise = [] {
    if constexpr (model == STATE_MODEL_TYPE::CA)
    {
      return CA_noise_comps;
    }
    if constexpr (model == STATE_MODEL_TYPE::CP)
    {
      return CP_noise_comps;
    }
    if constexpr (model == STATE_MODEL_TYPE::CTP)
    {
      return CTP_noise_comps;
    }
    if constexpr (model == STATE_MODEL_TYPE::CTRA)
    {
      return CTRA_noise_comps;
    }
    if constexpr (model == STATE_MODEL_TYPE::CTRV)
    {
      return CTRV_noise_comps;
    }
    if constexpr (model == STATE_MODEL_TYPE::CV)
    {
      return CV_noise_comps;
    }
    if constexpr (model == STATE_MODEL_TYPE::ISCATR)
    {
      return ISCATR_noise_comps;
    }
    LOG_FATAL("unknown state mode");
    assert(false);
    std::unreachable();
  }();
  Components extent_comps = [] -> Components {
    if constexpr (extent == EXTENT::NONE)
    {
      return Components({});
    }
    if constexpr (extent == EXTENT::RECTANGULAR)
    {
      return EXTENT_RECTANGULAR;
    }
    if constexpr (extent == EXTENT::CIRCULAR)
    {
      return EXTENT_CIRCULAR;
    }
    LOG_FATAL("unknown extent");
    assert(false);
    std::unreachable();
  }();

  Components extent_noise = [] -> Components {
    if constexpr (extent == EXTENT::NONE)
    {
      return Components({});
    }
    if constexpr (extent == EXTENT::RECTANGULAR)
    {
      return EXTENT_RECTANGULAR_NOISE;
    }
    if constexpr (extent == EXTENT::CIRCULAR)
    {
      return EXTENT_CIRCULAR_NOISE;
    }
    LOG_FATAL("unknown extent");
    assert(false);
    std::unreachable();
  }();
  return { base_comps.merge(extent_comps), base_noise.merge(extent_noise) };
}

std::pair<Components, Components> state_and_noise_comps(STATE_MODEL_TYPE model, EXTENT extent);

template <STATE_MODEL_TYPE model>
std::pair<Components, Components> state_and_noise_comps(EXTENT extent)
{
  switch (extent)
  {
    case EXTENT::NONE:
    {
      return state_and_noise_comps<model, EXTENT::NONE>();
    }
    case EXTENT::RECTANGULAR:
    {
      return state_and_noise_comps<model, EXTENT::RECTANGULAR>();
    }
    case EXTENT::CIRCULAR:
    {
      return state_and_noise_comps<model, EXTENT::CIRCULAR>();
    }
  }
  LOG_FATAL("error");
  assert(false);
  std::unreachable();
}

inline std::unordered_map<STATE_MODEL_TYPE, Components> state_model_type_2_state_comps_map{
  { STATE_MODEL_TYPE::CA, CA_state_comps },        { STATE_MODEL_TYPE::CP, CP_state_comps },
  { STATE_MODEL_TYPE::CV, CV_state_comps },        { STATE_MODEL_TYPE::CTP, CTP_state_comps },
  { STATE_MODEL_TYPE::CTRA, CTRA_state_comps },    { STATE_MODEL_TYPE::CTRV, CTRV_state_comps },
  { STATE_MODEL_TYPE::ISCATR, ISCATR_state_comps }
};

inline std::unordered_map<STATE_MODEL_TYPE, Components> state_model_type_2_noise_comps_map{
  { STATE_MODEL_TYPE::CA, CA_noise_comps },        { STATE_MODEL_TYPE::CP, CP_noise_comps },
  { STATE_MODEL_TYPE::CV, CV_noise_comps },        { STATE_MODEL_TYPE::CTP, CTP_noise_comps },
  { STATE_MODEL_TYPE::CTRA, CTRA_noise_comps },    { STATE_MODEL_TYPE::CTRV, CTRV_noise_comps },
  { STATE_MODEL_TYPE::ISCATR, ISCATR_noise_comps }
};

}  // namespace ttb

inline std::size_t boost::hash_value(ttb::Components const& comps)
{
  boost::hash<std::vector<ttb::COMPONENT>> constexpr hasher;
  return hasher(comps._comps);
}


template <>
struct std::formatter<ttb::Components>
{
  constexpr auto parse(std::format_parse_context& ctx)
  {
    return ctx.begin();
  }
  auto format(ttb::Components const& comps, std::format_context& ctx) const
  {
    return std::format_to(ctx.out(), "{}", comps.toString());
  }
};