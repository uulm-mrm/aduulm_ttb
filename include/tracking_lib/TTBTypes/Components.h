#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"

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
  [[nodiscard]] std::optional<Indices> indexOf(std::vector<COMPONENT> const& comps) const;
  /// Return the intersection of my Components and the other Components
  [[nodiscard]] Components intersection(Components const& otherComps) const;
  /// Check, whether I contain all of the other Components
  [[nodiscard]] bool contains(Components const& other) const;
  /// Set Difference between this and other; return = this\other
  [[nodiscard]] Components diff(Components const& other) const;
  /// Merge all other Components into myself
  [[nodiscard]] Components merge(Components const& others) const;
  /// data
  std::vector<COMPONENT> _comps;
};

}  // namespace ttb