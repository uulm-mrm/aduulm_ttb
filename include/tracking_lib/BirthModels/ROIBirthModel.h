#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/States/State.h"
#include <tracking_lib/Misc/Region.h>

namespace ttb
{

/// Static Birth model that sets up new tracks at points relative to center of ROI
template <class UDT>
class ROIBirthModel
{
public:
  explicit ROIBirthModel(TTBManager<UDT>* manager);
  [[nodiscard]] static BIRTH_MODEL_TYPE type() noexcept;
  [[nodiscard]] TTBManager<UDT>* manager() const;
  /// Returns birth Dists at predefined birth state distributions
  [[nodiscard]] BirthTracks<UDT> getBirthTracks(DataSourceContainer<UDT> const& measContainer,
                                                std::map<MeasurementId, UDT> const& /*unused*/,
                                                std::vector<State<UDT>> const& existingTracks);
  [[nodiscard]] std::map<COMPONENT, double> getCenterOfROI(Components const& comps, Region const& roi);
  Eigen::Matrix2d computeCovariance(Point const& center, Polygon::SinglePolygon const& polygon, double confidence);

  Eigen::Matrix2d computeInnerCovariance(Point const& center,
                                         Polygon::SinglePolygon const& polygon,
                                         double confidence_level);

  void reset();
  /// read and store the static birth locations with respect to roi
  void loadBirthLocations(DataSourceContainer<UDT> const& measurementContainer);
  TTBManager<UDT>* _manager;
  /// vector holding a list of birth locations
  std::vector<std::map<COMPONENT, double>> _birthLocations;
  /// vector holding a list of variance matrices
  std::vector<std::map<COMPONENT, double>> _birthVariances;

  Polygon::SinglePolygon _roi = {};
  Point _point{ 0.0, 0.0 };
};
}  // namespace ttb
