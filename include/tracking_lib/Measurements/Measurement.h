#pragma once

#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/Classification/MeasClassification.h"
#include "tracking_lib/TTBTypes/Components.h"
#include "tracking_lib/Distributions/GaussianDistribution.h"

namespace ttb
{

/// This represents a single measurement.
class Measurement final
{
public:
  /// Ctor with a BaseDistribution
  Measurement(std::unique_ptr<BaseDistribution> dist, Time time, Components meas_comps);
  /// Ctor with a GaussianDistribution
  Measurement(GaussianDistribution const& gauss_dist, Time time, Components meas_comps);
  Measurement(Measurement const& other);
  Measurement(Measurement&& other) noexcept;
  Measurement& operator=(Measurement const& other);
  Measurement& operator=(Measurement&& other) noexcept;
  ~Measurement();
  /// string representation
  [[nodiscard]] std::string toString(std::string const& prefix = "") const;
  /// unique id
  MeasurementId _id{ _idGenerator.getID() };
  /// consistent! object id
  std::optional<ObjectId> _objectId;
  /// The underlying measured Distribution
  std::unique_ptr<BaseDistribution> _dist;
  /// Time of the measurement
  Time _time{};
  /// The measured components
  Components _meas_comps;
  /// whether the refPoint of the distribution is actually measured
  bool _ref_point_measured = false;
  /// The classification of the measurement
  classification::MeasClassification _classification;
  static IDGenerator<MeasurementId> _idGenerator;
};

}  // namespace ttb