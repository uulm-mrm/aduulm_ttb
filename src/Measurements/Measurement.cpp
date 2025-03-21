#include "tracking_lib/Measurements/Measurement.h"

#include "tracking_lib/Distributions/BaseDistribution.h"

namespace ttb
{
IDGenerator<MeasurementId> Measurement::_idGenerator{};

Measurement::Measurement(std::unique_ptr<BaseDistribution> dist, Time time, Components meas_comps)
  : _dist{ std::move(dist) }, _time{ time }, _meas_comps{ std::move(meas_comps) }
{
  assert([&] {  // NOLINT
    if (_dist->mean().rows() != _meas_comps._comps.size())
    {
      LOG_FATAL("Num of rows: " << _dist->mean().rows()
                                << " does not fit the number of Meas Comps: " << _meas_comps._comps.size());
      return false;
    }
    return true;
  }());
}

Measurement::Measurement(GaussianDistribution const& gauss_dist, Time time, Components meas_comps)
  : Measurement(std::make_unique<GaussianDistribution>(gauss_dist), time, std::move(meas_comps))
{
}

Measurement::Measurement(Measurement const& other)
  : _id{ other._id }
  , _objectId{ other._objectId }
  , _dist{ other._dist->clone() }
  , _time{ other._time }
  , _meas_comps{ other._meas_comps }
  , _ref_point_measured{ other._ref_point_measured }
  , _classification{ other._classification }
{
}

Measurement::Measurement(Measurement&& other) noexcept = default;

Measurement& Measurement::operator=(Measurement&& other) noexcept = default;

Measurement::~Measurement() = default;

Measurement& Measurement::operator=(Measurement const& other)
{
  Measurement tmp(other);
  *this = std::move(tmp);
  return *this;
}

std::string Measurement::toString(std::string const& prefix) const
{
  std::string out = prefix + "Measurement\n";
  out += prefix + _dist->toString("|\t") + prefix + "|\tComps: " + _meas_comps.toString() + "\n" + prefix +
         "|\tMeasurement ID: " + std::to_string(_id.value_) + "\n";
  if (_objectId.has_value())
  {
    out += prefix + "|\tObject ID: " + std::to_string(_objectId.value().value_) + "\n";
  }
  out += prefix + "|\tClassification: " + _classification.toString();

  return out;
}

}  // namespace ttb