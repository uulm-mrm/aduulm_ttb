#include "tracking_lib/DataContainer/DataSourceContainer.h"

namespace ttb
{

template class DataSourceContainer<double>;
template class DataSourceContainer<AD>;

template <class UDT>
IDGenerator<DataSourceContainerId> DataSourceContainer<UDT>::_idGenerator{};

template <class UDT>
std::string DataSourceContainer<UDT>::toString() const
{
  std::string out{ "DataSourceContainer\n\tSource ID: " + _source_id.value_ +
                   "\n"
                   "\tTime: " +
                   std::format("{}", _time) + "\n" + std::format("\tId: {}\n", _id) };
  for (State<UDT> const& meas : _measurements)
  {
    out += "\tMeasurement: " + meas.toString();
  }
  for (BernoulliState<UDT> const& state : _tracks)
  {
    out += "\tMeasurement: " + state.state.toString();
  }
  out += _sensorInfo.toString("\t");
  out += _predictionInfo._egoMotion.toString("\t");
  if (_predictionInfo._roi.has_value())
  {
    out += _predictionInfo._roi.value().toString("\t");
  }
  if (_ground_truth_info.has_value())
  {
    out += std::format("\tGround Truth Info:\n\t\tProcess model noise {}\n",
                       _ground_truth_info.value().state_model_info.noise_dev);

    out += std::format("\t\tMeasModelInfo: pD {} clutter {} meas noise {}\n",
                       _ground_truth_info.value().meas_model_info.detection_prob,
                       _ground_truth_info.value().meas_model_info.clutter_rate,
                       _ground_truth_info.value().meas_model_info.meas_xy_dev);
  }
  return out;
}

SE3Trafo default_tracking_cs_to_default_sensor_cs(SourceId const& id)
{
  if (id.value_.contains("camera"))
  {
    Matrix44 H = Matrix44::Zero();
    H(3, 3) = 1;
    H({ 0, 1, 2 }, { 0, 1, 2 }) = Matrix<double>{ { 0, -1, 0 }, { 0, 0, -1 }, { 1, 0, 0 } };
    SE3Trafo t;
    t.matrix() = H;
    return t;
  }
  return SE3Trafo::Identity();
}

}  // namespace ttb
