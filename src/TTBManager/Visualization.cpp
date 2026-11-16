#include "tracking_lib/TTBManager/Visualization.h"
#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/Transformations/Transformation.h"

#include <set>
#include <boost/geometry/algorithms/append.hpp>
#include <boost/geometry/algorithms/centroid.hpp>
#include <boost/geometry/algorithms/convex_hull.hpp>
#include <implot/implot.h>
#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>
#include <tracy/tracy/Tracy.hpp>
#include <tracy/tracy/TracyC.h>
#define GL_SILENCE_DEPRECATION
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif
#include <stacktrace>
#include <GLFW/glfw3.h>  // Will drag system OpenGL headers

#include <boost/math/distributions/gamma.hpp>
#include <boost/math/special_functions/beta.hpp>
#include <boost/math/distributions/beta.hpp>
#include <boost/math/distributions/chi_squared.hpp>

#include <unsupported/Eigen/src/MatrixFunctions/MatrixSquareRoot.h>
namespace ttb
{

template <class Data>
Time get_data_time(Data const& data)
{
  if constexpr (std::is_convertible_v<Data, Time>)
  {
    return data;
  }
  else if constexpr (requires {
                       { data._time } -> std::convertible_to<Time>;
                     })
  {
    return data._time;
  }
  else if constexpr (requires {
                       { std::get<0>(data) } -> std::convertible_to<Time>;
                     })
  {
    return std::get<0>(data);
  }
  else
  {
    NotDefined<Data> this_does_not_compile;
    static_assert(always_false_v<Data>, "unsupported data type");
  }
}

template <class Data>
auto find_data(Data const& timed_data, Time time) -> std::pair<std::optional<typename Data::value_type>, Duration>
{
  if (timed_data.empty())
  {
    return {};
  }
  assert([&] {
    if (not std::ranges::is_sorted(timed_data, {}, [](auto const& data_point) { return get_data_time(data_point); }))
    {
      LOG_FATAL("not sorted data .. ");
      for (auto const& data : timed_data)
      {
        LOG_FATAL(std::format("Time: {}", get_data_time(data)));
      }
      LOG_FATAL(std::stacktrace::current());
      return false;
    }
    return true;
  }());
  if (get_data_time(timed_data.back()) < time)
  {
    return { timed_data.back(), time - get_data_time(timed_data.back()) };
  }
  for (auto const& data : std::views::reverse(timed_data))
  {
    if (Time const data_time = get_data_time(data); data_time <= time)
    {
      return { data, std::chrono::abs(data_time - time) };
    }
  }
  return {};
}

struct Triangle
{
  Vector2 belief{ { 0, 0 } };
  Vector2 disbelief{ { 1, 0 } };
  Vector2 uncertainty{ { 0.5, std::sqrt(3) / 2.0 } };
};

void plot_sl_triangle(Triangle const& tri, std::string const& belief = "Ok", std::string const& disbelief = "Not Ok")
{
  ImPlot::PlotLine("SL Triangle",
                   std::array{ tri.belief.x(), tri.disbelief.x() }.data(),
                   std::array{ tri.belief.y(), tri.disbelief.y() }.data(),
                   2,
                   ImPlotItemFlags_NoLegend);
  ImPlot::PlotLine("SL Triangle",
                   std::array{ tri.disbelief.x(), tri.uncertainty.x() }.data(),
                   std::array{ tri.disbelief.y(), tri.uncertainty.y() }.data(),
                   2,
                   ImPlotItemFlags_NoLegend);
  ImPlot::PlotLine("SL Triangle",
                   std::array{ tri.uncertainty.x(), tri.belief.x() }.data(),
                   std::array{ tri.uncertainty.y(), tri.belief.y() }.data(),
                   2,
                   ImPlotItemFlags_NoLegend);
  ImPlot::PlotText(belief.c_str(), tri.belief.x(), tri.belief.y());
  ImPlot::PlotText(disbelief.c_str(), tri.disbelief.x(), tri.disbelief.y());
}

void plot_opinion(Opinion const& opinion, std::string const& name, Triangle const& tri, std::string const& label = "")
{
  double const x = tri.belief.x() * opinion.belief() + tri.disbelief.x() * opinion.disbelief() +
                   tri.uncertainty.x() * opinion.uncertainty();
  double const y = tri.belief.y() * opinion.belief() + tri.disbelief.y() * opinion.disbelief() +
                   tri.uncertainty.y() * opinion.uncertainty();
  ImVec4 const color = [&] -> ImVec4 {
    float const g = opinion.getProbability();
    float const u = opinion.uncertainty();
    return { 1.0f - g, 0.0f + g, 0.0f, 1.0f - u };
  }();
  ImPlot::SetNextLineStyle(color);
  ImPlot::PlotScatter(name.c_str(), std::array{ x }.data(), std::array{ y }.data(), 1);
  if (not label.empty())
  {
    ImPlot::PlotText(label.c_str(), x, y);
  }
}

template class Vizu<double>;
template class Vizu<AD>;

template <class UDT>
Vizu<UDT>::Vizu(TTBManager<UDT>* manager) : _manager{ manager }
{
}

template <class T>
std::vector<T> to_x_data(auto& vals)
{
  std::vector<T> xs;
  for (auto const& x : vals)
  {
    xs.push_back(x);
  }
  return xs;
}

std::pair<std::vector<double>, std::vector<double>> to_xy_data(auto& vals)
{
  std::vector<double> xs;
  std::vector<double> ys;
  for (auto const& [x, y] : vals)
  {
    xs.push_back(x);
    ys.push_back(y);
  }
  return std::pair{ std::move(xs), std::move(ys) };
}

auto to_xyy_data(auto& vals)
{
  std::vector<double> xs;
  std::vector<double> y1s;
  std::vector<double> y2s;
  for (auto const& [x, y1, y2] : vals)
  {
    xs.push_back(x);
    y1s.push_back(y1);
    y2s.push_back(y2);
  }
  return std::tuple{ std::move(xs), std::move(y1s), std::move(y2s) };
}

auto to_xyyy_data(auto& vals)
{
  std::vector<double> xs;
  std::vector<double> y1s;
  std::vector<double> y2s;
  std::vector<double> y3s;
  for (auto const& [x, y1, y2, y3] : vals)
  {
    xs.push_back(x);
    y1s.push_back(y1);
    y2s.push_back(y2);
    y3s.push_back(y3);
  }
  return std::tuple{ std::move(xs), std::move(y1s), std::move(y2s), std::move(y3s) };
}

auto to_xyyyy_data(auto& vals)
{
  std::vector<double> xs;
  std::vector<double> y1s;
  std::vector<double> y2s;
  std::vector<double> y3s;
  std::vector<double> y4s;
  for (auto const& [x, y1, y2, y3, y4] : vals)
  {
    xs.push_back(x);
    y1s.push_back(y1);
    y2s.push_back(y2);
    y3s.push_back(y3);
    y4s.push_back(y4);
  }
  return std::tuple{ std::move(xs), std::move(y1s), std::move(y2s), std::move(y3s), std::move(y4s) };
}

struct DefaultDataFormat
{
  std::pair<double, double> operator()(auto const& data)
  {
    if constexpr (requires {
                    { std::get<1>(data) } -> std::convertible_to<double>;
                  })
    {
      return std::pair{ to_seconds(get_data_time(data).time_since_epoch()), std::get<1>(data) };
    }
    if constexpr (requires {
                    { std::get<1>(data) } -> std::convertible_to<Duration>;
                  })
    {
      return std::pair{ to_seconds(get_data_time(data).time_since_epoch()), to_milliseconds(std::get<1>(data)) };
    }
    else
    {
      LOG_FATAL("do not know how to convert data to double");
      std::terminate();
    }
  }
};

template <class Data>
auto filter_data(Data const& data_vals, Time now, Duration until)
{
  assert([&] {  // NOLINT
    if (not std::ranges::is_sorted(data_vals, {}, [&](auto const& data) { return get_data_time(data); }))
    {
      LOG_FATAL("data is not sorted");
      LOG_FATAL(std::stacktrace::current());
      return true;
    }
    return true;
  }());
  auto start =
      std::ranges::lower_bound(data_vals, now - until, {}, [&](auto const& data) { return get_data_time(data); });
  auto end = std::ranges::upper_bound(data_vals, now, {}, [&](auto const& data) { return get_data_time(data); });
  return std::span(start, end);
}

template <typename T1 = double, typename T2 = double, class F = DefaultDataFormat>
std::pair<std::vector<T1>, std::vector<T2>>
to_xy(auto const& data_vals, Time now, Duration until, F transform = DefaultDataFormat{})
{
  auto filtered_data = filter_data(data_vals, now, until) | std::views::transform(transform);
  return to_xy_data(filtered_data);
}

template <typename T1 = double, typename T2 = double, typename T3 = double, class F = DefaultDataFormat>
std::tuple<std::vector<T1>, std::vector<T2>, std::vector<T3>>
to_xyy(auto const& data_vals, Time now, Duration until, F transform = DefaultDataFormat{})
{
  auto filtered_data = filter_data(data_vals, now, until) | std::views::transform(transform);
  return to_xyy_data(filtered_data);
}

template <typename T = double, class F = DefaultDataFormat>
std::vector<T> to_x(auto const& data_vals, Time now, Duration until, F transform = DefaultDataFormat{})
{
  auto filtered_data = filter_data(data_vals, now, until) | std::views::transform(transform);
  return to_x_data<T>(filtered_data);
}

template <class UDT>
std::optional<Vector2> to_xy(State<UDT> const& state, TTBManager<UDT>* manager)
{
  auto const [model, dist] = state.getEstimate();
  if (auto const ind = manager->state_comps(model).indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y }); ind.has_value())
  {
    return dist.mean()(ind.value());
  }
  return {};
}

std::pair<std::vector<double>, std::vector<double>> to_xy(Interval const& interval)
{
  std::vector<double> xs;
  std::vector<double> ys;
  for (auto const& line : interval._multi_line)
  {
    for (Point const& point : line)
    {
      xs.push_back(point._point.x());
      ys.push_back(point._point.y());
    }
  }
  return { std::move(xs), std::move(ys) };
}

std::pair<std::vector<double>, std::vector<double>> to_xy(Polygon::SinglePolygon const& poly)
{
  std::vector<double> xs;
  std::vector<double> ys;
  for (auto const& point : poly.outer())
  {
    xs.push_back(point._point.x());
    ys.push_back(point._point.y());
  }
  return { std::move(xs), std::move(ys) };
}

std::pair<std::vector<double>, std::vector<double>> to_xy(Polygon const& multi_poly)
{
  std::vector<double> xs;
  std::vector<double> ys;
  for (auto const& poly : multi_poly.multiPolygon)
  {
    for (auto const& point : poly.outer())
    {
      xs.push_back(point._point.x());
      ys.push_back(point._point.y());
    }
  }
  return { std::move(xs), std::move(ys) };
}

std::pair<std::vector<double>, std::vector<double>> to_xy(Region const& region)
{
  if (auto const it = region._polygons.find(Components({ COMPONENT::POS_X, COMPONENT::POS_Y }));
      it != region._polygons.end())
  {
    return to_xy(it->second);
  }
  return {};
}

template <class UDT>
void plot_fov(std::pair<Time, DataSourceContainer<UDT>> const& data, bool show_name)
{
  if (data.second._sensorInfo._sensor_fov.has_value())
  {
    auto [xs, ys] = to_xy(data.second._sensorInfo._sensor_fov.value());
    ImPlot::PlotLine(data.second._source_id.value_.c_str(), xs.data(), ys.data(), xs.size());
    if (show_name)
    {
      for (std::size_t i = 0; i < xs.size(); ++i)
      {
        ImPlot::PlotText(data.second._source_id.value_.c_str(), xs[i], ys[i]);
      }
    }
    if (data.second._predictionInfo._roi.has_value())
    {
      std::tie(xs, ys) = to_xy(data.second._predictionInfo._roi.value());
      ImPlot::PlotLine("ROI", xs.data(), ys.data(), xs.size());
    }
  }
}

template <class UDT>
std::pair<std::vector<double>, std::vector<double>> to_box(Vector<UDT> const& state, Components const& comps)
{
  std::vector<double> xs, ys;
  if (auto const ind = comps.indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y, COMPONENT::RADIUS }); ind.has_value())
  {
    Vector2 const m{ { to_val(state(ind.value()(0))), to_val(state(ind.value()(1))) } };
    double const r = to_val(state(ind.value()(2)));
    for (double ts = 0; ts < 2 * std::numbers::pi; ts += 0.05)
    {
      xs.push_back(m(0) + std::cos(ts) * r);
      ys.push_back(m(1) + std::sin(ts) * r);
    }
    return { std::move(xs), std::move(ys) };
  }
  Components const req_comps(
      { COMPONENT::POS_X, COMPONENT::POS_Y, COMPONENT::ROT_Z, COMPONENT::LENGTH, COMPONENT::WIDTH });
  std::optional<Vector<UDT>> const transformed_state = transformation::transform(state, comps, req_comps);

  if (transformed_state.has_value())
  {
    for (REFERENCE_POINT const rp : { REFERENCE_POINT::FRONT_RIGHT,
                                      REFERENCE_POINT::FRONT_LEFT,
                                      REFERENCE_POINT::BACK_LEFT,
                                      REFERENCE_POINT::BACK_RIGHT })
    {
      auto ind_xy = req_comps.indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y });
      Vector<UDT> corner =
          transformation::transform(transformed_state.value(), req_comps, REFERENCE_POINT::CENTER, rp)(ind_xy.value());
      xs.emplace_back(to_val(corner(0)));
      ys.emplace_back(to_val(corner(1)));
    }
    xs.emplace_back(xs.front());
    ys.emplace_back(ys.front());
    return { std::move(xs), std::move(ys) };
  }
  if (auto const xsys = comps.indexOf({ COMPONENT::HULL_POS_X_MIN,
                                        COMPONENT::HULL_POS_X_MAX,
                                        COMPONENT::HULL_POS_Y_MIN,
                                        COMPONENT::HULL_POS_Y_MAX });
      xsys.has_value())
  {
    xs = { to_val(state(xsys.value()(0))), to_val(state(xsys.value()(1))) };
    ys = { to_val(state(xsys.value()(2))), to_val(state(xsys.value()(3))) };
    return { std::move(xs), std::move(ys) };
  }
  if (auto const xy = comps.indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y }); xy.has_value())
  {
    xs.emplace_back(to_val(state(xy.value()(0))));
    ys.emplace_back(to_val(state(xy.value()(1))));
    return { std::move(xs), std::move(ys) };
  }
  return {};
}

template <class UDT>
std::pair<std::vector<double>, std::vector<double>> to_box(GaussianDistribution<UDT> const& dist,
                                                           Components const& comps)
{
  return to_box(dist.mean(), comps);
}

template <class UDT>
std::pair<std::vector<double>, std::vector<double>> to_cov(GaussianDistribution<UDT> const& dist,
                                                           Components const& comps)
{
  std::optional<Indices> ind_xy = comps.indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y });
  if (not ind_xy.has_value())
  {
    return {};
  }
  Index const ix = ind_xy.value()(0);
  Index const iy = ind_xy.value()(1);
  Matrix<UDT> const PT = dist.covariance();
  Matrix<double> P{ { to_val(PT(ix, ix)), to_val(PT(ix, iy)) }, { to_val(PT(iy, ix)), to_val(PT(iy, iy)) } };
  Matrix22 const P_xy = P.sqrt();

  std::vector<double> xs;
  std::vector<double> ys;
  Vector<double> const mean{ { to_val(dist.mean()(ind_xy.value()(0))), to_val(dist.mean()(ind_xy.value()(1))) } };
  for (double ts = 0; ts < 2 * std::numbers::pi; ts += 0.1)
  {
    Vector2 t = mean + P_xy * Vector2{ { std::sin(ts), -std::cos(ts) } };
    xs.push_back(t(0));
    ys.push_back(t(1));
  }
  return { std::move(xs), std::move(ys) };
}

template <class UDT>
std::pair<std::vector<double>, std::vector<double>> to_box(State<UDT> const& state)
{
  if (state.isEmpty())
  {
    return {};
  }
  auto const& [model_id, dist] = state.getEstimate();
  return to_box(dist, state._manager->state_comps(model_id));
}

template <class UDT>
std::pair<std::vector<double>, std::vector<double>> to_vel(State<UDT> const& state)
{
  auto const& [model_id, dist] = state.getEstimate();
  auto ind_xy = state._manager->state_comps(model_id).indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y });
  if (not ind_xy.has_value())
  {
    return {};
  }
  if (auto const vel = transformation::transform(
          dist.mean(), state._manager->state_comps(model_id), Components({ COMPONENT::VEL_X, COMPONENT::VEL_Y }));
      vel.has_value())
  {
    std::vector xvel{ to_val(dist.mean()(ind_xy.value()(0))), to_val(dist.mean()(ind_xy.value()(0)) + vel.value()(0)) };
    std::vector yvel{ to_val(dist.mean()(ind_xy.value()(1))), to_val(dist.mean()(ind_xy.value()(1)) + vel.value()(1)) };
    return { std::move(xvel), std::move(yvel) };
  }
  return {};
}

template <class UDT>
std::tuple<std::vector<double>, std::vector<double>> to_beam(State<UDT> const& meas,
                                                             COMPONENT comp,
                                                             SensorInformation const& sensor_information)
{
  if (meas.isEmpty())
  {
    return {};
  }
  Vector<UDT> const mean = meas.getEstimate().second.mean();
  std::tuple<std::vector<double>, std::vector<double>> beams;
  if (auto ind = meas._manager->state_comps(meas._state_dist.begin()->first).indexOf(comp); ind.has_value())
  {
    Vector3 const p1 = [&] -> Vector3 {
      if (comp == COMPONENT::AZIMUTH)
      {
        double const angle = to_val(mean(ind.value()));
        Vector3 out{ { std::cos(angle), std::sin(angle), 0 } };
        return out * 1500;
      }
      Vector3 out{ { to_val(mean(ind.value())), 0, 1 } };
      out.normalize();
      return out * 150;
    }();
    Vector3 const p0 = Vector3::Zero();
    Vector<double> const t0 = sensor_information._to_sensor_cs.inverse() * p0.homogeneous();
    Vector<double> const t1 = sensor_information._to_sensor_cs.inverse() * p1.homogeneous();
    std::get<0>(beams) = { t0(0), t1(0) };
    std::get<1>(beams) = { t0(1), t1(1) };
  }
  return beams;
}

template <class UDT>
Label select_track_label(std::map<Label, std::vector<State<UDT>>> const& tracks)
{
  static Label track_label = Label{ 0 };
  if (ImGui::BeginCombo("Select Track Label", std::to_string(track_label.value_).c_str()))
  {
    bool is_selected = (Label{ 0 } == track_label);
    if (ImGui::Selectable(std::format("{} ({})", "0", 0).c_str(), is_selected))
    {
      track_label = Label{ 0 };
    }
    if (is_selected)
    {
      ImGui::SetItemDefaultFocus();
    }
    for (auto const& [label, states] : tracks)
    {
      if (states.size() < 2)
      {
        continue;
      }
      is_selected = (label == track_label);
      if (ImGui::Selectable(std::format("{} ({})", label.value_, states.size()).c_str(), is_selected))
      {
        track_label = label;
      }
      if (is_selected)
      {
        ImGui::SetItemDefaultFocus();
      }
    }
    ImGui::EndCombo();
  }
  return track_label;
}

struct Header
{
  bool auto_follow;
  Time now;
  Duration until;
};

template <class UDT>
Header show_header(TTBManager<UDT>* _manager)
{
  ImGui::Text("Filter time: %f", to_seconds(_manager->filter_time().time_since_epoch()));
  ImGui::SameLine();
  bool static use_real_time = true;
  ImGui::Checkbox("use real time", &use_real_time);
  ImGui::SameLine();
  double static time_sec = to_seconds(_manager->filter_time().time_since_epoch());
  Time const now = [&] {
    if (use_real_time)
    {
      time_sec = std::max(to_seconds(_manager->filter_time().time_since_epoch()), time_sec);
      return _manager->filter_time();
    }
    ImGui::InputDouble("Time s", &time_sec, 0.1, 1.0);
    ImGui::SameLine();
    time_sec = std::min(to_seconds(_manager->filter_time().time_since_epoch()), time_sec);
    return Time(std::chrono::milliseconds(static_cast<Duration::rep>(time_sec * 1000)));
  }();
  bool static auto_follow = false;
  ImGui::Checkbox("auto follow", &auto_follow);
  ImGui::SameLine();
  int static len_window = 100000;
  ImGui::InputInt("window length ms", &len_window, 100, 5000);
  Duration until = std::chrono::milliseconds(len_window);
  return { .auto_follow = auto_follow, .now = now, .until = until };
}

void plot_way(Way const& way)
{
  auto const [x, y] = to_xy(way.centerline);
  ImPlot::PlotLine("Way", x.data(), y.data(), x.size());
  auto const [x1, y1] = to_xy(way.outline);
  ImPlot::PlotLine("Way Polygon", x1.data(), y1.data(), x1.size());
}

void plot_map(Map const& map)
{
  for (Building const& building : map.buildings)
  {
    auto const [x, y] = to_xy(building.footprint);
    ImPlot::PlotLine("Building", x.data(), y.data(), x.size());
  }
  for (ParkingArea const& parking_area : map.parking_areas)
  {
    auto const [x, y] = to_xy(parking_area.area);
    ImPlot::PlotLine("Parking Area", x.data(), y.data(), x.size());
  }
  for (Way const& way : map.ways)
  {
    plot_way(way);
  }
}

template <class UDT>
void Vizu<UDT>::show_grouping()
{
  auto const [auto_follow, now, until] = show_header(_manager);
  bool static show_details = false;
  ImGui::Checkbox("show details", &show_details);
  auto const cycle_lock = _cycle_data.rlock();
  if (ImGui::BeginTabBar("Measurement Models"))
  {
    for (auto const rlock = _meas_model_data.rlock(); auto const& [model_id, model_data] : rlock.data)
    {
      if (ImGui::BeginTabItem(model_id.value_.c_str()))
      {
        if (ImPlot::BeginPlot("Tracks", ImVec2(-1, -1), ImPlotFlags_Equal))
        {
          ImPlot::SetupAxes("x", "y");
          ImPlot::SetupFinish();
          auto const [best_fit, min_time_diff] = find_data(model_data._track_groups, now);
          if (best_fit.has_value())
          {
            for (auto const& [ind, group] : std::views::enumerate(best_fit.value().second))
            {
              Polygon::SinglePolygon poly;
              std::map<Label, std::set<StateId>> labels;
              for (State<UDT> const& state : std::get<0>(group))
              {
                labels[state._label].insert(state._id);
                auto const [model, dist] = state.getEstimate();
                if (auto const pos = _manager->state_comps(model).indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y });
                    pos.has_value())
                {
                  boost::geometry::append(
                      poly, Point(to_val(dist.mean()(pos.value()(0))), to_val(dist.mean()(pos.value()(1)))));
                }
              }
              Polygon::SinglePolygon hull;
              boost::geometry::convex_hull(poly, hull);
              if (show_details)
              {
                try
                {
                  Point const center = boost::geometry::return_centroid<Point>(hull);
                  std::string str = std::format("#Hypotheses {}\n", std::get<1>(group));
                  for (auto const& [label, n] : labels)
                  {
                    str += std::format("Label {} #States {}\n", label, n.size());
                  }
                  str += std::get<2>(group).to_string();
                  ImPlot::PlotText(str.c_str(), center._point.x(), center._point.y());
                }
                catch (boost::geometry::centroid_exception const& exp)
                {
                }
              }
              auto [xs, ys] = to_xy(hull);
              ImPlot::PlotLine(std::to_string(ind).c_str(), xs.data(), ys.data(), xs.size());
              for (Label label : labels | std::views::keys)
              {
                if (auto const it = cycle_lock.data._tracks.find(label); it != cycle_lock.data._tracks.end())
                {
                  auto const [xs2,
                              ys2] = to_xy<double, double>(it->second, now, until, [this](State<UDT> const& track) {
                    auto const& [state_model_id, dist] = track.getEstimate();
                    Vector<UDT> pos_xy = dist.mean()(
                        _manager->state_comps(state_model_id).indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y }).value());
                    return std::tuple{ to_val(pos_xy(0)), to_val(pos_xy(1)) };
                  });
                  ImPlot::PlotLine(
                      std::to_string(ind).c_str(), xs2.data(), ys2.data(), xs2.size(), ImPlotItemFlags_NoLegend);
                }
              }
            }
          }
          ImPlot::EndPlot();
        }
        ImGui::EndTabItem();
      }
    }
    ImGui::EndTabBar();
  }
}

template <class UDT>
void Vizu<UDT>::show_tracks()
{
  auto const [auto_follow, now, until] = show_header(_manager);
  bool static show_tracks = true;
  ImGui::Checkbox("show tracks", &show_tracks);
  ImGui::SameLine();
  bool static show_vanished_tracks = false;
  ImGui::Checkbox("show vanished tracks", &show_vanished_tracks);
  ImGui::SameLine();
  bool static show_covariance = false;
  ImGui::Checkbox("show covariance", &show_covariance);
  ImGui::SameLine();
  bool static show_vel = false;
  ImGui::Checkbox("show velocity", &show_vel);
  ImGui::SameLine();
  bool static show_all_tracks = false;
  ImGui::Checkbox("show all tracks", &show_all_tracks);
  ImGui::SameLine();
  bool static detailed_track_info = false;
  ImGui::Checkbox("detailed track info", &detailed_track_info);
  ImGui::SameLine();
  bool static show_ids = false;
  ImGui::Checkbox("show ids", &show_ids);
  ImGui::SameLine();
  bool static show_gt = true;
  ImGui::Checkbox("show ground truth", &show_gt);
  Label const show_track_label = select_track_label(_cycle_data.rlock().data._tracks);
  bool static show_tracks_assessment = false;
  ImGui::Checkbox("show tracks assessment", &show_tracks_assessment);
  bool static show_detections = false;
  ImGui::Checkbox("show detections", &show_detections);
  ImGui::SameLine();
  bool static show_detection_details = false;
  ImGui::Checkbox("show detection details", &show_detection_details);
  bool static show_measurement_prediction = false;
  ImGui::Checkbox("show measurement prediction", &show_measurement_prediction);
  bool static show_birth_tracks = false;
  ImGui::Checkbox("show birth tracks", &show_birth_tracks);
  bool static show_fov = true;
  ImGui::Checkbox("show fov", &show_fov);
  bool static show_fov_names = false;
  ImGui::Checkbox("show fov names", &show_fov_names);
  bool static show_map = true;
  ImGui::Checkbox("show map", &show_map);
  double const height = [&] {
    if (not show_gt)
    {
      return -1.0f;
    }
    return 4 * (ImGui::GetWindowHeight() - ImGui::GetCursorPosY()) / 5;
  }();
  if (ImPlot::BeginPlot("Tracks", ImVec2(-1, height), ImPlotFlags_Equal))
  {
    ImPlot::SetupAxes("x", "y");
    ImPlot::SetupFinish();
    if (show_map)
    {
      for (auto const rlock = _meas_model_data.rlock(); auto const& [model_id, model_data] : rlock.data)
      {
        auto [best_fit, min_time_diff] = find_data(model_data._measurements, now);
        if (min_time_diff < 300ms and best_fit.has_value())
        {
          if (best_fit.value().second._map.has_value())
          {
            plot_map(best_fit.value().second._map.value());
            break;
          }
        }
      }
    }
    if (show_gt)
    {
      std::vector<double> xs;
      std::vector<double> ys;
      for (auto const rlock = _cycle_data.rlock(); auto const& [label, positions] : rlock.data._gt_tracks)
      {
        std::tie(xs, ys) = to_xy<double, double>(positions, now, until, [this](State<UDT> const& track) {
          auto const& [model_id, dist] = track.getEstimate();
          Vector<UDT> pos_xy =
              dist.mean()(_manager->state_comps(model_id).indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y }).value());
          return std::tuple{ to_val(pos_xy(0)), to_val(pos_xy(1)) };
        });

        ImVec4 const color = [&] -> ImVec4 { return { 1.0f, 1.0f, 1.0f, 1.0f }; }();
        ImPlot::SetNextLineStyle(color);
        ImPlot::PlotLine(
            std::to_string(label.value_).c_str(), xs.data(), ys.data(), xs.size(), ImPlotItemFlags_NoLegend);
      }
    }
    if (show_tracks)
    {
      std::vector<double> xs;
      std::vector<double> ys;
      std::vector<Opinion> track_op;
      if (show_track_label == Label{ 0 })
      {
        for (auto const rlock = _cycle_data.rlock(); auto const& [label, positions] : rlock.data._tracks)
        {
          std::tie(xs, ys) = to_xy(positions, now, until, [this](State<UDT> const& track) {
            auto const& [model_id, dist] = track.getEstimate();
            Vector<UDT> pos_xy =
                dist.mean()(_manager->state_comps(model_id).indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y }).value());
            return std::tuple{ to_val(pos_xy(0)), to_val(pos_xy(1)) };
          });
          ImPlot::PlotLine(
              std::format("{}_", label).c_str(), xs.data(), ys.data(), xs.size(), ImPlotItemFlags_NoLegend);
          if (show_ids)
          {
            auto const ids = to_x<StateId>(positions, now, until, [](State<UDT> const& track) { return track._id; });
            for (auto const [x, y, id] : std::views::zip(xs, ys, ids))
            {
              ImPlot::PlotText(std::to_string(id.value_).c_str(), x, y);
            }
          }
        }
        if (show_all_tracks)
        {
          for (auto const rlock = _cycle_data.rlock(); auto const& [label, positions] : rlock.data._all_tracks)
          {
            std::tie(xs, ys) = to_xy(positions, now, until, [this](State<UDT> const& track) {
              auto const& [model_id, dist] = track.getEstimate();
              Vector<UDT> pos_xy =
                  dist.mean()(_manager->state_comps(model_id).indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y }).value());
              return std::tuple{ to_val(pos_xy(0)), to_val(pos_xy(1)) };
            });
            ImPlot::PlotLine(
                std::format("{}_", label).c_str(), xs.data(), ys.data(), xs.size(), ImPlotItemFlags_NoLegend);
            if (show_ids)
            {
              auto const ids = to_x<StateId>(positions, now, until, [](State<UDT> const& track) { return track._id; });
              for (auto const [x, y, id] : std::views::zip(xs, ys, ids))
              {
                ImPlot::PlotText(std::to_string(id.value_).c_str(), x, y);
              }
            }
          }
        }
      }
      else
      {
        std::tie(xs, ys) =
            to_xy(_cycle_data.rlock().data._tracks.at(show_track_label), now, until, [this](State<UDT> const& track) {
              auto const& [model_id, dist] = track.getEstimate();
              Vector<UDT> pos_xy =
                  dist.mean()(_manager->state_comps(model_id).indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y }).value());
              return std::pair(to_val(pos_xy(0)), to_val(pos_xy(1)));
            });
        ImPlot::PlotLine(
            std::format("{}_", show_track_label).c_str(), xs.data(), ys.data(), xs.size(), ImPlotItemFlags_NoLegend);
        if (show_ids)
        {
          auto const ids = to_x<StateId>(_cycle_data.rlock().data._tracks.at(show_track_label),
                                         now,
                                         until,
                                         [](State<UDT> const& track) { return track._id; });
          for (auto const [x, y, id] : std::views::zip(xs, ys, ids))
          {
            ImPlot::PlotText(std::to_string(id.value_).c_str(), x, y);
          }
        }
      }
      auto const rlock = _cycle_data.rlock();
      auto [overlapping_tracks, min_time_diff_overlapping_tracks] = find_data(rlock.data._overlapping_tracks, now);
      for (auto const& [label, states] : rlock.data._tracks)
      {
        xs.clear();
        ys.clear();
        auto [best_state, min_time_diff] = find_data(states, now);
        if (min_time_diff < 300ms and best_state.has_value())
        {
          auto const& [model_id, dist] = best_state.value().getEstimate();
          auto ind_xy = _manager->state_comps(model_id).indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y });
          if (not ind_xy.has_value() or (show_track_label != Label{ 0 } and label != show_track_label))
          {
            continue;
          }
          Vector<UDT> const& mean = dist.mean();
          std::tie(xs, ys) = to_box(best_state.value());
          if (overlapping_tracks.has_value())
          {
            if (std::ranges::contains(
                    std::get<1>(overlapping_tracks.value()), best_state.value()._label, &State<UDT>::_label))
            {
              ImVec4 const color = [&] -> ImVec4 { return { 1.0f, 0.0f, 0.0f, 1.0f }; }();
              ImPlot::SetNextLineStyle(color);
            }
            else
            {
              ImVec4 const color = [&] -> ImVec4 { return { 0.0f, 1.0f, 0.0f, 1.0f }; }();
              ImPlot::SetNextLineStyle(color);
            }
          }
          else
          {
            ImVec4 const color = [&] -> ImVec4 { return { 0.0f, 1.0f, 0.0f, 1.0f }; }();
            ImPlot::SetNextLineStyle(color);
          }

          ImPlot::PlotLine(
              std::to_string(label.value_).c_str(), xs.data(), ys.data(), xs.size(), ImPlotItemFlags_NoLegend);

          if (show_covariance)
          {
            std::optional<transformation::SigmaPoints<UDT>> const sigma_points =
                transformation::createSigmaPoints(dist.mean(), dist.covariance());
            if (sigma_points.has_value())
            {
              for (Index i = 0; i < sigma_points.value().sigma_points.cols(); ++i)
              {
                Vector<UDT> state_vector = sigma_points.value().sigma_points.col(i);
                std::tie(xs, ys) = to_box(state_vector, _manager->state_comps(model_id));
                ImPlot::PlotLine(
                    std::to_string(label.value_).c_str(), xs.data(), ys.data(), xs.size(), ImPlotItemFlags_NoLegend);
              }
            }
          }
          if (show_vel)
          {
            std::tie(xs, ys) = to_vel(best_state.value());
            ImPlot::PlotLine(std::to_string(label.value_).c_str(), xs.data(), ys.data(), static_cast<int>(xs.size()));
          }
          if (detailed_track_info)
          {
            ImPlot::PlotText(best_state.value().toString().c_str(),
                             to_val(mean(ind_xy.value()(0))),
                             to_val(mean(ind_xy.value()(1))));
          }
          if (show_tracks_assessment)
          {
            double const x = to_val(mean(ind_xy.value()(0)));
            double const y = to_val(mean(ind_xy.value()(1)));
            Triangle const tri{ .belief = Vector2{ { x, y } },
                                .disbelief = Vector2{ { x + 10, y } },
                                .uncertainty = Vector2{ { x + 5, y + 10 * std::sqrt(3) / 2 } } };

            ImVec4 const color{ [&] { return ImVec4{ 0.5f, 0.5f, 0.5f, 1.0f }; }() };
            ImPlot::SetNextLineStyle(color);
            plot_sl_triangle(tri, "", "");
            ImPlot::SetNextLineStyle(color);
            plot_opinion(best_state.value()._meta_data._assessment, "Track Assessment", tri);
            for (auto const& [type_str, op] : best_state.value()._meta_data._assessment_debug_info)
            {
              plot_opinion(op, type_str, tri);
            }
          }
        }
      }
    }
    if (show_vanished_tracks)
    {
      for (auto const rlock = _cycle_data.rlock(); auto const& [time, positions] : rlock.data._vanished_tracks)
      {
        if (time < now and now - time < until)
        {
          for (State<UDT> const& vanished_track : positions)
          {
            State predicted = vanished_track;
            predicted.predict(now - predicted._time, EgoMotionDistribution::zero());
            if (not predicted.isEmpty())
            {
              auto const& [model_id, dist] = predicted.getEstimate();
              auto const [xs, ys] = to_box(dist.mean(), _manager->state_comps(model_id));
              ImPlot::SetNextLineStyle(ImVec4{ 0.5f, 0.5f, 0.5f, 1.0f });
              ImPlot::PlotLine(std::format("{}_", predicted._label).c_str(),
                               xs.data(),
                               ys.data(),
                               xs.size(),
                               ImPlotItemFlags_NoLegend);
            }
          }
        }
      }
    }
    if (show_measurement_prediction)
    {
      auto const mlock = _meas_model_data.rlock();
      for (auto const rlock = _cycle_data.rlock(); auto const& [label, positions] : rlock.data._tracks)
      {
        auto const [fitting, time_diff] = find_data(positions, now);

        if (fitting.has_value() and time_diff < 100ms)
        {
          auto const& [model_id, dist] = fitting.value().getEstimate();

          Components hull_comps({ COMPONENT::HULL_POS_X_MIN,
                                  COMPONENT::HULL_POS_Y_MIN,
                                  COMPONENT::HULL_POS_X_MAX,
                                  COMPONENT::HULL_POS_Y_MAX });
          for (auto const& [meas_model, data] : mlock.data)
          {
            if (meas_model.value_.find("early_fusion") == std::string::npos)
            {
              continue;
            }
            auto const& data_container = find_data(mlock.data.at(meas_model)._measurements, fitting.value()._time);
            if (not data_container.first.has_value())
            {
              continue;
            }
            std::optional<Vector<UDT>> const transformed =
                transformation::transform(dist.mean(),
                                          _manager->state_comps(model_id),
                                          hull_comps,
                                          data_container.first.value().second._sensorInfo._to_sensor_cs);
            if (not transformed.has_value())
            {
              continue;
            }
            std::vector xs{ to_val(transformed.value())(hull_comps.indexOf(COMPONENT::HULL_POS_X_MIN).value()),
                            to_val(transformed.value())(hull_comps.indexOf(COMPONENT::HULL_POS_X_MAX).value()) };
            std::vector ys{ to_val(transformed.value())(hull_comps.indexOf(COMPONENT::HULL_POS_Y_MIN).value()),
                            to_val(transformed.value())(hull_comps.indexOf(COMPONENT::HULL_POS_Y_MAX).value()) };
            ImPlot::PlotLine(std::format("{}", meas_model).c_str(), xs.data(), ys.data(), xs.size());
          }
        }
      }
    }

    if (show_detections)
    {
      std::vector<double> xs;
      std::vector<double> ys;
      for (auto const rlock = _meas_model_data.rlock(); auto const& [model_id, detections] : rlock.data)
      {
        for (auto const& [time, single_time_detections] : detections._measurements)
        {
          if (time < now and now - time < until)
          {
            std::vector<State<UDT>> meass = single_time_detections._measurements;
            for (State<UDT> const& state : single_time_detections._tracks | member<&BernoulliState<UDT>::state>)
            {
              meass.push_back(state);
            }
            for (State<UDT> const& meas : meass)
            {
              std::tie(xs, ys) = to_box(meas);
              ImPlot::PlotLine(model_id.value_.c_str(), xs.data(), ys.data(), xs.size());
              if (not xs.empty() and show_detection_details)
              {
                ImPlot::PlotText(meas.toString().c_str(), xs.front(), ys.front());
              }
              for (COMPONENT comp : { COMPONENT::AZIMUTH, COMPONENT::X_CC_LOWER_LEFT, COMPONENT::X_CC_UPPER_RIGHT })
              {
                std::tie(xs, ys) = to_beam(meas, comp, single_time_detections._sensorInfo);
                ImPlot::PlotLine(model_id.value_.c_str(), xs.data(), ys.data(), xs.size());
              }
              if (meas._misc.contains("HULL"))
              {
                auto const& polygon = std::any_cast<Polygon::SinglePolygon>(meas._misc.at("HULL"));
                std::tie(xs, ys) = to_xy(polygon);
                ImPlot::PlotLine(model_id.value_.c_str(), xs.data(), ys.data(), xs.size());
              }
            }
            auto plot_positions = meass | std::views::filter([](State<UDT> const& meas) {
                                    return not meas.isEmpty() and meas._manager->state_comps(meas.getEstimate().first)
                                                                      .indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y })
                                                                      .has_value();
                                  }) |
                                  std::views::transform([](State<UDT> const& meas) {
                                    Vector<UDT> pos_xy = meas.getEstimate().second.mean()(
                                        meas._manager->state_comps(meas.getEstimate().first)
                                            .indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y })
                                            .value());
                                    return std::pair(to_val(pos_xy(0)), to_val(pos_xy(1)));
                                  });
            auto [x, y] = to_xy_data(plot_positions);
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Cross);
            ImPlot::PlotScatter(model_id.value_.c_str(), x.data(), y.data(), x.size());
          }
        }
      }
    }
    if (show_birth_tracks)
    {
      for (auto const rlock = _meas_model_data.rlock(); auto const& [model_id, model_data] : rlock.data)
      {
        auto time_filtered = filter_data(model_data._birth_tracks, now, until);
        for (std::pair<Time, std::vector<BernoulliState<UDT>>> const& birth_tracks : time_filtered)
        {
          for (BernoulliState<UDT> const& state : birth_tracks.second)
          {
            auto [xs, ys] = to_box(state.state);
            if (xs.size() > 1)
            {
              ImPlot::PlotLine(model_id.value_.c_str(), xs.data(), ys.data(), xs.size());
            }
            else
            {
              ImPlot::PlotScatter(model_id.value_.c_str(), xs.data(), ys.data(), xs.size());
            }
            auto const [state_model_id, dist] = state.state.getEstimate();
            std::tie(xs, ys) = to_cov(dist, state.state._manager->state_comps(state_model_id));
            ImPlot::PlotLine(model_id.value_.c_str(), xs.data(), ys.data(), std::ssize(xs));
          }
        }
      }
    }
    if (show_fov)
    {
      plot_fov(now, show_fov_names);
    }
    ImPlot::EndPlot();
  }
  if (show_gt)
  {
    if (ImPlot::BeginPlot("Metrics", ImVec2(-1, -1)))
    {
      ImPlot::SetupAxes("Time", "Metric");
      ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Time);
      ImPlot::SetupFinish();
      auto [x, missed, false_track] = to_xyy(_cycle_data.wlock().data.gospa, now, until, [](auto const& gospa) {
        return std::tuple{ to_seconds(std::get<0>(gospa).time_since_epoch()),
                           std::get<1>(gospa).missed_cost,
                           std::get<1>(gospa).false_cost };
      });
      ImPlot::PlotLine("GOSPA MISSED", x.data(), missed.data(), x.size());
      ImPlot::PlotLine("GOSPA FALSE", x.data(), false_track.data(), x.size());
      auto [x2, loc, gospa] = to_xyy(_cycle_data.wlock().data.gospa, now, until, [](auto const& gospa) {
        return std::tuple{ to_seconds(std::get<0>(gospa).time_since_epoch()),
                           std::get<1>(gospa).loc_cost,
                           std::get<1>(gospa).gospa };
      });
      ImPlot::PlotLine("GOSPA LOC", x2.data(), loc.data(), x2.size());
      ImPlot::PlotLine("GOSPA", x2.data(), gospa.data(), x2.size());
      auto [x_1, ospa] = to_xy(_cycle_data.wlock().data.ospa, now, until, [](auto const& ospa) {
        return std::tuple{ to_seconds(std::get<0>(ospa).time_since_epoch()), std::get<1>(ospa).ospa };
      });
      ImPlot::PlotLine("OSPA", x_1.data(), ospa.data(), x_1.size());
      auto [x_2, ospa2] = to_xy(_cycle_data.wlock().data.ospa2, now, until, [](auto const& ospa2) {
        return std::tuple{ to_seconds(std::get<0>(ospa2).time_since_epoch()), std::get<1>(ospa2).ospa2 };
      });
      ImPlot::PlotLine("OSPA2", x_2.data(), ospa2.data(), x_2.size());
      ImPlot::EndPlot();
    }
  }
}

template <class UDT>
void Vizu<UDT>::plot_fov(Time now, bool show_names)
{
  for (auto const rlock = _meas_model_data.rlock(); auto const& [model_id, model_data] : rlock.data)
  {
    if (model_data._measurements.empty())
    {
      continue;
    }
    auto [best_fit, min_time_diff] = find_data(model_data._measurements, now);
    if (min_time_diff < 300ms and best_fit.has_value())
    {
      ttb::plot_fov(best_fit.value(), show_names);
    }
  }
}

template <class UDT>
void Vizu<UDT>::show_tracks_in_sensor_cs()
{
  auto const [auto_follow, now, until] = show_header(_manager);
  bool static show_tracks = true;
  ImGui::Checkbox("show tracks", &show_tracks);
  bool static show_labels = false;
  ImGui::Checkbox("show labels", &show_labels);
  bool static show_fov = true;
  ImGui::Checkbox("show fov", &show_fov);

  if (ImGui::BeginTabBar("Measurement Models"))
  {
    for (auto const rlock = _meas_model_data.rlock(); auto const& [model_id, model_data] : rlock.data)
    {
      if (ImGui::BeginTabItem(model_id.value_.c_str()))
      {
        if (ImPlot::BeginPlot("Tracks", ImVec2(-1, -1), ImPlotFlags_Equal))
        {
          ImPlot::SetupAxes("x", "y");
          ImPlot::SetupFinish();
          std::vector<double> x_detected;
          std::vector<double> y_detected;
          std::vector<double> x_missed;
          std::vector<double> y_missed;

          for (auto const& [time, data] : model_data._states_in_sensor_cs)
          {
            if (time < now and now - time < until)
            {
              for (State<UDT> const& state : data)
              {
                auto [state_id, dist] = state.getEstimate();

                if (auto const ind_xy = _manager->state_comps(state_id).indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y });

                    ind_xy.has_value())
                {
                  if (auto const [mean, it_missed_weight] =
                          std::pair{ dist.mean()(ind_xy.value()),
                                     std::ranges::find(state._meta_data._meas_update_weight.begin()
                                                           ->second,  // todo: this is not correct. here the
                                                                      // information from datasource container id to
                                                                      // source_id is missing
                                                       NOT_DETECTED,
                                                       &std::pair<MeasurementId, UDT>::first) };
                      it_missed_weight != state._meta_data._meas_update_weight.begin()->second.end() and
                      it_missed_weight->second > 0.5)
                  {
                    ImPlot::SetNextLineStyle(ImColor{ 1.0f, 0.0f, 0.0f });
                    auto [x, y] = to_box(state);
                    ImPlot::PlotLine("Missed", x.data(), y.data(), x.size());
                    x_missed.push_back(to_val(mean.x()));
                    y_missed.push_back(to_val(mean.y()));
                  }
                  else
                  {
                    ImPlot::SetNextLineStyle(ImColor{ 0.0f, 1.0f, 0.0f });
                    auto [x, y] = to_box(state);
                    ImPlot::PlotLine("Detected", x.data(), y.data(), x.size());
                    x_detected.push_back(to_val(mean.x()));
                    y_detected.push_back(to_val(mean.y()));
                  }
                }
              }
            }
          }
          ImPlot::SetNextLineStyle(ImColor{ 1.0f, 0.0f, 0.0f });
          ImPlot::PlotScatter("Missed", x_missed.data(), y_missed.data(), x_missed.size());
          ImPlot::SetNextLineStyle(ImColor{ 0.0f, 1.0f, 0.0f });
          ImPlot::PlotScatter("Detected", x_detected.data(), y_detected.data(), x_detected.size());

          if (show_fov)
          {
            if (auto [fov_data, time_diff] = find_data(model_data._measurements, now); fov_data.has_value())
            {
              if (fov_data.has_value())
              {
                if (auto const& fov = fov_data.value().second._sensorInfo._sensor_fov_in_sensor_cs; fov.has_value())
                {
                  auto [x, y] = to_xy(fov.value());
                  ImPlot::PlotLine(model_id.value_.c_str(), x.data(), y.data(), x.size());
                }
              }
            }
          }
          ImPlot::EndPlot();
        }
        ImGui::EndTabItem();
      }
    }
    ImGui::EndTabBar();
  }
}

template <class UDT>
void Vizu<UDT>::show_birth_info()
{
  auto [auto_follow, now, until] = show_header(_manager);
  if (auto_follow)
  {
    ImPlot::SetNextAxesToFit();
  }
  double const height = ImGui::GetWindowHeight() - ImGui::GetCursorPosY();
  if (ImPlot::BeginPlot("#BirthTracks", ImVec2((ImGui::GetWindowWidth() - ImGui::GetCursorPosX()), height / 2)))
  {
    ImPlot::SetupAxes("Time", "#Tracks");
    ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Time);
    ImPlot::SetupFinish();
    for (auto const rlock = _meas_model_data.rlock(); auto const& [model_id, model_data] : rlock.data)
    {
      auto [x, y] = to_xy(
          model_data._birth_tracks, now, until, [&](std::pair<Time, std::vector<BernoulliState<UDT>>> const& data) {
            return std::pair{ to_seconds(data.first.time_since_epoch()), data.second.size() };
          });
      ImPlot::PlotLine(model_id.value_.c_str(), x.data(), y.data(), x.size());
    }
    ImPlot::EndPlot();
  }
  if (ImPlot::BeginPlot("BirthTracks", ImVec2((ImGui::GetWindowWidth() - ImGui::GetCursorPosX()), -1)))
  {
    for (auto const rlock = _meas_model_data.rlock(); auto const& [model_id, model_data] : rlock.data)
    {
      auto time_filtered = filter_data(model_data._birth_tracks, now, until);
      for (std::pair<Time, std::vector<BernoulliState<UDT>>> const& birth_tracks : time_filtered)
      {
        for (BernoulliState<UDT> const& state : birth_tracks.second)
        {
          auto [xs, ys] = to_box(state.state);
          if (xs.size() > 1)
          {
            ImPlot::PlotLine(model_id.value_.c_str(), xs.data(), ys.data(), xs.size());
          }
          else
          {
            ImPlot::PlotScatter(model_id.value_.c_str(), xs.data(), ys.data(), xs.size());
          }
          auto const [state_model_id, dist] = state.state.getEstimate();
          std::tie(xs, ys) = to_cov(dist, state.state._manager->state_comps(state_model_id));
          ImPlot::PlotLine(model_id.value_.c_str(), xs.data(), ys.data(), std::ssize(xs));
        }
      }
    }
    ImPlot::EndPlot();
  }
}

template <class UDT>
void Vizu<UDT>::show_overview()
{
  auto [auto_follow, now, until] = show_header(_manager);
  if (auto_follow)
  {
    ImPlot::SetNextAxesToFit();
  }
  double height = ImGui::GetWindowHeight() - ImGui::GetCursorPosY();
  if (ImPlot::BeginPlot("#Tracks", ImVec2((ImGui::GetWindowWidth() - ImGui::GetCursorPosX()) / 2, height / 4)))
  {
    ImPlot::SetupAxes("Time", "#Tracks");
    ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Time);
    ImPlot::SetupFinish();
    auto const lock = _cycle_data.rlock();
    auto [x, y] = to_xy<double>(lock.data._num_tracks, now, until);
    ImPlot::PlotLine("#Tracks", x.data(), y.data(), x.size());
    ImPlot::EndPlot();
  }
  ImGui::SameLine();
  if (auto_follow)
  {
    ImPlot::SetNextAxesToFit();
  }
  if (ImPlot::BeginPlot("#Measurements", ImVec2(ImGui::GetWindowWidth() - ImGui::GetCursorPosX(), height / 4)))
  {
    ImPlot::SetupAxes("Time", "#Measurements");
    ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Time);
    ImPlot::SetupFinish();
    for (auto const lock = _meas_model_data.rlock(); auto const& [source, meas_data] : lock.data)
    {
      auto [x, y] = to_xy(meas_data._measurements, now, until, [&](auto const& data) {
        return std::pair(to_seconds(std::get<0>(data).time_since_epoch()),
                         data.second._measurements.size() + data.second._tracks.size());
      });
      ImPlot::PlotLine(source.value_.c_str(), x.data(), y.data(), x.size());
    }
    auto [x1, y1] = to_xy(_cycle_data.rlock().data._num_sources, now, until);
    ImPlot::PlotLine("#Data Sources", x1.data(), y1.data(), x1.size());
    ImPlot::EndPlot();
  }
  if (auto_follow)
  {
    ImPlot::SetNextAxesToFit();
  }
  if (ImPlot::BeginPlot("Computation Time", ImVec2(-1, height / 4)))
  {
    ImPlot::SetupAxes("Time", "Computation Time ms");
    ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Time);
    for (auto const rlock = _meas_model_data.rlock(); auto const& [model_id, sensor_data] : rlock.data)
    {
      auto [x1, y1] = to_xy(sensor_data._computation_time, now, until);
      ImPlot::PlotLine(model_id.value_.c_str(), x1.data(), y1.data(), x1.size());
    }
    auto [x, y] = to_xy(_cycle_data.rlock().data.computation_time, now, until);
    ImPlot::PlotLine("Cycle Time", x.data(), y.data(), x.size());
    ImPlot::EndPlot();
  }
  ImGui::Text("#Discarded Data %zu", [&] {
    std::size_t tot_discarded = 0;
    for (auto const rlock = _meas_model_data.rlock(); auto const& [model, model_data] : rlock.data)
    {
      tot_discarded += model_data._discarded_data.size();
    };
    return tot_discarded;
  }());
  if (auto_follow)
  {
    ImPlot::SetNextAxesToFit();
  }
  double width = ImGui::GetWindowWidth() - ImGui::GetCursorPosX();
  if (ImPlot::BeginPlot("Buffer", ImVec2(width / 2, height / 4)))
  {
    ImPlot::SetupAxes("Time", "#Sources");
    ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Time);
    auto [x, y] = to_xy(_cycle_data.rlock().data._num_sources, now, until);
    ImPlot::PlotLine("#Sources", x.data(), y.data(), x.size());
    auto [x1, y1] = to_xy(_cycle_data.rlock().data._buffer_size, now, until);
    ImPlot::PlotLine("Buffer Size", x1.data(), y1.data(), x1.size());
    ImPlot::EndPlot();
  }
  ImGui::SameLine();
  if (auto_follow)
  {
    ImPlot::SetNextAxesToFit();
  }
  if (ImPlot::BeginPlot("Sources Meta Info", ImVec2(ImGui::GetWindowWidth() - ImGui::GetCursorPosX(), height / 4)))
  {
    ImPlot::SetupAxes("Time", "Duration ms");
    ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Time);
    for (auto const rlock = _meas_model_data.rlock(); auto const& [model_id, sensor_data] : rlock.data)
    {
      auto [x, y] = to_xy(sensor_data._delay, now, until);
      ImPlot::PlotLine(std::string(model_id.value_ + " Delay").c_str(), x.data(), y.data(), x.size());
    }
    auto [x3, y3] = to_xy(_cycle_data.rlock().data._filter_delay_first, now, until);
    ImPlot::PlotLine("Filter Delay First", x3.data(), y3.data(), x3.size());
    auto [x4, y4] = to_xy(_cycle_data.rlock().data._filter_delay_last, now, until);
    ImPlot::PlotLine("Filter Delay Last", x4.data(), y4.data(), x4.size());
    ImPlot::EndPlot();
  }
  if (auto_follow)
  {
    ImPlot::SetNextAxesToFit();
  }
  if (ImPlot::BeginPlot("Data Usage", ImVec2(-1, -1)))
  {
    ImPlot::SetupAxes("Time", "Data Source");
    ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Time);
    double model_height = 0;
    for (auto const rlock = _meas_model_data.rlock(); auto const& [model_id, sensor_data] : rlock.data)
    {
      auto [x, y] = to_xy(sensor_data._used_in_cycle, now, until, [model_height](auto const& data) {
        return std::pair(to_seconds(get_data_time(data).time_since_epoch()), model_height);
      });
      ImPlot::PlotScatter(model_id.value_.c_str(), x.data(), y.data(), x.size());
      auto [x1, y1] = to_xy(sensor_data._discarded_data, now, until, [model_height](auto const& data) {
        return std::pair(to_seconds(get_data_time(data).time_since_epoch()), model_height);
      });
      ImPlot::SetNextMarkerStyle(ImPlotMarker_Cross, 5);
      ImPlot::PlotScatter(model_id.value_.c_str(), x1.data(), y1.data(), x1.size());
      model_height += 1;
    }
    for (Time time : _cycle_data.rlock().data._trigger_time)
    {
      if (now - time < until and time < now)
      {
        std::array x = { to_seconds(time.time_since_epoch()), to_seconds(time.time_since_epoch()) };
        std::array y = { -0.5, model_height - 0.5 };
        ImPlot::PlotLine("##1", x.data(), y.data(), x.size(), ImPlotItemFlags_NoLegend);
      }
    }
    ImPlot::EndPlot();
  }
}

template <class UDT>
void Vizu<UDT>::reset()
{
  _cycle_data.wlock().data = {};
  _meas_model_data.wlock().data.clear();
}

template <class UDT>
void Vizu<UDT>::show_debug()
{
  auto const [auto_follow, now, until] = show_header(_manager);
  double height = ImGui::GetWindowHeight() - ImGui::GetCursorPosY();
  if (ImGui::BeginTabBar("Sensors"))
  {
    for (auto const rlock = _meas_model_data.rlock(); auto const& [model_id, sensor_data] : rlock.data)
    {
      if (ImGui::BeginTabItem(model_id.value_.c_str()))
      {
        height = ImGui::GetWindowHeight() - ImGui::GetCursorPosY();
        if (auto_follow)
        {
          ImPlot::SetNextAxesToFit();
        }
        if (ImPlot::BeginPlot("Delay", ImVec2(-1, height / 3)))
        {
          ImPlot::SetupAxes("Time", "Delay ms");
          ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Time);
          auto [x, y] = to_xy(sensor_data._delay, now, until);
          ImPlot::PlotLine("Data Delay", x.data(), y.data(), x.size());
          auto [x2, y2] = to_xy(sensor_data._estimated_delay, now, until);
          ImPlot::PlotLine("Estimated Data Delay", x2.data(), y2.data(), x2.size());
          auto [x1, y1] = to_xy(_cycle_data.rlock().data._buffer_delay, now, until);
          ImPlot::PlotLine("Buffer Delay", x1.data(), y1.data(), x1.size());
          auto [x3, y3] = to_xy(_cycle_data.rlock().data._filter_delay_first, now, until);
          ImPlot::PlotLine("Filter Delay First", x3.data(), y3.data(), x3.size());
          auto [x5, y5] = to_xy(_cycle_data.rlock().data._filter_delay_last, now, until);
          ImPlot::PlotLine("Filter Delay Last", x5.data(), y5.data(), x5.size());
          auto [x4, y4] = to_xy(sensor_data._duration_in_buffer, now, until);
          ImPlot::PlotLine("Duration in Buffer", x4.data(), y4.data(), x4.size());
          ImPlot::EndPlot();
        }
        if (auto_follow)
        {
          ImPlot::SetNextAxesToFit();
        }
        if (ImPlot::BeginPlot("#Measurements", ImVec2(-1, height / 3)))
        {
          ImPlot::SetupAxes("Time", "#Measurements");
          ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Time);
          auto [x, y] = to_xy(sensor_data._measurements, now, until, [&](auto const& data) {
            return std::pair(to_seconds(std::get<0>(data).time_since_epoch()),
                             data.second._measurements.size() + data.second._tracks.size());
          });
          ImPlot::PlotLine("#Measurements", x.data(), y.data(), x.size());
          auto [x1, y1] = to_xy(sensor_data._num_assoc_measurements, now, until);
          ImPlot::PlotLine("#Associated Measurements", x1.data(), y1.data(), x1.size());
          ImPlot::EndPlot();
        }
        std::size_t num_hypotheses{ 0 };
        Duration min_time_diff(1s);
        for (auto const& [time, num_hyps] : sensor_data._num_hypotheses)
        {
          if (Duration const diff = std::chrono::abs(time - now); diff < min_time_diff)
          {
            min_time_diff = diff;
            num_hypotheses = num_hyps;
          }
        }
        ImGui::Text("Number of hypotheses: %ld", num_hypotheses);
        if (ImPlot::BeginPlot("Cardinality Distribution", ImVec2(ImGui::GetWindowWidth() / 2, -1)))
        {
          if (auto const [cardinality_dist, _] = find_data(sensor_data._cardinality_distribution, now);
              cardinality_dist.has_value())
          {
            ImPlot::PlotBars("Cardinality Distribution",
                             cardinality_dist.value().second.data(),
                             cardinality_dist.value().second.rows());
          }
          ImPlot::EndPlot();
        }
        ImGui::SameLine();
        if (ImPlot::BeginPlot("#Hypotheses per Cardinality", ImVec2(-1, -1)))
        {
          if (auto const [cardinalities, _1] = find_data(sensor_data._num_hypotheses_per_cardinality, now);
              cardinalities.has_value())
          {
            ImPlot::PlotBars("#Hypotheses per Cardinality",
                             cardinalities.value().second.data(),
                             cardinalities.value().second.rows());
          }
          ImPlot::EndPlot();
        }

        ImGui::EndTabItem();
      }
    }
    if (ImGui::BeginTabItem("Tracks"))
    {
      Label const track_label = select_track_label(_cycle_data.rlock().data._tracks);
      if (ImPlot::BeginPlot("Track", ImVec2(-1, height / 2), ImPlotFlags_Equal))
      {
        ImPlot::SetupAxes("x", "y");
        ImPlot::SetupFinish();
        if (track_label != Label{ 0 })
        {
          auto const [xs, ys] =
              to_xy(_cycle_data.rlock().data._tracks.at(track_label), now, until, [this](State<UDT> const& track) {
                auto const& [model_id, dist] = track.getEstimate();
                Vector<UDT> pos_xy = dist.mean()(
                    _manager->state_comps(model_id).indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y }).value());
                return std::pair(to_val(pos_xy(0)), to_val(pos_xy(1)));
              });
          ImPlot::PlotLine(
              std::to_string(track_label.value_).c_str(), xs.data(), ys.data(), xs.size(), ImPlotItemFlags_NoLegend);
        }
        ImPlot::EndPlot();
      }
      ImGui::SameLine();
      if (ImPlot::BeginPlot("Track Assessment", ImVec2(-1, -1), ImPlotFlags_Equal))
      {
        ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Time);
        if (track_label != Label{ 0 })
        {
          auto [state, time_diff] = find_data(_cycle_data.rlock().data._tracks.at(track_label), now);
          if (state.has_value())
          {
            Triangle tri{};
            plot_sl_triangle(tri);
            for (auto const& [weight, op] : state.value()._meta_data._input_opinions_debug_info)
            {
              plot_opinion(op, "Input Data", tri, std::to_string(weight));
            }
            plot_opinion(state.value()._meta_data._assessment, "Track Assessment", tri);
          }
        }
        ImPlot::EndPlot();
      }
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Search Track ID"))
    {
      static int state_id;
      ImGui::InputInt("StateId", &state_id);
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Detection Modell"))
    {
      auto rlock = _meas_model_data.rlock();
      for (auto const& [source, data] : *rlock)
      {
        if (data._measurements.empty())
        {
          continue;
        }
        double minx = -500;
        double maxx = 500;
        double miny = -500;
        double maxy = 500;
        double step = 1;
        Eigen::Matrix<float, -1, -1, Eigen::RowMajor> static values(static_cast<Index>((maxy - miny) / step) + 1,
                                                                    static_cast<Index>((maxx - minx) / step) + 1);
        if (_manager->_num_cycles % 10 == 0)
        {
          for (Index i = 0; i < values.rows(); ++i)
          {
            double const y = maxy - i * (maxy - miny) / (values.rows() - 1);
            for (Index j = 0; j < values.cols(); ++j)
            {
              double const x = minx + j * (maxx - minx) / (values.cols() - 1);
              values(i, j) =
                  to_val(std::get<GaussianMeasurementModel<UDT>>(_manager->getMeasModel(source)._meas_model)
                             .getDetectionProbability(
                                 Vector<UDT>{ { x, y } }, Components({ COMPONENT::POS_X, COMPONENT::POS_Y }), {}));
            }
          }
        }
        ImPlot::PushColormap(ImPlotColormap_Plasma);
        ImPlotStyle& style = ImPlot::GetStyle();

        style.PlotPadding = ImVec2(20, 20);  // Space between axes and plot
        style.LabelPadding = ImVec2(5, 5);   // Space between axis and labels
        style.LegendPadding = ImVec2(5, 5);
        style.FitPadding = ImVec2(0.05f, 0.05f);
        if (ImPlot::BeginPlot("##Heatmap1",
                              ImVec2(ImGui::GetWindowWidth() * 8.0 / 9, ImGui::GetWindowHeight() * 4.0 / 5),
                              ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText))
        {
          ImPlot::SetupAxes("x", "y");
          ImPlot::SetupAxesLimits(minx, maxx, miny, maxy, ImPlotCond_Always);
          ImPlot::PlotHeatmap("heat",
                              values.data(),
                              values.rows(),
                              values.cols(),
                              0.0,
                              1.0,
                              nullptr,
                              ImPlotPoint(minx, miny),
                              ImPlotPoint(maxx, maxy));
          ImPlot::EndPlot();
        }
        ImGui::SameLine();
        ImPlot::ColormapScale("##HeatScale", 0.0, 1.0, ImVec2(-1, ImGui::GetWindowHeight() * 4.0 / 5));
        ImPlot::PopColormap();
      }
      ImGui::EndTabItem();
    }
    ImGui::EndTabBar();
  }
}

template <class UDT>
void Vizu<UDT>::show_assessment_overview()
{
  auto const [auto_follow, now, until] = show_header(_manager);
  ImGui::Text("Statistical Models");
  auto const lock = _meas_model_data.rlock();
  float const num_meas_models = lock.data.size();
  float const width = (ImGui::GetWindowWidth() - ImGui::GetCursorPosX()) / num_meas_models;
  float const height = ImGui::GetWindowHeight() - ImGui::GetCursorPosY();
  float constexpr num_rows = 5;
  ImGui::Text("-- Detection Model");
  std::size_t imguiid = 0;
  for (auto const& [source_id, sensor_data] : lock.data)
  {
    ImGui::PushID(imguiid++);
    auto [data, time_diff] = find_data(sensor_data._assessment, now);
    if (data.has_value())
    {
      if (ImPlot::BeginPlot("Detection Model", ImVec2(width, height / num_rows), ImPlotFlags_Equal))
      {
        Triangle tri{};
        plot_sl_triangle(tri);
        plot_opinion(data.value().second._detection, std::format("Detection Opinion {}", source_id), tri);
      }
      ImPlot::EndPlot();
    }
    ImGui::PopID();
    ImGui::SameLine();
  }
  ImGui::NewLine();
  ImGui::Text("-- Clutter Model");
  for (auto const& [source_id, sensor_data] : lock.data)
  {
    ImGui::PushID(imguiid++);
    auto [data, time_diff] = find_data(sensor_data._assessment, now);
    if (data.has_value())
    {
      if (ImPlot::BeginPlot("Clutter Model", ImVec2(width, height / num_rows), ImPlotFlags_Equal))
      {
        Triangle tri{};
        plot_sl_triangle(tri);
        plot_opinion(data.value().second._clutter, std::format("Clutter Opinion {}", source_id), tri);
      }
      ImPlot::EndPlot();
    }
    ImGui::PopID();
    ImGui::SameLine();
  }
  ImGui::NewLine();
  ImGui::Text("-- Single-Object Process and Measurement Model");
  for (auto const& [source_id, sensor_data] : lock.data)
  {
    auto [data, time_diff] = find_data(sensor_data._assessment, now);
    if (data.has_value())
    {
      ImGui::PushID(imguiid++);
      double const nis_width = width / data.value().second._nis.size();
      for (auto const& [comp, opinion] : data.value().second._nis)
      {
        ImGui::PushID(imguiid++);
        if (ImPlot::BeginPlot("NIS", ImVec2(nis_width, height / num_rows), ImPlotFlags_Equal))
        {
          Triangle tri{};
          plot_sl_triangle(tri);
          plot_opinion(opinion, std::format("Comp {}", comp), tri);
          ImPlot::EndPlot();
        }
        ImGui::PopID();
        ImGui::SameLine();
      }
      ImGui::PopID();
      ImGui::SameLine();
    }
  }
  ImGui::NewLine();
  ImGui::Text("Non-Modeled Knowledge");

  ImGui::Text("Implementation");
  auto const cycle_lock = _cycle_data.rlock();
  double const impl_width =
      (ImGui::GetWindowWidth() - ImGui::GetCursorPosX()) / cycle_lock.data.implementation_op.size();
  for (auto const& [type, opinions] : cycle_lock.data.implementation_op)
  {
    if (ImPlot::BeginPlot(type.c_str(), ImVec2(impl_width, height / num_rows), ImPlotFlags_Equal))
    {
      Triangle tri{};
      plot_sl_triangle(tri);
      auto const [opinion, timediff] = find_data(opinions, now);
      plot_opinion(opinion.value_or(std::pair{ Time::max(), Opinion::VacuousBeliefOpinion() }).second,
                   std::format("{}", type),
                   tri);
      ImPlot::EndPlot();
    }
    ImGui::SameLine();
  }
  ImGui::NewLine();
}

template <class UDT>
void Vizu<UDT>::show_assessment()
{
  auto const [auto_follow, now, until] = show_header(_manager);
  double static quantile = 0.9;
  ImGui::InputDouble("quantile", &quantile, 0.01, 0.1);
  quantile = std::clamp(quantile, 0.0001, 0.9999);
  float constexpr num_rows = 4;
  if (ImGui::BeginTabBar("Tab Bar"))
  {
    for (auto const rlock = _meas_model_data.rlock(); auto const& [model_id, sensor_data] : rlock.data)
    {
      if (ImGui::BeginTabItem(model_id.value_.c_str()))
      {
        if (auto_follow)
        {
          ImPlot::SetNextAxesToFit();
        }
        auto const height = ImGui::GetWindowHeight() - ImGui::GetCursorPosY();
        auto const width = ImGui::GetWindowWidth() - ImGui::GetCursorPosX();
        if (ImPlot::BeginPlot("Clutter Rate Estimation ", ImVec2(width / 3, height / num_rows)))
        {
          ImPlot::SetupAxes("Time", "Clutter Rate");
          ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Time);
          ImPlot::SetupFinish();
          ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.25f);
          auto [x, y1, y2, y3] = [&] {
            auto clutter_plot =
                sensor_data._clutter_estimation_bayes_approx | std::views::filter([&](auto const& data) {
                  return now - get_data_time(data) < until and get_data_time(data) < now;
                }) |
                std::views::transform([&](auto const& data) {
                  double const mean = std::get<1>(data);
                  double const lower = mean - std::get<2>(data);
                  double const upper = std::get<3>(data) - mean;
                  return std::tuple{ to_seconds(std::get<0>(data).time_since_epoch()), lower, upper, mean };
                });
            return to_xyyy_data(clutter_plot);
          }();

          ImPlot::PlotErrorBars("Clutter Estimation", x.data(), y3.data(), y1.data(), y2.data(), x.size());
          ImPlot::PlotLine("Clutter Estimation", x.data(), y3.data(), x.size());
          std::tie(x, y1, y2, y3) = [&] {
            auto clutter_plot =
                sensor_data._approx_clutter_estimation | std::views::filter([&](auto const& data) {
                  return now - get_data_time(data) < until and get_data_time(data) < now;
                }) |
                std::views::transform([&](auto const& data) {
                  double const mean = data.second.mean();
                  boost::math::gamma_distribution gamma(data.second._alpha, 1 / data.second._beta);
                  double const lower = mean - [&] {
                    try
                    {
                      return boost::math::quantile(gamma, (1 - quantile) / 2);
                    }
                    catch (...)
                    {
                      return std::numeric_limits<double>::quiet_NaN();
                    }
                  }();
                  double const upper = [&] {
                    try
                    {
                      return boost::math::quantile(gamma, quantile + (1 - quantile) / 2);
                    }
                    catch (...)
                    {
                      return std::numeric_limits<double>::quiet_NaN();
                    }
                  }() - mean;
                  return std::tuple{ to_seconds(std::get<0>(data).time_since_epoch()), lower, upper, mean };
                });
            return to_xyyy_data(clutter_plot);
          }();
          ImPlot::PlotErrorBars("Approx Clutter Estimation", x.data(), y3.data(), y1.data(), y2.data(), x.size());
          ImPlot::PlotLine("Approx Clutter Estimation", x.data(), y3.data(), x.size());
          std::tie(x, y1) = to_xy(sensor_data._clutter_rate, now, until);
          ImPlot::PlotLine("Clutter Rate Filter", x.data(), y1.data(), x.size());
          std::tie(x, y1) = to_xy(sensor_data._measurements,
                                  now,
                                  until,
                                  [&](std::pair<Time, DataSourceContainer<UDT>> const& time_container) {
                                    return std::pair{ to_seconds(time_container.first.time_since_epoch()),
                                                      time_container.second._ground_truth_info
                                                          .transform([&](GroundTruthScenarioInfo const& info) {
                                                            return info.meas_model_info.clutter_rate;
                                                          })
                                                          .value_or(std::numeric_limits<double>::quiet_NaN()) };
                                  });
          ImPlot::PlotLine("Detection Probability GT", x.data(), y1.data(), x.size());
          ImPlot::EndPlot();
        }
        if (auto_follow)
        {
          ImPlot::SetNextAxesToFit();
        }
        ImGui::SameLine();
        if (ImPlot::BeginPlot((std::string("Clutter Rate Bayes Factor ") + model_id.value_).c_str(),
                              ImVec2(width / 3, height / num_rows)))
        {
          ImPlot::SetupAxes("Time", "Bayes Factor");
          ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Time);
          ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
          ImPlot::SetupFinish();

          auto [x, y] = to_xy(sensor_data._clutter_bayes_factor, now, until);
          ImPlot::PlotLine("Bayes Factor", x.data(), y.data(), x.size());
          auto [x1, y1] = to_xy(sensor_data._clutter_cum_bayes_factor, now, until);
          ImPlot::PlotLine("Cum Bayes Factor", x1.data(), y1.data(), x1.size());
          if (not x.empty())
          {
            std::array xs = { std::ranges::min(x), std::ranges::max(x) };
            std::array ys = { 1.0, 1.0 };
            ImPlot::PlotLine("good/bad", xs.data(), ys.data(), 2);
            ys = { std::pow(10.0, -0.5), std::pow(10.0, -0.5) };
            ImPlot::PlotLine("substantial", xs.data(), ys.data(), 2);
            ys = { std::pow(10.0, -1), std::pow(10.0, -1) };
            ImPlot::PlotLine("strong", xs.data(), ys.data(), 2);
            ys = { std::pow(10.0, -1.5), std::pow(10.0, -1.5) };
            ImPlot::PlotLine(" very strong", xs.data(), ys.data(), 2);
            ys = { std::pow(10.0, -2), std::pow(10.0, -2) };
            ImPlot::PlotLine("decisive", xs.data(), ys.data(), 2);
          }
          ImPlot::EndPlot();
        }
        ImGui::SameLine();
        if (ImPlot::BeginPlot(
                "Clutter Opinion", ImVec2(-1, height / num_rows), ImPlotFlags_Equal))  // | ImPlotFlags_NoInputs))
        {
          Triangle tri{};
          plot_sl_triangle(tri);
          auto [data, time_diff] = find_data(sensor_data._assessment, now);
          if (data.has_value())
          {
            plot_opinion(data.value().second._clutter_bf, "Clutter BF Opinion", tri);
            plot_opinion(data.value().second._clutter_estimation, "Clutter Estimation Opinion", tri);
            plot_opinion(data.value().second._clutter, "Clutter Opinion", tri);
          }
          ImPlot::EndPlot();
        }
        if (auto_follow)
        {
          ImPlot::SetNextAxesToFit();
        }
        if (ImPlot::BeginPlot("Detection Probability Estimation ", ImVec2(width / 3, height / num_rows)))
        {
          ImPlot::SetupAxes("Time", "Detection Probability");
          ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Time);
          ImPlot::SetupFinish();
          auto [x, y1, y2, y3] = [&] {
            auto clutter_plot =
                sensor_data._detection_estimation_bayes_approx | std::views::filter([&](auto const& data) {
                  return now - get_data_time(data) < until and get_data_time(data) < now;
                }) |
                std::views::transform([&](auto const& data) {
                  double const mean = std::get<1>(data);
                  double const lower = mean - std::get<2>(data);
                  double const upper = std::get<3>(data) - mean;
                  return std::tuple{ to_seconds(std::get<0>(data).time_since_epoch()), lower, upper, mean };
                });
            return to_xyyy_data(clutter_plot);
          }();

          ImPlot::PlotErrorBars("Detection Estimation", x.data(), y3.data(), y1.data(), y2.data(), x.size());
          ImPlot::PlotLine("Detection Estimation", x.data(), y3.data(), x.size());
          std::tie(x, y1, y2, y3) = [&] {
            auto detection_plot =
                sensor_data._approx_detection_estimation | std::views::filter([&](auto const& data) {
                  return now - get_data_time(data) < until and get_data_time(data) < now;
                }) |
                std::views::transform([&](auto const& data) {
                  boost::math::beta_distribution beta(data.second._alpha(0), data.second._alpha(1));
                  double const mean = data.second.mean()(0);
                  double const lower = mean - [&] {
                    try
                    {
                      return boost::math::quantile(beta, (1 - quantile) / 2);
                    }
                    catch (...)
                    {
                      return std::numeric_limits<double>::quiet_NaN();
                    }
                  }();
                  double const upper = [&] {
                    try
                    {
                      return boost::math::quantile(beta, quantile + (1 - quantile) / 2);
                    }
                    catch (...)
                    {
                      return std::numeric_limits<double>::quiet_NaN();
                    }
                  }() - mean;
                  return std::tuple{ to_seconds(std::get<0>(data).time_since_epoch()), lower, upper, mean };
                });
            return to_xyyy_data(detection_plot);
          }();
          ImPlot::PlotErrorBars("Approx Detection Estimation", x.data(), y3.data(), y1.data(), y2.data(), x.size());
          ImPlot::PlotLine("Approx Detection Probability", x.data(), y3.data(), x.size());
          std::tie(x, y1) = to_xy(sensor_data._detection_probability, now, until);
          ImPlot::PlotLine("Detection Probability Filter", x.data(), y1.data(), x.size());
          std::tie(x, y1) = to_xy(sensor_data._measurements,
                                  now,
                                  until,
                                  [&](std::pair<Time, DataSourceContainer<UDT>> const& time_container) {
                                    assert(time_container.first == time_container.second._time);
                                    return std::pair{ to_seconds(time_container.first.time_since_epoch()),
                                                      time_container.second._ground_truth_info
                                                          .transform([&](GroundTruthScenarioInfo const& info) {
                                                            return info.meas_model_info.detection_prob;
                                                          })
                                                          .value_or(std::numeric_limits<double>::quiet_NaN()) };
                                  });
          ImPlot::PlotLine("Detection Probability GT", x.data(), y1.data(), x.size());
          ImPlot::EndPlot();
        }
        if (auto_follow)
        {
          ImPlot::SetNextAxesToFit();
        }
        ImGui::SameLine();
        if (ImPlot::BeginPlot((std::string("Detection Prob Bayes Factor ") + model_id.value_).c_str(),
                              ImVec2(width / 3, height / num_rows)))
        {
          ImPlot::SetupAxes("Time", "Bayes Factor");
          ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Time);
          ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
          ImPlot::SetupFinish();
          auto [x, y] = to_xy(sensor_data._detection_bayes_factor, now, until);
          ImPlot::PlotLine("Bayes Factor", x.data(), y.data(), x.size());
          auto [x1, y1] = to_xy(sensor_data._detection_cum_bayes_factor, now, until);
          ImPlot::PlotLine("Cum Bayes Factor", x1.data(), y1.data(), x1.size());
          if (not x.empty())
          {
            std::array xs = { std::ranges::min(x), std::ranges::max(x) };
            std::array ys = { 1.0, 1.0 };
            ImPlot::PlotLine("good/bad", xs.data(), ys.data(), 2);
            ys = { std::pow(10.0, -0.5), std::pow(10.0, -0.5) };
            ImPlot::PlotLine("substantial", xs.data(), ys.data(), 2);
            ys = { std::pow(10.0, -1), std::pow(10.0, -1) };
            ImPlot::PlotLine("strong", xs.data(), ys.data(), 2);
            ys = { std::pow(10.0, -1.5), std::pow(10.0, -1.5) };
            ImPlot::PlotLine(" very strong", xs.data(), ys.data(), 2);
            ys = { std::pow(10.0, -2), std::pow(10.0, -2) };
            ImPlot::PlotLine("decisive", xs.data(), ys.data(), 2);
          }
          ImPlot::EndPlot();
        }
        ImGui::SameLine();
        if (ImPlot::BeginPlot(
                "Detection Opinion", ImVec2(width / 6, height / num_rows), ImPlotFlags_Equal | ImPlotFlags_NoInputs))
        {
          Triangle tri{};
          plot_sl_triangle(tri);
          auto [data, time_diff] = find_data(sensor_data._assessment, now);
          if (data.has_value())
          {
            plot_opinion(data.value().second._detection_bf, "Detection BF Opinion", tri);
            plot_opinion(data.value().second._detection_estimation, "Detection Estimation Opinion", tri);
            plot_opinion(data.value().second._detection, "Detection Opinion", tri);
          }
          ImPlot::EndPlot();
        }
        if (auto_follow)
        {
          ImPlot::SetNextAxesToFit();
        }
        if (ImPlot::BeginPlot((std::string("Nis ") + model_id.value_).c_str(), ImVec2(-1, height / num_rows)))
        {
          ImPlot::SetupAxes("Time", "Nis");
          ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Time);
          ImPlot::SetupFinish();
          for (auto const& [comp, comp_wise_nis] : sensor_data._nis_dof)
          {
            auto [x, y1, y2, y3] = [&] {
              auto nis_plot =
                  comp_wise_nis | std::views::filter([&](auto const& data) {
                    return now - get_data_time(data) < until and get_data_time(data) < now;
                  }) |
                  std::views::transform([&](auto const& data) {
                    double const val = std::get<1>(data);
                    boost::math::chi_squared_distribution chi2(std::get<2>(data));
                    double const lower = val - [&] {
                      try
                      {
                        return boost::math::quantile(chi2, (1 - quantile) / 2);
                      }
                      catch (...)
                      {
                        return std::numeric_limits<double>::quiet_NaN();
                      }
                    }();
                    double const upper = [&] {
                      try
                      {
                        return boost::math::quantile(chi2, quantile + (1 - quantile) / 2);
                      }
                      catch (...)
                      {
                        return std::numeric_limits<double>::quiet_NaN();
                      }
                    }() - val;

                    return std::tuple{ to_seconds(std::get<0>(data).time_since_epoch()), lower, upper, val };
                  });
              return to_xyyy_data(nis_plot);
            }();
            ImPlot::PlotErrorBars(to_string(comp).c_str(), x.data(), y3.data(), y1.data(), y2.data(), x.size());
            ImPlot::PlotLine(to_string(comp).c_str(), x.data(), y3.data(), x.size());
          }
          ImPlot::EndPlot();
        }

        ImGui::EndTabItem();
      }
    }

    ImGui::EndTabBar();
  }
  if (auto_follow)
  {
    ImPlot::SetNextAxesToFit();
  }
  if (ImPlot::BeginPlot("Persistence Model", ImVec2(-1, -1)))
  {
    ImPlot::SetupAxes("Time", "Persistence Model");
    ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Time);
    ImPlot::SetupFinish();
    auto const cycle_lock = _cycle_data.rlock();
    auto [x, y1, y2, y3] = [&] {
      auto nis_plot = cycle_lock.data.persistence_prob_estimation | std::views::filter([&](auto const& data) {
                        return now - get_data_time(data) < until and get_data_time(data) < now;
                      }) |
                      std::views::transform([&](auto const& data) {
                        double const val = std::get<1>(data);
                        double const lower = val - std::get<2>(data);
                        double const upper = std::get<3>(data) - val;
                        return std::tuple{ to_seconds(std::get<0>(data).time_since_epoch()), lower, upper, val };
                      });
      return to_xyyy_data(nis_plot);
    }();
    ImPlot::PlotErrorBars("Persistence Prob Quantile", x.data(), y3.data(), y1.data(), y2.data(), x.size());
    ImPlot::PlotLine("Persistence Prob Mean", x.data(), y3.data(), x.size());
    ImPlot::EndPlot();
  }
}

template <class UDT>
void Vizu<UDT>::show()
{
  ZoneScopedN("Vizu::show");
  if (ImGui::Begin("Main Window"))
  {
    ImGui::PushItemWidth(200);
    ImGui::Text("%s", _manager->params()->name.c_str());
    if (ImGui::BeginTabBar("Tab Bar"))
    {
      if (ImGui::BeginTabItem("Overview"))
      {
        show_overview();
        ImGui::EndTabItem();
      }
      if (ImGui::BeginTabItem("Tracks and Detections"))
      {
        show_tracks();
        ImGui::EndTabItem();
      }
      if (ImGui::BeginTabItem("Grouping"))
      {
        show_grouping();
        ImGui::EndTabItem();
      }
      if (ImGui::BeginTabItem("Tracks in Sensor Coords"))
      {
        show_tracks_in_sensor_cs();
        ImGui::EndTabItem();
      }
      if (ImGui::BeginTabItem("Birth Info"))
      {
        show_birth_info();
        ImGui::EndTabItem();
      }
      if (ImGui::BeginTabItem("Self Assessment Overview"))
      {
        show_assessment_overview();
        ImGui::EndTabItem();
      }
      if (_manager->params()->filter.enable)
      {
        if (ImGui::BeginTabItem("Self Assessment"))
        {
          show_assessment();
          ImGui::EndTabItem();
        }
      }
      if (ImGui::BeginTabItem("Debug"))
      {
        show_debug();
        ImGui::EndTabItem();
      }
      ImGui::EndTabBar();
    }
    ImGui::PopItemWidth();
  }
#if __has_include("minimal_latency_buffer/mlb_visu.hpp")
  if (_manager->_data_buffer_visualizer.has_value())
  {
    _manager->_data_buffer_visualizer.value().render_content();
  }
#endif

  ImGui::End();
}

template <class UDT>
int Vizu<UDT>::viz_loop()
{
  glfwSetErrorCallback([](int error, char const* description) {
    throw std::runtime_error("glfw Error Code: " + std::to_string(error) + description);
  });
  if (glfwInit() != GLFW_TRUE)
  {
    return 1;
  }
  // GL 3.0 + GLSL 130
  const char* glsl_version = "#version 130";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);
  // Create window with graphics context
  std::string name = _manager->params()->name;
  GLFWwindow* window = glfwCreateWindow(2000, 1500, name.c_str(), nullptr, nullptr);
  if (window == nullptr)
  {
    return 1;
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);  // Enable vsync
  // Setup Dear ImGui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImPlot::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;   // Enable Gamepad Controls
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;      // IF using Docking Branch
  // Setup Dear ImGui style
  ImGui::StyleColorsDark();
  ImPlot::GetStyle().PlotPadding = { 0, 0 };
  ImPlot::GetStyle().LabelPadding = { 0, 0 };
  // Setup Platform/Renderer backends
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init(glsl_version);
  // available are Cousine-Regular.ttf Karla-Regular.ttf ProggyTiny.ttf DroidSans.ttf ProggyClean.ttf
  // Roboto-Medium.ttf
  const std::string font_name = "DroidSans.ttf";
  const std::string font_path = "/usr/share/fonts/imgui/" + font_name;
  io.Fonts->AddFontFromFileTTF(font_path.c_str(), 12);
  ImVec4 clear_color = ImVec4(0.45F, 0.55F, 0.60F, 1.00F);

  while (not glfwWindowShouldClose(window) and _manager->params()->show_gui)
  {
    // Poll and handle events (inputs, window resize, etc.)
    glfwPollEvents();

    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(viewport->Size);
    ImGui::SetNextWindowViewport(viewport->ID);
    ImGui::SetNextWindowBgAlpha(0.0F);

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0F);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0F);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
    std::string window_name = "Root window";
    ImGui::Begin(window_name.c_str(), nullptr, ImGuiWindowFlags_NoTitleBar);
    ImGui::PopStyleVar(3);

    ImGuiID dockspace_id = ImGui::GetID(window_name.c_str());
    ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_PassthruCentralNode;
    ImGui::DockSpace(dockspace_id, ImVec2(0.0F, 0.0F), dockspace_flags);

    if (ImGui::DockBuilderGetNode(dockspace_id) == nullptr)
    {
      ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);  // add empty node
      ImGui::DockBuilderSetNodeSize(dockspace_id, viewport->Size);
      ImGui::DockBuilderFinish(dockspace_id);
    }

    show();

    ImGui::End();

    // Rendering
    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(
        clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);

    ImGui::EndFrame();
  }
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImPlot::DestroyContext();
  ImGui::DestroyContext();

  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}

}  // namespace ttb
