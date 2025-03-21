#include "tracking_lib/TTBManager/Visualization.h"
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "tracking_lib/TTBManager/TTBManager.h"
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <implot/implot.h>

#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>
#define GL_SILENCE_DEPRECATION
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif
#include <GLFW/glfw3.h>  // Will drag system OpenGL headers

#include <boost/math/distributions/gamma.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/beta.hpp>
#include <boost/math/distributions/beta.hpp>
#include <boost/math/distributions/chi_squared.hpp>

#include <tracy/tracy/Tracy.hpp>
#include <unsupported/Eigen/src/MatrixFunctions/MatrixSquareRoot.h>
namespace ttb
{

Vizu::Vizu(TTBManager* manager) : _manager{ manager }
{
}

auto to_x_data(auto& vals)
{
  std::vector<double> xs;
  for (auto const& x : vals)
  {
    xs.push_back(x);
  }
  return xs;
}

auto to_xy_data(auto& vals)
{
  std::vector<double> xs;
  std::vector<double> ys;
  for (auto const& [x, y] : vals)
  {
    xs.push_back(x);
    ys.push_back(y);
  }
  return std::tuple{ std::move(xs), std::move(ys) };
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

std::pair<std::vector<double>, std::vector<double>> to_box(BaseDistribution const& dist, Components const& comps)
{
  std::vector<double> xs, ys;
  if (auto ind = comps.indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y, COMPONENT::RADIUS }); ind.has_value())
  {
    Vector2 const m = dist.mean()(ind.value()({ 0, 1 }));
    double const r = dist.mean()(ind.value()(2));
    for (double ts = 0; ts < 2 * std::numbers::pi; ts += 0.05)
    {
      xs.push_back(m(0) + std::cos(ts) * r);
      ys.push_back(m(1) + std::sin(ts) * r);
    }
    return { std::move(xs), std::move(ys) };
  }
  for (REFERENCE_POINT const rp : { REFERENCE_POINT::FRONT_RIGHT,
                                    REFERENCE_POINT::FRONT_LEFT,
                                    REFERENCE_POINT::BACK_LEFT,
                                    REFERENCE_POINT::BACK_RIGHT })
  {
    auto ind_xy = comps.indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y });
    if (not ind_xy.has_value())
    {
      return {};
    }
    Vector2 const& corner = transformation::transform(dist.mean(), comps, dist.refPoint(), rp)(ind_xy.value());
    xs.emplace_back(corner(0));
    ys.emplace_back(corner(1));
  }
  xs.emplace_back(xs.front());
  ys.emplace_back(ys.front());
  return { std::move(xs), std::move(ys) };
}

std::pair<std::vector<double>, std::vector<double>> to_box(Measurement const& meas)
{
  return to_box(*meas._dist->dists().front(), meas._meas_comps);
}

std::pair<std::vector<double>, std::vector<double>> to_box(State const& state)
{
  auto const& [model_id, dist] = state.getEstimate();
  return to_box(*dist, state._manager->getStateModel(model_id).state_comps());
}

std::pair<std::vector<double>, std::vector<double>> to_vel(State const& state)
{
  auto const& [model_id, dist] = state.getEstimate();
  auto ind_xy = state._manager->getStateModel(model_id).state_comps().indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y });
  if (not ind_xy.has_value())
  {
    return {};
  }
  if (auto const vel = transformation::transform(dist->mean(),
                                                 state._manager->getStateModel(model_id).state_comps(),
                                                 Components({ COMPONENT::VEL_X, COMPONENT::VEL_Y }));
      vel.has_value())
  {
    std::vector xvel{ dist->mean()(ind_xy.value()(0)), dist->mean()(ind_xy.value()(0)) + vel.value()(0) };
    std::vector yvel{ dist->mean()(ind_xy.value()(1)), dist->mean()(ind_xy.value()(1)) + vel.value()(1) };
    return { std::move(xvel), std::move(yvel) };
  }
  return {};
}

std::tuple<std::vector<double>, std::vector<double>> to_beam(Measurement const& meas,
                                                             COMPONENT comp,
                                                             SensorInformation const& sensor_information)
{
  Vector const& mean = meas._dist->dists().front()->mean();
  std::tuple<std::vector<double>, std::vector<double>> beams;
  if (auto ind = meas._meas_comps.indexOf(comp); ind.has_value())
  {
    Vector3 const p1 = [&] -> Vector3 {
      if (comp == COMPONENT::AZIMUTH)
      {
        double const angle = mean(ind.value());
        Vector3 out{ { std::cos(angle), std::sin(angle), 0 } };
        return out * 150;
      }
      Vector3 out{ { mean(ind.value()), 0, 1 } };
      out.normalize();
      return out * 150;
    }();
    Vector3 const p0 = Vector3::Zero();
    Vector const t0 = sensor_information._to_sensor_cs.inverse() * p0.homogeneous();
    Vector const t1 = sensor_information._to_sensor_cs.inverse() * p1.homogeneous();
    std::get<0>(beams) = { t0(0), t1(0) };
    std::get<1>(beams) = { t0(1), t1(1) };
  }
  return beams;
}

void Vizu::show_tracks()
{
  int static len_window = 3000;
  ImGui::InputInt("window length ms", &len_window, 10, 100);
  len_window = std::max(0, len_window);
  Duration const until = std::chrono::milliseconds(len_window);
  bool static use_real_time = true;
  ImGui::Checkbox("use real time", &use_real_time);
  double static time_sec = to_seconds(_manager->filter_time().time_since_epoch());
  Time const now = [&] {
    if (use_real_time)
    {
      time_sec = std::max(to_seconds(_manager->filter_time().time_since_epoch()), time_sec);
      return _manager->filter_time();
    }
    ImGui::InputDouble("Time s", &time_sec, 0.1, 1.0);
    time_sec = std::min(to_seconds(_manager->filter_time().time_since_epoch()), time_sec);
    return Time(std::chrono::milliseconds(static_cast<Duration::rep>(time_sec * 1000)));
  }();
  bool static show_tracks = true;
  ImGui::Checkbox("show tracks", &show_tracks);
  bool static show_labels = false;
  ImGui::Checkbox("show labels", &show_labels);
  bool static detailed_track_info = false;
  ImGui::Checkbox("detailed track info", &detailed_track_info);
  bool static show_detections = true;
  ImGui::Checkbox("show detections", &show_detections);
  bool static show_detection_details = false;
  ImGui::Checkbox("show detection details", &show_detection_details);
  bool static show_fov = true;
  ImGui::Checkbox("show fov", &show_fov);
  bool static show_fov_names = false;
  ImGui::Checkbox("show fov names", &show_fov_names);
  bool static show_covariance = false;
  ImGui::Checkbox("show covariance", &show_covariance);
  ImGui::Checkbox("add static tracks", &_manager->next_params().static_tracks.enable);
  if (ImPlot::BeginPlot("Tracks", ImVec2(-1, -1), ImPlotFlags_Equal))
  {
    ImPlot::SetupAxes("x", "y");
    ImPlot::SetupFinish();
    ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 5);
    std::vector<double> xs;
    std::vector<double> ys;
    if (show_tracks)
    {
      for (auto const& [label, positions] : _cycle_data._tracks)
      {
        std::tie(xs, ys) = [&] {
          auto plot_positions =
              positions |
              std::views::filter([&](State const& track) { return track._time < now and now - track._time < until; }) |
              std::views::transform([this](State const& track) {
                auto const& [model_id, dist] = track.getEstimate();
                Vector2 pos_xy = dist->mean()(_manager->getStateModel(model_id)
                                                  .state_comps()
                                                  .indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y })
                                                  .value());
                return std::pair(pos_xy(0), pos_xy(1));
              });
          return to_xy_data(plot_positions);
        }();
        ImPlot::PlotLine(
            std::to_string(label.value_).c_str(), xs.data(), ys.data(), xs.size(), ImPlotItemFlags_NoLegend);
        if (show_labels and not xs.empty())
        {
          ImPlot::PlotText(std::to_string(label.value_).c_str(), xs.back(), ys.back());
        }
      }
      for (auto const& [label, states] : _cycle_data._tracks)
      {
        xs.clear();
        ys.clear();
        Duration min_time_diff = Duration(1s);
        State best_state = states.front();
        for (State const& track : states)
        {
          if (std::chrono::abs(track._time - now) < min_time_diff)
          {
            min_time_diff = std::chrono::abs(track._time - now);
            best_state = track;
          }
        }
        if (min_time_diff < 300ms)
        {
          auto const& [model_id, dist] = best_state.getEstimate();
          auto ind_xy = _manager->getStateModel(model_id).state_comps().indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y });
          if (not ind_xy.has_value())
          {
            continue;
          }
          Vector const& mean = dist->mean();
          std::tie(xs, ys) = to_box(best_state);
          bool const is_cam_assoc = best_state._misc.contains("planner");
          if (is_cam_assoc)
          {
            ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 25);
          }
          ImPlot::PlotLine(
              std::to_string(label.value_).c_str(), xs.data(), ys.data(), xs.size(), ImPlotItemFlags_NoLegend);
          if (is_cam_assoc)
          {
            ImPlot::PopStyleVar();
          }
          Matrix22 P_xy = dist->covariance()(ind_xy.value(), ind_xy.value()).sqrt();
          if (show_covariance)
          {
            xs.clear();
            ys.clear();
            for (double ts = 0; ts < 2 * std::numbers::pi; ts += 0.1)
            {
              Vector2 t = mean(ind_xy.value()) + P_xy * Vector2{ { std::sin(ts), -std::cos(ts) } };
              xs.push_back(t(0));
              ys.push_back(t(1));
            }
            ImPlot::PlotLine(
                std::to_string(label.value_).c_str(), xs.data(), ys.data(), xs.size(), ImPlotItemFlags_NoLegend);
          }
          std::tie(xs, ys) = to_vel(best_state);
          ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 2);
          ImPlot::PlotLine(std::to_string(label.value_).c_str(), xs.data(), ys.data(), static_cast<int>(xs.size()));
          ImPlot::PopStyleVar();
          if (detailed_track_info)
          {
            ImPlot::PlotText(best_state.toString().c_str(), mean(ind_xy.value()(0)), mean(ind_xy.value()(1)));
          }
        }
      }
      ImPlot::PopStyleVar();
    }
    if (show_detections)
    {
      for (auto const& [model_id, detections] : _meas_model_data)
      {
        for (auto const& [time, single_time_detections] : detections._measurements)
        {
          if (time < now and now - time < until)
          {
            if (std::size_t const ind = single_time_detections.index(); ind == 0)
            {
              std::vector<Measurement> const& meass = std::get<0>(single_time_detections)._data;
              for (Measurement const& meas : meass)
              {
                std::tie(xs, ys) = to_box(meas);
                ImPlot::PlotLine(model_id.value_.c_str(), xs.data(), ys.data(), xs.size());
                if (not xs.empty() and show_detection_details)
                {
                  ImPlot::PlotText(meas.toString().c_str(), xs.front(), ys.front());
                }
                for (COMPONENT comp : { COMPONENT::AZIMUTH, COMPONENT::X_CC_LOWER_LEFT, COMPONENT::X_CC_UPPER_RIGHT })
                {
                  std::tie(xs, ys) = to_beam(meas, comp, std::get<0>(single_time_detections)._sensorInfo);
                  ImPlot::PlotLine(model_id.value_.c_str(), xs.data(), ys.data(), xs.size());
                }
              }
              auto plot_positions =
                  meass | std::views::filter([](Measurement const& meas) {
                    return meas._meas_comps.indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y }).has_value();
                  }) |
                  std::views::transform([this](Measurement const& meas) {
                    Vector2 pos_xy =
                        meas._dist->mean()(meas._meas_comps.indexOf({ COMPONENT::POS_X, COMPONENT::POS_Y }).value());
                    return std::pair(pos_xy(0), pos_xy(1));
                  });
              auto [x, y] = to_xy_data(plot_positions);
              ImPlot::SetNextMarkerStyle(ImPlotMarker_Cross);
              ImPlot::PlotScatter(model_id.value_.c_str(), x.data(), y.data(), x.size());
            }
            else if (ind == 1)
            {
              std::vector<State> const& states = std::get<1>(single_time_detections)._data;
              for (State const& state : states)
              {
                std::tie(xs, ys) = to_box(state);
                if (not xs.empty() and show_detection_details)
                {
                  ImPlot::PlotText(state.toString().c_str(), xs.front(), ys.front());
                }
                ImPlot::PlotLine(
                    std::get<1>(single_time_detections)._id.value_.c_str(), xs.data(), ys.data(), xs.size());
                std::tie(xs, ys) = to_vel(state);
                ImPlot::PlotLine(
                    std::get<1>(single_time_detections)._id.value_.c_str(), xs.data(), ys.data(), xs.size());
              }
            }
          }
        }
      }
    }
    if (show_fov)
    {
      for (auto const& [model_id, model_data] : _meas_model_data)
      {
        Duration min_time_diff = Duration(1s);
        if (model_data._measurements.empty())
        {
          continue;
        }
        std::variant<MeasurementContainer, StateContainer> best_fit = model_data._measurements.front().second;
        for (auto const& [time, data] : model_data._measurements)
        {
          if (std::chrono::abs(time - now) < min_time_diff)
          {
            min_time_diff = std::chrono::abs(time - now);
            best_fit = data;
          }
        }
        if (min_time_diff < 300ms)
        {
          if (auto ind = best_fit.index(); ind == 0)
          {
            MeasurementContainer const& meas_container = std::get<0>(best_fit);
            if (meas_container._sensorInfo._sensor_fov.has_value())
            {
              xs.clear();
              ys.clear();
              if (meas_container._sensorInfo._sensor_fov.value()._polygons.contains(
                      { COMPONENT::POS_X, COMPONENT::POS_Y }))
              {
                for (auto const& point : meas_container._sensorInfo._sensor_fov.value()
                                             ._polygons.at({ COMPONENT::POS_X, COMPONENT::POS_Y })
                                             .multiPolygon.front()
                                             .outer())
                {
                  xs.push_back(point.x());
                  ys.push_back(point.y());
                }
              }
              ImPlot::PlotLine(model_id.value_.c_str(), xs.data(), ys.data(), xs.size());
              if (show_fov_names)
              {
                for (std::size_t i = 0; i < xs.size(); ++i)
                {
                  ImPlot::PlotText(model_id.value_.c_str(), xs[i], ys[i]);
                }
              }
            }
          }
          else
          {
            // pass
          }
        }
      }
    }
    ImPlot::EndPlot();
  }
}

void Vizu::show_overview()
{
  bool static auto_follow = false;
  ImGui::Text("Filter time: %f", to_seconds(_manager->filter_time().time_since_epoch()));
  ImGui::Checkbox("auto follow", &auto_follow);
  Time const now = _manager->filter_time();
  Duration until;
  if (auto_follow)
  {
    int static len_window = 5;
    ImGui::InputInt("window length s", &len_window, 1, 5);
    until = std::chrono::seconds(len_window);
  }
  else
  {
    until = std::chrono::seconds(1000000);
  }
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
    auto [x, y] = [&] {
      auto num_tracks_plot = _cycle_data._num_tracks |
                             std::views::filter([&](auto const& data) { return now - std::get<0>(data) < until; }) |
                             std::views::transform([](auto const& data) {
                               return std::pair(to_seconds(std::get<0>(data).time_since_epoch()), std::get<1>(data));
                             });
      return to_xy_data(num_tracks_plot);
    }();
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
    auto [x, y] = [&] {
      auto num_tracks_plot = _cycle_data._num_measurements |
                             std::views::filter([&](auto const& data) { return now - std::get<0>(data) < until; }) |
                             std::views::transform([](auto const& data) {
                               return std::pair(to_seconds(std::get<0>(data).time_since_epoch()), std::get<1>(data));
                             });
      return to_xy_data(num_tracks_plot);
    }();
    ImPlot::PlotLine("#Measurements", x.data(), y.data(), x.size());
    auto [x1, y1] = [&] {
      auto num_tracks_plot = _cycle_data._num_sources |
                             std::views::filter([&](auto const& data) { return now - std::get<0>(data) < until; }) |
                             std::views::transform([](auto const& data) {
                               return std::pair(to_seconds(std::get<0>(data).time_since_epoch()), std::get<1>(data));
                             });
      return to_xy_data(num_tracks_plot);
    }();
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
    for (auto const& [model_id, sensor_data] : _meas_model_data)
    {
      auto [x1, y1] = [&] {
        auto comp_time =
            sensor_data._computation_time |
            std::views::filter([&](auto const& data) { return now - std::get<0>(data) < until; }) |
            std::views::transform([](auto const& data) {
              return std::pair(to_seconds(std::get<0>(data).time_since_epoch()), to_milliseconds(std::get<1>(data)));
            });
        return to_xy_data(comp_time);
      }();
      ImPlot::PlotLine(model_id.value_.c_str(), x1.data(), y1.data(), x1.size());
    }
    auto [x, y] = [&] {
      auto comp_time =
          _cycle_data.computation_time |
          std::views::filter([&](auto const& data) { return now - std::get<0>(data) < until; }) |
          std::views::transform([](auto const& data) {
            return std::pair(to_seconds(std::get<0>(data).time_since_epoch()), to_milliseconds(std::get<1>(data)));
          });
      return to_xy_data(comp_time);
    }();
    ImPlot::PlotLine("Cycle Time", x.data(), y.data(), x.size());
    ImPlot::EndPlot();
  }
  ImGui::Text("#Discarded Data %zu", [&] {
    std::size_t tot_discarded = 0;
    for (auto const& [model, model_data] : _meas_model_data)
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
    auto [x, y] = [&] {
      auto comp_time = _cycle_data._num_sources |
                       std::views::filter([&](auto const& data) { return now - std::get<0>(data) < until; }) |
                       std::views::transform([](auto const& data) {
                         return std::pair(to_seconds(std::get<0>(data).time_since_epoch()), std::get<1>(data));
                       });
      return to_xy_data(comp_time);
    }();
    ImPlot::PlotLine("#Sources", x.data(), y.data(), x.size());
    auto [x1, y1] = [&] {
      auto comp_time = _cycle_data._buffer_size |
                       std::views::filter([&](auto const& data) { return now - std::get<0>(data) < until; }) |
                       std::views::transform([](auto const& data) {
                         return std::pair(to_seconds(std::get<0>(data).time_since_epoch()), std::get<1>(data));
                       });
      return to_xy_data(comp_time);
    }();
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
    for (auto const& [model_id, sensor_data] : _meas_model_data)
    {
      auto [x, y] = [&] {
        auto delay_plot =
            sensor_data._delay | std::views::filter([&](auto const& data) { return now - std::get<0>(data) < until; }) |
            std::views::transform([](auto const& data) {
              return std::pair(to_seconds(std::get<0>(data).time_since_epoch()), to_milliseconds(std::get<1>(data)));
            });
        return to_xy_data(delay_plot);
      }();
      ImPlot::PlotLine(std::string(model_id.value_ + " Delay").c_str(), x.data(), y.data(), x.size());
    }
    auto [x1, y1] = [&] {
      auto delay_plot =
          _cycle_data._buffer_delay |
          std::views::filter([&](auto const& data) { return now - std::get<0>(data) < until; }) |
          std::views::transform([](auto const& data) {
            return std::pair(to_seconds(std::get<0>(data).time_since_epoch()), to_milliseconds(std::get<1>(data)));
          });
      return to_xy_data(delay_plot);
    }();
    ImPlot::PlotLine("Buffer Delay", x1.data(), y1.data(), x1.size());
    auto [x3, y3] = [&] {
      auto delay_plot =
          _cycle_data._filter_delay |
          std::views::filter([&](auto const& data) { return now - std::get<0>(data) < until; }) |
          std::views::transform([](auto const& data) {
            return std::pair(to_seconds(std::get<0>(data).time_since_epoch()), to_milliseconds(std::get<1>(data)));
          });
      return to_xy_data(delay_plot);
    }();
    ImPlot::PlotLine("Filter Delay", x3.data(), y3.data(), x3.size());

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
    for (auto const& [model_id, sensor_data] : _meas_model_data)
    {
      auto [x, y] = [&] {
        auto comp_time = sensor_data._used_in_cycle |
                         std::views::filter([&](auto const& data) { return now - data < until; }) |
                         std::views::transform([model_height](auto const& data) {
                           return std::pair(to_seconds(data.time_since_epoch()), model_height);
                         });
        return to_xy_data(comp_time);
      }();
      ImPlot::PlotScatter(model_id.value_.c_str(), x.data(), y.data(), x.size());
      auto [x1, y1] = [&] {
        auto comp_time = sensor_data._discarded_data |
                         std::views::filter([&](auto const& data) { return now - data < until; }) |
                         std::views::transform([model_height](auto const& data) {
                           return std::pair(to_seconds(data.time_since_epoch()), model_height);
                         });
        return to_xy_data(comp_time);
      }();
      ImPlot::SetNextMarkerStyle(ImPlotMarker_Cross, 5);
      ImPlot::PlotScatter(model_id.value_.c_str(), x1.data(), y1.data(), x1.size());
      model_height += 1;
    }
    for (Time time : _cycle_data._trigger_time)
    {
      if (now - time < until)
      {
        std::array x = { to_seconds(time.time_since_epoch()), to_seconds(time.time_since_epoch()) };
        std::array y = { -0.5, model_height - 0.5 };
        ImPlot::PlotLine("##1", x.data(), y.data(), x.size(), ImPlotItemFlags_NoLegend);
      }
    }
    ImPlot::EndPlot();
  }
}

void Vizu::reset()
{
  _cycle_data = {};
  _meas_model_data.clear();
}

void Vizu::show_debug()
{
  bool static auto_follow = false;
  ImGui::Checkbox("auto follow", &auto_follow);
  Time const now = _manager->filter_time();
  Duration until;
  if (auto_follow)
  {
    int static len_window = 5;
    ImGui::InputInt("window length s", &len_window, 1, 5);
    until = std::chrono::seconds(len_window);
  }
  else
  {
    until = std::chrono::seconds(1000000);
  }
  double height = ImGui::GetWindowHeight() - ImGui::GetCursorPosY();
  if (ImGui::BeginTabBar("Sensors"))
  {
    for (auto const& [model_id, sensor_data] : _meas_model_data)
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
          auto [x, y] = [&] {
            auto delay_plot = sensor_data._delay |
                              std::views::filter([&](auto const& data) { return now - std::get<0>(data) < until; }) |
                              std::views::transform([](auto const& data) {
                                return std::pair(to_seconds(std::get<0>(data).time_since_epoch()),
                                                 to_milliseconds(std::get<1>(data)));
                              });
            return to_xy_data(delay_plot);
          }();
          ImPlot::PlotLine("Data Delay", x.data(), y.data(), x.size());
          auto [x2, y2] = [&] {
            auto delay_plot = sensor_data._estimated_delay |
                              std::views::filter([&](auto const& data) { return now - std::get<0>(data) < until; }) |
                              std::views::transform([](auto const& data) {
                                return std::pair(to_seconds(std::get<0>(data).time_since_epoch()),
                                                 to_milliseconds(std::get<1>(data)));
                              });
            return to_xy_data(delay_plot);
          }();
          ImPlot::PlotLine("Estimated Data Delay", x2.data(), y2.data(), x2.size());
          auto [x1, y1] = [&] {
            auto delay_plot = _cycle_data._buffer_delay |
                              std::views::filter([&](auto const& data) { return now - std::get<0>(data) < until; }) |
                              std::views::transform([](auto const& data) {
                                return std::pair(to_seconds(std::get<0>(data).time_since_epoch()),
                                                 to_milliseconds(std::get<1>(data)));
                              });
            return to_xy_data(delay_plot);
          }();
          ImPlot::PlotLine("Buffer Delay", x1.data(), y1.data(), x1.size());
          auto [x3, y3] = [&] {
            auto delay_plot = _cycle_data._filter_delay |
                              std::views::filter([&](auto const& data) { return now - std::get<0>(data) < until; }) |
                              std::views::transform([](auto const& data) {
                                return std::pair(to_seconds(std::get<0>(data).time_since_epoch()),
                                                 to_milliseconds(std::get<1>(data)));
                              });
            return to_xy_data(delay_plot);
          }();
          ImPlot::PlotLine("Filter Delay", x3.data(), y3.data(), x3.size());
          auto [x4, y4] = [&] {
            auto delay_plot = sensor_data._duration_in_buffer |
                              std::views::filter([&](auto const& data) { return now - std::get<0>(data) < until; }) |
                              std::views::transform([](auto const& data) {
                                return std::pair(to_seconds(std::get<0>(data).time_since_epoch()),
                                                 to_milliseconds(std::get<1>(data)));
                              });
            return to_xy_data(delay_plot);
          }();
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
          auto [x, y] = [&] {
            auto num_measurements =
                sensor_data._measurements |
                std::views::filter([&](auto const& data) { return now - std::get<0>(data) < until; }) |
                std::views::transform([](auto const& data) {
                  if (std::size_t ind = data.second.index(); ind == 0)
                  {
                    return std::pair(to_seconds(std::get<0>(data).time_since_epoch()),
                                     std::get<0>(data.second)._data.size());
                  }
                  return std::pair(to_seconds(std::get<0>(data).time_since_epoch()),
                                   std::get<1>(data.second)._data.size());
                });
            return to_xy_data(num_measurements);
          }();
          ImPlot::PlotLine("#Measurements", x.data(), y.data(), x.size());
          auto [x1, y1] = [&] {
            auto num_assoc = sensor_data._num_assoc_measurements |
                             std::views::filter([&](auto const& data) { return now - std::get<0>(data) < until; }) |
                             std::views::transform([](auto const& data) {
                               return std::pair(to_seconds(std::get<0>(data).time_since_epoch()), std::get<1>(data));
                             });
            return to_xy_data(num_assoc);
          }();
          ImPlot::PlotLine("#Associated Measurements", x1.data(), y1.data(), x1.size());
          ImPlot::EndPlot();
        }
        ImGui::EndTabItem();
      }
    }
    ImGui::EndTabBar();
  }
}

void Vizu::show_params()
{
  if (ImGui::BeginTabBar("Tab Bar"))
  {
    if (ImGui::BeginTabItem("State Models"))
    {
      if (ImGui::BeginTabBar("State Models"))
      {
        for (auto const& [state_model_id, state_model] : _manager->getStateModelMap())
        {
          if (ImGui::BeginTabItem(to_string(_manager->state_model_next_params(state_model_id).type).c_str()))
          {
            for (auto& [comp, std_noise] : _manager->state_model_next_params(state_model_id).model_noise_std_dev)
            {
              ImGui::InputDouble(comp.c_str(), &std_noise, 0.05, 0.5);
              std_noise = std::clamp(std_noise, 0.0, 100.0);
            }
            ImGui::EndTabItem();
          }
        }
        ImGui::EndTabBar();
      }
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Static Tracks"))
    {
      if (ImGui::BeginTabBar("Static Tracks"))
      {
        ImGui::Checkbox("add static tracks", &_manager->next_params().static_tracks.enable);
        for (StaticTrackParams& static_track : _manager->next_params().static_tracks.tracks)
        {
          for (auto& [comp, val] : static_track.mean)
          {
            ImGui::InputDouble(comp.c_str(), &val, 0.05, 0.5);
          }
        }
        ImGui::EndTabBar();
      }
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Measurement Models"))
    {
      if (ImGui::BeginTabBar("Meas Models"))
      {
        for (auto const& [meas_model_id, meas_model] : _manager->getMeasModelMap())
        {
          if (ImGui::BeginTabItem(meas_model_id.value_.c_str()))
          {
            ImGui::Checkbox("Enable", &_manager->meas_model_next_params(meas_model_id).enable);
            if (_manager->meas_model_next_params(meas_model_id).enable)
            {
              ImGui::Checkbox("Override Variance",
                              &_manager->meas_model_next_params(meas_model_id).overwrite_meas_vars);
              if (_manager->meas_model_params(meas_model_id).overwrite_meas_vars)
              {
                //                for (COMPONENT comp : _manager->getMeasModel(meas_model_id).meas_model_comps()._comps)
                //                {
                //                  double& var =
                //                  _manager->meas_model_next_params(meas_model_id).default_var.at(to_string(comp));
                //                  ImGui::InputDouble(to_string(comp).c_str(), &var, 0.05, 0.5);
                //                  var = std::clamp(var, 0.001, 10.0);
                //                }
              }
              ImGui::Checkbox("Enable Clutter Rate Estimation",
                              &_manager->meas_model_next_params(meas_model_id).clutter.rate_estimation.enable);
              ImGui::Checkbox("Use Clutter Rate Estimation in Filter",
                              &_manager->meas_model_next_params(meas_model_id).clutter.rate_estimation.use_in_filter);
              if (_manager->meas_model_next_params(meas_model_id).clutter.rate.has_value())
              {
                double& rate = _manager->meas_model_next_params(meas_model_id).clutter.rate.value();
                ImGui::InputDouble("Clutter Rate", &rate, 0.05, 0.5);
                rate = std::clamp(rate, 0.001, 500.0);
              }
              ImGui::Checkbox("Enable Detection Probability Estimation",
                              &_manager->meas_model_next_params(meas_model_id).detection.prob_estimation.enable);
              ImGui::Checkbox("Use Detection Probability Estimation in Filter",
                              &_manager->meas_model_next_params(meas_model_id).detection.prob_estimation.use_in_filter);
              double& detection_prob = _manager->meas_model_next_params(meas_model_id).detection.prob;
              ImGui::InputDouble("Detection Probability", &detection_prob, 0.01, 0.05);
              detection_prob = std::clamp(detection_prob, 0.001, 1.0);
            }
            ImGui::EndTabItem();
          }
        }
        ImGui::EndTabBar();
      }
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("LMB Distribution"))
    {
      ImGui::Checkbox("Use Grouping", &_manager->next_params().lmb_distribution.use_grouping);
      bool& post_process_prediction = _manager->next_params().lmb_distribution.post_process_prediction.enable;
      ImGui::Checkbox("Post Process Prediction", &post_process_prediction);
      if (post_process_prediction)
      {
        std::size_t& max_tracks = _manager->next_params().lmb_distribution.post_process_prediction.max_tracks;
        auto max_tracks_int = static_cast<int>(max_tracks);
        ImGui::InputInt("Max #Tracks", &max_tracks_int, 1, 10);
        max_tracks_int = std::clamp(max_tracks_int, 0, 10000);
        max_tracks = static_cast<std::size_t>(max_tracks_int);

        double& pruning_threshold = _manager->next_params().lmb_distribution.post_process_prediction.pruning_threshold;
        ImGui::InputDouble("Pruning Threshold", &pruning_threshold, 0.01, 0.1);
        pruning_threshold = std::clamp(pruning_threshold, 0.0, 1.0);

        std::size_t& max_last_assoc_duration_ms =
            _manager->next_params().lmb_distribution.post_process_prediction.max_last_assoc_duration_ms;
        auto max_last_int = static_cast<int>(max_last_assoc_duration_ms);
        ImGui::InputInt("Max Last Association Duration ms", &max_last_int, 1, 10);
        max_last_int = std::clamp(max_last_int, 0, 100000);
        max_last_assoc_duration_ms = static_cast<std::size_t>(max_last_int);
      }
      bool& post_process_update = _manager->next_params().lmb_distribution.post_process_update.enable;
      ImGui::Checkbox("Post Process Update", &post_process_update);
      if (post_process_update)
      {
        std::size_t& max_tracks = _manager->next_params().lmb_distribution.post_process_update.max_tracks;
        auto max_tracks_int = static_cast<int>(max_tracks);
        ImGui::InputInt("Max #Tracks##", &max_tracks_int, 1, 10);
        max_tracks_int = std::clamp(max_tracks_int, 0, 100000);
        max_tracks = static_cast<std::size_t>(max_tracks_int);

        double& pruning_threshold = _manager->next_params().lmb_distribution.post_process_update.pruning_threshold;
        ImGui::InputDouble("Pruning Threshold##", &pruning_threshold, 0.01, 0.1);
        pruning_threshold = std::clamp(pruning_threshold, 0.0, 1.0);
      }
      static const char* update_methods_str[]{ "GLMB", "LBP" };
      static int selected = static_cast<int>(_manager->next_params().lmb_distribution.update_method);
      if (ImGui::Combo("Update Method", &selected, update_methods_str, 2))
      {
        _manager->next_params().lmb_distribution.update_method = static_cast<LMB_UPDATE_METHOD>(selected);
      }
      if (_manager->next_params().lmb_distribution.extraction.type ==
          MO_DISTRIBUTION_EXTRACTION_TYPE::EXISTENCE_PROBABILITY)
      {
        double& extraction_threshold = _manager->next_params().lmb_distribution.extraction.threshold;
        ImGui::InputDouble("Extraction Threshold", &extraction_threshold, 0.01, 0.1);
        extraction_threshold = std::clamp(extraction_threshold, 0.0, 1.0);
      }
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Filter"))
    {
      ImGui::Text("%s", to_string(_manager->next_params().filter.type).c_str());
      ImGui::EndTabItem();
    }
    ImGui::EndTabBar();
  }
}

void Vizu::show_assessment()
{
  bool static auto_follow = false;
  ImGui::Checkbox("auto follow", &auto_follow);
  Time const now = _manager->filter_time();
  Duration until;
  if (auto_follow)
  {
    int static len_window = 5;
    ImGui::InputInt("window length s", &len_window, 1, 5);
    until = std::chrono::seconds(len_window);
  }
  else
  {
    until = std::chrono::seconds(1000000);
  }
  double static quantile = 0.9;
  ImGui::InputDouble("quantile", &quantile, 0.01, 0.1);
  quantile = std::clamp(quantile, 0.0001, 0.9999);
  if (ImGui::BeginTabBar("Tab Bar"))
  {
    for (auto const& [model_id, sensor_data] : _meas_model_data)
    {
      if (ImGui::BeginTabItem(model_id.value_.c_str()))
      {
        if (auto_follow)
        {
          ImPlot::SetNextAxesToFit();
        }
        double const height = ImGui::GetWindowHeight() - ImGui::GetCursorPosY();
        double const width = ImGui::GetWindowWidth() - ImGui::GetCursorPosX();
        if (ImPlot::BeginPlot((std::string("Clutter Rate Estimation ") + model_id.value_).c_str(),
                              ImVec2(width / 2, height / 3)))
        {
          ImPlot::SetupAxes("Time", "Clutter Rate");
          ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Time);
          ImPlot::SetupFinish();
          ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.25f);
          auto [x, y1, y2, y3] = [&] {
            auto clutter_plot =
                sensor_data._clutter_estimation |
                std::views::filter([&](auto const& data) { return now - std::get<0>(data) < until; }) |
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
          ImPlot::PlotErrorBars("Clutter Estimation", x.data(), y3.data(), y1.data(), y2.data(), x.size());
          ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 5);
          ImPlot::PlotLine("Clutter Estimation", x.data(), y3.data(), x.size());
          ImPlot::PopStyleVar();
          ImPlot::EndPlot();
        }
        if (auto_follow)
        {
          ImPlot::SetNextAxesToFit();
        }
        ImGui::SameLine();
        if (ImPlot::BeginPlot((std::string("Clutter Rate Bayes Factor ") + model_id.value_).c_str(),
                              ImVec2(-1, height / 3)))
        {
          ImPlot::SetupAxes("Time", "Bayes Factor");
          ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Time);
          ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
          ImPlot::SetupFinish();
          auto [x, y] = [&] {
            auto clutter_plot =
                sensor_data._clutter_bayes_factor |
                std::views::filter([&](auto const& data) { return now - std::get<0>(data) < until; }) |
                std::views::transform([&](auto const& data) {
                  return std::tuple{ to_seconds(std::get<0>(data).time_since_epoch()), std::get<1>(data) };
                });
            return to_xy_data(clutter_plot);
          }();
          ImPlot::PlotLine("Bayes Factor", x.data(), y.data(), x.size());
          if (not x.empty())
          {
            std::array xs = { std::ranges::min(x), std::ranges::max(x) };
            std::array ys = { 1.0, 1.0 };
            ImPlot::PlotLine("good/bad", xs.data(), ys.data(), 2);
          }
          ImPlot::EndPlot();
        }
        if (auto_follow)
        {
          ImPlot::SetNextAxesToFit();
        }
        if (ImPlot::BeginPlot((std::string("Detection Probability Estimation ") + model_id.value_).c_str(),
                              ImVec2(width / 2, height / 3)))
        {
          ImPlot::SetupAxes("Time", "Detection Probability");
          ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Time);
          ImPlot::SetupFinish();
          auto [x, y1, y2, y3] = [&] {
            auto detection_plot =
                sensor_data._detection_estimation |
                std::views::filter([&](auto const& data) { return now - std::get<0>(data) < until; }) |
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
          ImPlot::PlotErrorBars("Detection Estimation", x.data(), y3.data(), y1.data(), y2.data(), x.size());
          ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 5);
          ImPlot::PlotLine("Detection Estimation", x.data(), y3.data(), x.size());
          ImPlot::PopStyleVar();
          ImPlot::EndPlot();
        }
        if (auto_follow)
        {
          ImPlot::SetNextAxesToFit();
        }
        ImGui::SameLine();
        if (ImPlot::BeginPlot((std::string("Detection Prob Bayes Factor ") + model_id.value_).c_str(),
                              ImVec2(-1, height / 3)))
        {
          ImPlot::SetupAxes("Time", "Bayes Factor");
          ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Time);
          ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
          ImPlot::SetupFinish();
          auto [x, y] = [&] {
            auto clutter_plot =
                sensor_data._detection_bayes_factor |
                std::views::filter([&](auto const& data) { return now - std::get<0>(data) < until; }) |
                std::views::transform([&](auto const& data) {
                  return std::tuple{ to_seconds(std::get<0>(data).time_since_epoch()), std::get<1>(data) };
                });
            return to_xy_data(clutter_plot);
          }();
          ImPlot::PlotLine("Bayes Factor", x.data(), y.data(), x.size());
          if (not x.empty())
          {
            std::array xs = { std::ranges::min(x), std::ranges::max(x) };
            std::array ys = { 1.0, 1.0 };
            ImPlot::PlotLine("good/bad", xs.data(), ys.data(), 2);
          }
          ImPlot::EndPlot();
        }
        if (auto_follow)
        {
          ImPlot::SetNextAxesToFit();
        }
        if (ImPlot::BeginPlot((std::string("Nis ") + model_id.value_).c_str(), ImVec2(-1, -1)))
        {
          ImPlot::SetupAxes("Time", "Nis");
          ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Time);
          ImPlot::SetupFinish();
          for (auto const& [comp, comp_wise_nis] : sensor_data._nis_dof)
          {
            auto [x, y1, y2, y3] = [&] {
              auto nis_plot =
                  comp_wise_nis |
                  std::views::filter([&](auto const& data) { return now - std::get<0>(data) < until; }) |
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
            ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 5);
            ImPlot::PlotLine(to_string(comp).c_str(), x.data(), y3.data(), x.size());
            ImPlot::PopStyleVar();
          }
          ImPlot::EndPlot();
        }
        ImGui::EndTabItem();
      }
    }
    ImGui::EndTabBar();
  }
}

void Vizu::show_evaluation()
{
  for (auto const& [model_id, sensor_data] : _meas_model_data)
  {
    for (auto const& [reference_id, distances] : sensor_data._distance_to_reference)
    {
      Vector2 tot_distance = Vector2::Zero();
      Vector2 tot_normed_distance = Vector2::Zero();
      double tot_vabs_distance = 0;
      std::size_t num_distances = 0;
      Vector2 max = Vector2::Zero();
      for (auto const& [time, values] : distances)
      {
        auto const& [dist, normed_dist, vabs] = values;
        if (dist.norm() > 0)
        {
          tot_distance += dist;
          tot_normed_distance += normed_dist;
          tot_vabs_distance += vabs;
          num_distances++;
        }
        if (dist.norm() > max.norm())
        {
          max = dist;
        }
      }
      Vector2 mean = tot_distance / num_distances;
      Vector2 normed_mean = tot_normed_distance / num_distances;
      Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
      Eigen::Matrix2d normed_cov = Eigen::Matrix2d::Zero();
      for (auto const& [time, values] : distances)
      {
        auto const& [dist, normed_dist, vabs] = values;
        if (dist.norm() > 0)
        {
          cov += (dist - mean) * (dist - mean).transpose();
          normed_cov += (normed_dist - normed_mean) * (normed_dist - normed_mean).transpose();
        }
      }
      cov *= 1.0 / (static_cast<double>(num_distances) - 1);
      normed_cov *= 1.0 / (static_cast<double>(num_distances) - 1);
      if (tot_distance.norm() > 0)
      {
        std::stringstream ss;
        ss << "---- Sensor: " << model_id.value_ << " \t --> Reference: " << reference_id.value_ << "\nMean:\n"
           << mean << "\n\nMax:\n"
           << max << "\n\nV:\n"
           << tot_vabs_distance / static_cast<double>(num_distances) << "\n\nCov:\n"
           << cov << "\n\nNormed Cov:\n"
           << normed_cov;

        ImGui::Text("%s", ss.str().c_str());
      }
    }
  }
}

void Vizu::show()
{
  std::scoped_lock lock(add_data_mutex, _manager->_params_mutex);
  ZoneScopedN("gui::render");
  if (ImGui::Begin("Main Window"))
  {
    ImGui::PushItemWidth(200);
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
      if (_manager->params().filter.enable)
      {
        if (ImGui::BeginTabItem("Self Assessment"))
        {
          show_assessment();
          ImGui::EndTabItem();
        }
      }
      if (_manager->params().ttt_filter.enable and _manager->params().ttt_filter.type == TTT_FILTER_TYPE::EVAL)
      {
        if (ImGui::BeginTabItem("Evaluation"))
        {
          show_evaluation();
          ImGui::EndTabItem();
        }
      }
      if (ImGui::BeginTabItem("Params"))
      {
        show_params();
        ImGui::EndTabItem();
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
  ImGui::End();
}

int Vizu::viz_loop()
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
  std::string name = _manager->params().name;
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

  while (not glfwWindowShouldClose(window) and _manager->params().show_gui)
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
      //      should_reset_layout = false;
      //      ImGui::DockBuilderRemoveNode(dockspace_id);                             // clear out existing layout
      ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);  // add empty node
      ImGui::DockBuilderSetNodeSize(dockspace_id, viewport->Size);

      // specify
      //      ImGuiID left = 0;
      //      ImGuiID remaining = 0;
      //      ImGuiID right_top = 0;
      //      ImGuiID right_bottom = 0;
      //      ImGui::DockBuilderSplitNode(dockspace_id, ImGuiDir_Left, 0.75, &left, &remaining);
      //      ImGui::DockBuilderSplitNode(remaining, ImGuiDir_Up, 0.5, &right_top, &right_bottom);
      //      ImGui::DockBuilderDockWindow("Patches", left);
      //      ImGui::DockBuilderDockWindow("Heuristics", right_top);
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
  _manager->next_params().show_gui = false;
  return 0;
}

}  // namespace ttb