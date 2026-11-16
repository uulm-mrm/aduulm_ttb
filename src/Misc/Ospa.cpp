#include "tracking_lib/Misc/Ospa.h"
#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/States/State.h"
#include "tracking_lib/Misc/HungarianMethod.h"

#include <tracy/tracy/Tracy.hpp>

namespace ttb
{

template OspaResult eval_ospa(TTBManager<double> const& manager,
                              std::vector<State<double>> const& tracks,
                              std::vector<State<double>> const& ground_truths,
                              double c,
                              double p,
                              std::vector<COMPONENT> const& components);
template OspaResult eval_ospa(TTBManager<AD> const& manager,
                              std::vector<State<AD>> const& tracks,
                              std::vector<State<AD>> const& ground_truths,
                              double c,
                              double p,
                              std::vector<COMPONENT> const& components);
template <typename UDT>
OspaResult eval_ospa(TTBManager<UDT> const& manager,
                     std::vector<State<UDT>> const& tracks,
                     std::vector<State<UDT>> const& ground_truths,
                     double c,
                     double p,
                     std::vector<COMPONENT> const& components)
{
  ZoneScopedN("ospa");
  if (tracks.empty() and ground_truths.empty())
  {
    return { 0 };
  }
  if (tracks.size() > ground_truths.size())
  {
    return eval_ospa(manager, ground_truths, tracks, c, p, components);
  }
  assert(tracks.size() <= ground_truths.size());
  Matrix<double> cost(tracks.size(), ground_truths.size());
  for (auto const& [i, track] : std::views::enumerate(tracks))
  {
    Vector<UDT> const track_xy = [&] -> Vector<UDT> {
      auto const& [model, dist] = track.getEstimate();
      if (auto ind = manager.state_comps(model).indexOf(components); ind.has_value())
      {
        return dist.mean()(ind.value());
      }
      return Vector<UDT>::Ones(components.size()) * std::numeric_limits<double>::infinity();
    }();
    for (auto const& [j, gt] : std::views::enumerate(ground_truths))
    {
      Vector<UDT> const gt_xy = [&] -> Vector<UDT> {
        auto const& [model, dist] = gt.getEstimate();
        if (auto ind = manager.state_comps(model).indexOf(components); ind.has_value())
        {
          return dist.mean()(ind.value());
        }
        return Vector<UDT>::Ones(components.size()) * std::numeric_limits<double>::infinity();
      }();
      double diff = to_val(track_xy - gt_xy).norm();
      if (std::isnan(diff))
      {
        diff = std::numeric_limits<double>::infinity();
      }
      diff = std::min(diff, c);
      assert(std::isfinite(diff));
      cost(i, j) = diff;
    }
  }
  assert(cost.allFinite());

  auto [ass, ass_cost] = hungarian::solve(cost);
  if (tracks.empty())
  {
    ass_cost = 0;
  }
  OspaResult const ospa{ std::pow(1.0 / ground_truths.size() *
                                      (std::pow(ass_cost, p) + std::pow(c, p) * (ground_truths.size() - tracks.size())),
                                  1 / p) };
  assert([&] {
    if (not std::isfinite(ospa.ospa))
    {
      LOG_FATAL("not finite");
      LOG_FATAL(cost);
      return false;
    }
    return true;
  }());
  return ospa;
}

/// The base distance between two trajectories.
/// The first are the tracked one with potential gaps, the second is the gt state
template <typename UDT>
[[nodiscard]] UDT trajectory_distance(TTBManager<UDT> const& manager,
                                      SlidingWindow<std::optional<State<UDT>>> const& track_trajectory,
                                      SlidingWindow<std::optional<State<UDT>>> const& gt_trajectory,
                                      double c = 10,
                                      std::vector<COMPONENT> const& components = { COMPONENT::POS_X, COMPONENT::POS_Y })
{
  ZoneScopedN("ospa2::trajectory_distance");
  UDT trajectory_dist = 0;
  std::vector<Time> gt_times;
  for (auto const& state : gt_trajectory.data_buffer)
  {
    gt_times.push_back(state.time);
  }
  std::vector<Time> track_times;
  for (auto const& state : track_trajectory.data_buffer)
  {
    track_times.push_back(state.time);
  }
  std::vector<Time> common_times;
  std::ranges::sort(gt_times);
  std::ranges::sort(track_times);

  std::ranges::set_intersection(gt_times, track_times, std::back_inserter(common_times));
  if (common_times.empty())
  {
    return UDT{ std::numeric_limits<double>::infinity() };
  }
  for (Time time : common_times)
  {
    assert([&] {
      if (not std::ranges::contains(gt_trajectory.data_buffer, time, &TimedData<std::optional<State<UDT>>>::time))
      {
        LOG_FATAL("Time " << time << "not in gt data buffer");
        for (auto const& data : gt_trajectory.data_buffer)
        {
          if (data.data.has_value())
          {
            LOG_FATAL(std::format("Time {}, sTate: {}", data.time, data.data->toString()));
          }
          else
          {
            LOG_FATAL(std::format("Time {}, sTate: empty", data.time));
          }
        }
        return false;
      }
      return true;
    }());
    assert([&] {
      if (not std::ranges::contains(track_trajectory.data_buffer, time, &TimedData<std::optional<State<UDT>>>::time))
      {
        LOG_FATAL("Time " << time << "not in track trajectory data buffer");
        for (auto const& data : track_trajectory.data_buffer)
        {
          if (data.data.has_value())
          {
            LOG_FATAL(std::format("Time {}, sTate: {}", data.time, data.data->toString()));
          }
          else
          {
            LOG_FATAL(std::format("Time {}, sTate: empty", data.time));
          }
        }
        return false;
      }
      return true;
    }());
    std::optional<State<UDT>> const& gt =
        std::ranges::find(gt_trajectory.data_buffer, time, &TimedData<std::optional<State<UDT>>>::time)->data;
    std::optional<State<UDT>> const& track =
        std::ranges::find(track_trajectory.data_buffer, time, &TimedData<std::optional<State<UDT>>>::time)->data;
    if (not gt.has_value() and not track.has_value())
    {
      trajectory_dist += 0;
    }
    else if (not gt.has_value() and track.has_value())
    {
      trajectory_dist += c;
    }
    else if (gt.has_value() and not track.has_value())
    {
      trajectory_dist += c;
    }
    else
    {
      assert(track.value()._time == gt.value()._time);
      Vector<UDT> const track_xy = [&] -> Vector<UDT> {
        auto const& [model, dist] = track.value().getEstimate();
        if (auto ind = manager.state_comps(model).indexOf(components); ind.has_value())
        {
          return dist.mean()(ind.value());
        }
        return Vector<UDT>::Ones(components.size()) * std::numeric_limits<double>::infinity();
      }();

      Vector<UDT> const gt_xy = [&] -> Vector<UDT> {
        auto const& [model, dist] = gt.value().getEstimate();
        if (auto ind = manager.state_comps(model).indexOf(components); ind.has_value())
        {
          return dist.mean()(ind.value());
        }
        return Vector<UDT>::Ones(components.size()) * std::numeric_limits<double>::infinity();
      }();
      double diff = (to_val(track_xy - gt_xy)).norm();
      if (std::isnan(diff))
      {
        diff = std::numeric_limits<double>::infinity();
      }
      trajectory_dist += std::min(c, diff);
    }
  }
  return trajectory_dist / common_times.size();
}

template Ospa2Result eval_ospa2(TTBManager<double> const& manager,
                                std::map<Label, SlidingWindow<std::optional<State<double>>>> const& gt_trajectories,
                                std::map<Label, SlidingWindow<std::optional<State<double>>>> const& track_trajectories,
                                double c,
                                double p,
                                std::vector<COMPONENT> const& components);
template Ospa2Result eval_ospa2(TTBManager<AD> const& manager,
                                std::map<Label, SlidingWindow<std::optional<State<AD>>>> const& gt_trajectories,
                                std::map<Label, SlidingWindow<std::optional<State<AD>>>> const& track_trajectories,
                                double c,
                                double p,
                                std::vector<COMPONENT> const& components);
template <typename UDT>
Ospa2Result eval_ospa2(TTBManager<UDT> const& manager,
                       std::map<Label, SlidingWindow<std::optional<State<UDT>>>> const& gt_trajectories,
                       std::map<Label, SlidingWindow<std::optional<State<UDT>>>> const& track_trajectories,
                       double c,
                       double p,
                       std::vector<COMPONENT> const& components)
{
  ZoneScopedN("ospa2");
  if (gt_trajectories.size() < track_trajectories.size())
  {
    return eval_ospa2(manager, track_trajectories, gt_trajectories, c, p, components);
  }
  assert(track_trajectories.size() <= gt_trajectories.size());
  Matrix<double> cost(track_trajectories.size(), gt_trajectories.size());
  for (auto const& [i, track] : std::views::enumerate(track_trajectories))
  {
    for (auto const& [j, gt] : std::views::enumerate(gt_trajectories))
    {
      cost(i, j) = to_val(trajectory_distance(manager, track.second, gt.second, c, components));
    }
  }
  assert(cost.allFinite());
  auto [ass, ass_cost] = hungarian::solve(cost);
  if (track_trajectories.empty())
  {
    ass_cost = 0;
  }
  Ospa2Result const ospa2{ std::pow(
      1.0 / gt_trajectories.size() *
          (std::pow(ass_cost, p) + std::pow(c, p) * (gt_trajectories.size() - track_trajectories.size())),
      1 / p) };
  return ospa2;
}

template GospaResult eval_gospa(TTBManager<double> const& manager,
                                std::vector<State<double>> const& tracks,
                                std::vector<State<double>> const& ground_truths,
                                double c,
                                double p,
                                std::vector<COMPONENT> const& components);
template GospaResult eval_gospa(TTBManager<AD> const& manager,
                                std::vector<State<AD>> const& tracks,
                                std::vector<State<AD>> const& ground_truths,
                                double c,
                                double p,
                                std::vector<COMPONENT> const& components);
template <typename UDT>
GospaResult eval_gospa(TTBManager<UDT> const& manager,
                       std::vector<State<UDT>> const& tracks,
                       std::vector<State<UDT>> const& ground_truths,
                       double c,
                       double p,
                       std::vector<COMPONENT> const& components)
{
  ZoneScopedN("gospa");
  std::size_t card_gt = ground_truths.size();
  std::size_t card_est = tracks.size();
  if (ground_truths.size() < tracks.size())
  {
    GospaResult const res = eval_gospa(manager, ground_truths, tracks, c, p, components);
    return GospaResult{ res.false_cost, res.missed_cost, res.loc_cost, res.gospa, card_gt, card_est };
  }
  assert(tracks.size() <= ground_truths.size());
  Matrix<double> cost = std::pow(c, p) * Matrix<double>::Ones(tracks.size(), ground_truths.size());
  for (auto const& [i, track] : std::views::enumerate(tracks))
  {
    Vector2 const track_xy = [&] -> Vector2 {
      auto const& [model, dist] = track.getEstimate();
      if (auto ind = manager.state_comps(model).indexOf(components); ind.has_value())
      {
        return to_val(dist.mean())(ind.value());
      }
      return Vector2::Ones() * std::numeric_limits<double>::infinity();
    }();
    for (auto const& [j, gt] : std::views::enumerate(ground_truths))
    {
      Vector2 const gt_xy = [&] -> Vector2 {
        auto const& [model, dist] = gt.getEstimate();
        if (auto ind = manager.state_comps(model).indexOf(components); ind.has_value())
        {
          return to_val(dist.mean())(ind.value());
        }
        return Vector2::Ones() * std::numeric_limits<double>::infinity();
      }();
      double diff = (track_xy - gt_xy).norm();
      if (std::isnan(diff))
      {
        diff = std::numeric_limits<double>::infinity();
      }
      diff = std::pow(diff, p);
      diff = std::min(diff, std::pow(c, p));
      assert(std::isfinite(diff));
      cost(i, j) = diff;
    }
  }
  auto [ass, ass_cost] = hungarian::solve(cost);
  if (tracks.empty())
  {
    ass_cost = 0;
  }
  std::size_t num_assigned = 0;
  double loc_cost = 0;
  for (Index track_i = 0; track_i < ass.rows(); ++track_i)
  {
    if (double loc = cost(track_i, ass(track_i)); loc < std::pow(c, p))
    {
      ++num_assigned;
      loc_cost += loc;
    }
  }
  double const fac = std::pow(c, p) / 2;
  double const missed = (ground_truths.size() - num_assigned) * fac;
  double const false_t = (tracks.size() - num_assigned) * fac;

  return GospaResult{ std::pow(missed, 1 / p),
                      std::pow(false_t, 1 / p),
                      std::pow(loc_cost, 1 / p),
                      std::pow(missed + false_t + loc_cost, 1 / p),
                      card_gt,
                      card_est };
}

}  // namespace ttb
