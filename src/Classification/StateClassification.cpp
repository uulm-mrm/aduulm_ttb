#include "tracking_lib/Classification/StateClassification.h"
// ######################################################################################################################
#include "tracking_lib/Classification/MeasClassification.h"
#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/Measurements/Measurement.h"
// ######################################################################################################################
#include <sstream>

namespace ttb::classification
{

StateClassification::StateClassification(TTBManager* manager)
  : _manager(manager), _sl_evid{ [&] {
    std::map<CLASS, double> evid;
    for (auto const clazz : _manager->params().state.classification.classes)
    {
      evid[clazz] = 1;
    }
    return evid;
  }() }
{
}

// Updates the current (state) classification based on (measurement) classification
void StateClassification::update(MeasClassification const& other)
{
  if (other.getEstimate() == CLASS::NOT_CLASSIFIED)
  {
    return;
  }
  std::map<CLASS, double> m, v;
  const double S = std::accumulate(
      _sl_evid.begin(), _sl_evid.end(), 0.0, [](auto prev, auto& map_entry) { return prev + map_entry.second; });
  for (auto const class_evid : _sl_evid)
  {
    auto [prob_other, found] = other.getProb(class_evid.first);
    if (found)
    {
      m[class_evid.first] = (prob_other + class_evid.second) / (1 + S);
      v[class_evid.first] = (1 + class_evid.second) * (class_evid.second + 2 * prob_other) * 1 / (1 + S) * 1 / (2 + S);
    }
    else
    {  // if typ not found in other -> distribute remaining prob equally
      auto tmp = prob_other / (_manager->params().state.classification.classes.size() - other.getSize());
      m[class_evid.first] = (tmp + class_evid.second) / (1 + S);
      v[class_evid.first] = (1 + class_evid.second) * (class_evid.second + 2 * tmp) * 1 / (1 + S) * 1 / (2 + S);
    }
  }
  double num = 0;
  double denum = 0;
  for (auto clazz : _manager->params().state.classification.classes)
  {
    num += (m[clazz] - v[clazz]) * m[clazz] * (1 - m[clazz]);
    denum += (v[clazz] - m[clazz]) * (v[clazz] - m[clazz]) * m[clazz] * (1 - m[clazz]);
  }
  for (auto& class_evid : _sl_evid)
  {
    class_evid.second = m[class_evid.first] * num / denum;
  }
}

// merge the current (state) classification with other (state) classification
// as far as I know this happens only within Track... when approx./marginalize the GLMB to LMB
void StateClassification::merge(const StateClassification& other)
{
  LOG_DEB("Merge State Classifications");

  for (auto& class_evid : _sl_evid)
  {
    if (other._sl_evid.find(class_evid.first) != other._sl_evid.end())
    {
      class_evid.second = (class_evid.second + other._sl_evid.at(class_evid.first)) / 2;
    }
  }
}

void StateClassification::discount(double alpha)
{
  assert([&] {  // NOLINT
    if (alpha < 0 or alpha > 1)
    {
      LOG_FATAL("invalid discount value alpha: " << alpha << ". Not in [0, 1]");
      return false;
    }
    return true;
  }());
  for (auto& class_evid : _sl_evid)
  {
    class_evid.second = (class_evid.second - 1) * alpha + 1;
  }
}

void StateClassification::discount()
{
  discount(_manager->params().state.classification.discount_factor_prediction);
}

CLASS StateClassification::getEstimate() const
{
  if (auto const max_it =
          std::ranges::max_element(_sl_evid, [](auto const& a, auto const& b) { return a.second < b.second; });
      max_it != _sl_evid.end())
  {
    return max_it->first;
  }
  return CLASS::NOT_CLASSIFIED;
}

std::size_t StateClassification::getSize() const
{
  return _manager->params().state.classification.classes.size();
}

std::string StateClassification::toString() const
{
  std::string out;
  for (auto const [clazz, evid] : _sl_evid)
  {
    out += to_string(clazz) + ": " + std::to_string(evid) + "\n";
  }
  return out;
}

double StateClassification::getProb(CLASS typ) const
{
  const double S = std::accumulate(
      _sl_evid.begin(), _sl_evid.end(), 0.0, [](auto prev, auto& map_entry) { return prev + map_entry.second; });
  if (auto const it = _sl_evid.find(typ); it != _sl_evid.end())
  {
    return it->second / S;
  }
  return 0;
}

void StateClassification::setSLEvid(std::map<CLASS, double> probs)
{
  for (CLASS const& elem : _manager->params().state.classification.classes)
  {
    if (probs.contains(elem))
    {
      _sl_evid[elem] = 1 + probs.at(elem);
    }
    else
    {
      _sl_evid[elem] = 1;
    }
  }
}

}  // namespace ttb::classification
