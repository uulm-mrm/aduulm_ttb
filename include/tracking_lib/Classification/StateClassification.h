#pragma once

#include "tracking_lib/TTBTypes/Params.h"
#include "tracking_lib/TTBTypes/TTBTypes.h"
#include <map>

namespace ttb::classification
{

class MeasClassification;

/// Represent the classification of a State, see
/// A. Scheible, T. Griebel, M. Herrmann, C. Hermann and M. Buchholz, "Track Classification for Random Finite Set Based
/// Multi-Sensor Multi-Object Tracking," 2023 IEEE Symposium Sensor Data Fusion and International Conference on
/// Multisensor Fusion and Integration (SDF-MFI), Bonn, Germany, 2023, pp. 1-8, doi: 10.1109/SDF-MFI59545.2023.10361438.
class StateClassification
{
public:
  explicit StateClassification(TTBManager* manager);
  //// update with a MeasClassification
  void update(MeasClassification const& other);
  /// merge another StateClassification into this
  void merge(StateClassification const& other);
  /// discount this classification.
  /// alpha in [0, 1] with alpha=1 the discount has no effect, with alpha=0 it discount to no information
  void discount(double alpha);
  /// discount with alpha as specified in the parameters
  void discount();
  /// return an estimation of the classification
  CLASS getEstimate() const;
  /// number of potential classes
  std::size_t getSize() const;
  /// probability of the given class
  double getProb(CLASS typ) const;
  /// set the subjective logic evidence of all classes
  void setSLEvid(std::map<CLASS, double> probs);
  /// return a string representation of the classification
  std::string toString() const;
  TTBManager* _manager;
  std::map<CLASS, double> _sl_evid;
};

}  // namespace ttb::classification
