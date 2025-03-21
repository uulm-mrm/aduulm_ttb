//
// Created by hermann on 3/22/24.
//

#ifndef NEW_INTERFACES_SANDBOX_TTTUNCORRELATEDTRACKS_H
#define NEW_INTERFACES_SANDBOX_TTTUNCORRELATEDTRACKS_H

#include "tracking_lib/TTBTypes/TTBTypes.h"
#include "tracking_lib/MultiObjectStateDistributions/LMBDistribution.h"
#include "tracking_lib/Measurements/SensorInformation.h"
#include "tracking_lib/StateModels/BaseStateModel.h"
#include "tracking_lib/MeasurementModels/BaseMeasurementModel.h"
#include "tracking_lib/Transformations/Transformation.h"
#include "tracking_lib/MeasurementModels/GaussianMeasurementModel.h"
#include "tracking_lib/TTBManager/TTBManager.h"
#include "tracking_lib/Misc/AngleNormalization.h"
#include "tracking_lib/TTBTypes/Params.h"
#include "tracking_lib/TTTFilters/TTTHelpers.h"

namespace ttb::uncorrelated_t2t_fusion
{
class TTTUncorrelatedTracks
{
public:
  TTTUncorrelatedTracks(TTBManager* manager,
                        TTTUncorrelatedTracksParams params,
                        std::vector<tttfusion::TracksSensor>&& lmbs);

  struct PseudoPredictedMeasurement
  {
    Vector z_pred;          ///< the predicted Measurement mean
    double detection_prob;  ///< the detection prob of the predicted Measurement
  };

  struct TTTUpdateInfo
  {
    Matrix T;         ///< T = P1+P2
    Matrix iT;        /// inverse of T
    Vector residuum;  ///< The residuum of both track means
    double distance;  ///< distance value considering the covariances of both tracks (comparable to the MHD2), dist =
                      ///< residuum^T*T^(-1)*residuum
    bool isGated;     ///< tracks belong together or not
  };

  [[nodiscard]] double getDistance(const Vector& residuum, const Matrix& T, const Matrix& iT) const
  {
    // Track to track fusion equations for uncorrelated tracks (Bar-Shalom: A Handbook of algorithms, p. 580-583)
    return residuum.transpose() * iT * residuum;
  };

  [[nodiscard]] Vector getResiduum(Vector const& x1, Vector const& x2, std::optional<std::size_t> rot_Ind) const
  {
    Vector res = x2 - x1;
    if (rot_Ind.has_value())
    {
      angles::normalizeAngle(res(rot_Ind.value()));
    }
    return res;
  };

  using GatingDistanceCache = std::map<StateModelId, double>;

  [[nodiscard]] TTTUpdateInfo calcUpdateInfoT2T(Measurement const& meas,        // pseudo meas
                                                BaseDistribution const* track,  // pseudo pred meas
                                                double const& gatingDistance,
                                                std::optional<std::size_t> rot_Ind) const
  {
    // Track to track fusion equations for uncorrelated tracks (Bar-Shalom: A Handbook of algorithms, p. 580-583)
    //    ZoneScopedNC("GaussianMeasurementModel::calcUpdateInfo", tracy_color);
    Vector const mean1 = meas._dist->mean();
    Vector const mean2 = track->mean();
    Matrix const cov1 = meas._dist->covariance();
    Matrix const cov2 = track->covariance();

    TTTUpdateInfo info;
    info.T = cov1 + cov2;
    info.iT = info.T.inverse();
    info.residuum = getResiduum(mean2, mean1, rot_Ind);
    info.distance = getDistance(info.residuum, info.T, info.iT);
    info.isGated = info.distance > gatingDistance;
    return info;
  };

  /// Calculates the spatial likelihood
  /// This method calculates the spatial likelihood of a measurement using the MHD and the
  /// Innovation matrix. Special versions (which e.g. do not use the normalization constant)
  /// have to be implemented in derived classes.
  [[nodiscard]] static double getSpatialLikelihood(double distance, double gatingConf, const Matrix& T)
  {
    return exp((-0.5) * distance) / std::sqrt(2 * M_PI * (T).determinant()) / gatingConf;
  };

  [[nodiscard]] static double getMeasurementLikelihood(double distance, double gatingConf, const Matrix& T)
  {
    return getSpatialLikelihood(distance, gatingConf, T);
  }

  /**
   * Adds remaining measurements as new birth tracks to density
   * @param measurementContainer
   * @param rzMap contains information
   * @param num_empty_updates
   * @return
   */
  [[nodiscard]] tttfusion::TracksSensor addPseudoBirthTracks(MeasurementContainer const& measurementContainer,
                                                             std::map<MeasurementId, double> const& rzMap,
                                                             std::size_t num_empty_updates,
                                                             bool gatingActivated = true,
                                                             bool deactivate_parallelization = false) const;

  /**
   * Does num_empty_updates of single sensor lmb updates (ic version) for an empty measurement container
   * @param dist lmb dist, which shall be updated
   * @param num_empty_updates number of empty updates which shall be done
   * @param id id of the measurement model
   * @return updated lmb distribution
   */
  [[nodiscard]] LMBDistribution doEmptyLMBUpdate(LMBDistribution&& dist,
                                                 std::size_t num_empty_updates,
                                                 MeasModelId id,
                                                 bool gatingActivated = true,
                                                 bool deactivate_parallelization = false) const;

  /**
   * overwrites covariance if given in config
   * @param lmbDist lmb distribution
   */
  void overwriteCovariances(LMBDistribution& lmbDist) const;

  [[nodiscard]] LMBDistribution fpm_version(LMBDistribution lmbDist,
                                            std::vector<MeasurementContainer>&& measContainerList);

  [[nodiscard]] std::pair<LMBDistribution, std::vector<tttfusion::TracksSensor>>
  fpmUpdateAndFusion(LMBDistribution&& lmbdist,
                     std::vector<MeasurementContainer>&& measContainerList,
                     bool gatingActivated = true,
                     bool deactivate_parallelization = false) const;

  [[nodiscard]] LMBDistribution ic_version(LMBDistribution lmbDist,
                                           std::vector<MeasurementContainer>&& measContainerList,
                                           bool gatingActivated = true,
                                           bool deactivate_parallelization = false);

  /**
   * performs fusion of birth lmb distributions
   * @return fused birth lmb distribution
   */
  [[nodiscard]] LMBDistribution fuseTracksOfDifferentSensors();

  [[nodiscard]] LMBDistribution fusion_without_t2tAssociation_before();

  [[nodiscard]] LMBDistribution fusion_with_t2tAssociation_before();

  /// calculate the state innovation for all tracks for all other birthTracks (t2t fusion of uncorrelated tracks) with
  /// respect to the fov of the sensors
  [[nodiscard]] LMBDistribution calcInnovationsTracks(LMBDistribution&& lmbdist,
                                                      MeasurementContainer const& Z,
                                                      bool gatingActivated = true,
                                                      bool deactivate_parallelization = false) const;

  [[nodiscard]] Innovation calculateInnovationT2T(MeasurementContainer const& measContainer,
                                                  State const& dist,
                                                  bool gatingActivated = true) const;

  TTBManager* _manager;
  TTTUncorrelatedTracksParams _t2t_params;
  // each vector element contains a vector of tracks, which shall be fused with the other track vectors
  std::vector<tttfusion::TracksSensor>&& _tracksVec;
  std::size_t _numTrackers;  // for fpm stuff it is always num_sensors-1!
  std::map<MeasurementId, StateModelId> _measId2stateModelId;
};

}  // namespace ttb::uncorrelated_t2t_fusion

#endif  // NEW_INTERFACES_SANDBOX_TTTUNCORRELATEDTRACKS_H
