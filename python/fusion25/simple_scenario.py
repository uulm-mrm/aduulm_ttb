#!/usr/bin/env python
# %%
"""
=========================================================================
2 - Multi Target - two target tracking and clutter tutorial with one sensor
=========================================================================
"""

from itertools import tee

import numpy as np
from datetime import timedelta, datetime

import matplotlib

matplotlib.use('TkAgg')


# %%
"""
==============================
CREATING GROUND TRUTH SCENARIO
==============================
"""

# Simulate a target
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.state import GaussianState
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity, RandomWalk
from ordered_set import OrderedSet

# np.random.seed(1991)

truths = OrderedSet()

q_x = q_y = 2
qz = 0
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(q_x),    # [x, v_x,
                                                          ConstantVelocity(q_y),    #  y, v_y,
                                                          RandomWalk(qz)])           #  z]

start_time = datetime(1970, 1, 1, 0, 0, 0)

truth = [GroundTruthState([100, -10, 100, -10, 0], timestamp=start_time),
         GroundTruthState([100, -10, -100, 10, 0], timestamp=start_time),
         GroundTruthState([-100, 10, -100, 10, 0], timestamp=start_time),
         GroundTruthState([-100, 10, 100, -10, 0], timestamp=start_time),
         GroundTruthState([100, -10, 0, 0, 0], timestamp=start_time),
         GroundTruthState([0, 0, -100, 10, 0], timestamp=start_time),
         GroundTruthState([0, 0, 100, -10, 0], timestamp=start_time),
         GroundTruthState([-100, 10, 0, 0, 0], timestamp=start_time),]

number_of_steps = 120
timestep_size = timedelta(milliseconds=100)

from stonesoup.simulator.simple import MultiTargetGroundTruthSimulator
ground_truth_simulator = MultiTargetGroundTruthSimulator(
    transition_model=transition_model,
    initial_state=GaussianState([0, 0, 0, 0, 0],
                                np.diag([2, 0.1, 2, 0.1, 0]),
                                timestamp=start_time),
    birth_rate= 0,
    death_probability= 0,
    number_steps=number_of_steps,
    timestep=timestep_size,
    preexisting_states=[t.state_vector for t in truth],
    initial_number_targets=0
)

ground_truths, gt_sim_clutter, *gt_sims = tee(ground_truth_simulator, 3)

# %%
"""
========================================
CREATING ONE SENSOR PLATFORM WITH CLUTTER
========================================
"""
from stonesoup.models.measurement.linear import LinearGaussian

from stonesoup.models.clutter.clutter import ClutterModel


from stonesoup_bridge.bridge import Sensor_TTB

cov = 2
position_mapping = (0, 2)
measurement_model = LinearGaussian(
    ndim_state=5,
    mapping=(0, 2),
    noise_covar=np.array([[cov, 0],
                          [0, cov]])
    )


# Define the clutter model which will be the same for both sensors
clutter_model = ClutterModel(
    clutter_rate=1,
    distribution=np.random.default_rng().uniform,
    dist_params=((-100, 100), (-100, 100))
    # dist_params=(x_range, y_range)
)


sensor = Sensor_TTB(measurement_model=measurement_model,
                    clutter_model=clutter_model,
                    detection_probability=0.9)

# Import the platform to place the sensors on
from stonesoup.platform.base import FixedPlatform
from stonesoup.types.state import GaussianState

# Instantiate the first sensor platform and add the sensor
sensor_platform = FixedPlatform(
    states=GaussianState([0, 0, 0, 0],
                         np.diag([1, 0, 1, 0])),
    position_mapping=(0, 2),
    sensors=[sensor])


from stonesoup.simulator.platform import PlatformDetectionSimulator


platform_simulator = PlatformDetectionSimulator(
    groundtruth=gt_sims[0],
    platforms=[sensor_platform])

detector, platform_simulator = tee(platform_simulator, 2)

# %%
"""
==========================
DEFINE STONE SOUP TRACKER
==========================
"""

# We use a Distance hypothesiser
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis

# Use a GNN 2D assignment, time deleter and initiator
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.deleter.time import UpdateTimeDeleter
from stonesoup.initiator.simple import MultiMeasurementInitiator, GaussianParticleInitiator

# Load the EKF components
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.predictor.kalman import KalmanPredictor

# Load the multitarget tracker
from stonesoup.tracker.simple import MultiTargetTracker

# define a track deleter based on time measurements
deleter = UpdateTimeDeleter(timedelta(seconds=0.3), delete_last_pred=False)
from stonesoup.deleter.error import CovarianceBasedDeleter
covariance_limit_for_delete = 2
# deleter = CovarianceBasedDeleter(covar_trace_thresh=covariance_limit_for_delete)
#Extended Kalman Filter

# load the Extended Kalman filter predictor and updater
EKF_predictor = KalmanPredictor(transition_model)
EKF_updater = KalmanUpdater(measurement_model=None)

# define the hypothesiser
hypothesiser_EKF = DistanceHypothesiser(
    predictor=EKF_predictor,
    updater=EKF_updater,
    measure=Mahalanobis(),
    missed_distance=5)

# define the distance data associator
data_associator_EKF = GNNWith2DAssignment(hypothesiser_EKF)

EKF_initiator = MultiMeasurementInitiator(
    prior_state=GaussianState([[0], [0], [0], [0], [0]],
                              np.diag([0, 1, 0, 1, 0])),
    measurement_model=None,
    deleter=deleter,
    updater=EKF_updater,
    data_associator=data_associator_EKF,
    min_points=5)

# Instantiate each of the Trackers, without specifying the detector
EKF_tracker = MultiTargetTracker(
    initiator=EKF_initiator,
    deleter=deleter,
    data_associator=data_associator_EKF,
    updater=EKF_updater,
    detector=detector)


# %%
"""
======================
DEFINE METRICS MANAGER 
======================
"""

# Define a data associator between the tracks and the truths
from stonesoup.dataassociator.tracktotrack import TrackToTruth

# load a multi-metric manager
from stonesoup.metricgenerator.manager import MultiManager

c=10
p=2


from stonesoup.metricgenerator.ospametric import GOSPAMetric, OSPAMetric
# gospa_gen_name = "GOSPA-TTB"
gospa_ttb = GOSPAMetric(c=c, p=p, generator_name="GOSPA-TTB",
                        tracks_key="tracks_ttb", truths_key="truths", switching_penalty=1)


# GOSPA Stone Soup
gospa_EKF = GOSPAMetric(c=c, p=p, generator_name='GOSPA-Stone-Soup',
                            tracks_key='tracks',  truths_key='truths', switching_penalty=1)


# Use the track associator
associator = TrackToTruth(association_threshold=30)

# Use a metric manager to deal with the various metrics
metric_manager = MultiManager([gospa_ttb, gospa_EKF],
                              associator)

# %%
"""
==============
RUN SIMULATION
==============
"""

print("Run simulations...")

timesteps=[]

detections = []
all_detections = []
truths = set()
tracks = set()

detection_iter = iter(platform_simulator)
tracker_iter = iter(EKF_tracker)

for t in range(number_of_steps):  # loop over the various time-steps
    print(t)

    time, gt = next(ground_truths)
    timesteps.append(time)
    truths.update(gt)


    _, det = next(detection_iter)
    detections.append(det)
    all_detections.append([det])

    # Run the Extended Kalman filter
    _, track = next(tracker_iter)
    # print(track)
    tracks.update(track)


# Add data to the metric manager
metric_manager.add_data({'tracks': tracks}, overwrite=False)

metric_manager.add_data({'truths': truths}, overwrite=False)

# %%
"""
=============================
USING THE BRIDGE FOR TRACKING
=============================
"""
from time import sleep

all_measurements = all_detections

from stonesoup_bridge import bridge

tracks_ttb, _ = bridge.track_via_ttb(transition_model=transition_model,
                                  measurement_model=measurement_model,
                                  all_measurements=all_measurements,
                                  sensors=[sensor],
                                  timesteps=timesteps,
                                  position_mapping=position_mapping)
import time
time.sleep(1)
print("Number of Stone Soup Tracks:", len(tracks))
print("Number of TTB Tracks:", len(tracks_ttb))
print("Number of Truths:", len(truths))
# sleep(4)
metric_manager.add_data({'tracks_ttb' : tracks_ttb}, overwrite=False)
metrics = metric_manager.generate_metrics()

# %%
"""
==========================================
PLOTTING ALL TRACKS, TRUTHS AND DETECTIONS
==========================================
"""

from stonesoup.plotter import Plotterly
# sleep(5) # sleep for 2s to prevent buffer overflow

print("Plotting...")
plotter = Plotterly()

plotter.plot_measurements(detections, [0, 2],
                          measurements_label='Sensor  measurements')

plotter.plot_sensors({sensor_platform}, [0, 1],
                     marker=dict(color='black', symbol='1', size=10))

plotter.plot_ground_truths(truths, [0, 2])

plotter.plot_tracks(tracks, [0, 2], line=dict(color='cyan'), label='Stone Soup Tracks')
plotter.plot_tracks(tracks_ttb, [0, 2], line=dict(color='#A32638'), label='TTB Tracks')

plotter.fig.show(renderer="browser")
# %%
"""
==================
PLOT GOSPA METRICS
==================
"""
import matplotlib
matplotlib.use('TkAgg')

# plt = bridge.plot_gospa(metrics, gospa_gen_name=["GOSPA-TTB", "GOSPA-Stone-Soup"])
plt = bridge.plot_all_gospas(metrics, gospa_gen_name=["GOSPA-TTB", "GOSPA-Stone-Soup"])
plt.show()
