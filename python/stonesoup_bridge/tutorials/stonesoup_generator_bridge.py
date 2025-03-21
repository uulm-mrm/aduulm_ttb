#!/usr/bin/env python
# coding: utf-8

"""
==============================================================
Performance comparison between Tracking Toolbox and Stone Soup
==============================================================
"""

# 1. define the targets trajectories and the sensors specifics for collecting the measurements;
# 2. define the various filter components and build the trackers;
# 3. perform the measurement fusion algorithm and run the trackers;
# 4. plot the tracks and the track performances using the metric manager tool available.
#

# %%
# General imports
# ^^^^^^^^^^^^^^^
import numpy as np
# import datetime
from itertools import tee
from stonesoup.models.measurement.linear import LinearGaussian
from datetime import datetime, timedelta

# %%
"""
==============================
CREATING GROUND TRUTH SCENARIO
==============================
"""

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity, RandomWalk



np.random.seed(1991)

q_x = q_y = 2
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(q_x),
                                                          ConstantVelocity(q_y),
                                                          RandomWalk(0)])

from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.state import GaussianState, State

initial_state_mean = StateVector([[0], [0], [0], [0], [0]])
initial_state_covariance = CovarianceMatrix(np.diag([30, 1, 30, 1, 0]))
timestep_size = timedelta(milliseconds=100)
number_of_steps = 200
birth_rate = 0.6
death_probability = 0.05
start_time = datetime(2020, 12, 27, 0, 0, 0)
initial_state = GaussianState(initial_state_mean, initial_state_covariance, start_time)

from stonesoup.simulator.simple import MultiTargetGroundTruthSimulator
ground_truth_simulator = MultiTargetGroundTruthSimulator(
    transition_model=transition_model,
    initial_state=initial_state,
    timestep=timestep_size,
    number_steps=number_of_steps,
    birth_rate=birth_rate,
    death_probability=death_probability
)


# %%
# Generate clutter
# ^^^^^^^^^^^^^^^^
from stonesoup.models.clutter.clutter import ClutterModel

# Define the clutter model which will be the same for both sensors
# Keep the clutter rate low due to particle filter errors
clutter_model = ClutterModel(
    clutter_rate=1,
    distribution=np.random.default_rng().uniform,
    dist_params=((-105, 105), (-105, 105))
    # dist_params=((-105, 105), (-10, 10), (-105, 105), (-10, 10))
)

# Define the clutter area and spatial density for the tracker
# clutter_area = np.prod(np.diff(clutter_model.dist_params))
# clutter_spatial_density = clutter_model.clutter_rate/clutter_area

# %%
# Radar sensor and platform set-up
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
from stonesoup_bridge.bridge import Sensor_TTB

cov = 2
position_mapping = (0, 2)
measurement_model = LinearGaussian(
    ndim_state=5,
    mapping=(0, 2),
    noise_covar=np.array([[cov, 0],
                          [0, cov]])
    # noise_covar = np.identity(4)*cov
    )

sensor_1 = Sensor_TTB(measurement_model=measurement_model,
                            clutter_model=clutter_model)

sensor_2 = Sensor_TTB(measurement_model=measurement_model,
                      clutter_model=clutter_model)


# Import the platform to place the sensors on
from stonesoup.platform.base import FixedPlatform

# Instantiate the first sensor platform and add the sensor
sensor1_platform = FixedPlatform(
    states=GaussianState([-20, 0, 0, 0],
                         np.diag([1, 0, 1, 0])),
    position_mapping=(0, 2),
    sensors=[sensor_1])

sensor2_platform = FixedPlatform(
    states=GaussianState([20, 0, -0, 0],
                         np.diag([1, 0, 1, 0])),
    position_mapping=(0, 2),
    sensors=[sensor_2])

# Load the platform detection simulator - Let's use a simulator for each track
# Instantiate the simulators
from stonesoup.simulator.platform import PlatformDetectionSimulator

ground_truths, *gt_sims = tee(ground_truth_simulator, 3)
platform_simulator1 = PlatformDetectionSimulator(
    groundtruth=gt_sims[0],
    platforms=[sensor1_platform])

platform_simulator2 = PlatformDetectionSimulator(
    groundtruth=gt_sims[1],
    platforms=[sensor2_platform])


# %%
# Stone Soup imports
# ^^^^^^^^^^^^^^^^^^

# We use a Distance hypothesiser
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis

# Use a GNN 2D assignment, time deleter and initiator
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.deleter.time import UpdateTimeDeleter
from stonesoup.initiator.simple import MultiMeasurementInitiator, GaussianParticleInitiator

# Load the EKF components
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.predictor.kalman import ExtendedKalmanPredictor

# Load the multitarget tracker
from stonesoup.tracker.simple import MultiTargetTracker

# Load a detection reader
from stonesoup.feeder.multi import MultiDataFeeder

# %%
# Design the trackers components
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# %%
# Detection reader and track deleter

# Detection reader
detection_reader = MultiDataFeeder([platform_simulator1, platform_simulator2])

# define a track deleter based on time measurements
deleter = UpdateTimeDeleter(timedelta(seconds=0.3), delete_last_pred=False)

# %%
#Extended Kalman Filter

# load the Extended Kalman filter predictor and updater
EKF_predictor = ExtendedKalmanPredictor(transition_model)
EKF_updater = ExtendedKalmanUpdater(measurement_model=None)

# define the hypothesiser
hypothesiser_EKF = DistanceHypothesiser(
    predictor=EKF_predictor,
    updater=EKF_updater,
    measure=Mahalanobis(),
    missed_distance=5)

# define the distance data associator
data_associator_EKF = GNNWith2DAssignment(hypothesiser_EKF)

EKF_initiator = MultiMeasurementInitiator(
    prior_state=GaussianState([0, 0, 0, 0, 0],
                              np.diag([0, 1, 0, 1, 0])),
    measurement_model=None,
    deleter=deleter,
    updater=EKF_updater,
    data_associator=data_associator_EKF,
    min_points=5)

#Create multiple copies of the detection reader for each tracker
detector, detectors = tee(detection_reader, 2)

# %%
#Instantiate each of the Trackers, without specifying the detector
EKF_tracker = MultiTargetTracker(
    initiator=EKF_initiator,
    deleter=deleter,
    data_associator=data_associator_EKF,
    updater=EKF_updater,
    detector=detector)


# Load the plotter
from stonesoup.plotter import Plotterly

# Load the OSPA metric managers
from stonesoup.metricgenerator.ospametric import OSPAMetric

# Define a data associator between the tracks and the truths
from stonesoup.dataassociator.tracktotrack import TrackToTruth

# load a multi-metric manager
from stonesoup.metricgenerator.manager import MultiManager

# Loaded the plotter for the various metrics.
from stonesoup.plotter import MetricPlotter

# %%
"""
======================
DEFINE METRICS MANAGER 
======================
"""
c=10
p=2


from stonesoup.metricgenerator.ospametric import GOSPAMetric, OSPAMetric
# gospa_gen_name = "GOSPA-TTB"
gospa_ttb = GOSPAMetric(c=c, p=p, generator_name="GOSPA-TTB",
                        tracks_key="tracks_ttb", truths_key="truths", switching_penalty=1)


# GOSPA Stone Soup
gospa_EKF = GOSPAMetric(c=c, p=p, generator_name='GOSPA-EKF',
                            tracks_key='tracks',  truths_key='truths', switching_penalty=1)


# Use the track associator
associator = TrackToTruth(association_threshold=30)

# Use a metric manager to deal with the various metrics
metric_manager = MultiManager([gospa_ttb, gospa_EKF],
                              associator)

# %%
# Run simulations
# ^^^^^^^^^^^^^^^
print("Run simulations...")

timesteps=[]

# Lists to hold the detections from each sensor
s1_detections = []
s2_detections = []

# list for all detections
full_detections = []

truths = set()
tracks = set()

tracker_iter = iter(EKF_tracker)

for t in range(number_of_steps):  # loop over the various time-steps
    print(t)
    time, gt = next(ground_truths)
    timesteps.append(time)
    truths.update(gt)

    for idx in (0, 1):
        _, det = next(detectors)
        if idx % 2 == 0:
            s1_detections.append(det)

        elif idx % 2 == 1:
            s2_detections.append(det)
        full_detections.extend(det)

        # Run the Extended Kalman filter
        _, track = next(tracker_iter)
        # print(track)
        tracks.update(track)


# Add data to the metric manager
metric_manager.add_data({'tracks': tracks}, overwrite=False)

# metric_manager.add_data({'truths': truths,
#                          'detections': full_detections}, overwrite=False)
metric_manager.add_data({'truths': truths}, overwrite=False)



# %%
"""
=============================
USING THE BRIDGE FOR TRACKING
=============================
"""
all_measurements = list(zip(s1_detections, s2_detections))

from stonesoup_bridge import bridge

tracks_ttb = bridge.track_via_ttb(transition_model=transition_model,
                                  measurement_model=measurement_model,
                                  all_measurements=all_measurements,
                                  sensors=[sensor_1, sensor_2],
                                  timesteps=timesteps,
                                  position_mapping=position_mapping)

print("Number of Stone Soup Tracks:", len(tracks))
print("Number of TTB Tracks:", len(tracks_ttb))
print("Number of Truths:", len(truths))

metric_manager.add_data({'tracks_ttb' : tracks_ttb}, overwrite=False)

# %%
# counter = 0
# for track in tracks:
#     if counter <5:
#         print("EKF:", track)
#         counter+=1
# counter = 0
# for track in tracks_ttb:
#     if counter <5:
#         print("TTB:", track)
#         counter+=1
# %%
# 4. Plot the tracks and the track performances
# ---------------------------------------------
# We have obtained the tracks and the ground truths from the trackers. It is time
# to visualise the tracks and load the metric manager to evaluate the performances.
print("Plotting...")
plotter = Plotterly()

plotter.plot_measurements(s1_detections, [0, 2],
                          measurements_label='Sensor 1 measurements')
plotter.plot_measurements(s2_detections, [0, 2],
                          measurements_label='Sensor 2 measurements')
plotter.plot_sensors({sensor1_platform, sensor2_platform}, [0, 1],
                     marker=dict(color='black', symbol='1', size=10))
plotter.plot_ground_truths(truths, [0, 2])

plotter.plot_tracks(tracks, [0, 2], line=dict(color='cyan'), label='Stone Soup Tracks')
plotter.plot_tracks(tracks_ttb, [0, 2], line=dict(color='#A32638'), label='TTB Tracks')

plotter.fig.show(renderer="browser")

# %%
metrics = metric_manager.generate_metrics()

"""
==================
PLOT GOSPA METRICS
==================
"""
import matplotlib
matplotlib.use('TkAgg')

plt = bridge.plot_gospa(metrics, gospa_gen_name=["GOSPA-TTB", "GOSPA-EKF"])
plt.show()

