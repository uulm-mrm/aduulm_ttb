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
from stonesoup.models.measurement.linear import LinearGaussian

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
qz = 1
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(q_x),    # [x, v_x,
                                                          ConstantVelocity(q_y),    #  y, v_y,
                                                          RandomWalk(qz)])           #  z]

start_time = datetime(1970, 1, 1, 0, 0, 0)

truth = [GroundTruthState([100, -10, 100, -10, 0], timestamp=start_time),
         GroundTruthState([100, -10, -100, 10, 0], timestamp=start_time),
         GroundTruthState([-100, 10, -100, 10, 0], timestamp=start_time),
         GroundTruthState([-100, 10, 100, -10, 0], timestamp=start_time),
         GroundTruthState([70, -10, 70, -5, 0], timestamp=start_time),
         GroundTruthState([70, -10, -70, 5, 0], timestamp=start_time),
         GroundTruthState([-70, 10, -70, 5, 0], timestamp=start_time),
         GroundTruthState([-70, 10, 70, -5, 0], timestamp=start_time),
         GroundTruthState([70, -5, 70, -10, 0], timestamp=start_time),
         GroundTruthState([70, -5, -70, 10, 0], timestamp=start_time),
         GroundTruthState([-70, 5, -70, 10, 0], timestamp=start_time),
         GroundTruthState([-70, 5, 70, -10, 0], timestamp=start_time),
         GroundTruthState([100, -10, 0, 0, 0], timestamp=start_time),
         GroundTruthState([0, 0, -100, 10, 0], timestamp=start_time),
         GroundTruthState([0, 0, 100, -10, 0], timestamp=start_time),
         GroundTruthState([-100, 10, 0, 0, 0], timestamp=start_time),
         GroundTruthState([70, -10, 100, -15, 0], timestamp=start_time),
         GroundTruthState([70, -10, -100, 15, 0], timestamp=start_time),
         GroundTruthState([-100, 10, -70, 15, 0], timestamp=start_time),
         GroundTruthState([-100, 10, 100, -15, 0], timestamp=start_time),
         GroundTruthState([100, -15, 0, 0, 0], timestamp=start_time),
         GroundTruthState([0, 0, -100, 15, 0], timestamp=start_time),
         GroundTruthState([0, 0, 100, -15, 0], timestamp=start_time),
         GroundTruthState([-100, 15, 0, 0, 0], timestamp=start_time),
         GroundTruthState([100, -15, 70, -10, 0], timestamp=start_time),
         GroundTruthState([100, -15, -70, 10, 0], timestamp=start_time),
         GroundTruthState([-100, 15, -100, 10, 0], timestamp=start_time),
         GroundTruthState([-100, 15, 70, -10, 0], timestamp=start_time),]

# truth = [GroundTruthState([100, -10, 100, -10, 0], timestamp=start_time),
#          GroundTruthState([100, -10, -100, 10, 0], timestamp=start_time),
#          GroundTruthState([-100, 10, -100, 10, 0], timestamp=start_time),
#          GroundTruthState([-100, 10, 100, -10, 0], timestamp=start_time),
#          GroundTruthState([100, -10, 0, 0, 0], timestamp=start_time),
#          GroundTruthState([0, 0, -100, 10, 0], timestamp=start_time),
#          GroundTruthState([0, 0, 100, -10, 0], timestamp=start_time),
#          GroundTruthState([-100, 10, 0, 0, 0], timestamp=start_time),]

number_of_steps = 90
timestep_size = timedelta(milliseconds=100)

from stonesoup.simulator.simple import MultiTargetGroundTruthSimulator
ground_truth_simulator = MultiTargetGroundTruthSimulator(
    transition_model=transition_model,
    initial_state=GaussianState([-50, 0, 0, 0, 0],
                                np.diag([2, 0.1, 2, 0.1, 0.1]),
                                timestamp=start_time),
    birth_rate= 0,
    death_probability= 0,
    number_steps=number_of_steps,
    timestep=timestep_size,
    preexisting_states=[t.state_vector for t in truth],
    initial_number_targets=0
)
# %%
# Generate clutter
# ^^^^^^^^^^^^^^^^
from stonesoup.models.clutter.clutter import ClutterModel

# Define the clutter model which will be the same for both sensors
# Keep the clutter rate low due to particle filter errors
clutter_model = ClutterModel(
    clutter_rate=25,
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
                      clutter_model=clutter_model,
                      detection_probability=0.7)

sensor_2 = Sensor_TTB(measurement_model=measurement_model,
                      clutter_model=clutter_model,
                      detection_probability=0.7 )


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
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.predictor.kalman import KalmanPredictor

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
from stonesoup.deleter.error import CovarianceBasedDeleter
# deleter = CovarianceBasedDeleter(covar_trace_thresh=2)
# %%
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
gospa_EKF = GOSPAMetric(c=c, p=p, generator_name='GOSPA-SOUP',
                        tracks_key='tracks',  truths_key='truths', switching_penalty=1)


# Use the track associator
associator = TrackToTruth(association_threshold=c)

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

tracks_ttb, _ = bridge.track_via_ttb(transition_model=transition_model,
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

def to_tikz_plot(x, y, note="", tol=0.1):
    import shapely
    line = shapely.LineString([(x, y) for x, y in zip(x, y)])
    line = line.simplify(tol)
    s = ""
    s += "\\addplot [black]\ntable {%" + note + "\n"
    for x, y in line.coords:
        s += f"{x} {y}\n"
    s += "};\n"
    s += f"\\filldraw[black] ({line.coords[0][0]}, {line.coords[0][1]}) circle (2pt);"
    return s

gt_mapping = (0, 2)
with open("simple_scenario.txt", 'w') as file:
    for truth in truths:
        file.write(to_tikz_plot([state.state_vector[gt_mapping[0]] for state in truth], [state.state_vector[gt_mapping[1]] for state in truth]))


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

plt = bridge.plot_all_gospas(metrics, gospa_gen_name=["GOSPA-TTB", "GOSPA-SOUP"])
plt.show()
