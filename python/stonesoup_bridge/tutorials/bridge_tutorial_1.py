#!/usr/bin/env python
# %%
"""
=================================================================
1 - An introduction to Stone Soup Bridge: using the Kalman filter
=================================================================
"""


# %%
from itertools import tee

import numpy as np
from datetime import timedelta, datetime

import matplotlib
import plotly.io as pio
from stonesoup.deleter.time import UpdateTimeDeleter
from stonesoup.plotter import Plotterly
from stonesoup.tracker.simple import SingleTargetTracker


# %%
"""
==============================
CREATING GROUND TRUTH SCENARIO
==============================
"""

from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity, RandomWalk

seed_number = 1991
np.random.seed(seed_number)

q_x = q_y = 2
q_z = 0
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(q_x),
                                                          ConstantVelocity(q_y),
                                                          RandomWalk(q_z)])

start_time = datetime(2020, 12, 27, 0, 0, 0)
truth = GroundTruthPath([GroundTruthState([-40,3,0, 4, 0], timestamp=start_time)])

number_of_steps = 100
timestep_size = timedelta(milliseconds=100)

from stonesoup.simulator.simple import SingleTargetGroundTruthSimulator
ground_truth_simulator = SingleTargetGroundTruthSimulator(
    transition_model=transition_model,
    initial_state=truth,
    timestep = timestep_size,
    number_steps=number_of_steps
)

# %%
"""
========================================
CREATING ONE SENSOR PLATFORM WITH CLUTTER
========================================
"""

from stonesoup.models.clutter.clutter import ClutterModel

# Define the clutter model which will be the same for both sensors
# Keep the clutter rate low due to particle filter errors
clutter_model = ClutterModel(
    clutter_rate=1,
    distribution=np.random.default_rng().uniform,
    dist_params=((-50, 20), (-10, 30))
)


from stonesoup.models.measurement.linear import LinearGaussian


cov = 2
position_mapping=(0, 2)
measurement_model = LinearGaussian(
    ndim_state=5,  # Number of state dimensions (position and velocity in 2D and Z Pos = 0)
    mapping=(0, 2),  # Mapping measurement vector index to state index
    noise_covar=np.array([[cov, 0],  # Covariance matrix for Gaussian PDF
                          [0, cov]])
)

from stonesoup_bridge.bridge import Sensor_TTB

sensor = Sensor_TTB(measurement_model=measurement_model,
                    clutter_model=clutter_model,
                    detection_probability=0.99)

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

ground_truths, *gt_sims = tee(ground_truth_simulator, 2)
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
from stonesoup.predictor.kalman import KalmanPredictor
predictor = KalmanPredictor(transition_model)

from stonesoup.updater.kalman import KalmanUpdater
updater = KalmanUpdater(measurement_model)

from stonesoup.types.state import GaussianState
prior = GaussianState([[0], [1], [0], [1], [0]], np.diag([1.5, 0.5, 1.5, 0.5, 0]), timestamp=start_time)

from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
hypothesiser = DistanceHypothesiser(predictor, updater, measure=Mahalanobis(), missed_distance=3)

from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
data_associator = GNNWith2DAssignment(hypothesiser)

# define a track deleter based on time measurements
deleter = UpdateTimeDeleter(timedelta(seconds=0.3), delete_last_pred=False)

from stonesoup.initiator.simple import MultiMeasurementInitiator
initiator = MultiMeasurementInitiator(
    prior_state=prior,
    measurement_model=measurement_model,
    data_associator=data_associator,
    updater=updater,
    deleter=deleter
)

tracker = SingleTargetTracker(
    initiator=initiator,
    data_associator=data_associator,
    updater=updater,
    detector=detector,
    deleter=deleter
)


# %%
"""
======================
DEFINE METRICS MANAGER 
======================
"""
# load a multi-metric manager
from stonesoup.metricgenerator.manager import MultiManager
# Define a data associator between the tracks and the truths
from stonesoup.dataassociator.tracktotrack import TrackToTruth


c=10
p=2


from stonesoup.metricgenerator.ospametric import GOSPAMetric
# gospa_gen_name = "GOSPA-TTB"
gospa_ttb = GOSPAMetric(c=c, p=p, generator_name="GOSPA-TTB",
                        tracks_key="tracks_ttb", truths_key="truths", switching_penalty=1)


# GOSPA Stone Soup
gospa_kalman = GOSPAMetric(c=c, p=p, generator_name='GOSPA-Stone-Soup',
                            tracks_key='tracks',  truths_key='truths', switching_penalty=1)


# Use the track associator
associator = TrackToTruth(association_threshold=30)

# Use a metric manager to deal with the various metrics
metric_manager = MultiManager([gospa_ttb, gospa_kalman],
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
tracker_iter = iter(tracker)

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

all_measurements = all_detections

from stonesoup_bridge import bridge

tracks_ttb, _ = bridge.track_via_ttb(transition_model=transition_model,
                                  measurement_model=measurement_model,
                                  all_measurements=all_measurements,
                                  sensors=[sensor],
                                  timesteps=timesteps,
                                  position_mapping=position_mapping)

print("Number of Stone Soup Tracks:", len(tracks))
print("Number of TTB Tracks:", len(tracks_ttb))
print("Number of Truths:", len(truths))

metric_manager.add_data({'tracks_ttb' : tracks_ttb}, overwrite=False)
metrics = metric_manager.generate_metrics()
# %%
"""
==========================================
PLOTTING ALL TRACKS, TRUTHS AND DETECTIONS
==========================================
"""
from time import sleep
sleep(5) # sleep for 5s to prevent buffer overflow

PLOT = False
if PLOT:
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