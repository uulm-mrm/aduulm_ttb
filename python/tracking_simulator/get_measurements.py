import matplotlib.pyplot as plt
import numpy as np
from tracking_lib.utils import Detection
from tracking_lib import _tracking_lib_python_api as _api
import pathlib


def get_pos_measurement_target(target, time, meas_pos_variance):
    created_path = target["path"]
    mean = np.zeros((2))
    mean[0] = created_path[0][time-target["birth_time"]] + np.sqrt(meas_pos_variance)*np.random.randn()
    mean[1] = created_path[1][time-target["birth_time"]] + np.sqrt(meas_pos_variance)*np.random.randn()
    return mean

def get_pos_measurements_sensor(targets, detection_prob, time, meas_pos_variance, components):
    
    meas_list = []
    ctr = 0
    for target in targets:
        random_number = np.random.uniform(0,1)
        # print(f'target nmbr: {ctr}, birth_time: {target["birth_time"]}, last_alive_time: {target["last_alive_time"]}')
        # print(f'detection_prob: {detection_prob}, rnd_nmbr: {random_number}')
        ctr = ctr + 1
        if target["birth_time"] <= time <= target["last_alive_time"] and random_number < detection_prob:
            mean = get_pos_measurement_target(target, time, meas_pos_variance)
            cov = np.zeros((len(mean), len(mean)))
            cov[0,0] = meas_pos_variance
            cov[1,1] = meas_pos_variance
            class_prob = 0.95
            meas = Detection(mean,cov,components,_api.ClassLabel.UNKNOWN,class_prob)
            meas_list.append(meas)
    return meas_list

def get_clutter(region_x, region_y, diff_x, diff_y):
    x_val = region_x[0] + diff_x * np.random.uniform(0,1) 
    y_val = region_y[0] + diff_y * np.random.uniform(0,1)
    return x_val, y_val

def get_pos_clutter_measurements_sensor(scenario,clutter_rate, meas_pos_variance, components):
    # the number of clutter measurements follows a poisson distribution and the clutter measurements are distributed following a uniform distribution
    meas_list = []

    if clutter_rate == 0:
        print("No clutter measurements are created!")
        return meas_list

    num_clutter = np.random.poisson(clutter_rate)
    region_x = scenario["region_x"]
    region_y = scenario["region_y"]
    if len(region_x) > 2 or len(region_y) > 2:
        print("Lenght of region x or y is wrong. Must be 2! The following clutter creation will probably be wrong since only the first two entries are considered!")
    diff_x = region_x[1] - region_x[0]
    diff_y = region_y[1] - region_y[0]
    for c in range(0, num_clutter):
        mean = get_clutter(region_x, region_y, diff_x, diff_y)
        cov = np.zeros((len(mean), len(mean)))
        cov[0,0] = meas_pos_variance
        cov[1,1] = meas_pos_variance
        class_prob = 0.95
        meas = Detection(mean,cov,components,_api.ClassLabel.UNKNOWN,class_prob)
        meas_list.append(meas)

    return meas_list
    

def get_pos_measurements(scenario, num_sensors, detection_prob, clutter_rate, time, meas_pos_variance, components_sensors):
    Z_sensors = []
    for s in range(0,num_sensors):
        if False and s==0 and 50<=time<=420:
            meas_target = get_pos_measurements_sensor(scenario["targets"], 0.7*detection_prob, time, meas_pos_variance, components_sensors[s])
        elif False and s==1 and 50<=time<=390:
            meas_target = get_pos_measurements_sensor(scenario["targets"], 0.5*detection_prob, time, meas_pos_variance, components_sensors[s])
        else:
            meas_target = get_pos_measurements_sensor(scenario["targets"], detection_prob, time, meas_pos_variance, components_sensors[s])
        if False and s==1 and 200<=time<=600:
            meas_clutter = get_pos_clutter_measurements_sensor(scenario,clutter_rate*7, meas_pos_variance, components_sensors[s])
        elif False and s==0 and 200<=time<=700:
            meas_clutter = get_pos_clutter_measurements_sensor(scenario,clutter_rate*6, meas_pos_variance, components_sensors[s])
        else:
            meas_clutter = get_pos_clutter_measurements_sensor(scenario,clutter_rate, meas_pos_variance, components_sensors[s])
        joint_measurement = meas_target + meas_clutter
        Z_sensor = dict(id=s,measurements = joint_measurement)
        Z_sensors.append(Z_sensor)
    
    return Z_sensors

def plot_measurements_of_sensor(Z, sensor_id, save_plot, save_path, fov):

    x = []
    y = []
    for time in Z:
        for sensor in time:
            if sensor_id == sensor["id"]:
                for meas in sensor["measurements"]:
                    x.append(meas.mean[0])
                    y.append(meas.mean[1])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x_max = fov[0,0]
    x_min = fov[2,0]
    y_min = fov[0,1]
    y_max = fov[1,1]
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_title('Measurements Sensor' + str(sensor_id))
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.plot(x, y, 'bo', markersize=2)
    if save_plot:
        plt.savefig(str(pathlib.Path(save_path)) + '/measurements'+str(sensor_id)+'.pdf')
    # plt.show()


def get_measurements(scenario, num_sensors, detection_prob, clutter_rate, type, meas_pos_variance, components_sensors):
    Z = []
    for time in range(0,scenario["total_sim_time"]):
        if type == 1:
            Z_sensors = get_pos_measurements(scenario, num_sensors, detection_prob, clutter_rate, time, meas_pos_variance, components_sensors)
            Z.append(Z_sensors)
    return Z
