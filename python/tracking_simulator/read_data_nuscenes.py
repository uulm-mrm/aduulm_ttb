# This file needs to be processed by python3.6 or python3.7 since Nuscenes cann only be installed for those version (python3.12 does not work...)
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits

from dataclasses import dataclass
from pyquaternion import Quaternion


nusc_to_ttb_class_label_map = {
    "barrier"               : _api.ClassLabel.UNKNOWN,
    "traffic_cone"          : _api.ClassLabel.UNKNOWN,
    "bicycle"               : _api.ClassLabel.BICYCLE,
    "motorcycle"            : _api.ClassLabel.MOTORBIKE,
    "pedestrian"            : _api.ClassLabel.PEDESTRIAN,
    "car"                   : _api.ClassLabel.CAR,
    "bus"                   : _api.ClassLabel.BUS,
    "construction_vehicle"  : _api.ClassLabel.UNKNOWN,
    "trailer"               : _api.ClassLabel.UNKNOWN,
    "truck"                 : _api.ClassLabel.TRUCK
}

ttb_params = {

    "seed_number" : 10,
    "detection_prob" : 0.5,
    "meas_pos_variance" : 1.5,
    "clutter_rate" : 0.001,
    "num_sensors" : 1,
    "measurement_type" : 1,
    "sensor_id_list" : ["sim0"],
    "components_sensors" : [("POS_X", "POS_Y","POS_Z","Yaw","LENGTH","WIDTH", "HEIGHT")],

    "cov_matrix" : np.array([[1.5, 0], [0, 1.5]]),
}

@dataclass(slots=True)
class Detection:
    mean: np.array # n x 1 vector
    components: tuple(str)         # Measured components of the sensor. The order must be the same as the order in mean and cov. e.g. (_api.Component.POS_X,_api.Component.POS_Y)
    detection_score: float
    class_label: str
    classification_prob: float = 0.95


def preprocess(nusc, my_scene, detections, ttb_params):

    first_sample_token = my_scene['first_sample_token']
    first_sample = nusc.get('sample', first_sample_token)


    current_sample = first_sample
    current_sample_token = first_sample['token']
    has_next = True
    k = 0

    annotations = []
    x_pos_annotations = []
    y_pos_annotations = []
    annotation_class = []

    measurements = []

    # iterate over the annotated detections
    while has_next:

        # get detections at current timestep
        detection = detections[current_sample_token]

        # get the current time
        timestamp = current_sample['timestamp']

        Z = []
        # create measurement dictionaries for detections
        for entry in detection:
            x, y, z = entry['translation']
            quaternion = Quaternion(entry['rotation'])
            yaw = quaternion.angle if quaternion.axis[2] > 0 else -quaternion.angle
            width, length, height = entry['size']
            mean = np.zeros((6))
            mean[0] = entry['translation'][0] # x
            mean[1] = entry['translation'][1] # y
            mean[2] = entry['translation'][2] # z
            mean[3] = yaw # z
            mean[4] = entry['translation'][0]
            meas = Detection(mean,ttb_params["components_sensors"][0],nusc_to_ttb_class_label_map[entry['detection_name']],entry['detection_score'])
            Z.append(meas)

        measurements.append([dict(id=0, measurements = Z, timestamp=timestamp)])

        annotations_timestep = []

        # get annotation for those detections (GT)
        for anno_token in current_sample['anns']:

            annotation = nusc.get('sample_annotation', anno_token)
            #print(annotation)
            x_pos = annotation['translation'][0]
            y_pos = annotation['translation'][1]

            x_pos_annotations.append(x_pos)
            y_pos_annotations.append(y_pos)

            annotation_class.append(annotation['category_name'])

            annotations_timestep.append([x_pos, y_pos])

        annotations.append(annotations_timestep)

        # check for next sample, terminate if theres none
        next_sample_token = current_sample['next']

        #print("----------------------------------------------------------")

        if (next_sample_token == ""):
            print("num steps = ", k)
            break

        current_sample_token = next_sample_token
        current_sample = nusc.get('sample', current_sample_token)
        k += 1

    # annotations in different format with classes, used it for generating plot
    gt = (x_pos_annotations, y_pos_annotations, annotation_class)

    return measurements, annotations, gt

if __name__ == "__main__":

    # get metadata of validation dataset
    # supposing the directory contains at least metadata of the dataset
    # [TODO] path to nuScenes metadata
    dataset_version = 'v1.0-mini'
    dataset_folder = '/home/hermann/Sequenzen_lokal/Nuscenes/mini_dataset/v1.0-mini'
    nusc = NuScenes(version=dataset_version, dataroot=dataset_folder)

    # get training and validation data splits
    split_data = splits.create_splits_scenes()
    train_scenes  = split_data['train']
    val_scenes = split_data['val']

    #[TODO] select a scene
    # useful command to find a scene: nusc.list_scenes()
    my_scene_token = nusc.field2token('scene', 'name', 'scene-0061')[0]
    my_scene = nusc.get('scene', my_scene_token)

    data_dir = os.path.abspath("/home/hermann/Sequenzen_lokal/Felicia/sep 24/odet")
    detections_path = os.path.join(data_dir, "results_nusc.json")

    with open(detections_path, 'r') as f:
        detections_json = json.load(f)

    detections = detections_json['results']

    # preprocess
    measurements, annotations, gt = preprocess(nusc, my_scene, detections, ttb_params)