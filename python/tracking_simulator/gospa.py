import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Callable, Dict, List, Tuple


class GOSPA:
    def __init__(self, c: float = 10, p: float = 2, alpha: float = 2, measure=None, mapping=None):
        """
        Initializes the Generalized Optimal SubPattern Assignment (GOSPA) metric with given parameters.

        Parameters:
            c (float): Cutoff distance.
            p (float): Order parameter.
            alpha (float): Cost factor for missed/false tracks (cardinality mismatch).
            measure (function): Distance function, default is Euclidean.
            mapping (List): State components to be considered for GOSPA evaluation.

        References:
            Rahmathullah, Abu Sajana, Ángel F. García-Fernández, and Lennart Svensson.
            "Generalized optimal sub-pattern assignment metric."
            2017 20th International Conference on Information Fusion (Fusion). IEEE, 2017.
        """
        self.c = c
        self.p = p
        self.alpha = alpha
        self.mapping = mapping if mapping is not None else [0,1]
        self.measure = measure if measure else self.euclidean_distance
        self.check_gospa_config_parameters()

    def euclidean_distance(self, track, truth):
        """Computes the Euclidean distance between track and truth states.
        Parameters:
            track: track state.
            truth: ground truth state.

        Returns:
            Euclidean distance between track and truth states.
        """
        return np.linalg.norm(np.array(track)[self.mapping] - np.array(truth)[self.mapping])

    def check_gospa_config_parameters(self):
        """Checks if the GOSPA parameters are within valid bounds.
        """
        if not (0 < self.alpha <= 2):
            raise ValueError("Alpha must be in the range (0, 2].")
        if self.c <= 0:
            raise ValueError("Cutoff distance c must be positive, i.e., in (0, inf).")
        if self.p < 1:
            raise ValueError("Order parameter p must be at least 1, i.e., in [1, inf).")

    def generate_cost_matrix(self, track_list, truth_list):
        """Computes the cost matrix between tracks and truth states.

        Parameters:
            track_list (List): List of tracks.
            truth_list (List): List of ground truths.

        Returns:
            cost matrix (np.ndarray) of distances.
        """
        num_tracks = len(track_list)
        num_truths = len(truth_list)

        cost_matrix = np.zeros((num_tracks, num_truths))
        for idx_track, track in enumerate(track_list):
            for idx_truth, truth in enumerate(truth_list):
                distance = self.measure(track, truth)
                if distance < self.c:
                    cost_matrix[idx_track, idx_truth] = distance
                else:
                    cost_matrix[idx_track, idx_truth] = self.c

        return cost_matrix

    def compute_gospa_metric(self, track_list, truth_list):
        """Computes the GOSPA metric at current time step.

        Parameters:
            track_list (List): List of tracked states.
            truth_list (List): List of ground truth states.

        Returns:
            GOSPA metric components and additional data:
                {'distance', 'localization', 'missed', 'false',
                 'truths_to_tracks_assignment',
                 'num_tracks', 'num_truths',
                 'num_missed', 'num_false'}
        """
        gospa_results = {'distance': 0.0, 'localization': 0.0, 'missed': 0, 'false': 0,
                         'truths_to_tracks_assignment': {},
                         'num_tracks': 0, 'num_truths': 0,
                         'num_missed': 0, 'num_false': 0}
        if not track_list and not truth_list:
            return gospa_results

        cost_matrix = self.generate_cost_matrix(track_list, truth_list).T

        unassigned_factor = (self.c ** self.p) / self.alpha

        num_tracks = len(track_list)
        num_truths = len(truth_list)
        truths_to_tracks_assignment = {}

        gospa_results['num_tracks'] = num_tracks
        gospa_results['num_truths'] = num_truths

        if num_tracks == 0:
            # There are no estimated tracks. All ground truth tracks are missed.
            num_missed = num_truths
            num_false = 0
        elif num_truths == 0:
            # There are no ground truth tracks. All estimated tracks are false tracks.
            num_missed = 0
            num_false = num_tracks
        else:
            cost_matrix = np.power(cost_matrix, self.p)
            truth_assignment, track_assignment = linear_sum_assignment(cost_matrix)

            for truth_idx, track_idx in zip(truth_assignment, track_assignment):
                if cost_matrix[truth_idx, track_idx] < self.c ** self.p:
                    gospa_results['localization'] += cost_matrix[truth_idx, track_idx]
                    truths_to_tracks_assignment[truth_idx] = track_idx

            num_assigned = len(truths_to_tracks_assignment)
            num_missed = num_truths - num_assigned
            num_false = num_tracks - num_assigned

        gospa_results['missed'] = unassigned_factor * num_missed
        gospa_results['false'] = unassigned_factor * num_false
        gospa_results['truths_to_tracks_assignment'] = truths_to_tracks_assignment
        gospa_results['num_missed'] = num_missed
        gospa_results['num_false'] = num_false

        gospa_results['distance'] = np.power(gospa_results['localization'] + gospa_results['missed']
                                             + gospa_results['false'], 1 / self.p)

        return gospa_results
