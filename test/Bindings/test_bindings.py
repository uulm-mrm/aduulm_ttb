import unittest
import pathlib



class HPX:
    def __init__(self, api, n_threads):
        self.api = api
        self.n_threads = n_threads
    def __enter__(self):
        self.api.start_hpx(self.n_threads)
    def __exit__(self, exc_type, exc_value, traceback):
        self.api.stop_hpx()


class TestTrackingPythonBindings(unittest.TestCase):

    def test_import(self):
        from tracking_lib import _tracking_lib_python_api as _api

    def test_manager(self):
        from tracking_lib import _tracking_lib_python_api as _api
        with HPX(_api, 2):
            config_file = pathlib.Path("src/tracking/library/test/DefaultTrackingSimulation/tracking_config.yaml")
            _api.set_log_level("Debug")
            manager = _api.TTBManager(config_file)
            self.assertEqual(len(manager.getEstimate()), 0)

if __name__ == '__main__':
    unittest.main()