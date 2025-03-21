import unittest
import pathlib


class TestTrackingPythonBindings(unittest.TestCase):

    def test_import(self):
        from tracking_lib import _tracking_lib_python_api as _api

    def test_manager(self):
        from tracking_lib import _tracking_lib_python_api as _api
        config_file = pathlib.Path("src/tracking/library/test/DefaultTrackingSimulation/tracking_config.yaml")
        _api.set_log_level("Debug")
        manager = _api.TTBManager(config_file)
        self.assertEqual(len(manager.getEstimate()), 0)

if __name__ == '__main__':
    unittest.main()