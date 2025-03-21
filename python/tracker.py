from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tracking_lib import _tracking_lib_python_api as _api

@dataclass
class TTB:
    manager: _api.TTBManager
    def __init__(self, config: Path):
        self.manager = _api.TTBManager(config)
    
    def reset(self):
        self.manager.reset()

    def cycle(self, time: datetime):
        self.manager.cycle(time)
    def cycle_tracking(self, time: datetime, meas: list):
        return self.manager.cycle(time, meas, True)


    def getEstimate(self):
        return self.manager.getEstimate()
    
    def addMeasurement(self, meas: _api.MeasurementContainer, received_time: datetime):
        self.manager.addData(meas, received_time)

    def getMeasModelMap(self):
        return self.manager.getMeasModelMap()

    
