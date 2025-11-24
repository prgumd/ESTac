from multiprocessing import Value
from multiprocessing.sharedctypes import Synchronized
import time
from typing import Optional

        
class TimeSource:
    """Universal Timesource for all of the data collection"""

    def __init__(self, sim: bool =False, t0: Optional[float] = None):
        self.sim = sim
        if self.sim:
            self._t: Synchronized[float] = Value("d", 0.0 if t0 is None else t0)
        else:
            self._start_time: Synchronized[float] = Value("d",  time.time() if t0 is None else t0)

    def time(self) -> float:
        """Returns the current time (simulated or real)."""
        if self.sim:
            return float(self._t.value)
        return time.time() - self._start_time.value

    def reset(self) -> None:
        """Resets the time to zero."""
        if self.sim:
            self._t.value = 0.0
        else:
            self._start_time.value = time.time()

    def set(self, t: float)  -> None:
        """Sets the current simulated time (only in sim mode)."""
        if not self.sim:
            raise RuntimeError("Cannot set time in real-time mode.")
        self._t.value = t

    def increment(self, dt: float) -> None:
        """Increments simulated time by dt seconds (only in sim mode)."""
        if not self.sim:
            raise RuntimeError("Cannot increment time in real-time mode.")
        self._t.value += dt
