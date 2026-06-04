# src/tokam2d/interfaces/memory_saver.py
import numpy as np


class MemorySaver:
    """In-memory drop-in for OutputSaver.

    Implements the same ``save_output(fields, step, t)`` interface that
    ``SimulationRunner`` calls at each diagnostic step, but keeps the
    real-space fields in memory instead of writing HDF5. ``get_output`` returns
    the stacked per-step fields together with the spatial coordinates.
    """

    def __init__(self, params):
        self.x = np.asarray(params.x_1d)
        self.y = np.asarray(params.y_1d)
        self._steps = {}
        self._times = []

    def save_output(self, fields, step, t, overwrite=False):
        for name, arr in fields.items():
            self._steps.setdefault(name, []).append(np.asarray(arr))
        self._times.append(float(t))

    def get_output(self):
        fields = {k: np.stack(v) for k, v in self._steps.items()}
        return dict(fields=fields, time=np.asarray(self._times),
                    x=self.x, y=self.y)

    # No-ops so MemorySaver can transparently replace OutputSaver.
    def save_output_data_file(self, *args, **kwargs):
        pass

    def save_metadata_file(self, *args, **kwargs):
        pass

    def save_user_input_file(self, *args, **kwargs):
        pass
