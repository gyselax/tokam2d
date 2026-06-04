# src/tokam2d/api.py
"""In-process entry point for running Tokam2D simulations."""

import jax.numpy as jnp
import numpy as np

from tokam2d.interfaces.input_reader import StaticParams
from tokam2d.interfaces.output_saver import OutputSaver
from tokam2d.interfaces.memory_saver import MemorySaver
from tokam2d.simulation.initialize_fields import FieldInitiator
from tokam2d.simulation.run_simulation import SimulationRunner


def run_simulation(params, initial_fields=None, *, save_dir=None,
                   n_iter=None, quiet=True):
    """Run a Tokam2D simulation in-process and return its diagnostics.

    Parameters
    ----------
    params : dict | str | pathlib.Path | StaticParams
        Configuration: a parsed YAML dict, a path to a YAML file, or a
        prebuilt :class:`StaticParams`.
    initial_fields : dict[str, ndarray] | None
        Real-space fields to start from (restart), e.g.
        ``{"density": d, "potential": p}``. If ``None``, the analytic initial
        condition from ``params`` is used.
    save_dir : str | pathlib.Path | None
        When given, write the usual HDF5 outputs there (uses ``OutputSaver``);
        otherwise the run is kept fully in memory (``MemorySaver``).
    n_iter : int | None
        Override the number of diagnostic steps (``time.Nt_diag``); the number
        of RK4 steps is set to ``n_iter * rk4_per_diag`` accordingly.
    quiet : bool
        Reduce logging verbosity.

    Returns
    -------
    dict
        ``fields`` (dict of ``[time, Ny, Nx]`` real arrays), ``time``, ``x``,
        ``y`` (present when running in memory), and ``final_fields`` (the last
        real-space frame as a dict of ``[Ny, Nx]`` arrays).
    """
    if not isinstance(params, StaticParams):
        params = StaticParams(params, quiet=quiet)

    if n_iter is not None:
        rk4_per_diag = params.user["time"]["rk4_per_diag"]
        params.user["time"]["Nt_diag"] = int(n_iter)
        params.user["time"]["Nt_rk4"] = int(n_iter) * rk4_per_diag
        params.Nt_diag = int(n_iter)
        params.Nt_rk4 = int(n_iter) * rk4_per_diag

    if initial_fields is None:
        fields = FieldInitiator(params).initial_fields()
    else:
        # Restart from real-space arrays: move them to Fourier space, which is
        # the state representation the solver evolves.
        fields = {f"{name}_fft": jnp.fft.fft2(jnp.asarray(arr))
                  for name, arr in initial_fields.items()}

    saver = OutputSaver(params, save_dir) if save_dir is not None \
        else MemorySaver(params)
    runner = SimulationRunner(params, saver=saver)
    final = runner.run(fields)

    out = saver.get_output() if isinstance(saver, MemorySaver) else {}
    out["final_fields"] = {name[:-4]: np.asarray(jnp.fft.ifft2(arr).real)
                           for name, arr in final.items()}
    return out
