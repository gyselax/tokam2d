# src/simulation/run_simulation.py
import jax
import jax.numpy as jnp
import functools
from jax import jit
from functools import partial

@jit
def add_states(*states):
    """
    Element-wise addition of multiple PyTree states. E.g. add_states(a, b, c, ...)
    returns a + b + c + ... in a single PyTree.
    """
    # If there's only one state, just return it:
    if len(states) == 1:
        return states[0]

    # Reduce from left to right with a smaller binary tree_map sum
    return functools.reduce(lambda x, y: jax.tree_util.tree_map(jnp.add, x, y), states)

@jit
def mul_states(a, factor):
    """Element-wise scaling of a state dict."""
    return jax.tree_util.tree_map(lambda x: factor * x, a)
    # return {k: factor * v for k, v in a.items()} #Numpy friendly equivalent

@jit
def fft_states(a):
    """Element-wise scaling of a state dict."""
    return jax.tree_util.tree_map(jnp.fft.fft2, a)

@jit
def ifft_real_states(a):
    """Element-wise scaling of a state dict."""
    return jax.tree_util.tree_map(lambda x: jnp.fft.ifft2(x).real, a)

# @jit
@partial(jax.jit, static_argnames=('pde',))
def step(state, pde, dt, t):
    k1 = pde.compute_rhs(state, t)
    k2 = pde.compute_rhs(add_states(state, mul_states(k1, 0.5*dt)), t + 0.5*dt)
    k3 = pde.compute_rhs(add_states(state, mul_states(k2, 0.5*dt)), t + 0.5*dt)
    k4 = pde.compute_rhs(add_states(state, mul_states(k3, dt)), t + dt)
    
    increment = add_states(
        k1,
        mul_states(k2, 2.0),
        mul_states(k3, 2.0),
        k4
    )
    increment = mul_states(increment, dt / 6.0)
    new_state = add_states(state, increment)
    return new_state

class SimulationRunner:
    """
    A runner that loops over time steps.

    For each iteration, it calls
    - "callbacks" using fields as inputs without modyfing them (e.g. saving, crash checking)
    - "inline_operations" using fields as inputs and modifying them (e.g. FFT filtering)
    - "inline_displays" not using fields (e.g. progress display)
    """
    def __init__(self, params, saver=None):

        # Set the saver and logger
        self.saver = saver
        self.logger = params.logger

        # Retrieve the PDE parameters
        self.eq = params.user["pde"]["eq"]
        self.pde = params.pde

        # Retrieve the time parameters and initialize the clocks
        self.Nt_rk4 = params.user["time"]["Nt_rk4"]
        self.Nt_diag = params.user["time"]["Nt_diag"]
        self.dt_rk4 = params.user["time"]["dt_rk4"]
        self.rk4_per_diag = params.user["time"]["rk4_per_diag"]

        self.time = 0.0
        self.step_rk4_count = 0
        self.step_diag_count = 0
        self._stop = False

        # Retrieve and initialize the operation occuring during the loop 
        self.user_callbacks = params.user.get("callbacks", {})
        self.user_inline_operations = params.user.get("inline_operations", {})

        self.callbacks = []
        self.inline_operations = []
        self.inline_displays = []

        self.mask_fft = params.mask_fft
        self._save_real = self.user_callbacks.get('save_real', False)
        self._save_fft = self.user_callbacks.get('save_fft', False)
        self._save_text = self._save_real*'real' + (self._save_real&self._save_fft)*' and ' + self._save_fft*'fft'
        
        self._init_callbacks()
        self._init_inline_operations()
        self._init_inline_display()

    def run(self, fields):
        """
        Run the simulation for Nt_rk4 steps, calling callbacks and inline operations
        at each diagnostic step.
        """
        while self.step_rk4_count < self.Nt_rk4 and not self._stop:

            if self.step_rk4_count % self.rk4_per_diag == 0:
                # Trigger callbacks (e.g. saving, crash checking) at diagnostic time step
                for cb in self.callbacks:
                    cb(fields)

                # Trigger inline display (e.g. progress) at diagnostic time step
                for disp in self.inline_displays:
                    disp()

                # Trigger inline operations (e.g. FFT filtering) at diagnostic time step
                for op in self.inline_operations:
                    fields = op(fields)

                self.step_diag_count += 1
            fields = step(fields, self.pde, self.dt_rk4, t=self.time)
            self.time += self.dt_rk4
            self.step_rk4_count += 1
        return fields

    def _init_callbacks(self):
        """Read config['callbacks'] and add any desired methods to self.callbacks."""

        # Enable inline HDF5 saving of fields in real space
        if self._save_real:
            self.logger.info("Enabling inline HDF5 saving of fields in real space...")
            self.callbacks.append(self._save_real_data_callback)

        # Enable inline HDF5 saving of fields in fourier space
        if self._save_fft:
            self.logger.info("Enabling inline HDF5 saving of fields in fourier space...")
            self.callbacks.append(self._save_fft_data_callback)

        # Enable crash checking
        if self.user_callbacks.get('check_crash', False):
            self.logger.info("Enabling crash checking...")
            self.callbacks.append(self._check_crash_callback)

    def _init_inline_operations(self):
        """Read config['inline_operation'] and add any desired methods to self.inline_operation."""

        # Enable numerical noise accumulation filter for HW advection
        if self.eq in ["HW", "mHW", "BHW"]:
            self.logger.info("Enabling numerical noise accumulation filter for HW advection...")
            self.inline_operations.append(self._numerical_noise_accumulation_filter)
        # Enable 2/3 de-aliasing rule
        if self.user_inline_operations.get('fft_filter', False):
            self.logger.info("Enabling 2/3 de-aliasing rule...")
            self.inline_operations.append(self._apply_fft_mask)

    def _init_inline_display(self):
        # Display simulation progress
        self.inline_displays.append(self._display_progress)

    # Callbacks
    def _save_real_data_callback(self, fields):
        # Convert the fields to real space and save them
        # name[:-4] removes the "_fft" suffix from the field name
        real_fields_dic = {name[:-4]: jnp.fft.ifft2(field).real for name, field in fields.items()}

        self.saver.save_output(real_fields_dic, step=self.step_diag_count, t=self.time)

    def _save_fft_data_callback(self, fields):
        self.saver.save_output(fields, step=self.step_diag_count, t=self.time)

    def _check_crash_callback(self, fields):
        # Field is a dict of 2D array, we to check the first array
        if jnp.any(jnp.isnan(fields[list(fields.keys())[0]])):
            self.logger.critical("\n !!! NUMERICAL CRASH !!!\n")
            self.logger.critical("NaNs detected in field data. Stopping the simulation properly...")
            self._stop = True

    # Inline operations
    def _apply_fft_mask(self, fields):
        return mul_states(fields, self.mask_fft)

    def _numerical_noise_accumulation_filter(self, fields):
        return fft_states(ifft_real_states(fields))

    # Inline display
    def _display_progress(self):
        self.logger.info(f"Save {self.step_diag_count}/{self.Nt_diag} of {self._save_text} fields | t={self.time:.3f} (RK4 step {self.step_rk4_count}/{self.Nt_rk4}).")