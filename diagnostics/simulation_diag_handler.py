# Import from built-in modules
import io
import logging
import operator
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

# Import from external modules
import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import imageio
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Map strings to operator functions
operator_map = {
    "*": operator.mul,
    "/": operator.truediv,
    "+": operator.add,
    "-": operator.sub,
}

class Simulation:
    """Class to load and access data from a simulation file.

    It makes use of lazy loading and caching for fast visualization.
    """

    # Ensure singleton behavior per simulation path
    # TODO(RV): Fix behaviour when an instantiation fails and leaves a broken instance in the cache
    _cache = {}

    def __new__(cls, myparam):
        if myparam in cls._cache:
            return cls._cache[myparam]

        instance = super().__new__(cls)
        cls._cache[myparam] = instance
        return instance

    def __init__(self, sim_data_path) -> None:
        if not hasattr(self, "_initialized"):
            self._initialized = True
            
            self.sim_folder_path = Path(sim_data_path)
            self.name = self.sim_folder_path.name
            self.data = {}
            #TODO(RV) Account for restarts seemlessly
            self.sim_data = h5py.File(self._get_first_data_file(), "r")
            try:
                self.sim_inputs = h5py.File(
                    sim_data_path+"/input_params_TOKAM_00.h5", "r")
            except:
                self.sim_inputs = h5py.File(
                    sim_data_path+"/user_inputs.h5", "r")
            try:
                self.sim_metadata = h5py.File(
                    sim_data_path+"/metadata_TOKAM_00.h5", "r")
            except:
                self.sim_metadata = h5py.File(
                    sim_data_path+"/metadata.h5", "r")
            self.kx = self._get_kx()[()]
            self.ky = self._get_ky()[()]
            self.kx2D = np.array(self.kx)[np.newaxis,:]*np.ones_like(self.ky)[:,np.newaxis]
            self.ky2D = np.array(self.ky)[:,np.newaxis]*np.ones_like(self.kx)[np.newaxis,:]
            self.k2D  = np.sqrt(self.kx2D**2 + self.ky2D**2)

            self.time = np.array(self["time"])
            self.t = self.time
            self.x = np.array(self["x"])
            self.y = np.array(self["y"])

            self.Nx = len(self.x)
            self.Ny = len(self.y)

            self.Ly = self["Ly"][()]
            self.Lx = self["Lx"][()]

            self.dx = self["Lx"][()]/self.Nx
            self.dy = self["Ly"][()]/self.Ny

            self.simulation_duration = self["time"][-1] - self["time"][0]
            self.dt_diag = self.simulation_duration / len(self["time"])

            self.dt_RK4 = [self["dt_rk4"][()] if "dt_rk4" in self.sim_inputs else None][0]

            # Dictionnary of rule to compute new fields
            self.field_mapping = {
                "VEy":{
                    "base_fields": ["phi"],
                    "dx": [1],
                    "dy": [0],
                    "power": [1],
                    "operators": None,
                    "sign": 1},
                "VEx":{
                    "base_fields": ["phi"],
                    "dx": [0],
                    "dy": [1],
                    "power": [1],
                    "operators": None,
                    "sign": -1},
                "vorticity":{
                    "base_fields": ["phi"],
                    "dx": [2],
                    "dy": [2],
                    "power": [1],
                    "operators": None,
                    "sign": 1},
                # Combinatory fields
                "reynolds_stress":{
                    "base_fields": ["phi", "phi"],
                    "dx": [0, 1],
                    "dy": [1, 0],
                    "power": [1, 1],
                    "operators":  ["*"],
                    "sign": -1},
                "flux":{
                    "base_fields": ["n", "phi"],
                    "dx": [0, 0],
                    "dy": [0, 1],
                    "power": [1, 1],
                    "operators": ["*"],
                    "sign": -1},
                "Isq":{
                    "base_fields": ["n", "n", "n"],
                    "dx": [1, 0, 0],
                    "dy": [0, 1, 0],
                    "power": [2, 2, 2],
                    "operators": ["+", "/"],
                    "sign": 1},
            }

    def __repr__(self) -> str:
        """Print usefull information about the simulation."""
        sim_comput_duration, dx, dy, sim_duration, start_time, end_time = [format(x, '.2f') for x in [self['simulation_duration'][()], self.dx, self.dy, self.simulation_duration, self['time'][0], self['time'][-1]]]  # noqa: E501

        meta_data_str = f"Simulation has been run on the {green(self['simulation_date'])} on {green(self['gpu_or_cpu'])} for {green(sim_comput_duration)} seconds on the {green(self['architecture'])} architecture on the branch {green(self['git_branch_name'])} (with corresponding commit hash {green(self['git_commit_hash'])}).\n\n"  # noqa: E501


        spatial_mesh_str = f"The spatial mesh is Nx x Ny = {green(self.Nx)}x{green(self.Ny)} with a spatial domain of Lx x Ly = {green(self['Lx'][()])} ρ0 x {green(self['Ly'][()])} {yellow('ρ0')}, i.e. a resolution of {green(self.dx)} {yellow('ρ0')} in the x direction and {green(self.dy)} {yellow('ρ0')} in the y direction.\n\n"  # noqa: E501

        t_norm = yellow("L/c₀") if self["k_HW"][()] != 0 else yellow("ω₀⁻¹")

        temporal_mesh_str = f"The simulation spans a time from {green(start_time)} {t_norm} to {green(end_time)} {t_norm} (i.e. for {green(sim_duration)} {t_norm}) with a resolution for the RK4 scheme of {green(self.dt_RK4)} {t_norm} and the data outputted every {green(self.dt_diag)} {t_norm} {red('(ps: if number seems sketchy, do not trust them.)')}.\n\n"  # noqa: E501

        return meta_data_str + spatial_mesh_str + temporal_mesh_str

    def _get_first_data_file(self) -> Path:
        """Temporary method to get the first data file in the simulation folder.

        This will be updated to account for restarts.
        """
        try:
            return sorted([file for file in list(self.sim_folder_path.iterdir()) if 'data_TOKAM_run' in file.stem], key=lambda x: int(x.stem.split('_')[3]))[0]  # noqa: E501
        except:
            return self.sim_folder_path / 'simulation_fields.h5'

    def __setitem__(self, key: str, value: Any) -> None:  # noqa: ANN401
        """Assign new data."""
        self.data[key] = value

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401
        """Dictionary-style access to simulation data and inputs."""
        available_keys = list(self.sim_data.keys()) \
                         + list(self.sim_inputs.keys()) \
                         + list(self.sim_metadata.keys()) \
                         + list(self.data.keys())
        # General case
        if key in self.sim_data:
            return self.sim_data[key]
        elif key in self.data:
            return self.data[key]
        elif key in self.sim_inputs:
            return self.sim_inputs[key]
        elif key in self.sim_metadata:
            if type(val := self.sim_metadata[key][()]) is bytes:
                return val.decode("utf-8")
            else:
                return val
        # Specific for Philippe"s version
        elif key == "x" and key not in available_keys:
            return self.sim_data["x_pos"]
        elif key == "y" and key not in available_keys:
            return self.sim_data["y_pos"]
        elif key == "time" and key not in available_keys:
            return self.sim_data["time_evol"]
        elif key == "density" and key not in available_keys:
            try:
                return self.sim_data["dens_evol"]
            except KeyError:
                raise KeyError(f"Key {key}, use the get_data_slice method to compute it on the fly.")
        elif key == "potential" and key not in available_keys:
            try:
                return self.sim_data["phi_evol"]
            except KeyError:
                raise KeyError(f"Key {key}, use the get_data_slice method to compute it on the fly.")
        # Handle previous version
        elif key == "potential_fft" and key not in available_keys:
            try:
                return self.sim_data["phi_fft"]
            except KeyError:
                raise KeyError(f"Key {key}, use the get_data_slice method to compute it on the fly.")
        elif key == "density_fft" and key not in available_keys:
            try:
                return self.sim_data["n_fft"]
            except KeyError:
                raise KeyError(f"Key {key}, use the get_data_slice method to compute it on the fly.")
        else:
            # raise KeyError(f"Key {key} not found.")
            print(f"Key {key} not found.")
            return self.data.get(key, None)

    def get_data_slice(self, field, it=None, iy=None, ix=None):
        """Fetch a data slice for a specific field, time step, spatial indices.

        Supports slices in time (it), x (ix), and y (iy) dimensions.
        Makes use of field_map for quantities not in simulation data.
        "field_map" is a dictionnary defined in __init__ that gives the rules
        to compute new fields, still making use of the lazy loading
        """
        # Define slices
        x_slice = ix if ix is not None else slice(None)
        y_slice = iy if iy is not None else slice(None)
        it_slice = it if it is not None else slice(None)

        # If the field is already in simulation output, return it
        if field in self.sim_data:
            return self[field][it_slice, y_slice, x_slice]
        
        # If only real or fft fields have been saved, compute the other one on the fly
        if field == "potential":
            return (np.fft.ifft2(self["potential_fft"][it_slice], axes=(-2, -1)).real)[..., y_slice, x_slice]
        if field == "density":
            return (np.fft.ifft2(self["density_fft"][it_slice], axes=(-2, -1)).real)[..., y_slice, x_slice]
        if field == "potential_fft":
            return np.fft.fft2(self["potential"][it_slice], axes=(-2, -1))[..., y_slice, x_slice]
        if field == "density_fft":
            return np.fft.fft2(self["density"][it_slice], axes=(-2, -1))[..., y_slice, x_slice]

        # If the name of the field is not in the mapping, raise an error
        if field not in self.field_mapping:
            raise KeyError(f"Field {field} not found in mapping.")

        # Else, compute the field using the mapping rules
        params = self.field_mapping[field]
        base_fields, dx, dy, power, operators, sign = [params[key] for key in ["base_fields", "dx", "dy", "power", "operators", "sign"]]

        # Handle single field (no operators)
        if operators is None:
            return sign * self.derive_field(
                field=base_fields[0], dx=dx[0], dy=dy[0], power=power[0],
                it=it_slice, iy=y_slice, ix=x_slice
            )

        # Handle combinatory fields
        return sign * self.derive_combination(
            fields=base_fields, operators=operators, 
            derivatives=list(zip(dx, dy)), powers=power,
            it=it_slice, iy=y_slice, ix=x_slice
        )

    @staticmethod
    def _field_key_fft(field_key: str) -> str:
        """Convert a real field key to its corresponding FFT key."""
        if field_key in ["density","n"]: field_key = "density_fft"
        elif field_key in ["potential","phi"]: field_key = "potential_fft"
        return field_key

    @staticmethod
    def _field_key_real(field_key: str) -> str:
        """Convert an FFT field key to its corresponding real key."""
        if field_key in ["n_fft","n","density_fft"]: field_key = "density"
        elif field_key in ["phi_fft","phi","potential_fft"]: field_key = "potential"
        return field_key

    def derive_field(self, field, dx, dy, power, it=None, iy=None, ix=None):
        """Compute any derivative of a field using spectral differentiation.

        Parameters
        ----------
        - field: Field name as a string (e.g., "density", "potential").
        - dx, dy: Derivative orders in the x and y directions (integers).
        - power: Power to raise the derivative to (integer).

        Returns
        -------
        - 3D array (if it is a slice) or 2D array (if it is an integer).

        """
        ikx = (1j * self.kx2D)**dx if dx != 0 else 0
        iky = (1j * self.ky2D)**dy if dy != 0 else 0
        ik = ikx + iky

        if dx == dy:
            if dx == 0:
                return self[self._field_key_real(field)][it, iy, ix]**power
            else:
                ik = (1j * self.k2D)**dx

        field_fft = self.get_data_slice(self._field_key_fft(field), it=it)
        derivative_fft = ik * field_fft
        return ((np.fft.ifft2(derivative_fft, axes=(-2, -1)).real)**power)[..., iy, ix]

    def derive_combination(self, fields, operators, derivatives, powers, it=None, iy=None, ix=None):
        """Combine derivatives of multiple fields using specified operators.

        Parameters
        ----------
        - fields: List of field names (e.g., ["density", "potential", "density"]).
        - operators: List of operators as strings (e.g., ["*", "/"]).
        - derivatives: List of tuples for (dx, dy) orders (e.g., [(1, 0), (0, 1), (2, 0)]).
        - powers: List of powers for each field's derivative (e.g., [1, 1, 1]).

        Returns
        -------
        - Combined result as a 2D array.

        """
        if len(fields) != len(derivatives) or len(fields) != len(powers):
            raise ValueError("fields, derivatives, and powers must have the same length")
        if len(operators) != len(fields) - 1:
            raise ValueError("Number of operators must be one less than the number of fields")

        # Compute the first field's derivative
        result = self.derive_field(fields[0], *derivatives[0], powers[0], it, iy, ix)

        # Apply operators sequentially
        for i, op in enumerate(operators):
            next_field = self.derive_field(fields[i + 1], *derivatives[i + 1], powers[i + 1], it, iy, ix)
            op_func = operator_map[op]  # Map operator string to function
            result = op_func(result, next_field)

        return result

    def get_spatiotemporal_slice(self, field, it=None, ix=None):
        """Compute the spatiotemporal slice of a field.

        Parameters
        ----------
        - field: Field name as a string (e.g., "density", "potential").

        Returns
        -------
        - Result as a 2D array.

        TODO: Optimize the FFT of field exist or appears in derive_field to just
        keep the ky=0 component and then do the ifft

        """
        ix_slice = ix if ix is not None else slice(None)  # Full x dimension if ix is None
        it_slice = it if it is not None else slice(None)  # Full time dimension if it is None
        return np.mean(self.get_data_slice(field, it=it_slice, ix=ix_slice), axis=1)

    def _get_kx(self) -> np.ndarray:
        """Compute the kx grid if not present in the data."""
        if "kx" in self.sim_data:
            return self.sim_data["kx"]
        x = (self["x"] if "x" in self.sim_data else self["x_pos"])
        xgrid = np.arange(0, len(x))
        kx_norm = 2*np.pi/np.array(x[-1])
        return np.fft.ifftshift(kx_norm*(xgrid - len(x)/2))

    def _get_ky(self) -> np.ndarray:
        """Compute the ky grid if not present in the data."""
        if "ky" in self.sim_data:
            return self.sim_data["ky"]
        y = (self["y"] if "y" in self.sim_data else self["y_pos"])
        ygrid = np.arange(0, len(y))
        ky_norm = 2*np.pi/np.array(y[-1])
        return np.fft.ifftshift(ky_norm*(ygrid - len(y)/2))

    def export_to_h5(self, field, path=None, filename=None, it=None, ix=None, iy=None, save_grid=False):
        """Export a specific field to an HDF5 file.

        Supports exporting a full 3D array, a time step, or a spatial slice.
        """
        if path is None:
            path = self.sim_folder_path
        
        field_name = ''
        for val in (field if isinstance(field, list) else [field]):
            field_name += f"{val}_"
        field_name = field_name[:-1] 

        if filename is None:
            filename = f"{field_name}_{self.name}.h5"

        with h5py.File(path/filename, "a") as f:
            for val in (field if isinstance(field, list) else [field]):
                data = self.get_data_slice(val, it, iy, ix)
                f.create_dataset(val, data=data)

        if save_grid:
            with h5py.File(path/filename, "a") as f:
                f.create_dataset("x", data=self.x)
                f.create_dataset("y", data=self.y)
                f.create_dataset("time", data=self.time)

        print(f"Exported {field_name} to {path/filename}.")


    # Function to generate and save a single frame, optimized for parallel execution
    @staticmethod
    def _save_frame(params):
        """Generate and save a single frame, optimized for parallel execution."""
        for_IA, scheme, save_folder_path, acronym_simu, it, time_slice, data_slice, data_name, cmap, vmin, vmax, Nx, Ny, dpi = params
        frame_name = f'{acronym_simu}_{data_name}_{it:05d}.png'
        filepath = os.path.join(Path(save_folder_path), frame_name)
        logging.info(f'Generating frame {it}')
        time_norm = '$[\\omega_{{c0}}^{{-1}}]$'
        if scheme=='HW': time_norm = '$[L/c_0]$'
        # Use BytesIO as an in-memory buffer, so that the disk is not accessed when generating the frames
        with io.BytesIO() as buf:
            if Nx > Ny:
                size_x = Nx / dpi
                size_y = size_x * (Ny / Nx)
            else:
                size_y = Ny / dpi
                size_x = size_y * (Nx / Ny)
            size_x *= 1.05

            fig, ax = plt.subplots(figsize=(size_x, size_y), dpi=dpi)

            if vmin == None: vmin = np.min(data_slice)
            if vmax == None: vmax = np.max(data_slice)

            p = ax.imshow(data_slice, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
            if not for_IA:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = fig.colorbar(p, cax=cax)
                cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2e}'))
                ax.set_title(f'{data_name.capitalize()} at time = {time_slice:.2f} {time_norm}')
                ax.set_xlabel(r'x $[\rho_0]$')
                ax.set_ylabel(r'y $[\rho_0]$')
            else:
                ax.axis('off')
            fig.tight_layout()
            fig.savefig(buf, format='png')
            plt.close(fig)  # Close the figure to free memory
            buf.seek(0)
            with open(filepath, 'wb') as f:
                f.write(buf.getvalue())
        return filepath

    # Function to generate and save frames, with optional parallel execution
    def _generate_and_save_frames(self, parallel, num_cores, for_IA, scheme, save_folder_path, acronym_simu, time_frames, data_frames, data_name, cmap, vmin=None, vmax=None, Nx=256, Ny=256, dpi=128):
        """Generate and save frames, with optional parallel execution."""
        frames = []
        logging.info(f"Running in parallel using {num_cores} cores.")
        save_folder_path_frame = Path(save_folder_path)/f'{acronym_simu}_{data_name}_frames'
        save_folder_path_frame.mkdir(parents=True, exist_ok=True)
        args = [(for_IA, scheme, save_folder_path_frame, acronym_simu, it, time_slice, data_frames[it, :, :], data_name, cmap, vmin, vmax, Nx, Ny, dpi) for it, time_slice in enumerate(time_frames)]

        if parallel:
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                frames = list(executor.map(self._save_frame, args))

        else:
            for it, time_slice in enumerate(time_frames):
                frames.append(self._save_frame(args[it]))

        return frames

    # Function to compile frames into a movie
    def _compile_movie(self, title, frames, save_folder_path, fps):
        """Compile frames into a movie."""
        movie_filename = f'{title}.mp4'
        movie_filename_path = os.path.join(save_folder_path, movie_filename)
        logging.info(f'Compiling movie {movie_filename_path}')
        with imageio.get_writer(movie_filename_path, fps=fps) as writer:
            for frame_path in frames:
                writer.append_data(imageio.imread(frame_path))
        logging.info(f'Movie {title} created successfully at {save_folder_path}!')

    def make_movie(self, field, path=None, filename=None, it_slice=None, parallel=True, num_cores=None, for_IA=False, scheme=False, cmap='plasma', vmin=None, vmax=None, fps=30, save_frames=False, custom_field_name='custom_field'):
        """Generate a movie for a specific field.

        Optional parallel execution.
        Supports slices in time (it).
        """
        original_backend = mpl.get_backend()
        mpl.use('Agg')  # This is necessary to generate images without a display

        it_slice = it_slice if it_slice is not None else slice(None)  # Full time dimension if it is None

        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Make the output directory if it does not exist
        if path is None:
            path = self.sim_folder_path / 'movies'
        else:
            path = Path(path) / 'movies'
        path.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = self.name

        if num_cores is None:
            num_cores = os.cpu_count()
        parallel_warning = "\nWARNING: Parallel execution might get stuck if the CPU usage is too high prior to running this script." + \
                        "\nIf this happens, you can try to kill the processes intensively using CPU, e.g. VScode (use can use the 'htop' shell command to see a list of current processes)."
        if parallel: logging.warning(parallel_warning)

        start_time = time.time()

        # **Load the time and field slices into memory** before passing them to parallel workers
        if isinstance(field, str):
            time_frames = np.array(self['time'][it_slice])
            data_frames = np.array(self.get_data_slice(field, it=it_slice))
        elif isinstance(field, np.ndarray):
            logging.warning("Custom field provided as ndarray, itslice will be ignored.")
            time_len = field.shape[0]
            time_frames = np.array(self['time'][:time_len])
            data_frames = field
            field = custom_field_name # Name for the custom field

        Nx = self.Nx
        Ny = self.Ny
        dpi = max(Nx,Ny)/12
        
        plt.rcParams.update({'font.size': 14 * (100 / dpi)})
        plt.rcParams.update({'axes.titlesize': 14 * (100 / dpi)})
        plt.rcParams.update({'axes.labelsize': 14 * (100 / dpi)})

        # Generate frames using the data already loaded into memory
        frames = self._generate_and_save_frames(parallel, num_cores, for_IA, scheme, path, filename, time_frames, data_frames, field, cmap, vmin, vmax, Nx, Ny, dpi)

        # Compile the movie from the generated frames
        self._compile_movie(f'{filename}_{field}_movie', frames, path, fps)

        if not save_frames:
            for frame in frames:
                os.remove(frame)
            os.rmdir(path / f'{filename}_{field}_frames')
            logging.info('Frames deleted successfully!')

        logging.info(f"Execution time: {time.time() - start_time:.2f} s")

        # return to the original backend
        mpl.use(original_backend)

def set_plot_defaults() -> None:
    """Set default plotting parameters."""
    plt.rcParams.update({
        "image.cmap": "Spectral_r",
        "axes.formatter.limits": (-3, 3),
        "lines.linewidth": 2.5,
        "axes.grid": True,
        "lines.markersize": 8,
        "lines.markeredgecolor": "k",
        "lines.markeredgewidth": 2.0,
        "font.size": 20,
    })

def red(text: str) -> str:
    return f"\033[91m{text!s}\033[0m"

def green(text: str) -> str:
    return f"\033[92m{text!s}\033[0m"

def yellow(text: str) -> str:
    return f"\033[93m{text!s}\033[0m"

def blue(text: str) -> str:
    return f"\033[94m{text!s}\033[0m"

def magenta(text: str) -> str:
    return f"\033[95m{text!s}\033[0m"

def cyan(text: str) -> str:
    return f"\033[96m{text!s}\033[0m"

def white(text: str) -> str:
    return f"\033[97m{text!s}\033[0m"

def black(text: str) -> str:
    return f"\033[90m{text!s}\033[0m"
