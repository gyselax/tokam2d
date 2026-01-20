# src/interfaces/output_saver.py
import h5py as h5
from pathlib import Path
import time
import logging
import shutil

def concatenate_h5_files(file_paths):
    """Concatenate arrays from many HDF5 files to single array in parallel."""
    from concurrent.futures import ThreadPoolExecutor
    
    # Pre-initialize the concatenated arrays
    concatenated_arrays = _initialize_concatenated_arrays(file_paths)

    # Use ThreadPoolExecutor to read files in parallel
    with ThreadPoolExecutor() as executor:
        # Submit tasks for each file
        futures = [executor.submit(
            _read_file_to_array, file_paths[i], concatenated_arrays, i
            ) for i in range(len(file_paths))]

    # Wait for all tasks to complete
    for future in futures:
        future.result()  # This is just to catch exceptions if any

    return concatenated_arrays

def _initialize_concatenated_arrays(files):
    """Pre-initialize the final output array by concatenated each array contained in the hdf5 files based on the first file.
    This assumes all files have the same structure and dtype for each array.
    Note that "standard" numpy is used (and not JAX numpy) as these arrays are memory consumming and GPU memory is limited. 
    """
    import h5py
    import numpy as np
    concatenated_arrays = {}
    if files:
        with h5py.File(files[0], "r") as f:
            for key in f:
                # Detect the array's shape and dtype
                sample_array = f[key][()]
                dtype = sample_array.dtype
                # Initialize concatenated array with an extra dimension for the files
                shape = (len(files), *sample_array.shape)
                concatenated_arrays[key] = np.zeros(shape, dtype=dtype)
    return concatenated_arrays

def _read_file_to_array(file_name, concatenated_arrays, index):
    """Read arrays from a single file and place them into the correct index
    of the pre-initialized concatenated arrays.
    """
    import h5py
    with h5py.File(file_name, 'r') as f:
        for key in f.keys():
            concatenated_arrays[key][index, ...] = f[key][()]

def save_h5_in_folder(folder, filename, data):
    """Save a dictionary of arrays to an HDF5 file."""
    from pathlib import Path
    import h5py as h5

    path = Path(folder).resolve() / f"{filename}.h5"
    hf = h5.File(path, "w")

    for key, value in data.items():
        hf.create_dataset(key, data = data[key])
    hf.close()

class OutputSaver:
    """Class to save the output of the simulation in HDF5 format."""
    def __init__(self, params, output_folder):
        # Create the output folder
        self.output_folder = self._make_output_folder(output_folder)

        # Get the loader and set the stdout to the output folder
        self.logger = params.logger
        self._set_logger_output_file(params)

        # Copy the input file to the output folder
        self._copy_input_file_to_output_folder(params)

        # Initialize the dictionnary containing the grid parameters
        self.grid_save = {}
        self.grid_save.update({'kx': params.kx_1d, 'ky': params.ky_1d})
        self.grid_save.update({'x': params.x_1d, 'y': params.y_1d})

        # Stuff for user input parameters saving
        self.user_params = params.user

        # Stuff for metadata saving
        self._start_time = time.time()

    def _set_logger_output_file(self, params):
        """Set the logger save output folder."""
        if params.quiet:
            return
        file_handler = logging.FileHandler(self.output_folder / "simulation.log", mode="w")
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        params.mem_handler.setTarget(file_handler)
        params.mem_handler.flush()
        self.logger.addHandler(file_handler)

    def _copy_input_file_to_output_folder(self, params):
        """Copy the input file to the output folder."""
        try:
            if params.filepath is not None and params.filepath.exists():
                shutil.copy(params.filepath, self.output_folder / "input.yaml")
        except Exception as e:
            self.logger.error(f"Unable to copy input file to output folder: {e}")

    def _make_output_folder(self, output_folder):
        if output_folder is None:
            default_name = 'Tokam2Drun_'+get_simulation_date()
            output_folder = Path(__file__).parent.parent/default_name

        # If file exist, add _1, _2, etc. to the name
        i = 2
        
        output_folder_temp = output_folder
        while output_folder_temp.exists(): 
            output_folder_temp = output_folder.parent / (output_folder.name + f"_{i}")
            i += 1

        if i > 1:
            print(f"Output folder {output_folder} already exists. Renaming to {output_folder_temp}...")
    
        output_folder = output_folder_temp

        Path(output_folder).mkdir(parents=True, exist_ok=True)
        return Path(output_folder)

    def save_output(self, fields, step, t):
        with h5.File(self.output_folder / f'fields_{step:05d}.h5', 'a') as f:
            for field in fields:
                f.create_dataset(field, data=fields[field])
            # f.create_dataset('time', data=t)
            f.require_dataset('time', shape=(), dtype='f', data=t)

    def save_output_data_file(self, erase_files=True):
        from pathlib import Path
        self.logger.info("Saving output fields...")

        # Find and sort files to ensure correct order
        file_paths = sorted(Path(self.output_folder).glob("fields_*.h5"))

        data_output = concatenate_h5_files(file_paths)
        self.logger.info(f" --> Concatenation of fields_*.h5 files successful.")
        data_output.update(self.grid_save)

        if erase_files:
            print(f" --> Erasing fields_*.h5 files...")
            for file in file_paths:
                file.unlink()

        save_h5_in_folder(folder=self.output_folder, 
                               filename="simulation_fields", 
                               data=data_output)

    def save_metadata_file(self):
        # Save the metadata in a hdf5 file
        self.logger.info("Saving metadata...")
        try:
            self._end_time = time.time()
            git_commit_hash, git_branch_name = get_git_commit_hash_and_branch()
            metadata_dic = {"simulation_duration": self._end_time-self._start_time,
                            "simulation_date": get_simulation_date(),
                            "git_commit_hash": git_commit_hash,
                            "git_branch_name": git_branch_name,
                            "architecture": get_architecture(),
                            "sim_name": self.output_folder.name}
            save_h5_in_folder(folder=self.output_folder, 
                                   filename="metadata", 
                                   data=metadata_dic)
        except:
            self.logger.error("Unable to save metadata.")

    def save_user_input_file(self):
        # Save the user input in a hdf5 file
        self.logger.info("Saving user inputs...")
        save_h5_in_folder(folder=self.output_folder, 
                               filename="user_inputs", 
                               data=remove_dict_first_level_nesting(self.user_params))

def get_git_commit_hash_and_branch() -> tuple:
    """Get the current Git commit hash and branch name."""
    try:
        import subprocess
        from pathlib import Path

        # Get path of current script, regardless of where it is called from
        script_dir = Path(__file__).resolve().parent

        # Run the git command to get the current commit hash
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=script_dir
            ).decode('utf-8').strip()

        branch_name = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=script_dir
            ).decode('utf-8').strip()
        return commit_hash, branch_name

    except:
        # Handle cases where the git command fails
        print("Failed to retrieve Git commit hash.")
        return 'NA', 'NA'

def get_architecture():
    """Get the architecture of the machine where the code is running."""
    try:
        import platform
        return platform.platform()
    except:
        print("Failed to retrieve the architecture.")
        return None

def get_simulation_date() -> str:
    """Get the date of simulation start."""
    import datetime
    import time
    return str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%Hh%M'))
    # return str(datetime.datetime.fromtimestamp(end_time).date())

def remove_dict_first_level_nesting(dic):
    dic_new = {}
    for key in dic:
        if isinstance(dic[key], dict):
            dic_new.update(dic[key])
        else:
            dic_new[key] = dic[key]
    return dic_new
