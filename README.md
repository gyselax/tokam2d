# Installation
1. Create a Python environment (recommended Python 3.11 or later). You can use either `venv` or `conda`:

Ex: With conda
```bash
conda create --name tokam_env python=3.11
conda activate tokam_env
```

2. A few packages are needed, in particular JAX (the code is optimized for GPU but also runs on CPUs). To install them:

```bash
cd <TOKAM2D_directory>
pip install -r requirements_gpu.txt or
pip install -r requirements_cpu.txt
```

# Run Tokam2D
```bash
cd <TOKAM2D_directory>
python3 main.py -i input_files/input_file.yaml (-o output_folder)
```

# Output data (saved as dictionnary stored in HDF5 format)
- Simulation results containing 
    - the evolved fields, (e.g. density, electric potential) as 3D arrays, as real signal and/or fourier components depending on the simulation input;
    - the time and space coordinates arrays.
- The parameters used for the simulation;
- The metadata (code version, date, etc.).
- The log file of the simulation.

# Use the diagnostic tools
Use the notebook in `diagnostics/diag_main.ipynb` to explore the results of a simulation or create movies.