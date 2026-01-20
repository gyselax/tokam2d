# src/interfaces/input_reader.py
import yaml
import jax.numpy as np
from pathlib import Path
import copy

#TODO: Unsatisfactoty: an IO module calling a core module, to be refactored
from src.model.pde import PDE_REGISTRY

class StaticParams:
    """Class to read the input file and set the simulation parameters."""
    def __init__(self, filepath, quiet=False):
        # Set the logger
        self.logger, self.mem_handler = set_logger()
        self.quiet = quiet
        if quiet: self.logger.setLevel("ERROR")

        # Deepcopy used so that new instantiation of StaticParams has no memory
        # of previous ones (in case of multiple simulations in the same script)
        if type(filepath) is dict:
            self.user = copy.deepcopy(filepath)
        else:
            # Convert input yaml file to a dictionary
            self.filepath = Path(filepath)
            self.user = copy.deepcopy(self.read_input())

        # Ensure retrocompatibility with old input files and set default values
        self._ensure_retrocompatibility()
        self._set_default()

        # Get some recurrent parameters
        self.Nx = self.user['grid']['Nx']
        self.Ny = self.user['grid']['Ny']
        self.Lx = self.user['grid']['Lx']
        self.Ly = self.user['grid']['Ly']
        self.dx = self.Lx/self.Nx
        self.dy = self.Ly/self.Ny
        self.x_1d = np.linspace(0, self.Lx, self.Nx, endpoint=False)
        self.y_1d = np.linspace(0, self.Ly, self.Ny, endpoint=False)

        # Prepare the wave numbers and set the precision
        self._prepare_wave_numbers()
        self._set_precision()

        # Make the FFT mask
        self.mask_fft = self._make_mask_fft()

        # Get the PDE parameters
        self.pde = PDE_REGISTRY[self.user['pde']['eq']](self)

        # Handle the time stepping
        if self.user['pde']['eq'] in ["HW", "mHW", "BHW"]:
            self._set_time_stepping_HW()

        self._time_stepping()
        self.Tsim = self.user['time']['Tsim']
        self.Nt_diag = self.user['time']['Nt_diag']
        self.Nt_rk4 = self.user['time']['Nt_rk4']
        self.rk4_per_diag = self.user['time']['rk4_per_diag']
        self.dt_rk4 = self.user['time']['dt_rk4']
        self.dt_diag = self.user['time']['dt_diag']

        #TODO: To be amended when restarting from a previous simulation

    def read_input(self):
        with open(self.filepath, 'r') as f:
            user_params = yaml.safe_load(f)
        return user_params

    def _prepare_wave_numbers(self):
        Nx_half = int(self.Nx/2)
        Ny_half = int(self.Ny/2)
        un_x = np.ones(self.Nx)
        un_y = np.ones(self.Ny)

        kx_struc = np.zeros(self.Nx)

        kx_struc_firsthalf = np.linspace(0,Nx_half-1,Nx_half)
        kx_struc_secondhalf = -np.linspace(Nx_half,1,Nx_half)
        kx_struc = np.concatenate((kx_struc_firsthalf, kx_struc_secondhalf), axis=0)

        ky_struc = np.zeros(self.Ny)

        ky_struc_firsthalf = np.linspace(0,Ny_half-1,Ny_half)
        ky_struc_secondhalf = -np.linspace(Ny_half,1,Ny_half)
        ky_struc = np.concatenate((ky_struc_firsthalf, ky_struc_secondhalf), axis=0)

        self.dkx = 2*np.pi/self.Lx
        self.dky = 2*np.pi/self.Ly

        self.kx_2d = (un_y[:,np.newaxis]*kx_struc[np.newaxis,:])*self.dkx
        self.ky_2d = (ky_struc[:,np.newaxis]*un_x[np.newaxis,:])*self.dky

        self.kx_1d = kx_struc*self.dkx
        self.ky_1d = ky_struc*self.dky

        self.kx_max = max(self.kx_1d)
        self.ky_max = max(self.ky_1d)

        self.k2_2d = self.kx_2d*self.kx_2d + self.ky_2d*self.ky_2d

        k2_2d_star = self.k2_2d.at[0, 0].set(1.)

        self.inv_k2_2d = 1/k2_2d_star

    def _make_mask_fft(self):
        """Initialize the FFT mask."""
        mask_fft = np.ones((self.Ny, self.Nx))

        if self.user["inline_operations"].get("fft_filter", False):
            Nx_half = int(self.Nx/2)
            Ny_half = int(self.Ny/2)

            Nx_f = int(Nx_half/3)
            Ny_f = int(Ny_half/3)

            #TODO : adapt for non JAX
            mask_fft = mask_fft.at[Nx_half-Nx_f:Nx_half+Nx_f + 1, :].set(0)
            mask_fft = mask_fft.at[:, Ny_half-Ny_f:Ny_half+Ny_f+1].set(0)

        return mask_fft

    def _set_precision(self):
        bool_prec = self.user["grid"].get("x64_precision", False)
        if bool_prec:
            # Set default dtype to complex128
            try:
                from jax.config import config
            except ImportError:
                from jax import config
            config.update("jax_enable_x64", bool_prec)
            self.logger.info("Running JAX with float64 precision")

    def get_2d_grid(self):
        un_x = np.ones(self.Nx)
        un_y = np.ones(self.Ny)
        x_2d = un_y[:,np.newaxis]*self.x_1d[np.newaxis,:]
        y_2d = self.y_1d[:,np.newaxis]*un_x[np.newaxis,:]
        return x_2d, y_2d

    def _set_time_stepping_HW(self):
        """Set the time stepping for the HW case.
        
        Based on the prediction of the most unstable mode of the linear problem.
        """
        #TODO: improve readability of the function

        # Retrocompatibility
        if self.user.get("k_HW") == 1: self.user["pde"]["eq"] = "HW"
        elif self.user.get("k_HW") == 2: self.user["pde"]["eq"] = "mHW"

        if self.user["pde"]["eq"] in ["HW", "mHW", "BHW"]:
            self.logger.info(f"""Simulation with {self.user["pde"]["eq"]} scheme, total simulation duration is set to "HW_duration" number of inverse linear growth rates.""")
            self.logger.info("""To override this, set "HW_duration=0" in the input file""")
            if "HW_duration" not in self.user["pde"]:
                self.user["pde"]["HW_duration"] = 50.
                self.logger.info(f'HW_duration is not specified, set to {self.user["pde"]["HW_duration"]} inverse linear growth rates')
            if self.user["pde"].get("HW_duration")!=0:
                gamma, omega_r = self.time_step_HW_interchange()
                self.logger.info(f"Linear growth rate of most unstable mode = {gamma:.2e}")
                self.logger.info(f"Linear frequency of most unstable mode = {omega_r:.2e}")
                # Set the new simulation time
                self.user['time']['Tsim'] = self.user["pde"]["HW_duration"] / gamma
                # If dt_diag and dt_rk4 are specified, remove them as they have
                # to be recomputed with the new Tsim
                self.user["time"].pop("dt_diag", None)
                self.user["time"].pop("dt_rk4", None)
            else:
                self.logger.info(f'Simulation duration set to Tsim={self.user["time"]["Tsim"]}')

    def _time_stepping(self):
        self.time_stepper = TimeStepping(self.user["time"])
        self.user["time"].update(self.time_stepper.params)

    def _ensure_retrocompatibility(self):
        """Ensure retrocompatibility with old input files."""
        caterories_list = ['grid', 'time', 'pde', 'source', 'initial_condition', 'inline_operations', 'callbacks']
        is_old_input = all(category not in self.user for category in caterories_list)
        
        for category in caterories_list:
            if category not in self.user:
                self.user[category] = {}

        def deprecated(old_key, new_key, category, new_value=None):
            if new_key in self.user[category]:
                self.logger.warning(f"{new_key} already set to {self.user[category][new_key]}, {old_key} will not be set")
                return
            if not is_old_input:
                self.logger.warning(f"{old_key} is deprecated, use {new_key} instead. Setting {new_key} to {new_value}")
            self.user[category][new_key] = new_value
            # self.user.pop(old_key, None)

        def retro_HW(k_HW):
            if k_HW == 1: return "HW"
            elif k_HW == 2: return "mHW"
            else: return "SOL"

        deprec_tuples = [
            # Grid
            ("nx"           , "Nx"                  , "grid"  , int(2**self.user.get("nx", 0))),
            ("ny"           , "Ny"                  , "grid"  , int(2**self.user.get("ny", 0))),
            # Time
            ("nt"           , "dt_rk4"              , "time"  , 1/(2**self.user.get("nt", 0))),
            ("mt"           , "dt_diag"             , "time"  , 2**self.user.get("mt", 0)),
            ("Nt"           , "Nt_diag"             , "time"  , self.user.get("Nt")),
            # PDE
            ("k_HW"         , "eq"                  , "pde",    retro_HW(self.user.get("k_HW"))),
            ("sigma_n"      , "sigma_nn"            , "pde"   , self.user.get("sigma_n")),
            ("sigma_phi"    , "sigma_phiphi"        , "pde"   , self.user.get("sigma_phi")),
            ("r_n"          , "sigma_nphi"          , "pde"   , self.user.get("r_n", 0.)*self.user.get("sigma_n", 0)),
            ("r_phi"        , "sigma_phin"          , "pde"   , self.user.get("r_phi", 0.)*self.user.get("sigma_phi", 0)),
            ("gradn"        , "kappa"               , "pde"   , -self.user.get("gradn", 0.)),
            ("D0"           , "Dn"                  , "pde"   , self.user.get("D0")),
            ("nu0"          , "Dphi"                , "pde"   , self.user.get("nu0")),
            # Initial condition
            ("kx0"          , "init_harm_kx"        , "initial_condition", [self.user.get("kx0")]),
            ("ky0"          , "init_harm_ky"        , "initial_condition", [self.user.get("ky0")]),
            ("ampl_phi0"    , "init_harm_ampl_phi"  , "initial_condition", [self.user.get("ampl_phi0")]),
            ("ampl_phi0rand", "init_rand_ampl_phi"  , "initial_condition", self.user.get("ampl_phi0rand")),
            ("dens0"        , "init_uni_ampl_n"     , "initial_condition", self.user.get("dens0")),
            # ("dens_S"       , "init_gauss_ampl_n"   , "initial_condition", [self.user.get("dens_S",0) * self.user.get("S0",0)]),
            ("dens_S"       , "init_gauss_ampl_n"   , "initial_condition", [0.]), #TODO: to be fixed
            ("wsx"          , "init_gauss_sigma_x"  , "initial_condition", [self.user.get("wsx",0)]),
            ("wsx"          , "init_gauss_sigma_y"  , "initial_condition", [0.]),
            # Source
            ("x_ovd"        , "source_gauss_x0"     , "source", [self.user.get("x_ovd")]),
            ("y_ovd"        , "source_gauss_y0"     , "source", [self.user.get("y_ovd")]),
            ("r_ovd"        , "source_gauss_sigma_x", "source", [self.user.get("r_ovd")]), #
            ("r_ovd"        , "source_gauss_sigma_y", "source", [self.user.get("r_ovd")]), #
            ("Lgrad_ovd"    , None                  , "source", [self.user.get("Lgrad_ovd")]), #No equivalent in new code yet
            ("S0"           , "source_gauss_ampl_n" , "source", [self.user.get("S0",0)/(np.sqrt(2*np.pi)*self.user.get("wsx",0))]),
            ("wsx"          , "source_gauss_sigma_x", "source", [self.user.get("wsx",0)]),
            ("wsx"          , "source_gauss_sigma_y", "source", [0.]),
            # Inline operations
            ("k_filter_fft" , "fft_filter"      , "inline_operations", self.user.get("k_filter_fft")),
        ]

        for old_key, new_key, category, new_value in deprec_tuples:
            if old_key in self.user or old_key in self.user[category]:
                deprecated(old_key, new_key, category, new_value)

        if is_old_input:
            self.logger.warning('Old input format detected, processing retrocompatibility...')
            self.user['initial_condition']['init_rand_ampl_phi'] = 1.e-16 # Initilization of the new code is "too precise" now so a very small seed noise is needed or some cases never reach non-linear phase

    def _set_default(self):

        default_tuples = [
            # Grid
            ("Nx"               , "grid"             , None),
            ("Ny"               , "grid"             , None),
            ("Lx"               , "grid"             , float(self.user['grid']["Nx"])),
            ("Ly"               , "grid"             , float(self.user['grid']["Ny"])),
            # PDE
            ("eq"               , "pde"              , "SOL"),
            ("sigma_nn"         , "pde"              , None),
            ("sigma_phiphi"     , "pde"              , None),
            ("sigma_nphi"       , "pde"              , None),
            ("sigma_phin"       , "pde"              , None),
            ("kappa"            , "pde"              , None),
            ("Dn"               , "pde"              , None),
            ("Dphi"             , "pde"              , None),
            ("C"                , "pde"              , None),
            ("g"                , "pde"              , None),
            # Initial condition
            ("load_init_fields"   , "initial_condition", False),
            ("init_gauss_ampl_n"  , "initial_condition", [0.]),
            ("init_gauss_ampl_phi", "initial_condition", [0.]),
            ("init_gauss_x0"      , "initial_condition", [0.]),
            ("init_gauss_y0"      , "initial_condition", [0.]),
            ("init_gauss_sigma_x" , "initial_condition", [0.]),
            ("init_gauss_sigma_y" , "initial_condition", [0.]),
            ("init_uni_ampl_n"    , "initial_condition", 1.),
            ("init_uni_ampl_phi"  , "initial_condition", 0.),
            ("init_harm_kx"       , "initial_condition", [0]),
            ("init_harm_ky"       , "initial_condition", [0]),
            ("init_harm_ampl_n"   , "initial_condition", [0.]),
            ("init_harm_ampl_phi" , "initial_condition", [0.]),
            ("init_rand_ampl_n"   , "initial_condition", 0.),
            ("init_rand_ampl_phi" , "initial_condition", 0.),
            # Source
            ("source_gauss_ampl_n"  , "source"         , [0.]),
            ("source_gauss_ampl_phi", "source"         , [0.]),
            ("source_gauss_x0"      , "source"         , [0.]),
            ("source_gauss_y0"      , "source"         , [0.]),
            ("source_gauss_sigma_x" , "source"         , [0.]),
            ("source_gauss_sigma_y" , "source"         , [0.]),
            # Inline operations
            ("fft_filter"                , "inline_operations", True),
            ("compute_time_derivatives"  , "inline_operations", False),
            # Callbacks
            ("check_crash"     , "callbacks", True),
            ("save_real"       , "callbacks", True),
            ("save_fft"        , "callbacks", False),
        ]

        def default(key, category, default_value):
            if key not in self.user[category]:
                # If the key not category, try to find it in the root (old inputs) else set it to the default value
                try:
                    self.user[category][key] = self.user[key]
                except:
                    self.user[category][key] = self.user.get(key, default_value)
                    self.logger.warning(f"{key} not found in {category} category, set to {key}={default_value}")

        for key, category, default_value in default_tuples:
            default(key, category, default_value)

    def time_step_HW_interchange(self) -> tuple:
        #TODO: To be moved to a more appropriate place like pde.py
        """Initialize of the time step in the HW case.

        Based on the prediction of the most unstable mode of the linear problem.
        Issue: Troubles occur when the linear growth rate is negative or too close
        to zero.
        Also, the poloidal velocity curvature is stabilizing but is not considered
        in the estimation (although it could, but simulation duration seems better
        tailored without it).
        """
        # Define parameters
        ky = np.sqrt(2) #Most unstable mode in the HW case
        kx = 0
        k_square = kx**2 + ky**2
        k_square_inv = 1/k_square
        c = self.user["pde"]["C"]
        kappa = self.user["pde"]["kappa"]

        ampl_phi0_list = self.user["initial_condition"]["init_harm_ampl_phi"]
        ampl_phi0 = max(ampl_phi0_list)
        kx0 = self.user["initial_condition"]["init_harm_kx"][ampl_phi0_list.index(ampl_phi0)]

        Vy_xcurv = abs(ampl_phi0 * kx0**3 * (2*np.pi/float(self.Lx))**3)

        g = self.user["pde"]["g"]

        # Define the proxies
        # a = ky * k_square_inv * Vy_xcurv
        a = 0 # /!\ Vy_xcurv is linearly stabilizing
        b = c*(1 + k_square_inv)
        # c1 = C * ky * k_square_inv * (gradn + Vy_xcurv - g)
        c1 = c * ky * k_square_inv * (kappa - g) # /!\ Vy_xcurv is linearly stabilizing
        c2 = ky**2*k_square_inv*g*kappa 
        d = np.sqrt( (a**2 - b**2 - 4*c2)**2 + 4*(a*b - 2*c1)**2 )

        # Frequency without the Doppler shift
        omega_r = 0.5*a + np.sqrt( (a**2 - b**2 - 4*c2 +d)/8 )
        # Growth rate
        gamma = -0.5*b  + np.sqrt( (b**2 - a**2 + 4*c2 +d)/8 )

        return gamma, omega_r

    def __repr__(self):
        #TODO: add unity based on pde

        def handle_spaces(string):
            maxline = max([len(line) for line in string.split('\n')])
            new_text = ""
            for line in string.split('\n'):
                while len(line) < maxline:
                    line += ' '
                new_text += line + '\n'
            return new_text

        def combine_str_blocks(str_a, str_b):
            return '\n'.join(map(str.__add__, str_a.split('\n'), str_b.split('\n')))

        spatial_grid= handle_spaces(f"""
          -- Spatial grid --

        Ly={self.Ly:.2f}
        |
        |
        | Ny x Nx = {self.Ny} x {self.Nx}
        |    _                   
        |   |_|dy = {self.dy:.2f}
        |   dx = {self.dx:.2f}   
       0|_______________ Lx={self.Lx:.2f}
        0    
        """)

        spectral_grid= handle_spaces(f"""
          -- Spectral grid --
            
        max(ky)={self.ky_max:.2f}         
        |                         
        |                                 
        |
        |    _                   
        |   |_|dky = {self.dky:.2f}	    
        |   dkx = {self.dkx:.2f}         
     __0|_______________ max(kx)={self.kx_max:.2f}  
        |0    
        """)

        time_grid= handle_spaces(f"""
                             -- Temporal grid --

            dt_rk4 = {self.dt_rk4:.2e}  and  Nt_rk4 = {self.Nt_rk4}
               <->
            --------|--------|--------|--------|-----...-----> Tsim={self.Tsim}
            <------>
            dt_diag = {self.rk4_per_diag} x dt_rk4 = {self.dt_diag:.2f}  and  Nt_diag = {self.Nt_diag}
        """)

        return combine_str_blocks(spatial_grid, spectral_grid) + time_grid

class TimeStepping():
    def __init__(self, params):
        self.required_keys = ['Tsim', 'Nt_rk4', 'Nt_diag', 'dt_rk4', 'dt_diag', 'rk4_per_diag']
        self.params = params
        self.verify_param_structure()
        self.nb_undefined = list(self.params.values()).count(None)
        self.undefined_keys = [key for key, value in self.params.items() if value is None]
        self.defined_keys = [key for key, value in self.params.items() if value is not None]
        self._check_underspecified()
        self._check_overspecified()
        self.compute_missing_entries()

    def verify_param_structure(self):
        for key in self.required_keys:
            self.params[key] = self.params.get(key, None)

    def _check_underspecified(self):
        str_keys = ', '.join(self.defined_keys)
        if self.nb_undefined > 3:
            raise ValueError(f"Time scheme underspecified with only {str_keys}")
        elif self.nb_undefined == 3:
            if self.params['rk4_per_diag'] is not None:
                if self.params['Nt_diag'] is not None and self.params['Nt_rk4'] is not None:
                    raise ValueError(f"Time scheme underspecified with {str_keys}: cannot determine simulation duration Tsim")
                elif self.params['dt_diag'] is not None and self.params['dt_rk4'] is not None:
                    raise ValueError(f"Time scheme underspecified with {str_keys}: cannot determine simulation duration Tsim")

    def _check_overspecified(self):
        str_keys = ', '.join(self.defined_keys)
        if self.nb_undefined < 3:
            raise ValueError(f"Time scheme overspecified with {str_keys}")

    def compute_missing_entries(self):

        time_stepping_rules = {
            "dt_diag": lambda x: (x["dt_rk4"] * x["rk4_per_diag"] if (x["dt_rk4"] is not None and x["rk4_per_diag"] is not None)
                else x["Tsim"] / x["Nt_diag"]
            ),
            "dt_rk4": lambda x: (x["dt_diag"] / x["rk4_per_diag"] if (x["dt_diag"] is not None and x["rk4_per_diag"] is not None)
                else x["Tsim"] / x["Nt_rk4"]
            ),
            "rk4_per_diag": lambda x: int(x["dt_diag"] / x["dt_rk4"]) if (x["dt_diag"] is not None and x["dt_rk4"] is not None) else int(x["Nt_diag"] / x["Nt_rk4"]),
            "Nt_diag": lambda x: int(x["Tsim"] // x["dt_diag"]) if (x["dt_diag"] is not None and x["Tsim"] is not None) else int(x["Nt_rk4"] // x["rk4_per_diag"]),
            "Nt_rk4": lambda x: int(x["Tsim"] // x["dt_rk4"]) if (x["dt_rk4"] is not None and x["Tsim"] is not None) else int(x["Nt_diag"] * x["rk4_per_diag"]),
            "Tsim": lambda x: x["Nt_diag"] * x["dt_diag"] if (x["dt_diag"] is not None and x["Nt_diag"] is not None) else x["Nt_rk4"] * x["dt_rk4"]
        }

        unresolved = set(self.undefined_keys)
        while unresolved:
            for key in (sorted(unresolved)):
                try:
                    self.params[key] = time_stepping_rules[key](self.params)
                    unresolved.discard(key)
                except:
                    pass

def set_logger():
    """Set up a logger."""
    import logging
    from logging.handlers import MemoryHandler
    logger = logging.getLogger("simulation_logger")
    logger.setLevel(logging.INFO)

    # Create a console handler (light format)
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)

    from typing import ClassVar

    class ColoredFormatter(logging.Formatter):
        COLORS: ClassVar[dict] = {
            logging.CRITICAL: "\033[1;31mERROR - ",  # bright red
            logging.ERROR: "\033[0;31m",  # red
            logging.WARNING: "\033[0;33mWARNING - ",  # yellow
            logging.INFO: "",  # normal
            logging.DEBUG: "\033[0;32m",  # green
        }

        def format(self, record):
            log_fmt = super().format(record)
            return f"{self.COLORS.get(record.levelno, '')}{log_fmt}\033[0;0m"

    console_handler.setFormatter(ColoredFormatter("%(message)s"))
    logger.addHandler(console_handler)

    # Memory Handler (to buffer logs in memory) and attach a 'target' handler later once we know the output file
    mem_handler = MemoryHandler(
        capacity=10_000_000,  # Huge buffer that should never flush before end of simulation
        flushLevel=999999,  # No automatic flush
        target=None,  # This will be set later
        flushOnClose=True
    )
    logger.addHandler(mem_handler)

    return logger, mem_handler