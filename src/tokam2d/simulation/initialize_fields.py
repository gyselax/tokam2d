# src/simulation/initialize_fields.py
import jax.numpy as np

class FieldInitiator():
    """
    Class to initialize the fields for the simulation.
    Each field can be initialized with these options:
    - A uniform background;
    - One or multiple 1D / 2D Gaussian perturbations;
    - One or multiple harmonic perturbations;
    - A random noise perturbation.
    The perturbations or cummulative and set in the input file.
    """
    def __init__(self, params):
        # Get the initialization parameters set in the input file 
        self.param_init = params.user['initial_condition']

        # Retrieve the grid parameters in real and spectral space
        self.Nx = params.Nx
        self.Ny = params.Ny
        self.Lx = params.Lx
        self.Ly = params.Ly

        self.kx_2d = params.kx_2d
        self.ky_2d = params.ky_2d

        self.x_2d, self.y_2d = params.get_2d_grid()

        # Initialize the fields in spectral space
        self.density_fft_init = np.zeros((self.Ny, self.Nx), dtype=complex)
        self.potential_fft_init = np.zeros((self.Ny, self.Nx), dtype=complex)

        # Initialize the mask for the 2/3 rule
        self.mask_fft = params.mask_fft

        # Initialize the logger
        self.logger = params.logger

        # Handle case where user provide custom initial density and potential
        self.bool_load_init = self.param_init.get('load_init_fields', False)

    def _load_custom_init(self):
        # Load custom initial fields from hdf5 file
        from pathlib import Path
        custom_init_path = Path(self.param_init.get('load_init_path', None)) / "simulation_fields.h5"
        if not custom_init_path.exists():
            raise ValueError(f"load_init_fields set to True but path {custom_init_path} does not exist.")

        self.logger.info(f"Loading custom initial fields from {custom_init_path}")
        import h5py
        with h5py.File(custom_init_path, 'r') as f:
            if 'density_fft' in f.keys() and 'potential_fft' in f.keys():
                density_fft_init = f['density_fft'][-1,...]
                potential_fft_init = f['potential_fft'][-1,...]
            elif 'density' in f.keys() and 'potential' in f.keys():
                density_real_init = f['density'][-1,...]
                potential_real_init = f['potential'][-1,...]
                density_fft_init = np.fft.fft2(density_real_init)
                potential_fft_init = np.fft.fft2(potential_real_init)
            else:
                raise ValueError("Custom initial fields file must contain either 'density_fft' and 'potential_fft' or 'density' and 'potential'.")
        self.density_fft_init = density_fft_init
        self.potential_fft_init = potential_fft_init

    def _uniform_fft_init(self):
        # Uniform initial parameters
        amp_n = self.param_init['init_uni_ampl_n']
        amp_phi = self.param_init['init_uni_ampl_phi']

        self.density_fft_init = self.density_fft_init.at[0,0].add(amp_n*self.Nx*self.Ny)
        self.potential_fft_init = self.potential_fft_init.at[0,0].add(amp_phi*self.Nx*self.Ny)

    def _2D_gaussian_fft_init(self):

        sigma_x_ar = self.param_init['init_gauss_sigma_x']
        sigma_y_ar = self.param_init['init_gauss_sigma_y']
        x0_ar = self.param_init['init_gauss_x0']
        y0_ar = self.param_init['init_gauss_y0']
        ampl_n_ar = self.param_init['init_gauss_ampl_n']
        ampl_phi_ar = self.param_init['init_gauss_ampl_phi']

        kx_2d = self.kx_2d
        ky_2d = self.ky_2d

        norm = (self.Nx/self.Lx * self.Ny/self.Ly)

        for sigma_x, sigma_y, x0, y0, ampl_n, ampl_phi in zip(sigma_x_ar, sigma_y_ar, x0_ar, y0_ar, ampl_n_ar, ampl_phi_ar):

            x0_shift = np.exp(-1j*kx_2d*x0)
            y0_shift = np.exp(-1j*ky_2d*y0)

            if sigma_x != 0:
                Cx = 1/(2*sigma_x**2)
                Gauss_x = np.sqrt(np.pi/Cx)*np.exp(-kx_2d**2 / (4*Cx))
            else:
                Gauss_x = np.zeros_like(kx_2d)
                Gauss_x = Gauss_x.at[:, 0].set(1.0) * self.Lx

            if sigma_y != 0:
                Cy = 1/(2*sigma_y**2)
                Gauss_y = np.sqrt(np.pi/Cy)*np.exp(-ky_2d**2 / (4*Cy))
            else:
                Gauss_y = np.zeros_like(ky_2d)
                Gauss_y = Gauss_y.at[0, :].set(1.0) * self.Ly

            Gauss_fourier = Gauss_x * Gauss_y * x0_shift * y0_shift * norm

            self.density_fft_init += ampl_n*Gauss_fourier
            self.potential_fft_init += ampl_phi*Gauss_fourier

    def _harmonic_fft_init(self):
        kx = self.param_init['init_harm_kx']
        ky = self.param_init['init_harm_ky']
        ampl_n = self.param_init['init_harm_ampl_n']
        ampl_phi = self.param_init['init_harm_ampl_phi']


        for kx, ky, ampl_n, ampl_phi in zip(kx, ky, ampl_n, ampl_phi):
            
            # Cosine perturbation, multiply by 1j for sine
            self.density_fft_init = self.density_fft_init.at[ky, kx].add(ampl_n*self.Nx*self.Ny)
            self.potential_fft_init = self.potential_fft_init.at[ky, kx].add(ampl_phi*self.Nx*self.Ny)

    def _random_fft_init(self):
        from numpy import random
        random.seed(0)

        ampl_rand_n = self.param_init['init_rand_ampl_n']
        ampl_rand_phi = self.param_init['init_rand_ampl_phi']

        # self.logger.info(f"Initial random perturbation: ampl_rand_n = {ampl_rand_n}, ampl_rand_phi = {ampl_rand_phi}")

        # With JAX numpy:
        # from jax import random
        # density_real_init_random = ampl_n0rand*(random.uniform(random.PRNGKey(42), (Nx, Ny)) - 0.5)
        # potential_real_init_random = ampl_phi0rand*(random.uniform(random.PRNGKey(42), (Nx, Ny)) - 0.5)

        # Random perturbation in real space
        density_real_init_random = ampl_rand_n*(random.rand(self.Ny,self.Nx)-0.5)
        potential_real_init_random = ampl_rand_phi*(random.rand(self.Ny,self.Nx)-0.5)

        # Transform to spectral space
        self.density_fft_init += np.fft.fft2(density_real_init_random)
        self.potential_fft_init += np.fft.fft2(potential_real_init_random)

    def initial_fields(self):

        if self.bool_load_init:
            self._load_custom_init()
        else:
            self._uniform_fft_init()
            self._2D_gaussian_fft_init()
            self._harmonic_fft_init()
            self._random_fft_init()

        return {"density_fft": self.mask_fft*self.density_fft_init, "potential_fft": self.mask_fft*self.potential_fft_init}

    def __repr__(self):

        repr_str = ""

        if self.bool_load_init:
            load_path = self.param_init.get('load_init_path', None)
            repr_str += f"Custom initial fields loaded from folder: {load_path}\n"
            return repr_str

        # Uniform initialization
        repr_str += f"Amplitude of density uniform perturbation: {self.param_init['init_uni_ampl_n']}\n"
        repr_str += f"Uniform of potential uniform perturbation: {self.param_init['init_uni_ampl_phi']}\n"

        # Gaussian initialization
        nb_gaussian_pert = len(self.param_init['init_gauss_sigma_x'])
        repr_str += f"Number of Gaussian perturbations: {nb_gaussian_pert}\n"
        for i in range(nb_gaussian_pert):
            repr_str += f"  Gaussian perturbation {i+1}:\n"
            repr_str += f"    ampl_n = {self.param_init['init_gauss_ampl_n'][i]}\n"
            repr_str += f"    ampl_phi = {self.param_init['init_gauss_ampl_phi'][i]}\n"
            repr_str += f"    sigma_x = {self.param_init['init_gauss_sigma_x'][i]}\n"
            repr_str += f"    sigma_y = {self.param_init['init_gauss_sigma_y'][i]}\n"
            repr_str += f"    x0 = {self.param_init['init_gauss_x0'][i]}\n"
            repr_str += f"    y0 = {self.param_init['init_gauss_y0'][i]}\n"

        # Harmonic initialization
        nb_harmonic_pert = len(self.param_init['init_harm_kx'])
        repr_str += f"Number of harmonic perturbations: {nb_harmonic_pert}\n"
        for i in range(nb_harmonic_pert):
            repr_str += f"  Harmonic perturbation {i+1}:\n"
            repr_str += f"    kx = {self.param_init['init_harm_kx'][i]}\n"
            repr_str += f"    ky = {self.param_init['init_harm_ky'][i]}\n"
            repr_str += f"    ampl_n = {self.param_init['init_harm_ampl_n'][i]}\n"
            repr_str += f"    ampl_phi = {self.param_init['init_harm_ampl_phi'][i]}\n"
        
        # Random initialization
        repr_str += f"Amplitude of density random perturbation = {self.param_init['init_rand_ampl_n']}\n"
        repr_str += f"Amplitude of potential random perturbation = {self.param_init['init_rand_ampl_phi']}\n"

        return repr_str