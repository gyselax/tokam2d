# src/model/source.py
import jax.numpy as np

class Source():
    def __init__(self, params):
        # Get the source parameters set in the input file 
        self.param_source = params.user['source']

        # Retrieve the grid parameters in real and spectral space
        self.Nx = params.Nx
        self.Ny = params.Ny
        self.Lx = params.Lx
        self.Ly = params.Ly

        self.x_2d, self.y_2d = params.get_2d_grid()
        self.kx_2d = params.kx_2d
        self.ky_2d = params.ky_2d

        # Get the PDE and initial condition parameters
        self.pde = params.user['pde']
        self.init_params = params.user['initial_condition']

        # Initialize the sources in spectral space
        self.density_fft_source = np.zeros((self.Ny, self.Nx), dtype=complex)
        self.potential_fft_source = np.zeros((self.Ny, self.Nx), dtype=complex)

    def gaussian_fft_source(self):
        sigma_x_ar = self.param_source['source_gauss_sigma_x']
        sigma_y_ar = self.param_source['source_gauss_sigma_y']
        x0_ar = self.param_source['source_gauss_x0']
        y0_ar = self.param_source['source_gauss_y0']
        ampl_n_ar = self.param_source['source_gauss_ampl_n']
        ampl_phi_ar = self.param_source['source_gauss_ampl_phi']

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

            self.density_fft_source += ampl_n*Gauss_fourier
            self.potential_fft_source += ampl_phi*Gauss_fourier

    def get_control_source(self):

        kx_2d = self.kx_2d
        ky_2d = self.ky_2d

        ampl_n_control = self.param_source['source_control_amp']
        sigma = self.param_source['source_control_sigma']
        x0 = self.param_source['source_control_x0']
        y0 = self.param_source['source_control_y0']

        norm = (self.Nx/self.Lx * self.Ny/self.Ly)
        
        sigma_x = sigma
        sigma_y = sigma

        x0_shift = np.exp(-1j*kx_2d*x0)
        y0_shift = np.exp(-1j*ky_2d*y0)

        Cx = 1/(2*sigma_x**2)
        Gauss_x = np.sqrt(np.pi/Cx)*np.exp(-kx_2d**2 / (4*Cx))

        Cy = 1/(2*sigma_y**2)
        Gauss_y = np.sqrt(np.pi/Cy)*np.exp(-ky_2d**2 / (4*Cy))

        Gauss_fourier = Gauss_x * Gauss_y * x0_shift * y0_shift * norm

        print(f"Control source: ampl_n_control = {ampl_n_control}, sigma = {sigma}, x0 = {x0}, y0 = {y0}")

        return {"density_source_fft": ampl_n_control*Gauss_fourier}

    def balance_par_losses_source(self):

        S0 = self.pde["sigma_nn"] if self.pde['eq']=='SOL' else 0.

        self.density_fft_source = self.density_fft_source.at[0,0].add(self.init_params['init_uni_ampl_n']*S0*self.Nx*self.Ny)

    def get_source(self):
        self.gaussian_fft_source()

        if self.param_source.get('source_balance_par_loss', False):
            self.balance_par_losses_source()

        return {"density_source_fft": self.density_fft_source, "potential_source_fft": self.potential_fft_source}