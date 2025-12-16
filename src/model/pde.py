# src/model/pde.py
from src.model.source import Source

from jax import jit
import jax.numpy as np
from functools import partial

from abc import ABC, abstractmethod

PDE_REGISTRY = {} # Registry for storing PDEs that can be solved by Tokam2D

def register_pde(name):
    """Decorator to register a PDE class."""
    def decorator(cls):
        PDE_REGISTRY[name] = cls
        return cls
    return decorator

# @jit
def poisson_bracket(f: np.ndarray,
                    vx: np.ndarray,
                    vy: np.ndarray,
                    i_kx_2d: np.ndarray,
                    i_ky_2d: np.ndarray,
                    ) -> np.ndarray:
    """Return the Poisson Brackets."""
    return i_kx_2d*np.fft.fft2(vx*f) + i_ky_2d*np.fft.fft2(vy*f)

class PDE_structure(ABC):
    """ Abstract class for PDEs.
    
    Each PDE should inherit from this class and implement the compute_rhs method.
    """
    def __init__(self, params):
        self.pde = params.user["pde"]
        self.Dn = self.pde["Dn"]
        self.Dphi = self.pde["Dphi"]
        self.dissipation_type = self.pde.get("dissipation_type", ["diffusion"])
        self.g = self.pde["g"]

        self.kx_2d = params.kx_2d
        self.ky_2d = params.ky_2d
        self.inv_k2_2d = params.inv_k2_2d
        self.k2_2d = params.k2_2d

        self.dissipation = []
        for dissipation_type in self.dissipation_type:
            if dissipation_type == "friction":
                self.dissipation.append(- np.ones_like(self.k2_2d))
            elif dissipation_type == "diffusion":
                self.dissipation.append(- self.k2_2d)
            elif dissipation_type == "hyperdiffusion":
                self.dissipation.append(- self.k2_2d**2)

        self.dens_dissip = sum([Dn*k for Dn, k in zip(self.Dn, self.dissipation)])
        self.phi_dissip  = sum([Dphi*k for Dphi, k in zip(self.Dphi, self.dissipation)])

    @abstractmethod
    def compute_rhs(self, state, t=None):
        pass

@register_pde("SOL")
class SOL(PDE_structure):
    def __init__(self, params):
        super().__init__(params)
        self.sigma_nn = params.user["pde"]["sigma_nn"]
        self.sigma_nphi = params.user["pde"]["sigma_nphi"]
        self.sigma_phin = params.user["pde"]["sigma_phin"]
        self.sigma_phiphi = params.user["pde"]["sigma_phiphi"]
        self.g = params.user["pde"]["g"]

        #TODO: reimplement gradn
        self.a_n = self.dens_dissip - self.sigma_nn
        self.b_n = self.sigma_nphi
        self.a_phi = self.phi_dissip - self.inv_k2_2d*self.sigma_phiphi
        self.b_phi = -self.inv_k2_2d*(- 1j*self.ky_2d*self.g - self.sigma_phin)

        self.b_phi = self.b_phi.at[0, :].set(0)

        # Source
        self.source = Source(params)
        self.source_dict = self.source.get_source()

    @partial(jit, static_argnums=(0,))
    def compute_rhs(self, state, t=None):
        # t could be removed
        potential_fft = state["potential_fft"]
        dens_fft = state["density_fft"]

        Sn_fft = self.source_dict["density_source_fft"]
        Sphi_fft = self.source_dict["potential_source_fft"]

        dens = np.real(np.fft.ifft2(dens_fft))
        vort = np.real(np.fft.ifft2(- self.k2_2d*potential_fft))
        vEx = - np.real(np.fft.ifft2(1j*self.ky_2d*potential_fft))
        vEy = np.real(np.fft.ifft2(1j*self.kx_2d*potential_fft))

        nl_term_rhs_dens = - poisson_bracket(
            dens, vEx, vEy, 1j*self.kx_2d, 1j*self.ky_2d)
        nl_term_rhs_vort = - poisson_bracket(
            vort, vEx, vEy, 1j*self.kx_2d, 1j*self.ky_2d)

        dens_fft_out      = self.a_n*dens_fft + self.b_n*potential_fft + nl_term_rhs_dens + Sn_fft
        potential_fft_out = self.a_phi*potential_fft + self.b_phi*dens_fft + (nl_term_rhs_vort+Sphi_fft)*(-self.inv_k2_2d)

        return {"density_fft": dens_fft_out, "potential_fft": potential_fft_out}

@register_pde("SOL_ZF")
class SOL_ZF(SOL):
    def __init__(self, params):
        super().__init__(params)

        self.b_n = self.sigma_nphi * np.ones_like(self.k2_2d)
        self.a_n = self.a_n.at[0, :].set(self.dens_dissip[0, :])
        self.b_n = self.b_n.at[0,:].set(0)
        self.a_phi = self.a_phi.at[0, :].set(self.phi_dissip[0, :])

@register_pde("HW")
class HasegawaWakatani(PDE_structure):
    """Original Hasegawa-Wakatani PDE class.
    
    Note that interchange instability is also implemented so original HW
    needs g=0 in the input file.
    """
    def __init__(self, params):
        super().__init__(params)
        self.C = params.user["pde"]["C"] # Adiabatic parameter
        self.kappa = params.user["pde"]["kappa"] # - \partial_x ln(n0)

        self.mask_fft = params.mask_fft

        self.a_n = self.dens_dissip - self.C
        self.b_n = -1j * self.ky_2d*self.kappa + self.C
        self.a_phi = self.phi_dissip - self.inv_k2_2d*self.C
        self.b_phi = self.inv_k2_2d*self.C - 1j*self.ky_2d*self.g*(-self.inv_k2_2d)        

        self.b_phi = self.b_phi.at[0, :].set(0)
        self.a_n = self.a_n.at[0, 0].set(self.dens_dissip[0, 0])
        self.a_phi = self.a_phi.at[0, 0].set(self.phi_dissip[0, 0])

    @partial(jit, static_argnums=(0,))
    def compute_rhs(self, state, t=None):
        # t could be removed
        potential_fft = state["potential_fft"]
        dens_fft = state["density_fft"]

        dens = np.real(np.fft.ifft2(dens_fft))
        vort = np.real(np.fft.ifft2(- self.k2_2d*potential_fft))
        vEx = - np.real(np.fft.ifft2(1j*self.ky_2d*potential_fft))
        vEy = np.real(np.fft.ifft2(1j*self.kx_2d*potential_fft))

        nl_term_rhs_dens = - poisson_bracket(
            dens, vEx, vEy, 1j*self.kx_2d, 1j*self.ky_2d)
        nl_term_rhs_vort = - poisson_bracket(
            vort, vEx, vEy, 1j*self.kx_2d, 1j*self.ky_2d)

        dens_fft_out      = (self.a_n*dens_fft + self.b_n*potential_fft + nl_term_rhs_dens)
        potential_fft_out = (self.a_phi*potential_fft + self.b_phi*dens_fft + nl_term_rhs_vort*(-self.inv_k2_2d))

        return {"density_fft": dens_fft_out, "potential_fft": potential_fft_out}

@register_pde("mHW")
class modifiedHasegawaWakatani(HasegawaWakatani):
    def __init__(self, params):
        super().__init__(params)

        self.a_n = self.a_n.at[0, :].set(self.dens_dissip[0, :])
        self.b_n = self.b_n.at[0,:].set(0)
        self.a_phi = self.a_phi.at[0, :].set(self.phi_dissip[0, :])

@register_pde("BHW")
class fluxBalancedHasegawaWakatani(modifiedHasegawaWakatani):
    @partial(jit, static_argnums=(0,))
    def compute_rhs(self, state, t=None):
        # t could be removed
        potential_fft = state["potential_fft"]
        dens_fft = state["density_fft"]

        dens = np.real(np.fft.ifft2(dens_fft))
        vort_fft = - self.k2_2d*potential_fft
        vort = np.real(np.fft.ifft2(vort_fft))
        vEx = - np.real(np.fft.ifft2(1j*self.ky_2d*potential_fft))
        vEy = np.real(np.fft.ifft2(1j*self.kx_2d*potential_fft))

        nl_term_rhs_dens = - poisson_bracket(
            dens, vEx, vEy, 1j*self.kx_2d, 1j*self.ky_2d)
        nl_term_rhs_vort = - poisson_bracket(
            vort, vEx, vEy, 1j*self.kx_2d, 1j*self.ky_2d)

        dens_y_avg = np.mean(dens, axis=0, keepdims=True)
        div_flux_dens_y_avg_fft = poisson_bracket(
            dens_y_avg, vEx, vEy, 1j*self.kx_2d, 1j*self.ky_2d)
        y_avg_div_flux_dens_fft = np.mean(
            -nl_term_rhs_dens, axis=0, keepdims=True)
        nl_term_rhs_vort += y_avg_div_flux_dens_fft - div_flux_dens_y_avg_fft

        dens_fft_out      = (self.a_n*dens_fft + self.b_n*potential_fft + nl_term_rhs_dens)
        potential_fft_out = (self.a_phi*potential_fft + self.b_phi*dens_fft + nl_term_rhs_vort*(-self.inv_k2_2d))

        return {"density_fft": dens_fft_out, "potential_fft": potential_fft_out}