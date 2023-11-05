from scipy.optimize import root_scalar
from numba import njit
import numpy as np


@njit(fastmath=True, error_model="numpy")
def dust_gas_cooling(nH=1, T=10, Tdust=20, Z=1, dust_coupling=1):
    """
    Rate of heat transfer from gas to dust in erg/s per H
    """
    return nH * gas_dust_heating_coeff(T, Z, dust_coupling) * (Tdust - T)


@njit(fastmath=True, error_model="numpy")
def gas_dust_heating_coeff(T, Z, dust_coupling):
    """Coefficient alpha such that the gas-dust heat transfer is alpha * (T-T_dust)

    Uses Hollenbach & McKee 1979 prescription, assuming 10 Angstrom min grain size.
    """
    return 1.1e-32 * dust_coupling * Z * np.sqrt(T) * (1 - 0.8 * np.exp(-75 / T))


def dust_temperature(
    nH=1, T=10, Z=1, NH=0, X_FUV=1, X_OPT=1, z=0, beta=2, dust_coupling=1
):
    """
    Equilibrium dust temperature obtained by solving the dust energy balance equation accounting for absorption, emission, and gas-dust heat transfer.
    """
    abs = dust_absorption_rate(NH, Z, X_FUV, X_OPT, z, beta)
    sigma_IR_0 = 2e-25
    Tdust_guess = 10 * (abs / (sigma_IR_0 * Z * 2.268)) ** (1.0 / (4 + beta))
    Tdust_guess = max(Tdust_guess, (abs / (4.5e-23 * Z * 2.268)) ** 0.25)
    Tdust_guess = max(
        Tdust_guess,
        T
        - 2.268
        * sigma_IR_0
        * Z
        * (min(T, 150) / 10) ** beta
        * (T / 10) ** 4
        / (gas_dust_heating_coeff(T, Z, dust_coupling) * nH),
    )

    func = lambda dT: net_dust_heating(
        dT, nH, T, NH, Z, X_FUV, X_OPT, z, beta, abs, dust_coupling
    )  # solving for the difference T - Tdust since that's what matters for dust heating
    result = root_scalar(
        func, x0=Tdust_guess - T, x1=(Tdust_guess * 1.1 - T), method="secant", xtol=1e-5
    )  # ,rtol=1e-3,xtol=1e-4*T)
    Tdust = T + result.root
    if not result.converged:
        func = lambda logT: net_dust_heating(
            10**logT - T, nH, T, NH, Z, X_FUV, X_OPT, z, beta, abs, dust_coupling
        )
        result = root_scalar(func, bracket=[-1, 8], method="brentq")
        Tdust = 10**result.root
    return Tdust


@njit(fastmath=True, error_model="numpy")
def net_dust_heating(
    dT, nH, T, NH, Z=1, X_FUV=1, X_OPT=1, z=0, beta=2, absorption=-1, dust_coupling=1
):
    """Derivative of the dust energy in the dust energy equation, solve this = 0 to get the equilibrium dust temperature."""
    Td = T + dT
    sigma_IR_0 = 2e-25
    sigma_IR_emission = (
        sigma_IR_0 * Z * (min(Td, 150) / 10) ** beta
    )  # dust cross section per H in cm^2
    lambdadust_thin = 2.268 * sigma_IR_emission * (Td / 10) ** 4
    lambdadust_thick = 2.268 * (Td / 10) ** 4 / (NH + 1e-100)
    p = 2.5  # how sharp the transition between optically thin and thick cooling is
    psi_IR = (lambdadust_thin**-p + lambdadust_thick**-p) ** -(1 / p)
    # 1/(1/lambdadust_thin + 1/lambdadust_thick) #interpolates
    #  the lower envelope of the optically-thin and -thick limits
    lambda_gd = dust_gas_cooling(nH, T, Td, Z, dust_coupling)

    if absorption < 0:
        absorption = dust_absorption_rate(NH, Z, X_FUV, X_OPT, z, beta)
    return absorption - lambda_gd - psi_IR


@njit(fastmath=True, error_model="numpy")
def dust_absorption_rate(NH, Z=1, X_FUV=1, X_OPT=1, z=0, beta=2):
    """Rate of radiative absorption by dust, per H nucleus in erg/s."""
    T_CMB = 2.73 * (1 + z)
    X_OPT_eV_cm3 = X_OPT * 0.54
    X_IR_eV_cm3 = X_OPT * 0.39
    X_FUV_eV_cm3 = X_FUV * 0.041
    T_IR = max(
        20, 3.8 * (X_OPT_eV_cm3 + X_IR_eV_cm3) ** 0.25
    )  # assume 20K dust emission or the blackbody temperature of the X_FUV energy density, whichever is greater

    sigma_UV = 1e-21 * Z
    sigma_OPT = 3e-22 * Z
    sigma_IR_0 = 2e-25
    sigma_IR_CMB = sigma_IR_0 * Z * (min(T_CMB, 150) / 10) ** beta
    sigma_IR_ISRF = sigma_IR_0 * Z * (min(T_IR, 150) / 10) ** beta

    tau_UV = min(sigma_UV * NH, 100)
    gamma_UV = X_FUV * np.exp(-tau_UV) * 5.965e-25 * Z

    tau_OPT = min(sigma_OPT * NH, 100)
    gamma_OPT = X_OPT * 7.78e-24 * np.exp(-tau_OPT) * Z
    gamma_IR = (
        2.268 * sigma_IR_CMB * (T_CMB / 10) ** 4
        + 0.048
        * (
            X_IR_eV_cm3
            + X_OPT_eV_cm3 * (-np.expm1(-tau_OPT))
            + X_FUV_eV_cm3 * (-np.expm1(-tau_UV))
        )
        * sigma_IR_ISRF
    )
    return gamma_IR + gamma_UV + gamma_OPT
