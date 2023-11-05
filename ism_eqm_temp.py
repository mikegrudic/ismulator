from scipy.optimize import brentq, root_scalar
from scipy.interpolate import interp1d
from numba import njit
import numpy as np
import dust
import coolingrates

all_processes = coolingrates.cooling_processes


def net_heating(
    T=10,
    nH=1,
    NH=0,
    X_FUV=1,
    X_OPT=1,
    Z=1,
    z=0,
    divv=None,
    zeta_CR=2e-16,
    Tdust=None,
    jeans_shielding=False,
    dust_beta=2.0,
    dust_coupling=1,
    sigma_GMC=100,
    processes=all_processes,
    attenuate_cr=True,
    co_prescription="Whitworth 2018",
    cii_prescription="Hopkins 2022 (FIRE-3)",
):
    if jeans_shielding:
        lambda_jeans = 8.1e19 * nH**-0.5 * (T / 10) ** 0.5
        NH = np.max([nH * lambda_jeans * jeans_shielding, NH], axis=0)

    if Tdust == None:
        Tdust = dust.dust_temperature(
            nH,
            T,
            Z,
            NH,
            X_FUV,
            X_OPT,
            z,
            dust_beta,
            dust_coupling * ("Dust-Gas Coupling" in processes),
        )

    rate = 0
    for process in processes:
        if process == "CR Heating":
            rate += coolingrates.CR_heating(zeta_CR, NH * attenuate_cr)
        if process == "Lyman cooling":
            rate -= coolingrates.lyman_cooling(nH, T)
        if process == "Photoelectric":
            rate += coolingrates.photoelectric_heating(X_FUV, nH, T, NH, Z)
        if process == "CII Cooling":
            rate -= coolingrates.CII_cooling(
                nH, Z, T, NH, X_FUV, prescription=cii_prescription
            )
        if process == "CO Cooling":
            rate -= coolingrates.CO_cooling(
                nH, T, NH, Z, X_FUV, divv, prescription=co_prescription
            )
        if process == "Dust-Gas Coupling":
            rate += coolingrates.dust_gas_cooling(nH, T, Tdust, Z, dust_coupling)
        if process == "H_2 Cooling":
            rate -= coolingrates.H2_cooling(nH, NH, T, X_FUV, Z)
        if process == "Grav. Compression":
            rate += coolingrates.compression_heating(nH, T)
        if process == "Turb. Dissipation":
            rate += coolingrates.turbulent_heating(sigma_GMC=sigma_GMC)

    return rate


def equilibrium_temp(
    nH=1,
    NH=0,
    X_FUV=1,
    X_OPT=1,
    Z=1,
    z=0,
    divv=None,
    zeta_CR=2e-16,
    Tdust=None,
    jeans_shielding=False,
    dust_beta=2.0,
    dust_coupling=1,
    sigma_GMC=100.0,
    processes=all_processes,
    attenuate_cr=True,
    co_prescription="Whitworth 2018",
    cii_prescription="Hopkins 2022 (FIRE-3)",
    return_Tdust=True,
    T_guess=None,
):
    if NH == 0:
        NH = 1e18
    params = (
        nH,
        NH,
        X_FUV,
        X_OPT,
        Z,
        z,
        divv,
        zeta_CR,
        Tdust,
        jeans_shielding,
        dust_beta,
        dust_coupling,
        sigma_GMC,
        processes,
        attenuate_cr,
        co_prescription,
        cii_prescription,
    )
    func = lambda logT: net_heating(
        10**logT, *params
    )  # solving vs logT converges a bit faster

    use_brentq = True
    if (
        T_guess is not None
    ):  # we have an initial guess that is supposed to be close (e.g. previous grid point)
        T_guess2 = T_guess * 1.01
        result = root_scalar(
            func,
            x0=np.log10(T_guess),
            x1=np.log10(T_guess2),
            method="secant",
            rtol=1e-3,
        )
        if result.converged:
            T = 10**result.root
            use_brentq = False

    if use_brentq:
        try:
            T = 10 ** brentq(func, -1, 5, rtol=1e-3, maxiter=500)
        except:
            try:
                T = 10 ** brentq(func, -1, 10, rtol=1e-3, maxiter=500)
            except:
                raise ("Couldn't solve for temperature! Try some other parameters.")

    if return_Tdust:
        Tdust = dust.dust_temperature(
            nH,
            T,
            Z,
            NH,
            X_FUV,
            X_OPT,
            z,
            dust_beta,
            dust_coupling * ("Dust-Gas Coupling" in processes),
        )
        return T, Tdust
    else:
        return T


def equilibrium_temp_grid(
    nH,
    NH,
    X_FUV=1,
    X_OPT=1,
    Z=1,
    z=0,
    divv=None,
    zeta_CR=2e-16,
    Tdust=None,
    jeans_shielding=False,
    dust_beta=2.0,
    dust_coupling=1,
    sigma_GMC=100.0,
    processes=all_processes,
    attenuate_cr=True,
    return_Tdust=False,
    co_prescription="Whitworth 2018",
    cii_prescription="Hopkins 2022 (FIRE-3)",
):
    params = (
        X_FUV,
        X_OPT,
        Z,
        z,
        divv,
        zeta_CR,
        Tdust,
        jeans_shielding,
        dust_beta,
        dust_coupling,
        sigma_GMC,
        processes,
        attenuate_cr,
        co_prescription,
        cii_prescription,
    )
    Ts = []
    Tds = []

    T_guess = None
    for i in range(
        len(nH)
    ):  # we do a pass on the grid where we use previously-evaluated temperatures to get good initial guesses for the next grid point
        if i == 1:
            T_guess = Ts[-1]
        elif i > 1:
            T_guess = 10 ** interp1d(
                np.log10(nH[:i]), np.log10(Ts), fill_value="extrapolate"
            )(
                np.log10(nH[i])
            )  # guess using linear extrapolation in log space

        sol = equilibrium_temp(
            nH[i], NH[i], *params, return_Tdust=return_Tdust, T_guess=T_guess
        )
        if return_Tdust:
            T, Tdust = sol
            Ts.append(T)
            Tds.append(Tdust)
        else:
            Ts.append(sol)
    if return_Tdust:
        return np.array(Ts), np.array(Tds)
    else:
        return np.array(Ts)
