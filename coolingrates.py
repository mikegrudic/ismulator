from scipy.interpolate import interpn
import numpy as np
import chemistry
from dust import dust_gas_cooling

cooling_processes = (
    "CR Heating",
    "Lyman cooling",
    "Photoelectric",
    "CII Cooling",
    "CO Cooling",
    "Dust-Gas Coupling",
    "Grav. Compression",
    "H_2 Cooling",
    "Turb. Dissipation",
)


def turbulent_heating(sigma_GMC=100.0, M_GMC=1e5):
    """
    Rate of tubulent dissipation per H in erg/s, assuming a turbulent GMC with virial parameter=1 of a certain surface density and mass.

    Note that much of the cooling can take place in shocks that are way out of equilibrium, so this doesn't necessarily capture the full effect.
    """
    return 5e-27 * (M_GMC / 1e6) ** 0.25 * (sigma_GMC / 100) ** (1.25)


def photoelectric_heating(X_FUV=1, nH=1, T=10, NH=0, Z=1):
    """
    Rate of photoelectric heating per H nucleus in erg/s.
    Weingartner & Draine 2001 prescription
    Grain charge parameter is a highly approximate fit vs. density - otherwise need to solve ionization.
    """
    psi = chemistry.grain_charge(nH)
    c0 = 5.22
    c1 = 2.25
    c2 = 0.04996
    c3 = 0.0043
    c4 = 0.147
    c5 = 0.431
    c6 = 0.692
    eps_PE = (c0 + c1 * T**c4) / (1 + c2 * psi**c5 * (1 + c3 * psi**c6))
    sigma_FUV = 1e-21 * Z
    return 1e-26 * X_FUV * eps_PE * np.exp(-NH * sigma_FUV) * Z


def CII_cooling(nH=1, Z=1, T=10, NH=1e21, X_FUV=1, prescription="Simple"):
    """Cooling due to atomic and/or ionized C. Uses either Hopkins 2022 FIRE-3 or simple prescription. Rate per H nucleus in erg/s."""
    if prescription == "Hopkins 2022 (FIRE-3)":
        return atomic_cooling_fire3(nH, NH, T, Z, X_FUV)
    T_CII = 91
    f_C = 1 - chemistry.f_CO(nH, NH, T, X_FUV, Z)
    xc = 1.1e-4
    return 8e-10 * 1.256e-14 * xc * np.exp(-T_CII / T) * Z * nH * f_C


def lyman_cooling(nH=1, T=1000):
    """Rate of Lyman-alpha cooling from Koyama & Inutsuka 2002 per H nucleus in erg/s. Actually a hard upper bound assuming xe ~ xH ~ xH+ ~ 1/2, see Micic 2013 for discussion."""
    return 2e-19 * np.exp(-1.184e5 / T) * nH


def atomic_cooling_fire3(nH, NH, T, Z, X_FUV):
    """Cooling due to atomic and ionized C. Uses Hopkins 2022 FIRE-3 prescription. Rate per H nucleus in erg/s."""
    f = chemistry.f_CO(nH, NH, T, X_FUV, Z)
    return 1e-27 * (0.47 * T**0.15 * np.exp(-91 / T) + 0.0208 * np.exp(-23.6 / T)) * (1 - f) * nH * Z


def get_tabulated_CO_coolingrate(T, NH, nH2):
    """Tabulated CO cooling rate from Omukai 2010, used for Gong 2017 implementation of CO cooling."""
    logT = np.log10(T)
    logNH = np.log10(NH)
    table = np.loadtxt("coolingtables/omukai_2010_CO_cooling_alpha_table.dat")
    T_CO_table = np.log10(table[0, 1:])
    NH_table = table[1:, 0]
    alpha_table = table[1:, 1:].T
    LLTE_table = np.loadtxt("coolingtables/omukai_2010_CO_cooling_LLTE_table.dat")[1:, 1:].T
    n12_table = np.loadtxt("coolingtables/omukai_2010_CO_cooling_n12_table.dat")[1:, 1:].T
    alpha = interpn(
        (T_CO_table, NH_table),
        alpha_table,
        [[logT, logNH]],
        bounds_error=False,
        fill_value=None,
    )
    LLTE = 10 ** -interpn(
        (T_CO_table, NH_table),
        LLTE_table,
        [[logT, logNH]],
        bounds_error=False,
        fill_value=None,
    )
    n12 = 10 ** interpn(
        (T_CO_table, NH_table),
        n12_table,
        [[logT, logNH]],
        bounds_error=False,
        fill_value=None,
    )
    L0 = 10 ** -np.interp(
        np.log10(T),
        T_CO_table,
        [24.77, 24.38, 24.21, 24.03, 23.89, 23.82, 23.42, 23.13, 22.91, 22.63, 22.28],
    )
    LM = (L0**-1 + nH2 / LLTE + (1 / L0) * (nH2 / n12) ** alpha * (1 - n12 * L0 / LLTE)) ** -1
    return LM


def CO_cooling(
    nH=1,
    T=10,
    NH=0,
    Z=1,
    X_FUV=1,
    divv=None,
    xCO=None,
    simple=False,
    prescription="Whitworth 2018",
):
    """
    Rate of CO cooling per H nucleus in erg/s.

    Three prescriptions are implemented: Gong 2017, Whitworth 2018, and Hopkins 2022 (FIRE-3).

    Prescriptions that require a velocity gradient will assume a standard ISM size-linewidth relation by default,
    unless div v is provided.2
    """
    fmol = chemistry.f_CO(nH, NH, T, X_FUV, Z)
    pc_to_cm = 3.08567758128e18
    if xCO is None:
        xCO = fmol * Z * 1.1e-4 * 2
    if divv is None:
        R_scale = NH / nH / pc_to_cm
        divv = 1.0 * R_scale**0.5 / R_scale  # size-linewidth relation

    if prescription == "Gong 2017":
        n_H2 = fmol * nH / 2
        neff = n_H2 + nH * 2**0.5 * (
            2.3e-15 / (3.3e-16 * (T / 1000) ** -0.25)
        )  # Eq. 34 from gong 2017, ignoring electrons
        NCO = xCO * nH / (divv / pc_to_cm)
        LCO = get_tabulated_CO_coolingrate(T, NCO, neff)
        return LCO * xCO * n_H2
    elif prescription == "Hopkins 2022 (FIRE-3)":
        sigma_crit_CO = 1.3e19 * T / Z
        ncrit_CO = 1.9e4 * T**0.5
        return 2.7e-31 * T**1.5 * (xCO / 3e-4) * nH / (1 + (nH / ncrit_CO) * (1 + NH / sigma_crit_CO))  # lambda_CO_HI)
    elif prescription == "Whitworth 2018":
        lambda_CO_LO = 5e-27 * (xCO / 3e-4) * (T / 10) ** 1.5 * (nH / 1e3)
        lambda_CO_HI = 2e-26 * divv * (nH / 1e2) ** -1 * (T / 10) ** 4
        beta = 1.23 * (nH / 2) ** 0.0533 * T**0.164
        if simple:
            return np.min([lambda_CO_LO, lambda_CO_HI], axis=0)
        else:
            return (lambda_CO_LO ** (-1 / beta) + lambda_CO_HI ** (-1 / beta)) ** (-beta)


def CR_heating(zeta_CR=2e-16, NH=None):
    """Rate of cosmic ray heating in erg/s/H, just assuming 10eV per H ionization."""
    if NH is not None:
        return 3e-27 * (zeta_CR / 2e-16) / (1 + (NH / 1e21))
    else:
        return 3e-27 * (zeta_CR / 2e-16)


def compression_heating(nH=1, T=10):
    """Rate of compressional heating per H in erg/s, assuming freefall collapse (e.g. Masunaga 1998)"""
    return 1.2e-27 * np.sqrt(nH / 1e6) * (T / 10)


def H2_cooling(nH, NH, T, X_FUV, Z):
    """
    Glover & Abel 2008 prescription for H_2 cooling; accounts for H2-H2 and H2-HD collisions.
    Rate per H nucleus in erg/s.
    """
    f_molec = 0.5 * chemistry.f_H2(nH, NH, X_FUV, Z)
    EXPmax = 90
    logT = np.log10(T)
    T3 = T / 1000
    Lambda_H2_thick = (
        6.7e-19 * np.exp(-min(5.86 / T3, EXPmax))
        + 1.6e-18 * np.exp(-min(11.7 / T3, EXPmax))
        + 3.0e-24 * np.exp(-min(0.51 / T3, EXPmax))
        + 9.5e-22 * pow(T3, 3.76) * np.exp(-min(0.0022 / (T3 * T3 * T3), EXPmax)) / (1.0 + 0.12 * pow(T3, 2.1))
    ) / nH
    #  super-critical H2-H cooling rate [per H2 molecule]
    Lambda_HD_thin = (
        (1.555e-25 + 1.272e-26 * pow(T, 0.77)) * np.exp(-min(128.0 / T, EXPmax))
        + (2.406e-25 + 1.232e-26 * pow(T, 0.92)) * np.exp(-min(255.0 / T, EXPmax))
    ) * np.exp(-min(T3 * T3 / 25.0, EXPmax))
    #  optically-thin HD cooling rate [assuming all D locked into HD at temperatures where this is relevant], per molecule

    q = logT - 3.0
    Y_Hefrac = 0.25
    X_Hfrac = 0.75
    #  variable used below
    Lambda_H2_thin = (
        max(nH - 2.0 * f_molec, 0)
        * X_Hfrac
        * np.power(
            10.0,
            max(
                -103.0
                + 97.59 * logT
                - 48.05 * logT * logT
                + 10.8 * logT * logT * logT
                - 0.9032 * logT * logT * logT * logT,
                -50.0,
            ),
        )
    )
    #  sub-critical H2 cooling rate from H2-H collisions [per H2 molecule]; this from Galli & Palla 1998
    Lambda_H2_thin += Y_Hefrac * np.power(
        10.0,
        max(
            -23.6892
            + 2.18924 * q
            - 0.815204 * q * q
            + 0.290363 * q * q * q
            - 0.165962 * q * q * q * q
            + 0.191914 * q * q * q * q * q,
            -50.0,
        ),
    )
    #  H2-He; often more efficient than H2-H at very low temperatures (<100 K); this and other H2-x terms below from Glover & Abel 2008
    Lambda_H2_thin += (
        f_molec
        * X_Hfrac
        * np.power(
            10.0,
            max(
                -23.9621
                + 2.09434 * q
                - 0.771514 * q * q
                + 0.436934 * q * q * q
                - 0.149132 * q * q * q * q
                - 0.0336383 * q * q * q * q * q,
                -50.0,
            ),
        )
    )
    #  H2-H2; can be more efficient than H2-H when H2 fraction is order-unity

    f_HD = min(0.00126 * f_molec, 4.0e-5 * nH)

    nH_over_ncrit = Lambda_H2_thin / Lambda_H2_thick
    Lambda_HD = f_HD * Lambda_HD_thin / (1.0 + f_HD / (f_molec + 1e-10) * nH_over_ncrit) * nH
    Lambda_H2 = f_molec * Lambda_H2_thin / (1.0 + nH_over_ncrit) * nH
    return Lambda_H2 + Lambda_HD
