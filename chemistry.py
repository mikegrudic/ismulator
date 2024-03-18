import numpy as np


def grain_charge(nH):
    return max((5e3 / nH), 50)


def cosmic_ray_ionization_rate(zeta_CR=2e-16, NH=1e21):
    return zeta_CR / (1 + (NH / 1e21))


def f_HII(nH=1, NH=1e21, T=10, zeta_CR=2e-16):
    """Fraction of ionized H - assumes ONLY cosmic ray ionization, all electrons are from H, and ionization is small"""
    zeta_CR_H = cosmic_ray_ionization_rate(zeta_CR, NH)
    alpha_rr_H = (
        2.753e-14 * (315614 / T) ** 1.5 * (1 + (115188 / T) ** 0.407) ** -2.242
    )  # case B recombination coeff
    frac = np.sqrt(zeta_CR_H / (alpha_rr_H * nH)).clip(0, 1)
    return frac


def f_Cplus(nH=1, NH=1e21, T=10, X_FUV=1, zeta_CR=2e-16, Z=1):
    """Equilibrium fraction of C in C+, following Kim 2022, Gong 2017"""

    # just account for dust shielding for now (C and H2 also important)
    zeta_FUV = 3.43e-10 * X_FUV * np.exp(-1e-21 * NH * Z)
    zeta_CR_H = cosmic_ray_ionization_rate(zeta_CR, NH)
    fH2 = f_H2(nH, NH, X_FUV, Z)
    zeta_H2_CR = 520 * 2 * fH2 * zeta_CR_H
    zeta_CR_C = 3.85 * zeta_CR_H
    psi = grain_charge(nH)

    # rate coefficient for grain-assisted recombination
    alpha_grain_C = (
        45.58
        * 1e-14
        / (
            1
            + 6.089
            * 1e-3
            * psi**1.128
            * (1 + 4.331 * T**0.04845)
            * np.power(psi, -0.4956 - 5.494e-7 * np.log(T))
        )
    ) * Z

    alpha = np.sqrt(T / 6.67e-3)
    beta = np.sqrt(T / 1.943e6)
    gamma = 0.7849 + 0.1597 * np.exp(-49550 / T)
    # radiative recombination
    k_rr = 2.995e-9 / (alpha * (1 + alpha) ** (1 - gamma) * (1 + beta) ** (1 + gamma))
    # dielectronic recombination
    k_dr = T**-1.5 * (
        6.346e-9 * np.exp(-12.17 / T)
        + 9.793e-9 * np.exp(-73.8 / T)
        + 1.634e-6 * np.exp(-15230 / T)
    )

    # C+ + H_2 -> CH_2+
    k_cplus_H2 = 2.31e-13 * T**-1.3 * np.exp(-23 / T)
    zeta_total = zeta_FUV + zeta_H2_CR + zeta_CR_C
    xe = 0  # electron abundance
    cplus_fraction = zeta_total / (
        zeta_total
        + alpha_grain_C * nH
        + k_cplus_H2 * 0.5 * fH2 * nH
        + (k_rr + k_dr) * nH * xe
    )

    return cplus_fraction


def f_CO(nH=1, NH=1e21, T=10, X_FUV=1, Z=1, prescription="Tielens 2005"):
    """Equilibrium fraction of C locked in CO"""

    if prescription == "Tielens 2005":
        G0 = 1.7 * X_FUV * np.exp(-1e-21 * NH * Z)
        if nH > 10000 * G0 * 340:
            return 1.0
        x = (nH / (G0 * 340)) ** 2 * T**-0.5
        return x / (1 + x)


def f_H2(nH=1, NH=1e21, X_FUV=1, Z=1):
    """Krumholz McKee Tumlinson 2008 prescription for fraction of neutral H in H_2 molecules"""
    surface_density_Msun_pc2 = NH * 1.1e-20
    tau_UV = min(1e-21 * Z * NH, 100.0)
    G0 = 1.7 * X_FUV * np.exp(-tau_UV)
    chi = 71.0 * X_FUV / nH
    psi = chi * (1.0 + 0.4 * chi) / (1.0 + 1.08731 * chi)
    s = (Z + 1.0e-3) * surface_density_Msun_pc2 / (1e-100 + psi)
    q = s * (125.0 + s) / (11.0 * (96.0 + s))
    fH2 = 1.0 - (1.0 + q * q * q) ** (-1.0 / 3.0)
    if q < 0.2:
        fH2 = q * q * q * (1.0 - 2.0 * q * q * q / 3.0) / 3.0
    elif q > 10:
        fH2 = 1.0 - 1 / q
    return fH2
