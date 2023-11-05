import numpy as np


def grain_charge(nH):
    return max((5e3 / nH), 50)


# def f_Cplus(nH=1, NH=1e21, T=10, X_FUV=1, Z=1):
#    """Equilibrium fraction of C locked in CO, from Tielens 2005"""


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
