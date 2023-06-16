from scipy.optimize import brentq
from scipy.interpolate import interpn
import numpy as np

def photoelectric_heating(ISRF=1, NH=0, Z=1):

    sigma_FUV = 1e-21 * Z
    return 1e-26 * ISRF * 4 * np.exp(-NH*sigma_FUV) * Z

def f_mol(nH=1, NH=1e21, T=10,ISRF=1, Z=1):
    G0 = 1.7 * ISRF * np.exp(-1e-21 * NH * Z)
    if nH > 10000*G0*340: return 1.
    x = (nH/(G0*340))**2*T**-0.5
    return x/(1+x)

f_mol = np.vectorize(f_mol)

def CII_cooling(nH=1, Z=1, T=10, NH=1e21, ISRF=1):
    T_CII = 91
    f_C = f_mol(nH,NH,T,ISRF,Z)
    xc = 1.1e-4
    return 8e-10 * 1.256e-14 * xc * np.exp(-T_CII/T) * Z * nH * (1-f_C)

def lyman_cooling(nH=1,T=1000):
    return 2e-19 * np.exp(-1.184e5/T) * nH

def atomic_cooling_fire3(nH,NH,T,Z,ISRF):
    f = f_mol(nH,NH,T,ISRF,Z)
    return 1e-27 * (0.47*T**0.15 * np.exp(-91/T) + 0.0208 * np.exp(-23.6/T)) * (1-f) * nH * Z

def get_tabulated_CO_coolingrate(T,NH,nH2):
    logT = np.log10(T)
    logNH = np.log10(NH)
    table = np.loadtxt("coolingtables/omukai_2010_CO_cooling_alpha_table.dat")
    T_CO_table = np.log10(table[0,1:])
    NH_table = table[1:,0]
    alpha_table = table[1:,1:].T
    LLTE_table = np.loadtxt("coolingtables/omukai_2010_CO_cooling_LLTE_table.dat")[1:,1:].T
    n12_table = np.loadtxt("coolingtables/omukai_2010_CO_cooling_n12_table.dat")[1:,1:].T
    alpha = interpn((T_CO_table, NH_table), alpha_table, [[logT,logNH]],bounds_error=False,fill_value=None)
    LLTE = 10**-interpn((T_CO_table, NH_table), LLTE_table, [[logT,logNH]],bounds_error=False,fill_value=None)
    n12 = 10**interpn((T_CO_table, NH_table), n12_table, [[logT,logNH]],bounds_error=False,fill_value=None)
    L0 = 10**-np.interp(np.log10(T),T_CO_table,[24.77, 24.38, 24.21, 24.03, 23.89 ,23.82 ,23.42 ,23.13 ,22.91 ,22.63, 22.28])
    LM = (L0**-1 + nH2/LLTE + (1/L0)*(nH2/n12)**alpha * (1 - n12*L0/LLTE))**-1
    return LM

get_tabulated_CO_coolingrate = np.vectorize(get_tabulated_CO_coolingrate)

def CO_cooling(nH=1, T=10, NH=0,Z=1,ISRF=1,divv=None,xCO=None,simple=False,prescription='whitworth'):
    fmol = f_mol(nH,NH,T,ISRF,Z)
    pc_to_cm = 3.08567758128E+18
    if xCO is None:
        xCO = fmol * Z * 1.1e-4 * 2
    if divv is None:
        R_scale = NH/nH/pc_to_cm
        divv = 1. * R_scale**0.5 / R_scale # size-linewidth relation

    if prescription=="gong17":
        n_H2 = fmol*nH/2
        neff = n_H2 + nH*2**0.5 * (2.3e-15/(3.3e-16*(T/1000)**-0.25)) # Eq. 34 from gong 2017, ignoring electrons 
        NCO = xCO * nH / (divv / pc_to_cm)
       # print(xCO,NCO)
        LCO = get_tabulated_CO_coolingrate(T,NCO,neff)
        return LCO * xCO * n_H2
    elif prescription=='fire3':
        sigma_crit_CO=  1.3e19 * T / Z
        ncrit_CO=1.9e4 * T**0.5
        #lambda_CO_HI = 2e-26 * divv * (nH/1e2)**-1 * (T/10)**4
        return 2.7e-31 * T**1.5 * (xCO/3e-4) * nH/(1 + (nH/ncrit_CO)*(1+NH/sigma_crit_CO)) #lambda_CO_HI)
    elif prescription=='whitworth':
        lambda_CO_LO = 5e-27 * (xCO/3e-4) * (T/10)**1.5 * (nH/1e3)
        lambda_CO_HI = 2e-26 * divv * (nH/1e2)**-1 * (T/10)**4
        beta = 1.23 * (nH/2)**0.0533 * T**0.164
        if simple: return np.min([lambda_CO_LO, lambda_CO_HI],axis=0)
        else: return (lambda_CO_LO**(-1/beta) + lambda_CO_HI**(-1/beta))**(-beta)

CO_cooling = np.vectorize(CO_cooling)

def CR_heating(zeta_CR=2e-16,NH=None):
    if NH is not None:
        return 3e-27 * (zeta_CR / 2e-16)/(1+(NH/1e22))
    else:
        return 3e-27 * (zeta_CR / 2e-16)
    
CR_heating = np.vectorize(CR_heating)

def dust_gas_cooling(nH=1,T=10,Tdust=20,Z=1):
    return nH * 1.1e-32 * T**0.5 * (Tdust-T) * (1-0.8*np.exp(-75/T)) #3e-26 * Z * (T/10)**0.5 * (Tdust-T)/10 * (nH/1e6)

def compression_heating(nH=1,T=10):
    #return 1.672e-24 * np.sqrt(nH/4.1e12) * (T/10) #
    #if 5*(T/10)**1.5 * (nH/1e4)**-0.5 > 10: return 0
    return 1.2e-27 * np.sqrt(nH/1e6) * (T/10)

compression_heating = np.vectorize(compression_heating)

def turbulent_heating(sigma_GMC=100., M_GMC=1e5):
    return 5e-27 * (M_GMC/1e6)**0.25 * (sigma_GMC/100)**(1.25)

def dust_temperature(nH=1,T=10,Z=1,NH=0, ISRF=1, T_CMB=2.7):
    return brentq(net_dust_heating,1,10**5,args=(nH,T,NH,Z,ISRF,T_CMB))

dust_temperature = np.vectorize(dust_temperature)

def net_dust_heating(Td,nH,T,NH,Z=1,ISRF=1,T_CMB=2.7, beta=2):
    ISRF_OPT_eV_cm3 = ISRF * 0.54
    ISRF_IR_eV_cm3 = ISRF * 0.39
    sigma_IR_0 = 2e-25
    sigma_IR_d = sigma_IR_0 * Z * (min(Td,150)/10)**beta # dust cross section per H in cm^2
    sigma_IR_CMB = sigma_IR_0 * Z * (T_CMB/10)**beta
    T_ISRF=20
    sigma_IR_ISRF = sigma_IR_0 * Z * (T_ISRF/10)**beta # assume 20K dust emission
    sigma_UV = 1e-21 * Z
    tau_d = sigma_UV * NH
    sigma_OPT = 3e-22 * Z
    psi_IR = min(2.268 * sigma_IR_d * (Td/10)**4, 2.268 * (Td/10)**4 / (NH+1e-100))
    gamma_UV = ISRF * np.exp(-tau_d) * 5.965e-25 * Z
    tau_OPT = sigma_OPT * NH
    gamma_OPT = ISRF * 7.78e-24 * np.exp(-tau_OPT) * Z
    gamma_IR =  2.268 * sigma_IR_CMB * (T_CMB/10)**4 + 0.048 * (ISRF_IR_eV_cm3 + ISRF_OPT_eV_cm3 * (1-np.exp(-tau_OPT))) * sigma_IR_ISRF
    gamma_ISRF = gamma_UV+gamma_OPT
    lambda_gd = dust_gas_cooling(nH,T,Td,Z)
    return (gamma_IR + gamma_ISRF - lambda_gd - psi_IR)/1e-25

def net_heating(T=10, nH=1, ISRF=1, NH=0, Z=1, divv=None, zeta_CR=2e-16, Tdust=None,jeans_shielding=False, compression=False):
    #print(NH)
    if jeans_shielding:
        lambda_jeans = 8.1e19 * nH**-0.5 * (T/10)**0.5
        NH = np.max([nH*lambda_jeans*jeans_shielding, NH],axis=0)
    if Tdust==None:
        Tdust = dust_temperature(nH,T,Z,NH,ISRF)
    rate = CR_heating(zeta_CR,NH) - lyman_cooling(T) + photoelectric_heating(ISRF, NH, Z) - CII_cooling(nH, Z, T, NH, ISRF) - CO_cooling(nH,T,NH,Z,ISRF,divv) + dust_gas_cooling(nH,T,Tdust,Z) + compression*compression_heating(nH,T) #+ turbulent_heating()
    return rate

net_heating = np.vectorize(net_heating)

def equilibrium_temp(nH=1, NH=0, ISRF=1, Z=1, divv=None, zeta_CR=2e-16, Tdust=None,jeans_shielding=False,compression=False):
    if NH==0: NH=1e18
    params = nH, ISRF, NH, Z, divv, zeta_CR, Tdust, jeans_shielding,compression
    func = lambda T: net_heating(T, *params)/1e-30
    sol = brentq(func, 1, 10**5)
    return sol

equilibrium_temp = np.vectorize(equilibrium_temp)
