from scipy.optimize import brentq, newton, root_scalar
from scipy.interpolate import interpn
from numba import njit
import numpy as np

def photoelectric_heating(ISRF=1, nH=1, T=10, NH=0, Z=1):
    grain_charge =  max((5e3/nH),50)
    c0=5.22; c1 =2.25; c2=0.04996; c3=0.0043; c4=0.147; c5=0.431; c6=0.692
    eps_PE = (c0+c1*T**c4)/(1+c2*grain_charge**c5 * (1+c5*grain_charge**c6))
    sigma_FUV = 1e-21 * Z
    return 1e-26 * ISRF * eps_PE * np.exp(-NH*sigma_FUV) * Z

def f_CO(nH=1, NH=1e21, T=10,ISRF=1, Z=1):
    G0 = 1.7 * ISRF * np.exp(-1e-21 * NH * Z)
    if nH > 10000*G0*340: return 1.
    x = (nH/(G0*340))**2*T**-0.5
    return x/(1+x)

def f_H2(nH=1, NH=1e21, ISRF=1, Z=1):
    """Krumholz McKee Tumlinson 2008 prescription for fraction of neutral H in molecules"""
    surface_density_Msun_pc2 = NH * 1.1e-20  
    tau_UV = np.exp(1e-21 * Z * NH)
    G0 = 1.7 * ISRF * np.exp(-tau_UV)
    chi = 71. * ISRF / nH
    psi = chi * (1.+0.4*chi)/(1.+1.08731*chi)
    s = (Z + 1.e-3) * surface_density_Msun_pc2 / (1e-100 + psi)
    q = s * (125. + s) / (11. * (96. + s))
    fH2 = 1. - (1.+q*q*q)**(-1./3.)
    if q < 0.2: fH2 = q*q*q * (1. - 2.*q*q*q/3.)/3.
    elif q>10: fH2 = 1. - 1/q
    return fH2

def H2_cooling(nH,NH,T,ISRF,Z):
    f_molec = 0.5 * f_H2(nH,NH,ISRF,Z)
    EXPmax = 90
    logT = np.log10(T)
    T3 = T/1000
    Lambda_H2_thick = (6.7e-19*np.exp(-min(5.86/T3,EXPmax)) + 1.6e-18*np.exp(-min(11.7/T3,EXPmax)) + 3.e-24*np.exp(-min(0.51/T3,EXPmax)) + 9.5e-22*pow(T3,3.76)*np.exp(-min(0.0022/(T3*T3*T3),EXPmax))/(1.+0.12*pow(T3,2.1))) / nH; #  super-critical H2-H cooling rate [per H2 molecule]
    Lambda_HD_thin = ((1.555e-25 + 1.272e-26*pow(T,0.77))*np.exp(-min(128./T,EXPmax)) + (2.406e-25 + 1.232e-26*pow(T,0.92))*np.exp(-min(255./T,EXPmax))) * np.exp(-min(T3*T3/25.,EXPmax)); #  optically-thin HD cooling rate [assuming all D locked into HD at temperatures where this is relevant], per molecule
    
    q = logT - 3.; Y_Hefrac=0.25; X_Hfrac=0.75; #  variable used below
    Lambda_H2_thin = max(nH-2.*f_molec,0) * X_Hfrac * pow(10., max(-103. + 97.59*logT - 48.05*logT*logT + 10.8*logT*logT*logT - 0.9032*logT*logT*logT*logT , -50.)); #  sub-critical H2 cooling rate from H2-H collisions [per H2 molecule]; this from Galli & Palla 1998
    Lambda_H2_thin += Y_Hefrac * pow(10., max(-23.6892 + 2.18924*q -0.815204*q*q + 0.290363*q*q*q -0.165962*q*q*q*q + 0.191914*q*q*q*q*q, -50.)); #  H2-He; often more efficient than H2-H at very low temperatures (<100 K); this and other H2-x terms below from Glover & Abel 2008
    Lambda_H2_thin += f_molec * X_Hfrac * pow(10., max(-23.9621 + 2.09434*q -0.771514*q*q + 0.436934*q*q*q -0.149132*q*q*q*q -0.0336383*q*q*q*q*q, -50.)); #  H2-H2; can be more efficient than H2-H when H2 fraction is order-unity
    
    f_HD = min(0.00126*f_molec , 4.0e-5*nH)

    nH_over_ncrit = Lambda_H2_thin / Lambda_H2_thick
    Lambda_HD = f_HD * Lambda_HD_thin / (1. + f_HD/(f_molec+1e-10)*nH_over_ncrit) * nH 
    Lambda_H2 = f_molec * Lambda_H2_thin / (1. + nH_over_ncrit) * nH
    return Lambda_H2 + Lambda_HD

def CII_cooling(nH=1, Z=1, T=10, NH=1e21, ISRF=1,prescription="Simple"):    
    if prescription=="Hopkins 2022 (FIRE-3)":
        return atomic_cooling_fire3(nH,NH,T,Z,ISRF)
    T_CII = 91
    f_C = 1-f_CO(nH,NH,T,ISRF,Z)
    xc = 1.1e-4
    return 8e-10 * 1.256e-14 * xc * np.exp(-T_CII/T) * Z * nH * f_C

def lyman_cooling(nH=1,T=1000):
    return 2e-19 * np.exp(-1.184e5/T) * nH

def atomic_cooling_fire3(nH,NH,T,Z,ISRF):
    f = f_CO(nH,NH,T,ISRF,Z)
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

def CO_cooling(nH=1, T=10, NH=0,Z=1,ISRF=1,divv=None,xCO=None,simple=False,prescription='Whitworth 2018'):
    fmol = f_CO(nH,NH,T,ISRF,Z)
    pc_to_cm = 3.08567758128E+18
    if xCO is None:
        xCO = fmol * Z * 1.1e-4 * 2
    if divv is None:
        R_scale = NH/nH/pc_to_cm
        divv = 1. * R_scale**0.5 / R_scale # size-linewidth relation

    if prescription=="Gong 2017":
        n_H2 = fmol*nH/2
        neff = n_H2 + nH*2**0.5 * (2.3e-15/(3.3e-16*(T/1000)**-0.25)) # Eq. 34 from gong 2017, ignoring electrons 
        NCO = xCO * nH / (divv / pc_to_cm)
       # print(xCO,NCO)
        LCO = get_tabulated_CO_coolingrate(T,NCO,neff)
        return LCO * xCO * n_H2
    elif prescription=='Hopkins 2022 (FIRE-3)':
        sigma_crit_CO=  1.3e19 * T / Z
        ncrit_CO=1.9e4 * T**0.5
        #lambda_CO_HI = 2e-26 * divv * (nH/1e2)**-1 * (T/10)**4
        return 2.7e-31 * T**1.5 * (xCO/3e-4) * nH/(1 + (nH/ncrit_CO)*(1+NH/sigma_crit_CO)) #lambda_CO_HI)
    elif prescription=='Whitworth 2018':
        lambda_CO_LO = 5e-27 * (xCO/3e-4) * (T/10)**1.5 * (nH/1e3)
        lambda_CO_HI = 2e-26 * divv * (nH/1e2)**-1 * (T/10)**4
        beta = 1.23 * (nH/2)**0.0533 * T**0.164
        if simple: return np.min([lambda_CO_LO, lambda_CO_HI],axis=0)
        else: return (lambda_CO_LO**(-1/beta) + lambda_CO_HI**(-1/beta))**(-beta)

def CR_heating(zeta_CR=2e-16,NH=None):
    if NH is not None:
        return 3e-27 * (zeta_CR / 2e-16)/(1+(NH/1e22))
    else:
        return 3e-27 * (zeta_CR / 2e-16)
    
@njit(fastmath=True,error_model='numpy')
def gas_dust_heating_coeff(T,Z):
    return 1.1e-32 * Z * np.sqrt(T) * (1-0.8*np.exp(-75/T))

@njit(fastmath=True,error_model='numpy')
def dust_gas_cooling(nH=1,T=10,Tdust=20,Z=1):
    return nH * gas_dust_heating_coeff(T,Z) * (Tdust-T)  #3e-26 * Z * (T/10)**0.5 * (Tdust-T)/10 * (nH/1e6)

def compression_heating(nH=1,T=10):
    return 1.2e-27 * np.sqrt(nH/1e6) * (T/10) # ceiling here to get reasonable results in diffuse ISM
        #return 0.

def turbulent_heating(sigma_GMC=100., M_GMC=1e5):
    return 5e-27 * (M_GMC/1e6)**0.25 * (sigma_GMC/100)**(1.25)

def dust_temperature(nH=1,T=10,Z=1,NH=0, ISRF=1, z=0,beta=2):
    abs = dust_absorption_rate(NH,Z,ISRF,z, beta)
    sigma_IR_0 = 2e-25
    Tdust_guess = 10*(abs / (sigma_IR_0 * Z * 2.268))**(1./(4+beta))
    Tdust_guess = max(Tdust_guess, (abs/(4.5e-23*Z*2.268))**0.25)
    Tdust_guess = max(Tdust_guess, T - 2.268 * sigma_IR_0 * Z * (min(T,150)/10)**beta * (T/10)**4 /  (gas_dust_heating_coeff(T,Z)*nH))

    func = lambda dT: net_dust_heating(dT, nH,T,NH,Z,ISRF,z,beta,abs) # solving for the difference T - Tdust since that's what matters for dust heating
    result = root_scalar(func, x0 = Tdust_guess-T,x1 =(Tdust_guess*1.1 - T), method='secant',xtol=1e-5)#,rtol=1e-3,xtol=1e-4*T)
    Tdust = T+result.root
    if not result.converged:
        func = lambda logT: net_dust_heating(10**logT - T, nH,T,NH,Z,ISRF,z,beta,abs)
        result = root_scalar(func, bracket=[-1,8], method='brentq')
        Tdust = 10**result.root
    return Tdust

@njit(fastmath=True,error_model='numpy')
def net_dust_heating(dT,nH,T,NH,Z=1,ISRF=1,z=0, beta=2, absorption=-1):
    Td = T + dT
    sigma_IR_0 = 2e-25
    sigma_IR_emission = sigma_IR_0 * Z * (min(Td,150)/10)**beta # dust cross section per H in cm^2
    lambdadust_thin = 2.268 * sigma_IR_emission * (Td/10)**4
    lambdadust_thick = 2.268 * (Td/10)**4 / (NH+1e-100)
    psi_IR = 1/(1/lambdadust_thin + 1/lambdadust_thick) #interpolates the lower envelope of the optically-thin and -thick limits
    lambda_gd = dust_gas_cooling(nH,T,Td,Z)
    
    if absorption < 0:
        absorption = dust_absorption_rate(NH,Z,ISRF,z, beta)
    return absorption - lambda_gd - psi_IR

@njit(fastmath=True,error_model='numpy')
def dust_absorption_rate(NH,Z=1,ISRF=1,z=0, beta=2):
    T_CMB = 2.73*(1+z)
    ISRF_OPT_eV_cm3 = ISRF * 0.54
    ISRF_IR_eV_cm3 = ISRF * 0.39
    T_ISRF=max(20,3.8*(ISRF_OPT_eV_cm3+ISRF_IR_eV_cm3)**0.25)  # assume 20K dust emission or the blackbody temperature of the ISRF energy density, whichever is greater

    sigma_UV = 1e-21 * Z
    sigma_OPT = 3e-22 * Z
    sigma_IR_0 = 2e-25
    sigma_IR_CMB = sigma_IR_0 * Z * (min(T_CMB,150)/10)**beta
    sigma_IR_ISRF = sigma_IR_0 * Z * (min(T_ISRF,150)/10)**beta
    
    tau_UV = sigma_UV * NH
    gamma_UV = ISRF * np.exp(-tau_UV) * 5.965e-25 * Z

    tau_OPT = sigma_OPT * NH
    gamma_OPT = ISRF * 7.78e-24 * np.exp(-tau_OPT) * Z
    gamma_IR =  2.268 * sigma_IR_CMB * (T_CMB/10)**4 + 0.048 * (ISRF_IR_eV_cm3 + ISRF_OPT_eV_cm3 * (1-np.exp(-tau_OPT))) * sigma_IR_ISRF
    return gamma_IR + gamma_UV + gamma_OPT

all_processes = "CR Heating", "Lyman cooling", "Photoelectric", "CII Cooling", "CO Cooling", "Dust-Gas Coupling", "Grav. Compression", "H_2 Cooling", "Turb. Dissipation"

def net_heating(T=10, nH=1, ISRF=1, NH=0, Z=1, z=0, divv=None, zeta_CR=2e-16, Tdust=None,jeans_shielding=False, 
compression=0,dust_beta=2.,processes=all_processes, attenuate_cr=True,co_prescription="Whitworth 2018",cii_prescription="Hopkins 2022 (FIRE-3)"):
    if jeans_shielding:
        lambda_jeans = 8.1e19 * nH**-0.5 * (T/10)**0.5
        NH = np.max([nH*lambda_jeans*jeans_shielding, NH],axis=0)
    if Tdust==None:
        Tdust = dust_temperature(nH,T,Z,NH,ISRF,z,dust_beta)

    rate = 0
    for process in processes:
        if process == "CR Heating": rate += CR_heating(zeta_CR,NH*attenuate_cr)
        if process == "Lyman cooling": rate -= lyman_cooling(nH,T)
        if process == "Photoelectric": rate += photoelectric_heating(ISRF, nH,T, NH, Z)
        if process == "CII Cooling": rate -= CII_cooling(nH, Z, T, NH, ISRF,prescription=cii_prescription)
        if process == "CO Cooling": rate -= CO_cooling(nH,T,NH,Z,ISRF,divv,prescription=co_prescription)
        if process == "Dust-Gas Coupling": rate += dust_gas_cooling(nH,T,Tdust,Z)
        if process == "H_2 Cooling": rate -= H2_cooling(nH,NH,T,ISRF,Z)
        if process == "Grav. Compression": rate += compression*compression_heating(nH,T)
        if process == "Turb. Dissipation": rate += turbulent_heating()
            
    # rate = CR_heating(zeta_CR,NH*attenuate_cr) - lyman_cooling(nH,T) \
    # + photoelectric_heating(ISRF, nH,T, NH, Z) - CII_cooling(nH, Z, T, NH, ISRF,prescription=cii_prescription) \
    # - CO_cooling(nH,T,NH,Z,ISRF,divv,prescription=co_prescription)  \
    # + dust_gas_cooling(nH,T,Tdust,Z) + compression*compression_heating(nH,T)  \
    # - H2_cooling(nH,NH,T,ISRF,Z)
    return rate

def equilibrium_temp(nH=1, NH=0, ISRF=1, Z=1, z=0, divv=None, zeta_CR=2e-16, Tdust=None,jeans_shielding=False,
compression=False,dust_beta=2.,processes=all_processes,attenuate_cr=True,return_Tdust=True,co_prescription="Whitworth 2018",cii_prescription="Hopkins 2022 (FIRE-3)"):
    if NH==0: NH=1e18
    params = nH, ISRF, NH, Z, z, divv, zeta_CR, Tdust, jeans_shielding,compression, dust_beta, processes, attenuate_cr, co_prescription, cii_prescription
    func = lambda logT: net_heating(10**logT, *params)/1e-30 # solving vs logT converges a bit faster

    try:
        T = 10**brentq(func, -1,5,rtol=1e-3,maxiter=500)
    except:
        try:
            T = 10**brentq(func, -1,10,rtol=1e-3,maxiter=500)
        except:
            raise("Couldn't solve for temperature! Try some other parameters.")

    if return_Tdust:
        Tdust = dust_temperature(nH,T,Z,NH,ISRF,z,dust_beta)
        return T, Tdust
    else:
        return T

equilibrium_temp = np.vectorize(equilibrium_temp,excluded=["processes"])
