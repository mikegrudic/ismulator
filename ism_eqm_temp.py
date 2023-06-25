from scipy.optimize import brentq, newton, root_scalar
from scipy.interpolate import interpn, interp1d
from numba import njit
import numpy as np

def photoelectric_heating(X_FUV=1, nH=1, T=10, NH=0, Z=1):
    """
    Rate of photoelectric heating per H nucleus in erg/s.
    Weingartner & Draine 2001 prescription
    Grain charge parameter is a highly approximate fit vs. density - otherwise need to solve ionization.
    """
    grain_charge =  max((5e3/nH),50)
    c0=5.22; c1 =2.25; c2=0.04996; c3=0.0043; c4=0.147; c5=0.431; c6=0.692
    eps_PE = (c0+c1*T**c4)/(1+c2*grain_charge**c5 * (1+c5*grain_charge**c6))
    sigma_FUV = 1e-21 * Z
    #print(X_FUV,nH,T, eps_PE, 1e-26 * X_FUV * eps_PE * np.exp(-NH*sigma_FUV) * Z)
    return 1e-26 * X_FUV * eps_PE * np.exp(-NH*sigma_FUV) * Z

def f_CO(nH=1, NH=1e21, T=10,X_FUV=1, Z=1):
    """Equilibrium fraction of C locked in CO, from Tielens 2005"""
    G0 = 1.7 * X_FUV * np.exp(-1e-21 * NH * Z)
    if nH > 10000*G0*340: return 1.
    x = (nH/(G0*340))**2*T**-0.5
    return x/(1+x)

def f_H2(nH=1, NH=1e21, X_FUV=1, Z=1):
    """Krumholz McKee Tumlinson 2008 prescription for fraction of neutral H in H_2 molecules"""
    surface_density_Msun_pc2 = NH * 1.1e-20  
    tau_UV = min(1e-21 * Z * NH,100.)
    G0 = 1.7 * X_FUV * np.exp(-tau_UV)
    chi = 71. * X_FUV / nH
    psi = chi * (1.+0.4*chi)/(1.+1.08731*chi)
    s = (Z + 1.e-3) * surface_density_Msun_pc2 / (1e-100 + psi)
    q = s * (125. + s) / (11. * (96. + s))
    fH2 = 1. - (1.+q*q*q)**(-1./3.)
    if q < 0.2: fH2 = q*q*q * (1. - 2.*q*q*q/3.)/3.
    elif q>10: fH2 = 1. - 1/q
    return fH2

def H2_cooling(nH,NH,T,X_FUV,Z):
    """
    Glover & Abel 2008 prescription for H_2 cooling; accounts for H2-H2 and H2-HD collisions.
    Rate per H nucleus in erg/s.
    """
    f_molec = 0.5 * f_H2(nH,NH,X_FUV,Z)
    EXPmax = 90
    logT = np.log10(T)
    T3 = T/1000
    Lambda_H2_thick = (6.7e-19*np.exp(-min(5.86/T3,EXPmax)) + 1.6e-18*np.exp(-min(11.7/T3,EXPmax)) + 3.e-24*np.exp(-min(0.51/T3,EXPmax)) + 9.5e-22*pow(T3,3.76)*np.exp(-min(0.0022/(T3*T3*T3),EXPmax))/(1.+0.12*pow(T3,2.1))) / nH; #  super-critical H2-H cooling rate [per H2 molecule]
    Lambda_HD_thin = ((1.555e-25 + 1.272e-26*pow(T,0.77))*np.exp(-min(128./T,EXPmax)) + (2.406e-25 + 1.232e-26*pow(T,0.92))*np.exp(-min(255./T,EXPmax))) * np.exp(-min(T3*T3/25.,EXPmax)); #  optically-thin HD cooling rate [assuming all D locked into HD at temperatures where this is relevant], per molecule
    
    q = logT - 3.; Y_Hefrac=0.25; X_Hfrac=0.75; #  variable used below
    Lambda_H2_thin = max(nH-2.*f_molec,0) * X_Hfrac * np.power(10., max(-103. + 97.59*logT - 48.05*logT*logT + 10.8*logT*logT*logT - 0.9032*logT*logT*logT*logT , -50.)); #  sub-critical H2 cooling rate from H2-H collisions [per H2 molecule]; this from Galli & Palla 1998
    Lambda_H2_thin += Y_Hefrac * np.power(10., max(-23.6892 + 2.18924*q -0.815204*q*q + 0.290363*q*q*q -0.165962*q*q*q*q + 0.191914*q*q*q*q*q, -50.)); #  H2-He; often more efficient than H2-H at very low temperatures (<100 K); this and other H2-x terms below from Glover & Abel 2008
    Lambda_H2_thin += f_molec * X_Hfrac * np.power(10., max(-23.9621 + 2.09434*q -0.771514*q*q + 0.436934*q*q*q -0.149132*q*q*q*q -0.0336383*q*q*q*q*q, -50.)); #  H2-H2; can be more efficient than H2-H when H2 fraction is order-unity
    
    f_HD = min(0.00126*f_molec , 4.0e-5*nH)

    nH_over_ncrit = Lambda_H2_thin / Lambda_H2_thick
    Lambda_HD = f_HD * Lambda_HD_thin / (1. + f_HD/(f_molec+1e-10)*nH_over_ncrit) * nH 
    Lambda_H2 = f_molec * Lambda_H2_thin / (1. + nH_over_ncrit) * nH
    return Lambda_H2 + Lambda_HD

def CII_cooling(nH=1, Z=1, T=10, NH=1e21, X_FUV=1,prescription="Simple"):    
    """Cooling due to atomic and/or ionized C. Uses either Hopkins 2022 FIRE-3 or simple prescription. Rate per H nucleus in erg/s."""
    if prescription=="Hopkins 2022 (FIRE-3)":
        return atomic_cooling_fire3(nH,NH,T,Z,X_FUV)
    T_CII = 91
    f_C = 1-f_CO(nH,NH,T,X_FUV,Z)
    xc = 1.1e-4
    return 8e-10 * 1.256e-14 * xc * np.exp(-T_CII/T) * Z * nH * f_C

def lyman_cooling(nH=1,T=1000):
    """Rate of Lyman-alpha cooling from Koyama & Inutsuka 2002 per H nucleus in erg/s. Actually a hard upper bound assuming xe ~ xH ~ xH+ ~ 1/2, see Micic 2013 for discussion."""
    return 2e-19 * np.exp(-1.184e5/T) * nH

def atomic_cooling_fire3(nH,NH,T,Z,X_FUV):
    """Cooling due to atomic and ionized C. Uses Hopkins 2022 FIRE-3 prescription. Rate per H nucleus in erg/s."""
    f = f_CO(nH,NH,T,X_FUV,Z)
    return 1e-27 * (0.47*T**0.15 * np.exp(-91/T) + 0.0208 * np.exp(-23.6/T)) * (1-f) * nH * Z

def get_tabulated_CO_coolingrate(T,NH,nH2):
    """Tabulated CO cooling rate from Omukai 2010, used for Gong 2017 implementation of CO cooling."""
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

def CO_cooling(nH=1, T=10, NH=0,Z=1,X_FUV=1,divv=None,xCO=None,simple=False,prescription='Whitworth 2018'):
    """
    Rate of CO cooling per H nucleus in erg/s.

    Three prescriptions are implemented: Gong 2017, Whitworth 2018, and Hopkins 2022 (FIRE-3).

    Prescriptions that require a velocity gradient will assume a standard ISM size-linewidth relation by default, 
    unless div v is provided.2
    """
    fmol = f_CO(nH,NH,T,X_FUV,Z)
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
        LCO = get_tabulated_CO_coolingrate(T,NCO,neff)
        return LCO * xCO * n_H2
    elif prescription=='Hopkins 2022 (FIRE-3)':
        sigma_crit_CO=  1.3e19 * T / Z
        ncrit_CO=1.9e4 * T**0.5
        return 2.7e-31 * T**1.5 * (xCO/3e-4) * nH/(1 + (nH/ncrit_CO)*(1+NH/sigma_crit_CO)) #lambda_CO_HI)
    elif prescription=='Whitworth 2018':
        lambda_CO_LO = 5e-27 * (xCO/3e-4) * (T/10)**1.5 * (nH/1e3)
        lambda_CO_HI = 2e-26 * divv * (nH/1e2)**-1 * (T/10)**4
        beta = 1.23 * (nH/2)**0.0533 * T**0.164
        if simple: return np.min([lambda_CO_LO, lambda_CO_HI],axis=0)
        else: return (lambda_CO_LO**(-1/beta) + lambda_CO_HI**(-1/beta))**(-beta)

def CR_heating(zeta_CR=2e-16,NH=None):
    """Rate of cosmic ray heating in erg/s/H, just assuming 10eV per H ionization."""
    if NH is not None:
        return 3e-27 * (zeta_CR / 2e-16)/(1+(NH/1e21))
    else:
        return 3e-27 * (zeta_CR / 2e-16)
    
@njit(fastmath=True,error_model='numpy')
def gas_dust_heating_coeff(T,Z,dust_coupling):
    """Coefficient alpha such that the gas-dust heat transfer is alpha * (T-T_dust)
    
    Uses Hollenbach & McKee 1979 prescription, assuming 10 Angstrom min grain size.
    """
    return 1.1e-32 * dust_coupling* Z * np.sqrt(T) * (1-0.8*np.exp(-75/T)) 

@njit(fastmath=True,error_model='numpy')
def dust_gas_cooling(nH=1,T=10,Tdust=20,Z=1,dust_coupling=1):
    """
    Rate of heat transfer from gas to dust in erg/s per H
    """
    return nH * gas_dust_heating_coeff(T,Z,dust_coupling) * (Tdust-T)  #3e-26 * Z * (T/10)**0.5 * (Tdust-T)/10 * (nH/1e6)

def compression_heating(nH=1,T=10):
    """Rate of compressional heating per H in erg/s, assuming freefall collapse (e.g. Masunaga 1998)"""
    return 1.2e-27 * np.sqrt(nH/1e6) * (T/10) # ceiling here to get reasonable results in diffuse ISM

def turbulent_heating(sigma_GMC=100., M_GMC=1e5):
    """
    Rate of tubulent dissipation per H in erg/s, assuming a turbulent GMC with virial parameter=1 of a certain surface density and mass.
    
    Note that much of the cooling can take place in shocks that are way out of equilibrium, so this doesn't necessarily capture the full effect.
    """
    return 5e-27 * (M_GMC/1e6)**0.25 * (sigma_GMC/100)**(1.25)

def dust_temperature(nH=1,T=10,Z=1,NH=0, X_FUV=1, X_OPT=1, z=0,beta=2,dust_coupling=1):
    """
    Equilibrium dust temperature obtained by solving the dust energy balance equation accounting for absorption, emission, and gas-dust heat transfer.    
    """
    abs = dust_absorption_rate(NH,Z,X_FUV,X_OPT,z, beta)
    sigma_IR_0 = 2e-25
    Tdust_guess = 10*(abs / (sigma_IR_0 * Z * 2.268))**(1./(4+beta))
    Tdust_guess = max(Tdust_guess, (abs/(4.5e-23*Z*2.268))**0.25)
    Tdust_guess = max(Tdust_guess, T - 2.268 * sigma_IR_0 * Z * (min(T,150)/10)**beta * (T/10)**4 /  (gas_dust_heating_coeff(T,Z,dust_coupling)*nH))

    func = lambda dT: net_dust_heating(dT, nH,T,NH,Z,X_FUV,X_OPT,z,beta,abs,dust_coupling) # solving for the difference T - Tdust since that's what matters for dust heating
    result = root_scalar(func, x0 = Tdust_guess-T,x1 =(Tdust_guess*1.1 - T), method='secant',xtol=1e-5)#,rtol=1e-3,xtol=1e-4*T)
    Tdust = T+result.root
    if not result.converged:
        func = lambda logT: net_dust_heating(10**logT - T, nH,T,NH,Z,X_FUV,X_OPT,z,beta,abs,dust_coupling)
        result = root_scalar(func, bracket=[-1,8], method='brentq')
        Tdust = 10**result.root
    return Tdust

@njit(fastmath=True,error_model='numpy')
def net_dust_heating(dT,nH,T,NH,Z=1,X_FUV=1,X_OPT=1,z=0, beta=2, absorption=-1,dust_coupling=1):
    """Derivative of the dust energy in the dust energy equation, solve this = 0 to get the equilibrium dust temperature."""
    Td = T + dT
    sigma_IR_0 = 2e-25
    sigma_IR_emission = sigma_IR_0 * Z * (min(Td,150)/10)**beta # dust cross section per H in cm^2
    lambdadust_thin = 2.268 * sigma_IR_emission * (Td/10)**4
    lambdadust_thick = 2.268 * (Td/10)**4 / (NH+1e-100)
    psi_IR = 1/(1/lambdadust_thin + 1/lambdadust_thick) #interpolates the lower envelope of the optically-thin and -thick limits
    lambda_gd = dust_gas_cooling(nH,T,Td,Z,dust_coupling)
    
    if absorption < 0:
        absorption = dust_absorption_rate(NH,Z,X_FUV,X_OPT,z, beta)
    return absorption - lambda_gd - psi_IR

@njit(fastmath=True,error_model='numpy')
def dust_absorption_rate(NH,Z=1,X_FUV=1,X_OPT=1,z=0, beta=2):
    """Rate of radiative absorption by dust, per H nucleus in erg/s."""
    T_CMB = 2.73*(1+z)
    X_OPT_eV_cm3 = X_OPT * 0.54
    X_IR_eV_cm3 = X_OPT * 0.39
    X_FUV_eV_cm3 = X_FUV * 0.041
    T_IR=max(20,3.8*(X_OPT_eV_cm3+X_IR_eV_cm3)**0.25)  # assume 20K dust emission or the blackbody temperature of the X_FUV energy density, whichever is greater

    sigma_UV = 1e-21 * Z
    sigma_OPT = 3e-22 * Z
    sigma_IR_0 = 2e-25
    sigma_IR_CMB = sigma_IR_0 * Z * (min(T_CMB,150)/10)**beta
    sigma_IR_ISRF = sigma_IR_0 * Z * (min(T_IR,150)/10)**beta
    
    tau_UV = min(sigma_UV * NH,100)
    gamma_UV = X_FUV * np.exp(-tau_UV) * 5.965e-25 * Z

    tau_OPT = min(sigma_OPT * NH,100)
    gamma_OPT = X_OPT * 7.78e-24 * np.exp(-tau_OPT) * Z
    gamma_IR =  2.268 * sigma_IR_CMB * (T_CMB/10)**4 + 0.048 * (X_IR_eV_cm3 + X_OPT_eV_cm3 * (-np.expm1(-tau_OPT)) + X_FUV_eV_cm3 * (-np.expm1(-tau_UV))) * sigma_IR_ISRF
    return gamma_IR + gamma_UV + gamma_OPT

all_processes = "CR Heating", "Lyman cooling", "Photoelectric", "CII Cooling", "CO Cooling", "Dust-Gas Coupling", "Grav. Compression", "H_2 Cooling", "Turb. Dissipation"

def net_heating(T=10, nH=1, NH=0, X_FUV=1,X_OPT=1,Z=1, z=0, divv=None, zeta_CR=2e-16, Tdust=None,jeans_shielding=False,dust_beta=2.,dust_coupling=1,sigma_GMC=100, processes=all_processes, attenuate_cr=True,co_prescription="Whitworth 2018",cii_prescription="Hopkins 2022 (FIRE-3)"):
    if jeans_shielding:
        lambda_jeans = 8.1e19 * nH**-0.5 * (T/10)**0.5
        NH = np.max([nH*lambda_jeans*jeans_shielding, NH],axis=0)
    if Tdust==None:
        Tdust = dust_temperature(nH,T,Z,NH,X_FUV,X_OPT,z,dust_beta,dust_coupling * ("Dust-Gas Coupling" in processes))

    rate = 0
    for process in processes:
        if process == "CR Heating": rate += CR_heating(zeta_CR,NH*attenuate_cr)
        if process == "Lyman cooling": rate -= lyman_cooling(nH,T)
        if process == "Photoelectric": rate += photoelectric_heating(X_FUV, nH,T, NH, Z)
        if process == "CII Cooling": rate -= CII_cooling(nH, Z, T, NH, X_FUV,prescription=cii_prescription)
        if process == "CO Cooling": rate -= CO_cooling(nH,T,NH,Z,X_FUV,divv,prescription=co_prescription)
        if process == "Dust-Gas Coupling": rate += dust_gas_cooling(nH,T,Tdust,Z,dust_coupling)
        if process == "H_2 Cooling": rate -= H2_cooling(nH,NH,T,X_FUV,Z)
        if process == "Grav. Compression": rate += compression_heating(nH,T)
        if process == "Turb. Dissipation": rate += turbulent_heating(sigma_GMC=sigma_GMC)
            
    return rate

def equilibrium_temp(nH=1, NH=0, X_FUV=1,X_OPT=1, Z=1, z=0, divv=None, zeta_CR=2e-16, Tdust=None,jeans_shielding=False,
                     dust_beta=2.,dust_coupling=1,sigma_GMC=100., processes=all_processes,attenuate_cr=True,
                     co_prescription="Whitworth 2018",cii_prescription="Hopkins 2022 (FIRE-3)",return_Tdust=True,T_guess=None):
    
    if NH==0: NH=1e18
    params = nH, NH, X_FUV,X_OPT, Z, z, divv, zeta_CR, Tdust, jeans_shielding, dust_beta, dust_coupling, sigma_GMC,processes, attenuate_cr, co_prescription, cii_prescription
    func = lambda logT: net_heating(10**logT, *params) # solving vs logT converges a bit faster
    
    use_brentq = True
    if T_guess is not None: # we have an initial guess that is supposed to be close (e.g. previous grid point)
        T_guess2 = T_guess * 1.01
        result = root_scalar(func, x0=np.log10(T_guess),x1=np.log10(T_guess2), method='secant',rtol=1e-3) #,rtol=1e-3,xtol=1e-4*T)
        if result.converged:
            T = 10**result.root; use_brentq = False

    if use_brentq: 
        try:
            T = 10**brentq(func, -1,5,rtol=1e-3,maxiter=500)
        except:
            try:
                T = 10**brentq(func, -1,10,rtol=1e-3,maxiter=500)
            except:
                raise("Couldn't solve for temperature! Try some other parameters.")

    if return_Tdust:
        Tdust = dust_temperature(nH,T,Z,NH,X_FUV,X_OPT,z,dust_beta,dust_coupling*("Dust-Gas Coupling" in processes))
        return T, Tdust
    else:
        return T

def equilibrium_temp_grid(nH, NH, X_FUV=1, X_OPT=1, Z=1, z=0, divv=None, zeta_CR=2e-16, Tdust=None,jeans_shielding=False,
    dust_beta=2.,dust_coupling=1,sigma_GMC=100., processes=all_processes,attenuate_cr=True,return_Tdust=False,
    co_prescription="Whitworth 2018",cii_prescription="Hopkins 2022 (FIRE-3)"):
        
    params = X_FUV, X_OPT, Z, z, divv, zeta_CR, Tdust, jeans_shielding, dust_beta, dust_coupling, sigma_GMC,processes, attenuate_cr, co_prescription, cii_prescription
    Ts = []
    Tds = []

    T_guess = None
    for i in range(len(nH)): # we do a pass on the grid where we use previously-evaluated temperatures to get good initial guesses for the next grid point
        if i==1:
            T_guess = Ts[-1]
        elif i>1:
            T_guess = 10**interp1d(np.log10(nH[:i]),np.log10(Ts),fill_value="extrapolate")(np.log10(nH[i])) # guess using linear extrapolation in log space

        sol = equilibrium_temp(nH[i],NH[i],*params,return_Tdust=return_Tdust,T_guess=T_guess)
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

    
