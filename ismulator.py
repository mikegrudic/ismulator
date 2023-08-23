import streamlit as st
import pandas as pd
import numpy as np
from ism_eqm_temp import *
import bokeh.plotting as bk
from bokeh.models import Range1d


def make_plot_bokeh(n,T,Tdust,x_var="Density", y_var="Temperature"):

    labeldict = {"Density": "nH (cm^-3)", #r"$$n_{\rm H}\,\left(\rm cm^{-3}\right)$$",
                 "Jeans Mass": "Jeans Mass (M☉)", ##r"$$M_{\rm J}\left(M_\odot\right)$$",
                 "Temperature": "Temperature (K)", #r"$$T\left(\rm K\right)$$",
                 "Pressure": "Pressure (kB K cm^-3)"#r"$$P\left(k_{\rm B} \mathrm{K\,cm^{-3}}\right)$$"
                }

    P = n/(2.3*0.7) * T
    M_J = 5*(T/10)**1.5*(n/1e4)**-0.5
    quantities = {"Density": n,
                 "Jeans Mass": M_J,
                 "Temperature": T,
                 "Pressure": P
                }
    
    X = quantities[x_var]
    Y = quantities[y_var]
    p = bk.figure(y_axis_type="log",x_axis_type="log",x_axis_label=labeldict[x_var],y_axis_label=labeldict[y_var])

    p.line(X,Y,color='black',line_width=2,legend_label="Gas")#,legend_label=("Gas" if "Temp",line_width=2)

    if x_var=="Temperature":
        p.line(Tdust,Y,color='black',line_width=2,legend_label="Dust",line_dash='dashed')
    elif y_var=="Temperature":
        p.line(X,Tdust,color='black',line_width=2,legend_label="Dust",line_dash='dashed')

    return p

st.set_page_config(page_title="ISMulator")
col1,col2 = st.columns([0.75,0.25])

NH2 = 10**st.sidebar.slider(r'Column density: $\log N_{\rm H}/\rm cm^{-2}$ at $n_{\rm H}=100\,\rm cm^{-3}$',min_value=19.,max_value=23., value=21.)
Z = 10**st.sidebar.slider(r'Metallicity: $\log Z$',min_value=-7.,max_value=1., value=0.)
z = st.sidebar.slider(r'Redshift: $z$',min_value=0.,max_value=20., value=0.)
zeta_CR = 10**st.sidebar.slider(r'$\log \zeta_{\rm CR}$: Cosmic ray ionization rate ($s^{-1}$)',min_value=-18.,max_value=-10., value=-15.7)
X_FUV = 10**st.sidebar.slider(r'$\log \chi_{\rm FUV}$: FUV radiation field strength',min_value=-2.,max_value=4.,value=0.)
X_OPT = 10**st.sidebar.slider(r'$\log \chi_{\rm OPT}$: Optical-NIR radiation field strength',min_value=-2.,max_value=4.,value=0.)
NH_alpha = st.sidebar.slider(r'Column density scaling: $\alpha$ where $N_{\rm H}=N_{\rm H,0}\left(n_{\rm H}/100\rm cm^{-3}\right)^{\alpha}$',min_value=0.,max_value=1., value=.3)
fJ = st.sidebar.slider(r'$f_{\rm J}=l/\lambda_{\rm J}$: Ratio of shielding length floor to Jeans wavelength',min_value=0.,max_value=1.,value=0.25)
dust_beta = st.sidebar.slider(r'$\beta$: Dust spectral index',min_value=1.,max_value=2.,value=2.)
dust_coeff = 10**st.sidebar.slider(r'$\log \alpha_{\rm gd}$: dust-gas coupling coefficient',min_value=-4.,max_value=4.,value=0.)
sigma_GMC = 10**st.sidebar.slider(r'$\log \Sigma_{\rm GMC}\left(M_\odot\rm pc^{-2}\right)$ for turb. dissipation',min_value=1.,max_value=5.,value=2.)
attenuate_cr = st.sidebar.checkbox(r"Cosmic ray attenuation",value=True)
x_var = col2.radio("X axis", ["Density", "Pressure","Temperature"])
y_var = col2.radio("Y axis", ["Temperature", "Jeans Mass","Pressure"])
process_dict ={}
for process in all_processes:
    process_dict[process] = col2.checkbox(process, value=True)
processes_to_use = [p  if process_dict[p]  else "" for p in process_dict.keys()]

CO_Rx = st.sidebar.selectbox("CO cooling prescription", ["Whitworth 2018","Gong 2017","Hopkins 2022 (FIRE-3)"])
CI_Rx = st.sidebar.selectbox("Atomic cooling prescription", ["Hopkins 2022 (FIRE-3)","Simple"])

# sliders for plot limits
Tmin = 1
Tmax = 1e4
nmin = 10**st.sidebar.number_input(r"Min. $\log n_{\rm H}\left(\rm cm^{-3}\right)$",min_value=-2., max_value=20.,value=0.)
nmax = 10**st.sidebar.number_input(r"Max. $\log n_{\rm H}\left(\rm cm^{-3}\right)$",min_value=0., max_value=20.,value=8.)
Tmin = 10**st.sidebar.number_input(r"Min. $\log T\left(\rm K\right)$",min_value=0., max_value=8.,value=0.)
Tmax = 10**st.sidebar.number_input(r"Max. $\log T\left(\rm K\right)$",min_value=0., max_value=7.,value=4.)
Ngrid = st.sidebar.number_input(r"Number of $n_{\rm H}$ grid points",min_value=10, max_value=1000,value=100)


n = np.logspace(np.log10(nmin),np.log10(nmax),Ngrid)
NH = NH2 * (n/1e2)**NH_alpha

try:
    T, Tdust = equilibrium_temp_grid(n,NH,jeans_shielding=fJ, Z=Z,z=z,X_FUV=X_FUV,X_OPT=X_OPT,zeta_CR=zeta_CR,                                                        
    attenuate_cr=attenuate_cr,dust_beta=dust_beta,dust_coupling=dust_coeff,sigma_GMC=sigma_GMC,co_prescription=CO_Rx,
    cii_prescription=CI_Rx,processes=processes_to_use,return_Tdust=True)
except:
    st.write("Couldn't solve for temperature!")
    T=Tdust=np.zeros_like(n)

col1.bokeh_chart(make_plot_bokeh(n,T,Tdust,x_var,y_var),use_container_width=True)
"ISMulator solves for the equilibrium of gas and dust heating and cooling in interstellar clouds as a function of density."
"Warning: results at >1000K are VERY approximate because solving ionization is deferred to Future Work ™️🙃"
"Source code available at https://github.com/mikegrudic/ISMulator"
"Contributions welcome via pull request."
