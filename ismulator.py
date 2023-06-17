import streamlit as st
#import SessionState
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from ism_eqm_temp import *

#session_initial_state = SessionState.get()

NH2 = 10**st.sidebar.slider(r'Column density: $\log N_{\rm H}/\rm cm^{-2}$ at $\left(n_{\rm H}=100\,\rm cm^{-3}\right)$',min_value=19.,max_value=23., value=21.)
Z = 10**st.sidebar.slider(r'Metallicity: $\log Z$',min_value=-4.,max_value=1., value=0.)
z = st.sidebar.slider(r'Redshift: $z$',min_value=0.,max_value=20., value=0.)
zeta_CR = 10**st.sidebar.slider(r'$\log \zeta_{\rm CR}$: Cosmic ray ionization rate ($s^{-1}$)',min_value=-18.,max_value=-10., value=-15.7)
ISRF = 10**st.sidebar.slider(r'$\log \chi$: Interstellar radiation field strength vs. Solar',min_value=-2.,max_value=4.,value=0.)
NH_alpha = st.sidebar.slider(r'Column density scaling: $\alpha$ where $N_{\rm H}=N_{\rm H,0}\left(n_{\rm H}/100\rm cm^{-3}\right)^{\alpha}$',min_value=0.,max_value=1., value=.3)
fJ = st.sidebar.slider(r'$f_{\rm J}=l/\lambda_{\rm J}$: Ratio of shielding length floor to Jeans wavelength',min_value=0.,max_value=1.,value=0.25)
dust_beta = st.sidebar.slider(r'$\beta$: Dust spectral index',min_value=1.,max_value=2.,value=2.)
compression = st.sidebar.checkbox(r"Gravitational collapse heating",value=True)
attenuate_cr = st.sidebar.checkbox(r"Cosmic ray attenuation",value=True)
CO_Rx = st.sidebar.selectbox("CO cooling prescription", ["Whitworth 2018","Gong 2017","Hopkins 2022 (FIRE-3)"])
CI_Rx = st.sidebar.selectbox("Atomic cooling prescription", ["Hopkins 2022 (FIRE-3)","Simple"])

#reset = st.button("Reset")
#if reset:
    #session_state = session_initial_state

n = np.logspace(0,13,100)
NH = NH2 * (n/1e2)**0.3
T, Tdust = equilibrium_temp(n,NH,jeans_shielding=fJ, compression=compression,Z=Z,z=z,ISRF=ISRF,zeta_CR=zeta_CR,
attenuate_cr=attenuate_cr,return_Tdust=True,dust_beta=dust_beta,co_prescription=CO_Rx,cii_prescription=CI_Rx)
fig,ax = plt.subplots()
ax.loglog(n,T,label="Gas",color='black')
ax.loglog(n,Tdust,label="Dust",color='black',ls='dashed')
ax.legend(loc=1)
ax.set(xlim=[1,1e13],ylim=[1,1e4],xlabel=r"$n_{\rm H}\,\left(\rm cm^{-3}\right)$",ylabel=r"$T\left(\rm K\right)$")
st.write(fig)