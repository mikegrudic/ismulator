import streamlit as st
import pandas as pd
import numpy as np
#from matplotlib import pyplot as plt
from ism_eqm_temp import *
#from PIL import Image
from time import sleep, time
#import plotly.express as px
import bokeh.plotting as bk
from bokeh.models import Range1d

#plt.clf()
#plt.close()

def make_plot_matplotlib(n,T,Tdust):
    print("Plotting...")
    plt.clf()
    fig,ax = plt.subplots()
    if jeans_mass:
        MJ = 5 * (T/10)**1.5 * (n/1e4)**-0.5
        ax.loglog(n,MJ,color='black')
        ax.set(xlim=[nmin,nmax],ylim=[1e-4,1e6],xlabel=r"$n_{\rm H}\,\left(\rm cm^{-3}\right)$",ylabel=r"$M_{\rm J}\left(M_\odot\right)$")
    else:
        ax.loglog(n,T,label="Gas",color='black')
        ax.loglog(n,Tdust,label="Dust",color='black',ls='dashed')
        ax.legend(loc=1)
        ax.set(xlim=[nmin,nmax],ylim=[Tmin,Tmax],xlabel=r"$n_{\rm H}\,\left(\rm cm^{-3}\right)$",ylabel=r"$T\left(\rm K\right)$")
    #plt.savefig("fig%g.png"%time(),dpi=400,bbox_inches='tight')
    print("Done plotting!")
    #plt.close()
    return fig

def make_plot_plotly(n,T,Tdust):
    df = pd.DataFrame(dict(
        n = n,
        T = T,
        Td = Tdust,
    ))
    labels = {"n": r"$n_{\rm H}\,\left(\rm cm^{-3}\right)$", "T": r"$T\left(\rm K\right)$"}
    fig = px.line(df,x='n', y=df.columns[1:], labels=labels,log_x=True, log_y=True) 
    return fig

def make_plot_bokeh(n,T,Tdust):
    if jeans_mass:
        p = bk.figure(y_axis_type="log",x_axis_type="log",x_axis_label=r"$$n_{\rm H}\,\left(\rm cm^{-3}\right)$$",y_axis_label=r"$$M_{\rm J}\left(M_\odot\right)$$")
        p.line(n,5*(T/10)**1.5*(n/1e4)**-0.5,color='black',line_width=2)
        p.y_range=Range1d(1e-4,1e6)
    else:
        p = bk.figure(y_axis_type="log",x_axis_type="log",x_axis_label=r"$$n_{\rm H}\,\left(\rm cm^{-3}\right)$$",y_axis_label=r"$$T\left(\rm K\right)$$")
        p.line(n,T,color='black',legend_label="Gas",line_width=2)
        p.line(n,Tdust,color='black',line_dash='dashed',legend_label="Dust",line_width=2)
        p.y_range=Range1d(Tmin,Tmax)

    p.x_range=Range1d(nmin,nmax)
    return p

st.set_page_config(page_title="ISMulator")
col1,col2 = st.columns([0.75,0.25])

NH2 = 10**st.sidebar.slider(r'Column density: $\log N_{\rm H}/\rm cm^{-2}$ at $n_{\rm H}=100\,\rm cm^{-3}$',min_value=19.,max_value=23., value=21.)
Z = 10**st.sidebar.slider(r'Metallicity: $\log Z$',min_value=-4.,max_value=1., value=0.)
z = st.sidebar.slider(r'Redshift: $z$',min_value=0.,max_value=20., value=0.)
zeta_CR = 10**st.sidebar.slider(r'$\log \zeta_{\rm CR}$: Cosmic ray ionization rate ($s^{-1}$)',min_value=-18.,max_value=-10., value=-15.7)
ISRF = 10**st.sidebar.slider(r'$\log \chi$: Interstellar radiation field strength vs. Solar',min_value=-2.,max_value=4.,value=0.)
NH_alpha = st.sidebar.slider(r'Column density scaling: $\alpha$ where $N_{\rm H}=N_{\rm H,0}\left(n_{\rm H}/100\rm cm^{-3}\right)^{\alpha}$',min_value=0.,max_value=1., value=.3)
fJ = st.sidebar.slider(r'$f_{\rm J}=l/\lambda_{\rm J}$: Ratio of shielding length floor to Jeans wavelength',min_value=0.,max_value=1.,value=0.25)
dust_beta = st.sidebar.slider(r'$\beta$: Dust spectral index',min_value=1.,max_value=2.,value=2.)
#compression = st.checkbox(r"Gravitational collapse heating",value=True)
attenuate_cr = col2.checkbox(r"Cosmic ray attenuation",value=True)
jeans_mass = col2.checkbox(r"Plot Jeans mass",value=False)
process_dict ={}
for process in all_processes:
    process_dict[process] = col2.checkbox(process, value=True)
processes_to_use = [p  if process_dict[p]  else "" for p in process_dict.keys()]
compression = process_dict["Grav. Compression"]

CO_Rx = st.sidebar.selectbox("CO cooling prescription", ["Whitworth 2018","Gong 2017","Hopkins 2022 (FIRE-3)"])
CI_Rx = st.sidebar.selectbox("Atomic cooling prescription", ["Hopkins 2022 (FIRE-3)","Simple"])

# sliders for plot limits
Tmin = 1
Tmax = 1e4
nmin = 10**st.sidebar.number_input(r"Min. $\log n_{\rm H}\left(\rm cm^{-3}\right)$",min_value=-2., max_value=20.,value=0.)
nmax = 10**st.sidebar.number_input(r"Max. $\log n_{\rm H}\left(\rm cm^{-3}\right)$",min_value=0., max_value=20.,value=12.)
Tmin = 10**st.sidebar.number_input(r"Min. $\log T\left(\rm K\right)$",min_value=0., max_value=8.,value=0.)
Tmax = 10**st.sidebar.number_input(r"Max. $\log T\left(\rm K\right)$",min_value=0., max_value=7.,value=4.)


#switches for different cooling physics?

n = np.logspace(np.log10(nmin),np.log10(nmax),100)
NH = NH2 * (n/1e2)**NH_alpha

try:
    T, Tdust = equilibrium_temp(n,NH,jeans_shielding=fJ, compression=compression,Z=Z,z=z,ISRF=ISRF,zeta_CR=zeta_CR,                                                        
    attenuate_cr=attenuate_cr,return_Tdust=True,dust_beta=dust_beta,co_prescription=CO_Rx,cii_prescription=CI_Rx,processes=processes_to_use)
except:
    st.write("Couldn't solve for temperature!")
    T=Tdust=np.zeros_like(n)

#fig = make_plot_matplotlib(n,T,Tdust)
#st.session_state

#st.image(Image.open("fig.png"))
#st.pyplot(fig,clear_figure=True)
#st.plotly_chart(make_plot_plotly(n,T,Tdust))
col1.bokeh_chart(make_plot_bokeh(n,T,Tdust),use_container_width=True)
"ISMulator solves for the equilibrium of gas and dust heating and cooling in interstellar clouds as a function of density."
"Warning: results at >1000K are VERY approximate because I didn't feel like solving for ionization 🙃."
"Source code available at https://github.com/mikegrudic/ISMulator"
