
from ism_eqm_temp import equilibrium_temp
import numpy as np
from matplotlib import pyplot as plt
from time import time

def n_vs_T_plot():
    n = np.logspace(1,13,100)
    NH = 1e21 * (n/1e2)**0.3
    t=  time()
    T = equilibrium_temp(n,NH,jeans_shielding=0.25, compression=True)    

    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(n,T)

    ngrid = np.logspace(0,13,100)
    # for MJ in 10.**np.arange(-1,7): 
    #     TJ = 10 * ((MJ/330) * ngrid**0.5)**(1/1.5)  #/(np.pi**2.5 / 6)/cs**3
    #     ax.plot(ngrid,TJ,ls='dotted',color='black',lw=0.3,zorder=-1000)

    #     n0 = min(ngrid[TJ<np.interp(ngrid,n,T)][-1]/4,3e6)
    #     if MJ > 1e4: n0 *= 8
    #     T3 = np.interp(n0,ngrid,TJ)
    #     if np.abs(np.log10(MJ))<=2:
    #         label = r"$%g M_\odot$"%MJ
    #     else:
    #         label = r"$10^{%g} M_\odot$"%np.log10(MJ)
    #     if T3 > 1:
    #         ax.text(n0,T3*1.2,label,rotation=18,fontsize=5)
    #     else:
    #         n0 = 5e3
    #         T3 = np.interp(n0,ngrid,TJ)
    #         if T3>=0.8:
    #             ax.text(n0,T3*1.2,label,rotation=18,fontsize=5)
    #         else:
    #             n0 = 2e6
    #             T3 = np.interp(n0,ngrid,TJ)
    #             ax.text(n0,T3*1.2,label,rotation=18,fontsize=5)

    # for n0, P in ((1.2,10),(120,1e3),(400,1e5),(30000,1e7),(3e6,1e9)):
    #     mu = 2.3; XH = 0.74
    #     ntot = ngrid / XH / mu
    #     T_P = P/ntot
    #     ax.loglog(ngrid,T_P,ls='dashed',color='grey',lw=0.3,zorder=-1000)
    #     if P>10: label = r"$10^{%g} k_{\rm B}\rm K cm^{-3}$"%np.log10(P)
    #     else: label = None # $\\ $\Sigma \sim 10^{%g}M_\odot \rm pc^{-2}$"%(np.log10(P),np.log10(P/1e5)*0.5 + 2)
    #     ax.text(n0,np.interp(n0,ngrid,T_P)*0.24,label,fontsize=4,rotation=-47)

    ax.set(xlim=[1,1e13],ylim=[3,1e4],xlabel=r"$n_{\rm H}\,\rm \left(cm^{-3}\right)$",ylabel=r"$T\,\left(\rm K\right)$",xscale='log',yscale='log')
    ax.grid(which='major',alpha=0.3)
    #ax.legend(fontsize=8,labelspacing=0.1,loc=1,frameon=True)
    #plt.savefig("nH_vs_T.pdf",bbox_inches='tight')
    plt.show()

def main():
    n_vs_T_plot()

if __name__=="__main__":
    main()
    