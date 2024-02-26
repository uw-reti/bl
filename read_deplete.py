# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 21:50:33 2023

@author: b9801
"""

import h5py
from pwr1 import get_pins
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import matplotlib.pyplot as plt
import numpy as np
import itertools

colorlist=['#648FFF','#785EF0','#DC267F','#FE6100','#FFB000']
colors = itertools.cycle(colorlist)
next(colors)


ks=list()
fis=['Case A (Reference)','Case D (BL324)','Case E (BL324 with ref. clad thickness)','Case K at +20% power (BL324, 5.95 wt%)']
for fi in ["264","324","324C",'324P']:
    f=h5py.File("deplete_"+fi+"/depletion_results.h5")
    ks.append([f['eigenvalues'][j][0][0] for j in range(45)])
    
    if fi=="264":
        t=[f['time'][j][0]/3600/24/365.25 for j in range(45)]
    elif fi != '324P':
        dk=[1e5*(k-k0)/k/k0 for k,k0 in zip(ks[-1],ks[0])]
        plt.plot(t,dk,color=next(colors),label='BL'+fi)

plt.xlabel('time (yrs)')
plt.ylabel('reactivity difference from reference (pcm)')
plt.legend(loc='lower left')
plt.show()

colors = itertools.cycle(colorlist)

for j in range(4):
    plt.plot(t,ks[j],color=next(colors),label=fis[j])

plt.plot([0,7],[1.03,1.03],color='black',linestyle='dashed',label='k-inf=1.03')

plt.xlabel('time (yrs)')
plt.ylabel('k-inf')
plt.legend(loc='upper right')
plt.show()

power=3411e6/193/4/366.0 #Power per cm of the assembly
base_V = (264/4)*0.4095**2*np.pi #cm2 of fuel in assembly
C_mod = ((0.475*np.sqrt(264/324)+0.4095-0.475)/(0.4095*np.sqrt(264/324)))**2 #modifier for the retained clad thickness
HM_rho_5 = 10.4*(235*0.0501+238*(1-0.0501))/(235*0.0501+238*(1-0.0501)+16*2) #HM density for 5% enrichment
HM_rho_6 = 10.4*(235*0.0602+238*(1-0.0602))/(235*0.0602+238*(1-0.0602)+16*2) #HM density for 6% enrichment
base_M = base_V*HM_rho_5
P_mod = HM_rho_6/HM_rho_5

mass=np.array([base_M,base_M,base_M*C_mod,base_M*P_mod])

bu=np.zeros((4,45))
power=[power,power,power,1.2*power]

time=np.array(t)*365.25 #time in days
for j in range(4):
    bu[j,:]=power[j]*time[:]/mass[j]/1e3 #1e3 = g to t and W to GW

colors = itertools.cycle(colorlist)
for j in range(4):
    plt.plot(bu[j],ks[j],color=next(colors),label=fis[j])

plt.plot([0,100],[1.03,1.03],color='black',linestyle='dashed',label='k-inf=1.03')
    
plt.xlabel('Burnup (GWd/t))')
plt.ylabel('k-inf')
plt.legend(loc='upper right')
plt.show()


#power 
colors = itertools.cycle(colorlist)
p0 = 1.26
p2 = 17/19*p0
P_max=dict()
for n,fi in enumerate(["264","324","324C",'324P']):
    P_max[fi]=[]
    for j in range(45):
        f=h5py.File("deplete_"+fi+"/openmc_simulation_n{}.h5".format(j))
        npins = 264 if fi=="264" else 324
        ff=np.sqrt(264/npins)
        pins = get_pins(p0,p2,ff,npins)
        
        P=[]
        total=0
        P_total=0
        for i in range(len(pins)):
            val = f['tallies']['tally {}'.format(i)]['results'][0][0][0]
            P_total += val
            if pins[i][0] == 0 or pins[i][1] == 0:
                val *= 2
                total += 0.5
            else:
                total += 1
            P.append(val)
        
        P=np.array(P)
        P /= P_total
        P *= total
        P_max[fi].append(np.max(P))
    plt.plot(t,P_max[fi],color=next(colors),label=fis[n])

plt.xlabel('time (yrs)')
plt.ylabel('Assembly power peaking')
plt.legend(loc='upper right')
plt.show()

ks=np.array(ks)
for j in range(4):
    print(np.interp(-1.03,-ks[j],t))

        
            
        
        