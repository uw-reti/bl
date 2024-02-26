# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 09:04:44 2023

@author: lindley2
"""

from pwr1 import get_pins,get_w
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

def cplot(color_weights,r,p0,pins,P,subchan=None,sc_out=False,vbounds=None,colorbar=False):
    cmap = plt.cm.get_cmap('jet')
    
    gt_rad = 0.61
    gt_x,gt_y = get_w(p0)
    
    figure, axes = plt.subplots()
    axes.set_xlim(0,11)
    axes.set_ylim(0,11)
    axes.set_aspect(1)
    
       
    if not sc_out:
        for j,_ in enumerate(gt_x):
            axes.add_artist(plt.Circle((gt_x[j],gt_y[j]),gt_rad,color='grey'))
        
        for j,pin in enumerate(pins):
            axes.add_artist(plt.Circle((pin[0],pin[1]),r,color=cmap(color_weights[j])))
            if P is not None and not colorbar:
                if np.max(P)<5:
                    axes.annotate("{0:.2f}".format(P[j]),xy=(pin[0],pin[1]),fontsize=6)
                else: #pin numbering
                    axes.annotate("{0:.0f}".format(P[j]),xy=(pin[0],pin[1]),fontsize=10)
    
    if subchan is not None:
        if not sc_out:
            for i,sc in enumerate(subchan):
                px = np.mean(np.array([s[0] for s in sc]))
                py = np.mean(np.array([s[1] for s in sc]))
    
                axes.annotate("{0:.0f}".format(i),xy=(px,py),fontsize=8)
                axes.add_artist(plt.Polygon(sc,fill=None,edgecolor='red'))
        else:
            for i,sc in enumerate(subchan):
                axes.add_artist(plt.Polygon(sc,color=cmap(color_weights[i])))
    if sc_out:
        for j,pin in enumerate(pins):
            axes.add_artist(plt.Circle((pin[0],pin[1]),r,color='white'))
        for j,_ in enumerate(gt_x):
            axes.add_artist(plt.Circle((gt_x[j],gt_y[j]),gt_rad,color='white'))
            
    if vbounds is None:
        vbounds=[np.min(P),np.max(P)]
    
    if colorbar:
        plt.colorbar(plt.cm.ScalarMappable(norm=Normalize(vmin=vbounds[0],vmax=vbounds[1]), cmap=cmap), ax=axes)
        
    # Hide X and Y axes label marks
    axes.xaxis.set_tick_params(labelbottom=False)
    axes.yaxis.set_tick_params(labelleft=False)
    
    # Hide X and Y axes tick marks
    axes.set_xticks([])
    axes.set_yticks([])
    plt.show()

def calc_P(npins,pins,tail="",three=False):

    with open(("" if not three else "3percent/")+str(npins)+tail+"/tallies.out") as fi:
        data=fi.readlines()

    P=[]
    count=0 #pins in the quadrant (halves counted fully)
    total=0 #total pins per quadrant (halves counted as half)
    P_total=0 #keep track of the total P of the quadrant
    for line in data:
        if "Kappa" in line:
            P.append(float(line.split()[2]))
            P_total+=P[-1]
            
            #symmetry line pin
            if pins[count][0] == 0 or pins[count][1] == 0:
                P[-1] *= 2
                total += 0.5
            else:
                total += 1
            
            count += 1
    
    #Normalize
    P=np.array(P)
    P /= P_total
    P *= total
    return P

if __name__=="__main__":
    
    npins=292
    tail=""
    three=False
    
    if npins != 324 or three:
        tail=""
    f=np.sqrt(264/npins)
    
    p0 = 1.26
    p2 = 17/19*p0
    pins = get_pins(p0,p2,f,npins)
    P=calc_P(npins,pins,tail=tail,three=three)
    print(np.max(P),np.min(P))
    
    #using fixed heat map limits
    color_weights=(P-0.75)/(1.2-0.75)
    r = 0.475*f 
    print(r,0.418*f,0.4095*f)
    cplot(color_weights,r,p0,pins,P,colorbar=True)
    

