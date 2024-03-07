# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 12:52:10 2023

@author: lindley2
"""

import numpy as np
import os

def get_w(p0):
    #returns positions of control rods in terms of pitch p0
    #   -1  -2  -3  -4    -5   -6   -7   -8  -9
    wx=[0,   0,3*p0,5*p0,   0,3*p0,3*p0,6*p0,6*p0]
    wy=[0,3*p0,3*p0,5*p0,6*p0,6*p0,   0,   0,3*p0]
    return wx,wy

def get_pins(p0,p2,f,npins):
    """returns coordinates of pins given original 17x17 pitch p0, modified pitch p2
    f is the relative radius of new pin to reference pin
    npins is number of pins - a few preset options"""
    
    if npins not in [264,280,292,324]:
        raise ValueError
    
    pins=[]
    
    if npins == 264:
        for x in range(9):
            for y in range(9):
                flag=False
                for X,Y in [(0,0),(0,3),(3,3),(5,5),(0,6),(3,6),(3,0),(6,0),(6,3)]:
                    if x==X and y==Y:
                        flag=True
                if not flag:
                    pins.append((x*p0,y*p0,f))
        return pins
    
    
    
    if npins == 280 or npins == 292:
        
        for x in range(4,7):
            pins.append((x*p0,6*p0,f))
        
        for x in range(5):
            pins.append((x*p0,5*p0,f))
        
        pins.append((6*p0,5*p0,f))
        
        for y in [8,9]:
            pins.append((8*p2,y*p2,f))
            pins.append((9*p2,y*p2,f))
        
        for y in range(8): 
             pins.append((8*p2,y*p2,f))
             pins.append((9*p2,y*p2,f))
        for x in range(8):
             pins.append((x*p2,8*p2,f))
             pins.append((x*p2,9*p2,f))
             
        if npins==292:
                
           for y in [1,2,4]: 
                for x in range(8):
                    pins.append((x*p2,y*p0,f))
                       
        else:
           for y in [1,2,4]:  
                for x in range(7):
                    pins.append((x*p0,y*p0,f))
        
        for x in [1,2,4,5]:
            pins.append((x*p0,0,f))
            pins.append((x*p0,3*p0,f))
        
        for x in [1,2]:
            pins.append((x*p0,6*p0,f))
            
    elif npins==324:
        for y in [8,9]:
            pins.append((8*p2,y*p2,f))
            pins.append((9*p2,y*p2,f))
        for y in range(8): 
            pins.append((8*p2,y*p2,f))
            pins.append((9*p2,y*p2,f))
        for x in range(8):
            pins.append((x*p2,8*p2,f))
            pins.append((x*p2,9*p2,f))
            
        pins.append((4.5*p2,7.2*p2,f))
        pins.append((7.2*p2,4.5*p2,f))
        
        pins.append((3.68*p0,6.05*p2,f))
        pins.append((6.05*p2,3.68*p0,f))
        
        #pins.append((4.3*p0,6.3*p2,f))
        #pins.append((6.3*p2,4.3*p0,f))
        
        for x in [1,2]:
            for y in [3,4,5,6,7]:
                pins.append((x*p2,y*p2,f))
                pins.append((y*p2,x*p2,f))
        
        for x in [1,2,4,5]:
            pins.append((x*p0,0,f))
            pins.append((0,x*p0,f))
        
        
        pins.append((7*p2,7*p2,f))
        
        
        pins.append((5.5*p2,7.2*p2,f))
        pins.append((7.2*p2,5.5*p2,f))
        
        pins.append((6.32*p2,6.32*p2,f))
        
        pins.append((5*p2,6.4*p2,f))
        pins.append((6.4*p2,5*p2,f))
        
        for x in [1,2]:
            for y in [1,2]:
                pins.append((x*p2,y*p2,f))
        
        for x in [4.5,5.5]:
            pins.append((x*p2,2.8*p2,f))
            pins.append((2.8*p2,x*p2,f))
        
        pins.append((5*p2,3.6*p2,f))
        pins.append((3.6*p2,5*p2,f))
        
        #pins.append((4.85*p2,4.85*p2,f))
        pins.append((4.1*p2,4.1*p2,f))
        
        pins.append((5.2*p2,4.53*p2,f))
        pins.append((4.53*p2,5.2*p2,f))
 
    
    return pins

if __name__ == "__main__":
    
    import openmc
    
    deplete = False #produce a depletion input 
    keep_clad = False #if set to true, clad thickness is maintained at reference rather than scaled with pin diameter
    sdm = False #produce a calculation with control rods in
    
        
    if deplete:
        cases=[324] #264, 324 - pick one
        leuplus=True #If true, enrichment is increased.
    
    else:
        #sweep through all cases. Four 324 cases: regular, variable enrichment, leuplus, variable enrichment & leuplus
        cases=[264,280,292,324,324,324,324] 
    
    for xxx,npins in enumerate(cases):

    
        p0 = 1.26
        p2 = p0*17/19
        asm_pitch = 21.5
        
        f=np.sqrt(264/npins) #preserve total pin area
            
        var_enr = True if (xxx==len(cases)-1 or xxx==len(cases)-3) and not deplete else False
        leuplus = True if  xxx>=len(cases)-2 else False
        
        pins = get_pins(p0,p2,f,npins)
        
        #derived manually by printing out powers - these are the pins with reduced enrichment in 324 sensitivty case
        drop_pins = [38,39,50,51,58,59,60,61,62,63,84]
        
        
        if keep_clad:
            clad_inner=0.475*f-0.057
            fuel_outer=clad_inner-(0.418-0.4095)*f
        else:
            clad_inner=0.418*f
            fuel_outer=0.4095*f
        
        if deplete:
            fmat = len(pins) #different fuel material for each pin
        else:
            fmat = 1
        
        uo2=[]
        for j in range(fmat):
            uo2.append(openmc.Material(name="uo2"))
            uo2[-1].add_nuclide('U235',0.0501 if not leuplus else 0.0602) #4.95wt%, 5.95 wt%
            uo2[-1].add_nuclide('U238',0.9499 if not leuplus else 0.9398)
            uo2[-1].add_nuclide('O16', 2.0)
            uo2[-1].set_density('g/cm3', 10.4)
            
            if deplete:
                uo2[-1].volume=np.pi*fuel_outer**2
                if pins[j][0] == 0 or pins[j][1] == 0:
                    uo2[-1].volume /= 2 #line of symmetry

            
        
        if var_enr:
            if deplete:
                raise ValueError("Var_enr and depletion not set up")
            uo2x=openmc.Material(name="uo2_drop")
            uo2x.add_nuclide('U235', 0.0481 if not leuplus else 0.0582) #4.75wt% or 5.75wt%
            uo2x.add_nuclide('U238', 0.9519 if not leuplus else 0.9418)
            uo2x.add_nuclide('O16', 2.0)
            uo2x.set_density('g/cm3', 10.4)
        
        zircaloy = openmc.Material(name="zircaloy")
        zircaloy.add_element('Zr', 0.985)
        zircaloy.add_element('Sn',0.015)
        zircaloy.set_density('g/cm3', 6.6)
        
        if sdm:
            aic=openmc.Material(name="aic")
            aic.add_element('Ag',0.8)
            aic.add_element('In',0.15)
            aic.add_element('Cd',0.05)
            aic.set_density('g/cm3',10.17)
            
            #http://www.metalspiping.com/ss316-ss316l.html#:~:text=Stainless%20steel%20316%20(SS316)%20is,resistance%20to%20chloride%20ion%20solutions.
            ss=openmc.Material(name="ss")
            ss.add_element('C',0.04)
            ss.add_element('Mn',1.0)
            ss.add_element('P',0.02)
            ss.add_element('S',0.01)
            ss.add_element('Cr',17.0)
            ss.add_element('Ni',12.0)
            ss.add_element('Mo',2.5)
            ss.add_element('Fe',100-0.04-1.0-0.02-0.01-17.0-12.0-2.5)
            ss.set_density('g/cm3',7.99)
        
        
        water = openmc.Material(name="h2o")
        water.add_nuclide('H1', 2.0)
        water.add_nuclide('O16', 1.0)
        water.set_density('g/cm3', 0.7)
        water.add_s_alpha_beta('c_H_in_H2O')
        
        moderator = openmc.Material(name="h2o_gt")
        moderator.add_nuclide('H1', 2.0)
        moderator.add_nuclide('O16', 1.0)
        moderator.set_density('g/cm3', 0.74)
        moderator.add_s_alpha_beta('c_H_in_H2O')
        
        mat=uo2+[zircaloy, water,moderator]
        if var_enr:
            mat.append(uo2x)
        if sdm:
            mat+=[aic,ss]
            
        materials = openmc.Materials(mat)
        materials.export_to_xml()
        
        fuel_outer_radius=[]
        clad_inner_radius=[]
        clad_outer_radius=[]
        fuel_region=[]
        gap_region=[]
        clad_region=[]
        fuel=[]
        gap=[]
        clad=[]
        
        
        box = openmc.model.rectangular_prism(origin=(asm_pitch/4,asm_pitch/4),width=asm_pitch/2, height=asm_pitch/2,
                                       boundary_type='reflective')


            
        count=0    
        for x,y,f in pins:
            fuel_outer_radius.append(openmc.ZCylinder(r=fuel_outer,x0=x,y0=y))
            clad_inner_radius.append(openmc.ZCylinder(r=clad_inner,x0=x,y0=y))
            clad_outer_radius.append(openmc.ZCylinder(r=0.475*f ,x0=x,y0=y))
            
            fuel_region.append(-fuel_outer_radius[-1] & box)
            gap_region.append(+fuel_outer_radius[-1] & -clad_inner_radius[-1] & box)
            clad_region.append(+clad_inner_radius[-1] & -clad_outer_radius[-1] & box)
            
            fuel.append(openmc.Cell(name='fuel'))
            fuel[-1].temperature=900
            if var_enr and count in drop_pins:
                fuel[-1].fill = uo2x
            else:
                if deplete:
                    fuel[-1].fill = uo2[count]
                else:
                    fuel[-1].fill = uo2[0]
            fuel[-1].region = fuel_region[-1]
            count+=1
            
            gap.append(openmc.Cell(name='air gap'))
            gap[-1].temperature=600
            gap[-1].region = gap_region[-1]
            
            clad.append(openmc.Cell(name='clad'))
            clad[-1].temperature=600
            clad[-1].fill = zircaloy
            clad[-1].region = clad_region[-1]
        
        if sdm:
            aic_radius=[]
            aic_gap=[]
            ss_radius=[]
            aic_region=[]
            aic_gap_region=[]
            ss_region=[]
            aic_cell = []
            aic_gap_cell = []
            ss_cell = []
        
        wrod_inner_radius=[]
        wrod_outer_radius=[]
        moderator_region=[]
        wrod_region=[]
        wrod_inner=[]
        wrod=[]
        
        wx,wy = get_w(p0)
        
        #water rod locations
        wrods=zip(wx,wy)
        
        for count,xy in enumerate(wrods):
            x,y=xy
            if sdm and count>0: #central guide tube not rodded
                aic_radius.append(openmc.ZCylinder(r=0.42672,x0=x,y0=y))
                aic_gap.append(openmc.ZCylinder(r=0.43688,x0=x,y0=y))
                ss_radius.append(openmc.ZCylinder(r=0.48387,x0=x,y0=y))
            wrod_inner_radius.append(openmc.ZCylinder(r=0.56,x0=x,y0=y))
            wrod_outer_radius.append(openmc.ZCylinder(r=0.61,x0=x,y0=y))
            
            if sdm and count>0:
                aic_region.append(-aic_radius[-1] & box)
                aic_gap_region.append(+aic_radius[-1] & -aic_gap[-1] & box)
                ss_region.append(+aic_gap[-1] & -ss_radius[-1] & box)
                moderator_region.append(+ss_radius[-1] & -wrod_inner_radius[-1] & box)
            else:
                moderator_region.append(-wrod_inner_radius[-1] & box)
            
            wrod_region.append(+wrod_inner_radius[-1] & -wrod_outer_radius[-1] & box)
            
            if sdm and count>0:
                aic_cell.append(openmc.Cell(name='aic_region'))
                aic_cell[-1].temperature=586
                aic_cell[-1].fill=aic
                aic_cell[-1].region = aic_region[-1]
                
                aic_gap_cell.append(openmc.Cell(name='aic_gap'))
                aic_gap_cell[-1].temperature=586
                aic_gap_cell[-1].region=aic_gap_region[-1]
                
                ss_cell.append(openmc.Cell(name='ss_region'))
                ss_cell[-1].temperature=586
                ss_cell[-1].fill=ss
                ss_cell[-1].region = ss_region[-1]
            
            wrod_inner.append(openmc.Cell(name="wrod_inner"))
            wrod_inner[-1].temperature=586
            wrod_inner[-1].fill=moderator
            wrod_inner[-1].region=moderator_region[-1]
            
            wrod.append(openmc.Cell(name="wrod"))
            wrod[-1].temperature=586
            wrod[-1].fill=zircaloy
            wrod[-1].region=wrod_region[-1]
        
        water_region = box
        #Create outer region
        for j in range(len(wx)):
            water_region = water_region & +wrod_outer_radius[j]
        for j in range(len(pins)):
            water_region = water_region & +clad_outer_radius[j]
        
        coolant = openmc.Cell(name='coolant')
        coolant.temperature=586
        coolant.fill = water
        coolant.region = water_region
        
        #Universe
        root_universe = openmc.Universe(cells=fuel+gap+clad+wrod_inner+wrod+[coolant]+([] if not sdm else aic_cell + aic_gap_cell + ss_cell))
        
        geometry = openmc.Geometry(root_universe)
        geometry.export_to_xml()
        
        #Source
        source=openmc.Source()
        source.space = openmc.stats.Box((0, 0, 0), (asm_pitch/2, asm_pitch/2, 1))
        
        settings = openmc.Settings()
        settings.source = source
        settings.batches = 100
        settings.inactive = 10
        settings.particles = 500000
        settings.temperature = {'method': 'interpolation'}
        settings.export_to_xml()
        
        #Power Tallies
        tallies=list()
        cell_filters=list()
        for j in range(len(fuel)):
            cell_filters.append(openmc.CellFilter(fuel[j]))
            tallies.append(openmc.Tally(j))
            tallies[-1].filters = [cell_filters[-1]]
            tallies[-1].scores = ['kappa-fission']
        tallies = openmc.Tallies(tallies)
        tallies.export_to_xml()
        
        if deplete: #directly go ahead and perform the calculation
            import openmc.deplete
            model=openmc.Model(geometry=geometry,settings=settings)
            operator=openmc.deplete.CoupledOperator(model,"./chain_endfb71_pwr.xml")
            power=3411e6/193/4/366.0
            if leuplus:
                power*=1.2
            time_steps=[1,4,25,30]+[60]*40
            integrator = openmc.deplete.PredictorIntegrator(operator, time_steps, power, timestep_units='d')
            integrator.integrate()
            
        else:
            #Plot geometry
            plot = openmc.Plot()
            plot.filename = str(npins)
            plot.origin=(asm_pitch/4,asm_pitch/4,0)
            plot.width = (asm_pitch/2, asm_pitch/2)
            plot.pixels = (1000, 1000)
            plot.color_by = 'material'
            plots = openmc.Plots([plot])
            plots.export_to_xml()
            openmc.plot_geometry()
            #openmc.run() - could uncomment this line to run the case.
        
        mystr=str(npins)
        if var_enr:
            mystr+="V"
        
        if leuplus:
            mystr+="P"
        
        if sdm:
            mystr+="_sdm"
            
        os.system("mkdir {0}".format(mystr))
        os.system("cp *.xml {0}/".format(mystr))
        os.system("mv {0}.png {1}/".format(npins,mystr))
    

