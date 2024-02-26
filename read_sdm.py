# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 06:51:51 2023

@author: b9801
"""

import os
import glob

keff={}

for case in ['264','324','324V','324VP','324P']:
    for subcase in ['','_sdm']:
        #find the output file in the directory
        matching_files=glob.glob(os.path.join(case+subcase,"mc_*.out"))
        matching_file=matching_files[0]
        
        #open it
        with open(matching_file,"r") as f:
            #extract k-eff
            for line in f:
                if "Combined k-effective" in line:
                    keff[case+subcase]=float(line.split()[3])
    
    k1=keff[case]
    k2=keff[case+'_sdm']
    CRW=(k2-k1)/k2/k1*1e5
    print(case+" {:.0f}".format(CRW))


        
            
        
        