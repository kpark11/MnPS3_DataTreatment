# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 15:43:01 2022

@author: Student
"""

import numpy as np
import pandas as pd
import os
import fnmatch
import matplotlib.pyplot as plt


file = r'C:\Users\Student\OneDrive - University of Tennessee\Desktop\Research\MPS (M = Cr, Mn)\MnPS3\Amal\1_28_2016_Amal_variable_temp_MnPS3'
os.chdir(file)
cwd = os.getcwd()
print(cwd)
list = os.listdir(file)
print(list)


for k in range(len(list)):
    if fnmatch.fnmatch(list[k],'*.jpg*'):
        pass
    else:
        data = np.loadtxt(list[k])
        #print(data)
        wavelength = data[:,0]
        #print(wavelength)
        frequency = (1/wavelength[:])*1e7
        #print(frequency)
        energy = frequency/8065.548
        #print(energy)
        percent = (data[:,1]/100)/1.6
        absorption = -(1/0.005)*np.log(percent)
        alpha_E = frequency*absorption/8065.548
        Direct_Gap = np.square(alpha_E)
        Indirect_Gap = np.sqrt(alpha_E)
        
        data = np.column_stack((data,energy))
        data = np.column_stack((data,frequency))
        data = np.column_stack((data,percent))
        data = np.column_stack((data,absorption))
        data = np.column_stack((data,alpha_E))
        data = np.column_stack((data,Direct_Gap))
        data = np.column_stack((data,Indirect_Gap))
        
        df = pd.DataFrame(data, columns = ['Wavelength','Siganl','Energy','Frequency','Signal in percentage','Absorption','alpha_E','Direct Gap','Indirect Gap'])
        name = list[k]
        new_name = "python" + list[k]
        print(new_name)
        df.to_csv(new_name,index=False)
        
        
        """
        fig = plt.figure()
        fig.suptitle(PeakFitName)
        ax1 = fig.add_subplot(2,2,2)
        ax2 = fig.add_subplot(2,2,1)
    
        ax1.plot(CutEnergy,CutAbsorption,'r')
        ax1.set_xlabel('Energy (eV)')
        ax1.set_ylabel('Absorption')
        
        
        ax2.plot(energy,absorption,'b')
        ax2.set_xlabel('Energy(eV)')
        ax2.set_ylabel('Absorption')
        
        """
#plt.show()
        



