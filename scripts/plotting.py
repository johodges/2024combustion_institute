# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 14:02:49 2023

@author: jhodges
"""

import matplotlib.pyplot as plt
import numpy as np

def getJHcolors():
    colors = np.array([[0, 65, 101],
                       [229, 114, 0],
                       [136, 139, 141],
                       [170, 39, 44],
                       [119, 197, 213],
                       [161, 216, 132],
                       [255, 200, 69],
                       [101, 0, 65],
                       [0, 229, 114],
                       [141, 136, 139],
                       [44, 170, 39],
                       [213, 119, 197],
                       [132, 161, 216],
                       [69, 255, 200],
                       [65, 101, 0],
                       [114, 0, 229]], dtype=np.float32)
    colors = colors/255
    return colors

def getNewColors():
    colors2 = np.array([[216, 27, 96],
                        [30, 136, 229],
                        [0, 77, 64],
                        [255, 193, 7],
                        [216, 27, 216],
                        [27, 216, 27],
                        ]) / 255
    return colors2

def finishSimulationFigure(ymax, exp_tmax, savefigure, closefigure, namespace, fs):
    plt.xlabel("Time (min)", fontsize=fs)
    plt.ylabel(r'HRRPUA ($\mathrm{kW/m^{2}}$)', fontsize=fs)
    plt.ylim(0, np.ceil(1.1*ymax/100)*100)
    plt.xlim(0, np.ceil(exp_tmax/60))
    plt.grid()
    plt.tick_params(labelsize=fs)
    plt.legend(fontsize=fs) #, bbox_to_anchor=(1.05,0.6))
    plt.tight_layout()
    
    if savefigure: plt.savefig(namespace, dpi=300)
    if closefigure: plt.close()

def getPlotLimits(material):
    xlim = 3000
    ylim = 1000
    if material == 'FAA_PVC':
        xlim = 1000
        ylim = 300
    if material == 'FAA_PC':
        ylim = 1000
        xlim = 1000
    if material == 'FAA_PMMA':
        ylim = 1500
        xlim = 2500
    if material == 'FAA_HIPS':
        ylim = 1500
        xlim = 3000
    if material == 'FAA_HDPE':
        ylim = 2500
        xlim = 3000
    return xlim, ylim