# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 08:15:39 2023

@author: jhodges
"""

import matplotlib.pyplot as plt
import numpy as np

from algorithms import getMaterials, processCaseData
from algorithms import developRepresentativeCurve
from algorithms import runSimulation, getTimeAveragedPeak
from algorithms import getTimeAveragedEnergy, getTimeAveraged_timeToEnergy
from algorithms import spyroScaling

from plotting import finishSimulationFigure, getPlotLimits, getJHcolors

def sortCases(cases):
    cases_to_plot = np.array(list(cases.keys()))
    thicknesses = np.array([cases[c]['delta'] for c in cases_to_plot])
    coneExposures = np.array([cases[c]['cone'] for c in cases_to_plot])
    inds = np.argsort(coneExposures)
    thicknesses = thicknesses[inds]
    coneExposures = coneExposures[inds]
    cases_to_plot = cases_to_plot[inds]
    
    inds = np.argsort(thicknesses)
    thicknesses = thicknesses[inds]
    coneExposures = coneExposures[inds]
    cases_to_plot = cases_to_plot[inds]
    return coneExposures, thicknesses, cases_to_plot

if __name__ == "__main__":
    
    spec_file_dict = getMaterials()
    
    materials = ['FSRI_Vinyl_Plank_Flooring']
    materials = ['FAA_PC','FAA_PVC', 'FAA_PMMA', 'FAA_HIPS', 'FAA_HDPE']
    materials =['FSRI_Mineral_Wool_Insulation']
    
    # Output parameters
    (savefigure, closefigure) = (False, False)
    (style, nondimtype) = ('md_mf', 'FoBi')
    
    # Initialize parameters
    (fs, lw, s, exp_num_points) = (48, 9, 100, 25)
    (windowSize, percentile) = (60, 90)
    lineStyles = ['--','-.',':']
    colors = getJHcolors()
    for material in materials:
        spec_file_dict[material] = processCaseData(spec_file_dict[material])
        mat = spec_file_dict[material]
        (density, conductivity, specific_heat) = (mat['density'], mat['conductivity'], mat['specific_heat'])
        (HoC, emissivity, nu_char) = (mat['heat_of_combustion'], mat['emissivity'], mat['nu_char'])
        (cases, case_basis, data) = (mat['cases'], mat['case_basis'], mat['data'])
        
        total_energy_per_deltas = [case_basis[c]['totalEnergy']/case_basis[c]['delta'] for c in case_basis]
        total_energy_per_delta_ref = np.mean(total_energy_per_deltas)
        
        nondim_t, ref_hog, ref_qrs, ref_mlrs, ref_times = developRepresentativeCurve(mat, nondimtype)
        xlim, ylim = getPlotLimits(material)
        times = np.linspace(0, xlim*2, 10001)
        
        coneExposures, thicknesses, cases_to_plot = sortCases(cases)
        (delta_old, exp_tmax, ymax, j, fig) = (-1, 0, 0, 0, False)
        
        for i, c in enumerate(cases_to_plot):
            (delta0, coneExposure, tign) = (cases[c]['delta'], cases[c]['cone'], cases[c]['tign'])
            totalEnergy = total_energy_per_delta_ref*delta0
            times, hrrpuas, totalEnergy2 = runSimulation(times, mat, delta0, coneExposure, totalEnergy, nondim_t, ref_hog, nondimtype=nondimtype)
                                           #runSimulation(times, mat, delta0, coneExposure, totalEnergy, fobi_out, hog_out, nondimtype='FoBi'):
            mod_peak = getTimeAveragedPeak(times, hrrpuas, windowSize)
            exp_peak = getTimeAveragedPeak(cases[c]['times'],cases[c]['HRRs'], windowSize)
            
            energyThreshold, exp_t90 = getTimeAveragedEnergy(cases[c]['times']-cases[c]['tign'],cases[c]['HRRs'], windowSize, percentile)
            mod_t90, timeAverage = getTimeAveraged_timeToEnergy(times, hrrpuas, windowSize, energyThreshold)
            
            print("%s & %0.1f & %0.0f & %0.1f & %0.1f \\\\ %0.1f & %0.1f"%(material, delta0*1000, coneExposure, exp_peak, mod_peak, exp_t90, mod_t90))
            
            label = r'%0.0f Exp'%(coneExposure) #'$\mathrm{kW/m^{2}}$'%(coneExposure)
            
            if delta0 != delta_old:
                namespace = '..//figures//simulation_' + style + '_' + material + '_%dmm.png'%(delta_old*1e3)
                if fig is not False: finishSimulationFigure(ymax, exp_tmax*1.3, savefigure, closefigure, namespace, fs)
                fig = plt.figure(figsize=(24,18))
                (exp_tmax, ymax, j, delta_old) = (0, 0, 0, delta0)
            
            exp_int = 5
            plt.scatter(cases[c]['times'][::exp_int]/60,cases[c]['HRRs'][::exp_int], s=s, label=label, linewidth=lw, color=colors[j])
            #plt.plot(cases[c]['times']/60,cases[c]['HRRs'], lineStyles[0], linewidth=lw, color=colors[j])
            plt.plot((times+tign)/60, hrrpuas, lineStyles[1], linewidth=lw, label='FoBi*', color=colors[j])
            
            
            
            t1 = mat['case_basis']['case-1000']['times_trimmed'] 
            t1 = t1-t1[0]
            hrr1 = mat['case_basis']['case-1000']['hrrs_trimmed']
            delta1 = mat['case_basis']['case-1000']['delta']
            ref_cone = mat['case_basis']['case-1000']['cone']
            scaled_t, scaled_hrrpua = spyroScaling(t1, hrr1, delta1, ref_cone, delta0, coneExposure, tign=tign, qflame=25, referenceTimes=times, averageWindow=False) #qflame='Empirical')
            plt.plot(scaled_t/60, scaled_hrrpua, lineStyles[2], linewidth=lw, label='simple', color=colors[j])
            
            j = j+1
            exp_tmax = max([exp_tmax, cases[c]['times'].max()])
            ymax = max([ymax, np.nanmax(cases[c]['HRRs']), np.nanmax(hrrpuas)])
        
        namespace = '..//figures//simulation_' + style + '_' + material + '_%dmm.png'%(delta_old*1e3)
        finishSimulationFigure(ymax, exp_tmax*1.3, savefigure, closefigure, namespace, fs)
