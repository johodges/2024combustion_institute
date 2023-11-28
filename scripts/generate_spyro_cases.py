# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 18:18:31 2023

@author: jhodges
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, shutil, subprocess

from plotting import getJHcolors, getPlotLimits, finishSimulationFigure
from algorithms import getMaterials, processCaseData, sortCases
from algorithms import getFixedModelParams, buildFdsFile, runModel, load_csv, findFds
from algorithms import calculateUncertainty, plotMaterialExtraction, calculateUncertaintyBounds
from algorithms import getTimeAveragedPeak
from algorithms import getTimeAveragedEnergy, getTimeAveraged_timeToEnergy
from algorithms import getRepresentativeHrrpua, estimateExposureFlux, estimateHrrpua

if __name__ == "__main__":
    
    fdsdir, fdscmd = findFds()
    fileDir = os.path.dirname(os.path.abspath(__file__))
    
    spec_file_dict = getMaterials()
    materials = list(spec_file_dict.keys())
    
    materials = ['FAA_PMMA', 'FAA_HDPE', 'FAA_HIPS', 'FAA_PC', 'FAA_PVC']
    materials = ['FSRI_Polyester_Microfiber_Sheet']
    nondimtype = 'FoBi_simple_fixed_d'
    figoutdir = "figures"
    runSimulations = False
    plotResults = True
    lineStyles = ['--','-.',':']
    
    if figoutdir is not None:
        if os.path.exists(figoutdir) is not True: os.mkdir(figoutdir)
        import matplotlib.pyplot as plt
    
    output_statistics = dict()
    params = getFixedModelParams()
    colors = getJHcolors()
    
    # Initialize stats outputs
    exp_points = []
    mod_points = []
    ms = []
    
    for material in materials:
        output_statistics[material] = dict()
        xlim, ylim = getPlotLimits(material)
        spec_file_dict[material] = processCaseData(spec_file_dict[material])
        
        mat = spec_file_dict[material]
        (density, conductivity, specific_heat) = (mat['density'], mat['conductivity'], mat['specific_heat'])
        (HoC, emissivity, nu_char) = (mat['heat_of_combustion'], mat['emissivity'], mat['nu_char'])
        
        (cases, case_basis, data) = (mat['cases'], mat['case_basis'], mat['data'])
        matClass = mat['materialClass']
        
        totalEnergyMax = np.nanmax([case_basis[c]['totalEnergy'] for c in case_basis])
        
        if totalEnergyMax < 100:
            print("Total energy for %s is %0.1f < 100, skipping"%(material, totalEnergyMax))
            continue
        
        #fobi_out, fobi_hog_out, qr_out, fobi_mlr_out, _ = developRepresentativeCurve(mat, nondimtype=nondimtype)
        
        cone_hf_ref = [case_basis[c]['cone'] for c in case_basis][0]
        cone_d_ref = [case_basis[c]['delta'] for c in case_basis][0]
        
        times_trimmed = [case_basis[c]['times_trimmed'] for c in case_basis][0]
        hrrs_trimmed = [case_basis[c]['hrrs_trimmed'] for c in case_basis][0]
        
        chid = material
        basis_summary = [[case_basis[c]['delta'], case_basis[c]['cone']] for c in case_basis]
        tend = np.nanmax([(cases[c]['times_trimmed'].max()+cases[c]['tign'])*2 for c in cases])
        
        fluxes, deltas, tigns, cases_to_plot = sortCases(cases)
        
        workingDir = fileDir + os.sep +'..' + os.sep + 'input_files' + os.sep+ material + os.sep
        
        if os.path.exists(workingDir) is False: os.mkdir(workingDir)
        # Calculate times to ignition
        
        txt = buildFdsFile(chid, cone_hf_ref, cone_d_ref, emissivity, conductivity, density, 
                               specific_heat, 20, times_trimmed, hrrs_trimmed, tend,
                               deltas, fluxes, 15.0, ignitionMode='Time', case_tigns=tigns,
                               calculateDevcDt=False)
        
        with open("%s%s%s.fds"%(workingDir, os.sep, chid), 'w') as f:
            f.write(txt)
        
        if runSimulations:
            runModel(workingDir, chid+".fds", 1, fdsdir, fdscmd, printLiveOutput=False)
        
        if plotResults:
            (savefigure, closefigure) = (True, False)
            data = load_csv(workingDir, chid)
            # Plot results
            if figoutdir is not None:
                
                (fs, lw, s, exp_num_points) = (48, 9, 100, 25)
                (windowSize, percentile) = (60, 90)
                (delta_old, exp_tmax, ymax, j, fig) = (-1, 0, 0, 0, False)
                for i in range(0, len(cases_to_plot)):
                    c = cases_to_plot[i]
                    namespace = '%02d-%03d'%(fluxes[i], deltas[i]*1e3)
                    #label = r'%s'%(namespace) #'$\mathrm{kW/m^{2}}$'%(coneExposure)
                    label = r'%0.0f $\mathrm{kW/m^{2}}$'%(fluxes[i])
                    delta0 = cases[c]['delta']
                    if (material == 'FAA_PC' or material == 'FAA_PVC') and delta0*1e3 < 4:
                        exp_int = 5
                    else:
                        exp_int = int(np.ceil(cases[c]['times'].shape[0]/exp_num_points))
                    
                    if delta0 != delta_old:
                        fig_namespace = '..//figures//fdsout_' + material + '_%dmm.png'%(delta_old*1e3)
                        if fig is not False: finishSimulationFigure(ymax, exp_tmax*1.3, savefigure, closefigure, fig_namespace, fs)
                        fig = plt.figure(figsize=(14,12))
                        (exp_tmax, ymax, j, delta_old) = (0, 0, 0, delta0)
                    
                    plt.scatter(cases[c]['times'][::exp_int]/60,cases[c]['HRRs'][::exp_int], s=s, label=label, linewidth=lw, color=colors[j])
                    times = data['Time']
                    hrrpuas = data['"HRRPUA-'+namespace+'"']
                    plt.plot(times/60, hrrpuas, lineStyles[1], linewidth=lw, color=colors[j])
                    
                    mod_peak = getTimeAveragedPeak(times.values, hrrpuas.values, 60)
                    exp_peak = getTimeAveragedPeak(cases[c]['times'],cases[c]['HRRs'], 60) #, referenceTimes=times)
                    
                    j = j+1
                    exp_tmax = max([exp_tmax, cases[c]['times'].max()])
                    ymax = max([ymax, np.nanmax(cases[c]['HRRs']), np.nanmax(hrrpuas)])
                    
                    exp_points.append(exp_peak)
                    mod_points.append(mod_peak)
                    ms.append(material)
                    
                    for percentile in [90]:
                        energyThreshold, exp_t = getTimeAveragedEnergy(cases[c]['times']-cases[c]['tign'],cases[c]['HRRs'], windowSize, percentile)
                        mod_t, timeAverage = getTimeAveraged_timeToEnergy(times.values-cases[c]['tign'], hrrpuas.values, windowSize, energyThreshold)
                        
                        print(material, namespace, exp_t, mod_t)
                    
                fig_namespace = '..//figures//fdsout_' + material + '_%dmm.png'%(delta_old*1e3)
                finishSimulationFigure(ymax, exp_tmax*1.3, savefigure, closefigure, fig_namespace, fs)
            
            '''
            plt.xlabel("Time (min)", fontsize=fs)
            plt.ylabel(r'HRRPUA ($\mathrm{kW/m^{2}}$)', fontsize=fs)
            plt.ylim(0, np.ceil(1.1*ymax/100)*100)
            plt.xlim(0, np.ceil(exp_tmax/60))
            plt.grid()
            plt.tick_params(labelsize=fs)
            plt.legend(fontsize=fs, bbox_to_anchor=(1.05,0.6))
            plt.tight_layout(rect=(0, 0, 1, 0.95))
            
            plt.savefig("..//figures//fdsout_%s.png"%(material), dpi=300)
            '''
            
            #if savefigure: plt.savefig(namespace, dpi=300)
            #if closefigure: plt.close()
            
