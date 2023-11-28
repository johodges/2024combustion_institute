# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 18:18:31 2023

@author: jhodges
"""
import matplotlib.pyplot as plt
import numpy as np
import os

from plotting import finishSimulationFigure, getPlotLimits, getJHcolors
from algorithms import getMaterials, processCaseData, sortCases
from algorithms import developRepresentativeCurve
from algorithms import runSimulation, getTimeAveragedPeak, buildFdsFile, findFds, runModel, load_csv
from algorithms import plotMaterialExtraction

if __name__ == "__main__":
    
    nondimtype = 'FoBi' # FoBi or FoBi_simple
    
    # Compare normalization schemes on model predictions
    style = 'md_mf'
    material = 'FAA_HDPE'
    
    spec_file_dict = getMaterials()
    spec_file_dict[material] = processCaseData(spec_file_dict[material])
    mat = spec_file_dict[material]
    (density, conductivity, specific_heat) = (mat['density'], mat['conductivity'], mat['specific_heat'])
    (HoC, emissivity, nu_char) = (mat['heat_of_combustion'], mat['emissivity'], mat['nu_char'])
    (cases, case_basis, data) = (mat['cases'], mat['case_basis'], mat['data'])
    
    total_energy_per_deltas = [case_basis[c]['totalEnergy']/case_basis[c]['delta'] for c in case_basis]
    total_energy_per_delta_ref = np.mean(total_energy_per_deltas)
    
    fobi_out, fobi_hog_out, qr_out, fobi_mlr_out, _ = developRepresentativeCurve(mat, nondimtype)
    fo_out, fo_hog_out, qr_out, fo_mlr_out, _ = developRepresentativeCurve(mat, 'Fo')
    xlim, ylim = getPlotLimits(material)
    times_out = np.linspace(0, 3000, 6001) #xlim*2, 10001)
    
    fs=24
    lw = 3
    colors = getJHcolors()
    
    #labels = ['3mm','8mm','27mm']
    labels = ['3','8','27']
    
    reference_time, reference_hrrpua, cone_hf_ref = [case_basis[c]['times_trimmed'] for c in case_basis][0], [case_basis[c]['hrrs_trimmed'] for c in case_basis][0], [case_basis[c]['cone'] for c in case_basis][0]
    plt.figure(figsize=(10,8))
    for i, c in enumerate(['case-003','case-004','case-005']): #enumerate(list(cases.keys())):
        delta0 = cases[c]['delta']
        coneExposure = cases[c]['cone']
        totalEnergy = total_energy_per_delta_ref*delta0
        times, hrrpuas, totalEnergy2 = runSimulation(times_out, mat, delta0, coneExposure, totalEnergy, fobi_out, fobi_hog_out, reference_hrrpua, reference_time, cone_hf_ref,nondimtype=nondimtype)
        fo_times, fo_hrrpuas, fo_totalEnergy2 = runSimulation(times_out, mat, delta0, coneExposure, totalEnergy, fo_out, fo_hog_out, reference_hrrpua, reference_time, cone_hf_ref, nondimtype='Fo')
        
        
        plt.scatter(cases[c]['times'][::5]/60,cases[c]['HRRs'][::5], label=labels[i]+': Exp', s=50, linewidth=lw, color=colors[i])
        plt.semilogx((fo_times+cases[c]['tign'])/60, fo_hrrpuas, '--', linewidth=lw, label=labels[i]+': Fo', color=colors[i])
        plt.semilogx((times+cases[c]['tign'])/60, hrrpuas, '-', linewidth=lw, label=labels[i]+': FoBi*', color=colors[i])
        
    
    plt.xlabel("Time (min)", fontsize=fs)
    plt.ylabel(r"$\dot{Q}''$ ($\mathrm{kW/m^{2}}$)", fontsize=fs)
    plt.ylim(0, 1800)
    plt.xlim(1, 60)
    plt.xticks([1, 2, 4, 6, 10, 20, 60], ['1','2','4','6','10','20','60'])
    plt.grid()
    plt.tick_params(labelsize=fs)
    plt.legend(fontsize=fs, bbox_to_anchor=(1.05,0.8))
    plt.tight_layout()
    plt.savefig('..//figures//time_normalization_' + style + '_' + material + '.png', dpi=300)
    
    
    
    # Compare normalization schemes on Hg
    fs=24
    lw = 6
    leg_loc = 3
    style = 'md_mf'
    material = 'FAA_HDPE'
    spec_file_dict[material] = processCaseData(spec_file_dict[material])
    mat = spec_file_dict[material]
    
    plt.figure(figsize=(10, 8))
    mat2 = dict(mat)
    mat2['case_basis'] = dict(mat2['cases'])
    fobi_out, hog_out, qr_out, mlr_out, times_out = developRepresentativeCurve(mat2, nondimtype, plot=True, lw=2, labelPlot=False)
    
    plt.loglog(fobi_out, hog_out, '--', linewidth=lw, color='k', label='Median')
    
    
    mat2['case_basis'] = dict(mat2['cases'])
    for key in list(mat2['case_basis'].keys()):
        if key != 'case-000': mat2['case_basis'].pop(key)
    fobi_out, hog_out, qr_out, mlr_out, _ = developRepresentativeCurve(mat2, nondimtype, plot=True, lw=4, labelPlot=True)
    
    
    fobi_max = fobi_out[np.where(mlr_out < 0)[0][-1]]*1.2
    
    if nondimtype == 'FoBi':
        plt.xlim(1, 2e5) #fobi_max)
    elif nondimtype == 'FoBi_simple':
        plt.xlim(0.01,100)
    plt.ylim(1000, 100000)
    plt.xlabel(r"$\mathrm{FoBi^{*}}$ ($-$)", fontsize=fs)
    plt.ylabel(r'$\Delta H_{g}$ ($\mathrm{kJ/kg}$)', fontsize =fs)
    lgd = plt.legend(fontsize=fs, loc=leg_loc)
    
    plt.grid()
    plt.tick_params(labelsize=fs)
    
    plt.tight_layout()
    plt.savefig('..//figures//DHg_%s_%s_collapsed.png'%(style, material), dpi=300)
    
    
    fs2 = 24
    lw2 = 6
    plt.figure(figsize=(10, 8))
    mat2 = dict(mat)
    mat2['case_basis'] = dict(mat2['cases'])
    fobi_out, hog_out, qr_out, mlr_out, times_out = developRepresentativeCurve(mat2, 'Time', plot=True, lw=2)
    plt.loglog(fobi_out/60, hog_out, '--', linewidth=lw2, label='Median', color='k')
    
    plt.xlim(1, 30)
    plt.xticks([0.5, 1, 2, 4, 6, 10, 20, 30], ['0.5','1','2','4','6','10','20','30'])
    plt.ylim(1000, 100000)
    plt.xlabel("Time (min)", fontsize=fs2)
    
    plt.ylabel(r'$\Delta H_{g}$ ($\mathrm{kJ/kg}$)', fontsize =fs2)
    lgd = plt.legend(fontsize=fs2, loc=leg_loc)
    plt.grid()
    plt.tick_params(labelsize=fs2)
    plt.tight_layout()
    plt.savefig('..//figures//DHg_%s_%s_uncollapsed.png'%(style, material), dpi=300)
    
    
    
    # Run simulations for all 5 materials
    materials = ['FAA_PC','FAA_PVC', 'FAA_PMMA', 'FAA_HIPS', 'FAA_HDPE']
    
    # Output parameters
    (savefigure, closefigure) = (True, True)
    (style, nondimtype) = ('md_mf', 'FoBi')
    (style, nondimtype) = ('md_mf', 'FoBi_simple_fixed_d')
    
    # Initialize parameters
    (fs, lw, s, exp_num_points) = (48, 9, 100, 25)
    (windowSize, percentile) = (60, 90)
    colors = getJHcolors()
    
    # Initialize stats outputs
    exp_points = []
    mod_points = []
    ms = []
    for material in materials:
        spec_file_dict[material] = processCaseData(spec_file_dict[material])
        mat = spec_file_dict[material]
        (density, conductivity, specific_heat) = (mat['density'], mat['conductivity'], mat['specific_heat'])
        (HoC, emissivity, nu_char) = (mat['heat_of_combustion'], mat['emissivity'], mat['nu_char'])
        (cases, case_basis, data) = (mat['cases'], mat['case_basis'], mat['data'])
        
        total_energy_per_deltas = [case_basis[c]['totalEnergy']/case_basis[c]['delta'] for c in case_basis]
        total_energy_per_delta_ref = np.mean(total_energy_per_deltas)
        
        fobi_out, fobi_hog_out, qr_out, fobi_mlr_out, times_out = developRepresentativeCurve(mat, nondimtype)
        xlim, ylim = getPlotLimits(material)
        times = np.linspace(0, xlim*2, 10001)
        
        cases_to_plot = np.array(list(cases.keys()))
        thicknesses = np.array([cases[c]['delta'] for c in cases_to_plot])
        coneExposures = np.array([cases[c]['cone'] for c in cases_to_plot])
        
        coneExposures, thicknesses, tigns, cases_to_plot = sortCases(cases)
        
        (delta_old, exp_tmax, ymax, j, fig) = (-1, 0, 0, 0, False)
        reference_time, reference_hrrpua, cone_hf_ref = [case_basis[c]['times_trimmed'] for c in case_basis][0], [case_basis[c]['hrrs_trimmed'] for c in case_basis][0], [case_basis[c]['cone'] for c in case_basis][0]
        
        for i, c in enumerate(cases_to_plot): #[cases_to_plot[caseToPlot]]):
            (delta0, coneExposure, tign) = (cases[c]['delta'], cases[c]['cone'], cases[c]['tign'])
            totalEnergy = total_energy_per_delta_ref*delta0
            times, hrrpuas, totalEnergy2 = runSimulation(times, mat, delta0, coneExposure, totalEnergy, fobi_out, fobi_hog_out, reference_hrrpua, reference_time, cone_hf_ref, nondimtype=nondimtype)
            
            mod_peak = getTimeAveragedPeak(times, hrrpuas, 60)
            exp_peak = getTimeAveragedPeak(cases[c]['times'],cases[c]['HRRs'], 60) #, referenceTimes=times)
            
            print("%s & %0.1f & %0.0f & %0.0f & %0.0f \\"%(material, delta0*1000, coneExposure, exp_peak, mod_peak))
            
            label = r'%0.0f $\mathrm{kW/m^{2}}$'%(coneExposure)
            
            if delta0 != delta_old:
                namespace = '..//figures//simulation_' + style + '_' + material + '_%dmm.png'%(delta_old*1e3)
                if fig is not False: finishSimulationFigure(ymax, exp_tmax*1.3, savefigure, closefigure, namespace, fs)
                fig = plt.figure(figsize=(14,12))
                (exp_tmax, ymax, j, delta_old) = (0, 0, 0, delta0)
            
            if (material == 'FAA_PC' or material == 'FAA_PVC') and delta0*1e3 < 4:
                exp_int = 5
            else:
                exp_int = int(np.ceil(cases[c]['times'].shape[0]/exp_num_points))
            plt.scatter(cases[c]['times'][::exp_int]/60,cases[c]['HRRs'][::exp_int], s=s, linewidth=lw, color=colors[j])
            #plt.plot(cases[c]['times']/60,cases[c]['HRRs'], '--', label=label, linewidth=lw, color=colors[j])
            plt.plot((times+tign)/60, hrrpuas, '-', label=label, linewidth=lw, color=colors[j])
            
            j = j+1
            exp_tmax = max([exp_tmax, cases[c]['times'].max()])
            ymax = max([ymax, np.nanmax(cases[c]['HRRs']), np.nanmax(hrrpuas)])
            
            exp_points.append(exp_peak)
            mod_points.append(mod_peak)
            ms.append(material)
        
        namespace = '..//figures//simulation_' + style + '_' + material + '_%dmm.png'%(delta_old*1e3)
        finishSimulationFigure(ymax, exp_tmax*1.3, savefigure, closefigure, namespace, fs)
    
    (axmin, axmax, loglog) = (0.0, 2500, False)
    split = ms
    diff2 = ms
    labelNames = {}
    for m in materials: labelNames[m] = m.replace("FAA_","")
    label = '60s Avg'
    
    fig, sigma_m, delta = plotMaterialExtraction(exp_points, mod_points, split, label, diff=diff2, axmin=axmin, axmax=axmax, loglog=loglog, labelName=labelNames)
    plt.savefig('..//figures//FAA_materials_stats.png', dpi=300)
    
    
    
    # Change to fds
    nondimtype = 'FDS'
    style = 'FDS'
    
    # Initialize stats outputs
    exp_points = []
    mod_points = []
    ms = []
    
    fdsdir, fdscmd = findFds()
    fileDir = os.path.dirname(os.path.abspath(__file__))
    for material in materials:
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
        
        
        if style != 'FDS':
            sim_times = np.linspace(0, 20000, 20001)
            
            total_energy_per_deltas = [case_basis[c]['totalEnergy']/case_basis[c]['delta'] for c in case_basis]
            total_energy_per_delta_ref = np.mean(total_energy_per_deltas)
            
            fobi_out, fobi_hog_out, qr_out, fobi_mlr_out, _ = developRepresentativeCurve(mat, nondimtype=nondimtype)
            
            basis_summary = [[case_basis[c]['delta'], case_basis[c]['cone']] for c in case_basis]
            reference_time, reference_hrrpua, cone_hf_ref = [case_basis[c]['times_trimmed'] for c in case_basis][0], [case_basis[c]['hrrs_trimmed'] for c in case_basis][0], [case_basis[c]['cone'] for c in case_basis][0]
            for i, c in enumerate(list(cases.keys())):
                delta0 = cases[c]['delta']
                coneExposure = cases[c]['cone']
                totalEnergy = total_energy_per_delta_ref*delta0
                
                if totalEnergy < 100:
                    continue
                
                times, hrrpuas, totalEnergy2 = runSimulation(sim_times, mat, delta0, coneExposure, totalEnergy, fobi_out, fobi_hog_out, reference_hrrpua, reference_time, cone_hf_ref, nondimtype=nondimtype)
                
                if totalEnergy2 > totalEnergy:
                    print("Warning %s case %s more energy released than implied by basis and thickness"%(material, c))
                
                mod_peak = getTimeAveragedPeak(times, hrrpuas, windowSize)
                exp_peak = getTimeAveragedPeak(cases[c]['times'],cases[c]['HRRs'], windowSize)
                
                exp_points.append(exp_peak)
                mod_points.append(mod_peak)
                ms.append(material)
                
        else:
            runSimulations = False

            #fobi_out, fobi_hog_out, qr_out, fobi_mlr_out, _ = developRepresentativeCurve(mat, nondimtype=nondimtype)
            
            cone_hf_ref = [case_basis[c]['cone'] for c in case_basis][0]
            cone_d_ref = [case_basis[c]['delta'] for c in case_basis][0]
            
            times_trimmed = [case_basis[c]['times_trimmed'] for c in case_basis][0]
            hrrs_trimmed = [case_basis[c]['hrrs_trimmed'] for c in case_basis][0]
            
            chid = material.replace(' ','_')
            basis_summary = [[case_basis[c]['delta'], case_basis[c]['cone']] for c in case_basis]
            tend = np.nanmax([(cases[c]['times_trimmed'].max()+cases[c]['tign'])*2 for c in cases])
            
            fluxes, deltas, tigns, cases_to_plot = sortCases(cases)
            
            workingDir = fileDir + os.sep +'..' + os.sep + 'input_files' + os.sep+ material + os.sep
            workingDir = workingDir.replace(' ', '_')
            
            if os.path.exists(workingDir) is False: os.mkdir(workingDir)
            # Calculate times to ignition
            
            txt = buildFdsFile(chid, cone_hf_ref, cone_d_ref, emissivity, conductivity, density, 
                                   specific_heat, 20, times_trimmed, hrrs_trimmed, 20000,
                                   deltas, fluxes, 15.0, ignitionMode='Time', case_tigns=tigns,
                                   calculateDevcDt=False)
            
            with open("%s%s%s.fds"%(workingDir, os.sep, chid), 'w') as f:
                f.write(txt)
            
            if runSimulations:
                runModel(workingDir, chid+".fds", 1, fdsdir, fdscmd, printLiveOutput=False)
                
            data = load_csv(workingDir, chid)
            for i in range(0, len(cases_to_plot)):
                c = cases_to_plot[i]
                namespace = '%02d-%03d'%(fluxes[i], deltas[i]*1e3)
                #label = r'%s'%(namespace) #'$\mathrm{kW/m^{2}}$'%(coneExposure)
                label = r'%0.0f $\mathrm{kW/m^{2}}$'%(fluxes[i])
                delta0 = cases[c]['delta']
                times = data['Time'].values
                hrrpuas = data['"HRRPUA-'+namespace+'"'].values
                
                mod_peak = getTimeAveragedPeak(times, hrrpuas, windowSize)
                exp_peak = getTimeAveragedPeak(cases[c]['times'],cases[c]['HRRs'], windowSize) #, referenceTimes=times)
                
                exp_points.append(exp_peak)
                mod_points.append(mod_peak)
                ms.append(material)
    
    (axmin, axmax, loglog) = (0.0, 2500, False)
    split = ms
    diff2 = ms
    labelNames = {}
    for m in materials: labelNames[m] = m.replace("FAA_","")
    label = '60s Avg'
    
    fig, sigma_m, delta = plotMaterialExtraction(exp_points, mod_points, split, label, diff=diff2, axmin=axmin, axmax=axmax, loglog=loglog, labelName=labelNames)
    plt.savefig('..//figures//FAA_materials_stats_fds.png', dpi=300)
