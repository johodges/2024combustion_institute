# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 18:18:31 2023

@author: jhodges
"""
import matplotlib.pyplot as plt
import numpy as np

from algorithms import getJHcolors
from algorithms import getMaterials, processCaseData
from algorithms import developRepresentativeCurve
from algorithms import runSimulation, getTimeAveragedPeak
from algorithms import plotMaterialExtraction

def getPlotLimits(material):
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


if __name__ == "__main__":
    
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
    
    fobi_out, fobi_hog_out, qr_out, fobi_mlr_out, _ = developRepresentativeCurve(mat, 'FoBi')
    fo_out, fo_hog_out, qr_out, fo_mlr_out, _ = developRepresentativeCurve(mat, 'Fo')
    xlim, ylim = getPlotLimits(material)
    times = np.linspace(0, 10000, 50001) #xlim*2, 10001)
    
    fs=24
    lw = 3
    colors = getJHcolors()
    
    #labels = ['3mm','8mm','27mm']
    labels = ['3','8','27']
    
    plt.figure(figsize=(10,8))
    for i, c in enumerate(['case-003','case-004','case-005']): #enumerate(list(cases.keys())):
        delta0 = cases[c]['delta']
        coneExposure = cases[c]['cone']
        totalEnergy = total_energy_per_delta_ref*delta0
        times, hrrpuas, totalEnergy2 = runSimulation(times, mat, delta0, coneExposure, totalEnergy, fobi_out, fobi_hog_out, nondimtype='FoBi')
        fo_times, fo_hrrpuas, fo_totalEnergy2 = runSimulation(times, mat, delta0, coneExposure, totalEnergy, fobi_out, fobi_hog_out, nondimtype='Fo')
        
        plt.scatter(cases[c]['times'][::5]/60,cases[c]['HRRs'][::5], label=labels[i]+': Exp', s=50, linewidth=lw, color=colors[i])
        plt.semilogx((fo_times+cases[c]['tign'])/60, fo_hrrpuas, '--', linewidth=lw, label=labels[i]+': Fo', color=colors[i])
        plt.semilogx((times+cases[c]['tign'])/60, hrrpuas, '-', linewidth=lw, label=labels[i]+': FoBi*', color=colors[i])
        
    
    plt.xlabel("Time (min)", fontsize=fs)
    plt.ylabel("$\dot{Q}''$ ($\mathrm{kW/m^{2}}$)", fontsize=fs)
    plt.ylim(0, 1800)
    plt.xlim(1, 60)
    plt.xticks([1, 2, 4, 6, 10, 20, 60], ['1','2','4','6','10','20','60'])
    plt.grid()
    plt.tick_params(labelsize=fs)
    plt.legend(fontsize=fs, bbox_to_anchor=(1.05,0.25))
    #plt.legend(fontsize=fs) #bbox_to_anchor=(1.05,0.1))
    plt.tight_layout()
    plt.savefig('..//figures//time_normalization_' + style + '_' + material + '.png', dpi=300)
    
    
    
    # Compare normalization schemes on Hg
    fs=24
    lw = 6
    leg_loc = 3
    style = 'md_mf'
    material = 'FAA_HDPE'
    #material = 'FAA_PC'
    spec_file_dict[material] = processCaseData(spec_file_dict[material])
    mat = spec_file_dict[material]
    
    plt.figure(figsize=(10, 8))
    mat2 = dict(mat)
    mat2['case_basis'] = dict(mat2['cases'])
    #mat2['case_basis'].pop('case-000')
    fobi_out, hog_out, qr_out, mlr_out, times_out = developRepresentativeCurve(mat2, 'FoBi', plot=True, lw=2, labelPlot=False)
    
    #density, conductivity, specific_heat, HoC, emissivity, nu_char, _, _, _ = getMaterial(material)
    #cases, case_basis = processCaseData(material, style=style)
    #case_basis = list(cases.keys())
    #fobi_out, hog_out, qrs_out, mlr_out = developRepresentativeCurve(material, cases, case_basis, 'FoBi', plot=True)
    plt.loglog(fobi_out, hog_out, '--', linewidth=lw, color='k', label='Median')
    
    
    mat2['case_basis'] = dict(mat2['cases'])
    for key in list(mat2['case_basis'].keys()):
        if key != 'case-000': mat2['case_basis'].pop(key)
    fobi_out, hog_out, qr_out, mlr_out, _ = developRepresentativeCurve(mat2, 'FoBi', plot=True, lw=4, labelPlot=True)
    
    
    fobi_max = fobi_out[np.where(mlr_out < 0)[0][-1]]*1.2
    plt.xlim(1, 2e5) #fobi_max)
    plt.ylim(1000, 100000)
    plt.xlabel("$\mathrm{FoBi^{*}}$ ($-$)", fontsize=fs)
    plt.ylabel('$\Delta H_{g}$ ($\mathrm{kJ/kg}$)', fontsize =fs)
    #lgd = plt.legend(fontsize=fs, bbox_to_anchor=(1.1, 1.0))
    lgd = plt.legend(fontsize=fs, loc=leg_loc)
    
    #plt.arrow(1250, 40000, -700, 0, linewidth=2, color='k', head_width=5000, head_length=100, length_includes_head=True)
    #plt.arrow(21000, 40000, 18000, 0, linewidth=2, color='k', head_width=5000, head_length=5000, length_includes_head=True)
    #plt.annotate(r"$\Delta=3\mathrm{mm}$"+"\n"+r"$\dot{q}''_{cone}=25\mathrm{kW/m^{2}}$", xy = (1250, 35000), size=24)
    #plt.annotate(r"$\Delta=3$"+"\n"+r"$\dot{q}''_{cone}=25$", xy = (1250, 33000), size=24)
    #plt.annotate('Outlier', xy=(1500,40000))
    
    plt.grid()
    plt.tick_params(labelsize=fs)
    
    plt.tight_layout()
    #plt.savefig('..//figures//DHg_%s_%s_collapsed.png'%(style, material), dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig('..//figures//DHg_%s_%s_collapsed.png'%(style, material), dpi=300)
    
    
    fs2 = 24
    lw2 = 6
    plt.figure(figsize=(10, 8))
    mat2 = dict(mat)
    mat2['case_basis'] = dict(mat2['cases'])
    fobi_out, hog_out, qr_out, mlr_out, times_out = developRepresentativeCurve(mat2, 'Time', plot=True, lw=2)
    #plt.loglog(np.nanmedian(times_out, axis=1)/60, hog_out, '--', linewidth=lw2, label='Median', color='k')
    plt.loglog(fobi_out/60, hog_out, '--', linewidth=lw2, label='Median', color='k')
    
    #density, conductivity, specific_heat, HoC, emissivity, nu_char, _, _, _ = getMaterial(material)
    #cases, case_basis = processCaseData(material, style=style)
    #case_basis = list(cases.keys())
    #fobi_out, hog_out, qrs_out, mlr_out = developRepresentativeCurve(material, cases, case_basis, 'FoBi', plot=True, collapse=False)
    
    
    #plt.xlim(1, 3000)
    plt.xlim(1, 30)
    plt.xticks([0.5, 1, 2, 4, 6, 10, 20, 30], ['0.5','1','2','4','6','10','20','30'])
    plt.ylim(1000, 100000)
    plt.xlabel("Time (min)", fontsize=fs2)
    
    plt.ylabel('$\Delta H_{g}$ ($\mathrm{kJ/kg}$)', fontsize =fs2)
    lgd = plt.legend(fontsize=fs2, loc=leg_loc)
    plt.grid()
    plt.tick_params(labelsize=fs2)
    #plt.legend(fontsize=fs)
    plt.tight_layout()
    plt.savefig('..//figures//DHg_%s_%s_uncollapsed.png'%(style, material), dpi=300)
    
    
    
    # Run simulations for all 5 materials
    materials = ['FAA_PC','FAA_PVC', 'FAA_PMMA', 'FAA_HIPS', 'FAA_HDPE']
    
    style = 'md_mf'
    nondimtype = 'FoBi'
    fs=48
    lw = 9
    s = 100
    exp_num_points = 25
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
        
        #density, conductivity, specific_heat, HoC, emissivity, nu_char, _, _, _ = getMaterial(material)
        #cases, case_basis = processCaseData(material, style=style)
        #total_energy_per_deltas = [cases[c]['totalEnergy']/cases[c]['delta'] for c in case_basis]
        #total_energy_per_delta_ref = np.mean(total_energy_per_deltas)
        fobi_out, fobi_hog_out, qr_out, fobi_mlr_out, _ = developRepresentativeCurve(mat, 'FoBi')
        xlim, ylim = getPlotLimits(material)
        times = np.linspace(0, 10000, 100001) #xlim*2, 10001)
        
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
        
        
        delta_old = -1
        exp_tmax = 0
        ymax = 0
        j = 0
        fig = False
        m = material.replace('2','')
        for i, c in enumerate(cases_to_plot):
            delta0 = cases[c]['delta']
            coneExposure = cases[c]['cone']
            totalEnergy = total_energy_per_delta_ref*delta0
            #print(delta0, coneExposure, totalEnergy)
            times, hrrpuas, totalEnergy2 = runSimulation(times, mat, delta0, coneExposure, totalEnergy, fobi_out, fobi_hog_out, nondimtype='FoBi')
            
            mod_peak = getTimeAveragedPeak(times, hrrpuas, 60)
            exp_peak = getTimeAveragedPeak(cases[c]['times'],cases[c]['HRRs'], 60, referenceTimes=times)
            
            print("%s & %0.1f & %0.0f & %0.0f & %0.0f \\"%(material, delta0*1000, coneExposure, exp_peak, mod_peak))
            
            label = '%0.0f $\mathrm{kW/m^{2}}$'%(coneExposure)
            
            if delta0 != delta_old:
                if fig is not False:
                    plt.xlabel("Time (min)", fontsize=fs)
                    plt.ylabel('HRRPUA ($\mathrm{kW/m^{2}}$)', fontsize=fs)
                    plt.ylim(0, np.ceil(1.1*ymax/100)*100+20)
                    plt.xlim(0, np.ceil(exp_tmax/60))
                    plt.grid()
                    plt.tick_params(labelsize=fs)
                    plt.legend(fontsize=fs) #, bbox_to_anchor=(1.05,0.6))
                    plt.tight_layout()
                    plt.savefig('..//figures//simulation_' + style + '_' + m + '_%dmm.png'%(delta_old*1e3), dpi=300)
                    plt.close()
                fig = plt.figure(figsize=(14,12))
                delta_old = delta0
                exp_tmax = 0
                ymax = 0
                j = 0
            
            if (m == 'FAA_PC' or m == 'FAA_PVC') and delta0*1e3 < 4:
                exp_int = 5
            else:
                exp_int = int(np.ceil(cases[c]['times'].shape[0]/exp_num_points))
            plt.scatter(cases[c]['times'][::exp_int]/60,cases[c]['HRRs'][::exp_int], s=s, linewidth=lw, color=colors[j])
            plt.plot((times+cases[c]['tign'])/60, hrrpuas, '-', linewidth=lw, label=label, color=colors[j])
            j = j+1
            exp_tmax = max([exp_tmax, cases[c]['times'].max()])
            ymax = max([ymax, np.nanmax(cases[c]['HRRs']), np.nanmax(hrrpuas)])
            
            exp_points.append(exp_peak)
            mod_points.append(mod_peak)
            ms.append(material)
            
        plt.xlabel("Time (min)", fontsize=fs)
        plt.ylabel("$\dot{Q}''$ ($\mathrm{kW/m^{2}}$)", fontsize=fs)
        plt.ylim(0, np.ceil(1.1*ymax/100)*100+20)
        plt.xlim(0, np.ceil(exp_tmax/60))
        plt.grid()
        plt.tick_params(labelsize=fs)
        plt.legend(fontsize=fs)#, bbox_to_anchor=(1.05,0.6))
        plt.tight_layout()
        plt.savefig('..//figures//simulation_' + style + '_' + m + '_%dmm.png'%(delta_old*1e3), dpi=300)
        plt.close()
    
    (axmin, axmax, loglog) = (0.0, 2500, False)
    split = ms
    diff2 = ms
    labelNames = {}
    for mat in materials: labelNames[mat] = mat.replace("FAA_","")
    label = '60s Avg'
    
    fig, sigma_m, delta = plotMaterialExtraction(exp_points, mod_points, split, label, diff=diff2, axmin=axmin, axmax=axmax, loglog=loglog, labelName=labelNames)
    plt.savefig('..//figures//FAA_materials_stats.png', dpi=300)