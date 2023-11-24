# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 18:18:31 2023

@author: jhodges
"""
import numpy as np
import pandas as pd
import os

from algorithms import getMaterialClass
from algorithms import interpolateExperimentalData, findLimits
from algorithms import getFixedModelParams

def getMaterial(material, style='md_lmhf'):
    thicknesses = style.split('_')[0]
    fluxes = style.split('_')[1]
    
    if material == 'PC':
        referenceCurve = "..\\data\\faa_polymers\\PC.csv"
        density = 1180.0
        conductivity = 0.22 
        specific_heat = 1.9
        heat_of_combustion = 25.6
        emissivity = 0.9
        nu_char = 0.21
        data = pd.read_csv(referenceCurve)
        cases = {
                 '3-75': {'Time' : 'Time_3_75', 'HRR' : 'HRR_3_75', 'delta' : 3, 'cone' : 75},
                 
                 '6-50': {'Time' : 'Time_6_50', 'HRR' : 'HRR_6_50', 'delta' : 5.5, 'cone' : 50},
                 '6-75': {'Time' : 'Time_6_75', 'HRR' : 'HRR_6_75', 'delta' : 5.5, 'cone' : 75},
                 '6-92': {'Time' : 'Time_6_92', 'HRR' : 'HRR_6_92', 'delta' : 5.5, 'cone' : 92},
                 
                 '9-75': {'Time' : 'Time_9_75', 'HRR' : 'HRR_9_75', 'delta' : 9, 'cone' : 75},
                 }
        
        case_basis = []
        if ('l' in fluxes) and ('l' in thicknesses): pass
        if ('m' in fluxes) and ('l' in thicknesses): case_basis.append('3-75')
        if ('h' in fluxes) and ('l' in thicknesses): pass
        if ('l' in fluxes) and ('m' in thicknesses): case_basis.append('6-50')
        if ('m' in fluxes) and ('m' in thicknesses): case_basis.append('6-75')
        if ('h' in fluxes) and ('m' in thicknesses): case_basis.append('6-92')
        if ('l' in fluxes) and ('h' in thicknesses): pass
        if ('m' in fluxes) and ('h' in thicknesses): case_basis.append('9-75')
        if ('h' in fluxes) and ('h' in thicknesses): pass
    
    elif material == 'PVC':
        referenceCurve = "..\\data\\faa_polymers\\PVC.csv"
        density = 1430.0
        conductivity = 0.17 
        specific_heat = 1.55
        heat_of_combustion = 36.5
        emissivity = 0.9
        nu_char = 0.21
        data = pd.read_csv(referenceCurve)
        cases = {
                 '3-75': {'Time' : 'Time_3_75', 'HRR' : 'HRR_3_75', 'delta' : 3, 'cone' : 75},
                 
                 '6-50': {'Time' : 'Time_6_50', 'HRR' : 'HRR_6_50', 'delta' : 6, 'cone' : 50},
                 '6-75': {'Time' : 'Time_6_75', 'HRR' : 'HRR_6_75', 'delta' : 6, 'cone' : 75},
                 '6-92': {'Time' : 'Time_6_92', 'HRR' : 'HRR_6_92', 'delta' : 6, 'cone' : 92},
                 
                 '9-75': {'Time' : 'Time_9_75', 'HRR' : 'HRR_9_75', 'delta' : 9, 'cone' : 75},
                 }
        case_basis = []
        if ('l' in fluxes) and ('l' in thicknesses): pass
        if ('m' in fluxes) and ('l' in thicknesses): case_basis.append('3-75')
        if ('h' in fluxes) and ('l' in thicknesses): pass
        if ('l' in fluxes) and ('m' in thicknesses): case_basis.append('6-50')
        if ('m' in fluxes) and ('m' in thicknesses): case_basis.append('6-75')
        if ('h' in fluxes) and ('m' in thicknesses): case_basis.append('6-92')
        if ('l' in fluxes) and ('h' in thicknesses): pass
        if ('m' in fluxes) and ('h' in thicknesses): case_basis.append('9-75')
        if ('h' in fluxes) and ('h' in thicknesses): pass
    elif material == 'PMMA':
        referenceCurve = "..\\data\\faa_polymers\\pmma.csv"
        density = 1100
        conductivity = 0.20
        specific_heat = 2.2
        heat_of_combustion = 33.5 #24.450
        emissivity = 0.85
        nu_char = 0.0
        data = pd.read_csv(referenceCurve)
        cases = {
                 '3-25': {'Time' : 'Time_3_25', 'HRR' : 'HRR_3_25', 'delta' : 3.2, 'cone' : 25},
                 '8-24': {'Time' : 'Time_8_24', 'HRR' : 'HRR_8_24', 'delta' : 8.1, 'cone' : 24},
                 '27-23': {'Time' : 'Time_27_23', 'HRR' : 'HRR_27_23', 'delta' : 27, 'cone' : 23},
                 '3-50': {'Time' : 'Time_3_50', 'HRR' : 'HRR_3_50', 'delta' : 3.2, 'cone' : 50},
                 '8-49': {'Time' : 'Time_8_49', 'HRR' : 'HRR_8_49', 'delta' : 8.1, 'cone' : 49},
                 '27-46': {'Time' : 'Time_27_46', 'HRR' : 'HRR_27_46', 'delta' : 27, 'cone' : 46},
                 '3-75': {'Time' : 'Time_3_75', 'HRR' : 'HRR_3_75', 'delta' : 3.2, 'cone' : 75},
                 '8-73': {'Time' : 'Time_8_73', 'HRR' : 'HRR_8_73', 'delta' : 8.1, 'cone' : 73},
                 '27-69': {'Time' : 'Time_27_69', 'HRR' : 'HRR_27_69', 'delta' : 27, 'cone' : 69}
                 }
        case_basis = []
        if ('l' in fluxes) and ('l' in thicknesses): case_basis.append('3-25')
        if ('m' in fluxes) and ('l' in thicknesses): case_basis.append('3-50')
        if ('h' in fluxes) and ('l' in thicknesses): case_basis.append('3-75')
        if ('l' in fluxes) and ('m' in thicknesses): case_basis.append('8-24')
        if ('m' in fluxes) and ('m' in thicknesses): case_basis.append('8-49')
        if ('h' in fluxes) and ('m' in thicknesses): case_basis.append('8-73')
        if ('l' in fluxes) and ('h' in thicknesses): case_basis.append('27-23')
        if ('m' in fluxes) and ('h' in thicknesses): case_basis.append('27-46')
        if ('h' in fluxes) and ('h' in thicknesses): case_basis.append('27-69')
    elif material == 'HIPS':
        referenceCurve = "..\\data\\faa_polymers\\hips.csv"
        density = 950
        conductivity = 0.22
        specific_heat = 2.0
        heat_of_combustion = 39.2 #38.1
        emissivity = 0.86
        nu_char = 0.0
        data = pd.read_csv(referenceCurve)
        cases = {
                 '3-25': {'Time' : 'Time_3_25', 'HRR' : 'HRR_3_25', 'delta' : 3.2, 'cone' : 25},
                 '8-24': {'Time' : 'Time_8_24', 'HRR' : 'HRR_8_24', 'delta' : 8.1, 'cone' : 24},
                 '27-23': {'Time' : 'Time_27_23', 'HRR' : 'HRR_27_23', 'delta' : 27, 'cone' : 23},
                 '3-50': {'Time' : 'Time_3_50', 'HRR' : 'HRR_3_50', 'delta' : 3.2, 'cone' : 50},
                 '8-49': {'Time' : 'Time_8_49', 'HRR' : 'HRR_8_49', 'delta' : 8.1, 'cone' : 49},
                 '27-46': {'Time' : 'Time_27_46', 'HRR' : 'HRR_27_46', 'delta' : 27, 'cone' : 46},
                 '3-75': {'Time' : 'Time_3_75', 'HRR' : 'HRR_3_75', 'delta' : 3.2, 'cone' : 75},
                 '8-73': {'Time' : 'Time_8_73', 'HRR' : 'HRR_8_73', 'delta' : 8.1, 'cone' : 73},
                 '27-69': {'Time' : 'Time_27_69', 'HRR' : 'HRR_27_69', 'delta' : 27, 'cone' : 69}
                 }
        case_basis = []
        if ('l' in fluxes) and ('l' in thicknesses): case_basis.append('3-25')
        if ('m' in fluxes) and ('l' in thicknesses): case_basis.append('3-50')
        if ('h' in fluxes) and ('l' in thicknesses): case_basis.append('3-75')
        if ('l' in fluxes) and ('m' in thicknesses): case_basis.append('8-24')
        if ('m' in fluxes) and ('m' in thicknesses): case_basis.append('8-49')
        if ('h' in fluxes) and ('m' in thicknesses): case_basis.append('8-73')
        if ('l' in fluxes) and ('h' in thicknesses): case_basis.append('27-23')
        if ('m' in fluxes) and ('h' in thicknesses): case_basis.append('27-46')
        if ('h' in fluxes) and ('h' in thicknesses): case_basis.append('27-69')
    elif material == 'HDPE':
        referenceCurve = "..\\data\\faa_polymers\\hdpe.csv"
        density = 860
        conductivity = 0.29
        specific_heat = 3.5
        heat_of_combustion = 47.5 #43.5
        emissivity = 0.92
        nu_char = 0.0
        data = pd.read_csv(referenceCurve)
        cases = {
                 '3-25': {'Time' : 'Time_3_25', 'HRR' : 'HRR_3_25', 'delta' : 3.2, 'cone' : 25},
                 '8-24': {'Time' : 'Time_8_24', 'HRR' : 'HRR_8_24', 'delta' : 8.1, 'cone' : 24},
                 '27-23': {'Time' : 'Time_27_23', 'HRR' : 'HRR_27_23', 'delta' : 27, 'cone' : 23},
                 '3-50': {'Time' : 'Time_3_50', 'HRR' : 'HRR_3_50', 'delta' : 3.2, 'cone' : 50},
                 '8-49': {'Time' : 'Time_8_49', 'HRR' : 'HRR_8_49', 'delta' : 8.1, 'cone' : 49},
                 '27-46': {'Time' : 'Time_27_46', 'HRR' : 'HRR_27_46', 'delta' : 27, 'cone' : 46},
                 '3-75': {'Time' : 'Time_3_75', 'HRR' : 'HRR_3_75', 'delta' : 3.2, 'cone' : 75},
                 '8-73': {'Time' : 'Time_8_73', 'HRR' : 'HRR_8_73', 'delta' : 8.1, 'cone' : 73},
                 '27-69': {'Time' : 'Time_27_69', 'HRR' : 'HRR_27_69', 'delta' : 27, 'cone' : 69}
                 }
        case_basis = []
        if ('l' in fluxes) and ('l' in thicknesses): case_basis.append('3-25')
        if ('m' in fluxes) and ('l' in thicknesses): case_basis.append('3-50')
        if ('h' in fluxes) and ('l' in thicknesses): case_basis.append('3-75')
        if ('l' in fluxes) and ('m' in thicknesses): case_basis.append('8-24')
        if ('m' in fluxes) and ('m' in thicknesses): case_basis.append('8-49')
        if ('h' in fluxes) and ('m' in thicknesses): case_basis.append('8-73')
        if ('l' in fluxes) and ('h' in thicknesses): case_basis.append('27-23')
        if ('m' in fluxes) and ('h' in thicknesses): case_basis.append('27-46')
        if ('h' in fluxes) and ('h' in thicknesses): case_basis.append('27-69')
    for c in list(cases.keys()):
        cases[c]['File'] = referenceCurve
    return density, conductivity, specific_heat, heat_of_combustion, emissivity, nu_char, data, cases, case_basis

def getPlotLimits(material):
    if material == 'PVC':
        xlim = 1000
        ylim = 300
    if material == 'PC':
        ylim = 1000
        xlim = 1000
    if material == 'PMMA':
        ylim = 1500
        xlim = 2500
    if material == 'HIPS':
        ylim = 1500
        xlim = 3000
    if material == 'HDPE':
        ylim = 2500
        xlim = 3000
    return xlim, ylim

def processCaseData_old(material, style='md_lmhf', save_csv=False):
    density, conductivity, specific_heat, HoC, emissivity, nu_char, data, cases, case_basis = getMaterial(material, style=style)
    
    for i, c in enumerate(list(cases.keys())):
        times = data[cases[c]['Time']]
        HRRs = data[cases[c]['HRR']]
        
        targetTimes, HRRs_interp = interpolateExperimentalData(times.values, HRRs.values, targetDt=15, filterWidth=False)
        tign, times_trimmed, hrrs_trimmed = findLimits(times.values, HRRs.values, 0.001, 0.9)
        
        if save_csv:
            pd.DataFrame(np.array([times_trimmed.values-tign, hrrs_trimmed.values]).T, columns=['Time','HRRPUA']).to_csv('%s_%s.csv'%(material, c))
        
        tmp = (HRRs.values*0.1016*0.1016)
        tmp[np.isnan(tmp)] = 0
        times[np.isnan(times)] = 0
        totalEnergy = np.trapz(tmp,  times)
        
        cases[c]['tign'] = tign
        cases[c]['times'] = times
        cases[c]['HRRs'] = HRRs
        cases[c]['times_trimmed'] = times_trimmed
        cases[c]['hrrs_trimmed'] = hrrs_trimmed
        cases[c]['totalEnergy'] = totalEnergy
        
        if ((times_trimmed[1:]-times_trimmed[:-1]).min() == 0):
            print("Warning %s case %s has a zero time step"%(material, c))
    return cases, case_basis

if __name__ == "__main__":
    
    # Compare normalization schemes on model predictions
    style = 'md_mf'
    nondimtype = 'FoBi'
    materials = ['PC','PVC', 'PMMA', 'HIPS', 'HDPE']
    
    resultDir = ""
    inputFileDir = ""
    expFileDir = ""
    
    txt = 'Code,Number,Material,MaterialClass,DataFile,ResultDir,InputFileDir,ExpFileDir,'
    txt = txt + 'ReferenceExposure,ReferenceThickness,ReferenceTime,ReferenceHRRPUA,'
    txt = txt + 'ValidationTimes,ValidationHrrpuaColumns,ValidationFluxes,'
    txt = txt + 'Density,Conductivity,SpecificHeat,Emissivity,Thickness,'
    txt = txt + 'CharFraction,HeatOfCombustion,'
    txt = txt + 'IgnitionTemperature,IgnitionTemperatureBasis,HeaderRows,FYI'
    
    # Get fixed parameters
    params = getFixedModelParams()
    cone_area = params['cone_area']
    cone_diameter = params['cone_diameter']
    xr = params['xr']
    qa = params['qa']
    sig = params['sig']
    Tf = params['Tf']
    xA = params['xA']
    hc = params['hc']
    
    # Initialize variables
    for material in materials:
        density, conductivity, specific_heat, heat_of_combustion, emissivity, nu_char, _, _, case_basis = getMaterial(material, style=style)
        cases, case_basis = processCaseData_old(material, style=style)
        
        thickness = cases[case_basis[0]]['delta']
        initial_mass = density
        final_mass = initial_mass*nu_char
        matClass= getMaterialClass(material)
        
        caseNames = list(cases.keys())
        fluxes = [cases[c]['cone'] for c in caseNames]
        thicknesses = [cases[c]['delta'] for c in caseNames]
        case_files = [cases[c]['File'] for c in caseNames]
        ones = [1 for c in caseNames]
        
        reference_flux = cases[case_basis[0]]['cone']
        reference_file = cases[case_basis[0]]['File']
        reference_time = cases[case_basis[0]]['Time']
        reference_hrr = cases[case_basis[0]]['HRR']
        
        dataFiles = [os.path.abspath(f) for f in list(set(case_files))]
        
        dataFiles_txt = '|'.join(dataFiles)
        
        code ='d'
        number = 1
        mat = 'FAA_%s'%(material)
        dataFiles = ''
        
        thickness_txt = ''
        initial_mass_txt = ''
        final_mass_txt = ''
        timeFiles = ''
        hrrFiles = ''
        flux_txt = ''
        for i in range(0, len(fluxes)):
            
            thickness_txt = '%s%0.8f|'%(thickness_txt, thicknesses[i]/1000)
            
            dataFile = '../data/fsri_materials_processed/scaling_pyrolysis/%s-%02d.csv'%(mat, fluxes[i])
            dataFiles = dataFiles + dataFile + '|'
            dataFiles = dataFiles[:-1]
            
            timeFiles = timeFiles+cases[caseNames[i]]['Time']+'|'
            hrrFiles = hrrFiles+cases[caseNames[i]]['HRR']+'|'
            
            flux_txt = '%s%0.0f|'%(flux_txt, fluxes[i])
            
        txt = txt + "\n" + "%s,%s,%s,%s,%s,%s,"%(code, number, mat, matClass, dataFiles_txt, resultDir)
        txt = txt + "%s,%s,%0.0f,%0.8f,%s,%s,"%(inputFileDir, expFileDir, reference_flux, thickness/1000, reference_time,reference_hrr)
        txt = txt + timeFiles[:-1] + ',' + hrrFiles[:-1] + ',' + flux_txt[:-1] + ','
        txt = txt + '%0.1f,%0.4f,%0.4f,%0.4f,'%(density, conductivity, specific_heat, emissivity)
        txt = txt + thickness_txt[:-1] + ','
        txt = txt + '%0.4f,%0.4f,'%(nu_char, heat_of_combustion)
        txt = txt + 'Calculate,' + flux_txt[:-1] +',1,FAA_materials'
        
    with open('../data/faa_spec_file.csv', 'w') as f:
        f.write(txt)
        
            
