# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 18:18:31 2023

@author: jhodges
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob, os

from algorithms import getJHcolors, getMaterial, getMaterialClass
from algorithms import interpolateExperimentalData, findLimits
from algorithms import developRepresentativeCurve, getFixedModelParams
from algorithms import runSimulation
from algorithms import calculateUncertainty, plotMaterialExtraction

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
        
        targetTimes, HRRs_interp = interpolateExperimentalData(times, HRRs, targetDt=15, filterWidth=False)
        tign, times_trimmed, hrrs_trimmed = findLimits(times, HRRs, 0.001, 0.9)
        
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
        
        if ((times_trimmed.values[1:]-times_trimmed.values[:-1]).min() == 0):
            print("Warning %s case %s has a zero time step"%(material, c))
    return cases, case_basis

if __name__ == "__main__":
    
    # Compare normalization schemes on model predictions
    style = 'md_mf'
    nondimtype = 'FoBi'
    materials = ['PC','PVC', 'PMMA', 'HIPS', 'HDPE']
    
    resultDir = "" #"E:\\projects\\1JLH-NIST2022\\out\\FSRI_materials_jlh\\"
    inputFileDir = "" #"E:\\projects\\1JLH-NIST2022\\Validation\\FSRI_materials\\"
    expFileDir = "" #"E:\\projects\\1JLH-NIST2022\\exp\\FSRI_Materials\\"
    
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
            
            dataFile = os.path.abspath('../data/fsri_materials_processed/scaling_pyrolysis/%s-%02d.csv'%(mat, fluxes[i]))
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
        
            
