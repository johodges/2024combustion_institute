# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 18:18:31 2023

@author: jhodges
"""
import pyfdstools as fds
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
import pandas as pd
from collections import defaultdict
import scipy
from matplotlib.lines import Line2D
import glob

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


def getTimeAveragedPeak(times, hrrpuas, windowSize, referenceTimes=False):
    
    while times[-1] < np.nanmax(times):
        times = times[:-1]
        hrrpuas = hrrpuas[:-1]
    
    if referenceTimes is not False:
        hrrpuas = np.interp(referenceTimes, times, hrrpuas)
        times = referenceTimes
        
    dt = np.median(times[1:] - times[:-1])
    N = int(np.round(windowSize/dt))
    
    f = np.zeros((N)) + 1
    f = f/f.sum()
    
    hrrpuas_time_avg = np.convolve(hrrpuas, f, mode='same')
    
    return np.nanmax(hrrpuas_time_avg)

def getTimeAveragedEnergy(times, hrrpuas, windowSize, percentile, referenceTimes=False, truncateHRRPUA=10, timeAverage=True):
    while times[-1] < np.nanmax(times):
        times = times[:-1]
        hrrpuas = hrrpuas[:-1]
    
    try:
        ind = np.where(hrrpuas > truncateHRRPUA)[0][-1]
    except:
        ind = -1
    times = times[:ind]
    hrrpuas = hrrpuas[:ind]
    
    if referenceTimes is not False:
        hrrpuas = np.interp(referenceTimes, times, hrrpuas, right=0.0)
        times = referenceTimes
    
    dt = np.median(times[1:] - times[:-1])
    
    
    if windowSize > times.max():
        print("Warning, windowSize > max time after truncating. Not time averaging.")
        timeAverage = False
    
    if timeAverage:
        N = int(np.round(windowSize/dt))
        f = np.zeros((N)) + 1
        f = f/f.sum()
        hrrpuas_time_avg = np.convolve(hrrpuas, f, mode='same')
    else:
        hrrpuas_time_avg = hrrpuas
    
    energy_with_time = np.zeros_like(times)
    energy_with_time[1:] = np.cumsum(hrrpuas_time_avg[1:]*(times[1:]-times[:-1]))
    
    energyThreshold = energy_with_time[-1]*percentile/100
    
    ind = np.argwhere(energy_with_time > energyThreshold)[0][0]
    
    return energyThreshold, times[ind]


def getTimeAveragedPercentile(times, hrrpuas, windowSize, percentile, referenceTimes=False, truncateHRRPUA=10, timeAverage=True):
    while times[-1] < np.nanmax(times):
        times = times[:-1]
        hrrpuas = hrrpuas[:-1]
    
    try:
        ind = np.where(hrrpuas > truncateHRRPUA)[0][-1]
    except:
        ind = -1
    times = times[:ind]
    hrrpuas = hrrpuas[:ind]
    
    if referenceTimes is not False:
        hrrpuas = np.interp(referenceTimes, times, hrrpuas, right=0.0)
        times = referenceTimes
    
    dt = np.median(times[1:] - times[:-1])
    
    
    if windowSize > times.max():
        print("Warning, windowSize > max time after truncating. Not time averaging.")
        timeAverage = False
    
    if timeAverage:
        N = int(np.round(windowSize/dt))
        f = np.zeros((N)) + 1
        f = f/f.sum()
        hrrpuas_time_avg = np.convolve(hrrpuas, f, mode='same')
    else:
        hrrpuas_time_avg = hrrpuas
    
    energy_with_time = np.zeros_like(times)
    energy_with_time[1:] = np.cumsum(hrrpuas_time_avg[1:]*(times[1:]-times[:-1]))
    
    ind = np.argwhere(energy_with_time > energy_with_time[-1]*percentile/100)[0][0]
    
    return times[ind], timeAverage


def getTimeAveraged_timeToEnergy(times, hrrpuas, windowSize, energyThreshold, referenceTimes=False, truncateHRRPUA=10, timeAverage=True):
    while times[-1] < np.nanmax(times):
        times = times[:-1]
        hrrpuas = hrrpuas[:-1]
    
    try:
        ind = np.where(hrrpuas > truncateHRRPUA)[0][-1]
    except:
        ind = -1
    times = times[:ind]
    hrrpuas = hrrpuas[:ind]
    
    if referenceTimes is not False:
        hrrpuas = np.interp(referenceTimes, times, hrrpuas, right=0.0)
        times = referenceTimes
    
    dt = np.median(times[1:] - times[:-1])
    
    
    if windowSize > times.max():
        print("Warning, windowSize > max time after truncating. Not time averaging.")
        timeAverage = False
    
    if timeAverage:
        N = int(np.round(windowSize/dt))
        f = np.zeros((N)) + 1
        f = f/f.sum()
        hrrpuas_time_avg = np.convolve(hrrpuas, f, mode='same')
    else:
        hrrpuas_time_avg = hrrpuas
    
    energy_with_time = np.zeros_like(times)
    energy_with_time[1:] = np.cumsum(hrrpuas_time_avg[1:]*(times[1:]-times[:-1]))
    
    try:
        ind = np.argwhere(energy_with_time > energyThreshold)[0][0]
    except:
        ind = -1
    
    return times[ind], timeAverage


def findLimits(times, HRRs, energyCutoff1=0.001, energyCutoff2=1.01):
    ''' This function extracts the burning duration data from
    a cone calorimeter dataset. This is based on two cutoffs for the
    total energy released. The energy cutoff for the time to ignition
    is the same as used in findIgnitionTime and is arbitrary. The
    energy cutoff used on the trailing end is dynamically calculated
    based on the data curve. A seed value can be set as a start cutoff.
    By default, finds the last time where HRPPUA > 0.
    '''
    v = np.cumsum(HRRs)
    ind1 = 0 
    counter = 0
    while ind1 == 0:
        try:
            ind1 = np.where(v < np.nanmax(v)*energyCutoff1)[0][-1]
        except:
            energyCutoff1 = energyCutoff1*2
        counter += 1
        if counter > 20:
            ind1 = 0
            break
    ind2 = v.shape[0]
    '''
    counter = 0
    while ind2 == v.shape[0]:
        try:
            ind2 = np.where(v > np.nanmax(v)*energyCutoff2)[0][0]
        except:
            energyCutoff2 = energyCutoff2*0.99
        counter += 1
        if counter > 20:
            ind2 = v.shape[0]
            break
    '''
    times_trimmed = times[ind1:ind2]
    hrrs_trimmed = HRRs[ind1:ind2]
    tign = times[ind1]
    while (times_trimmed[-1] < np.nanmax(times_trimmed)):
        hrrs_trimmed = hrrs_trimmed[:-1]
        times_trimmed = times_trimmed[:-1]
    
    return tign, times_trimmed, hrrs_trimmed

def interpolateExperimentalData(times, HRRs, targetDt=False, filterWidth=False):
    dt = np.nanmedian(times[1:]-times[:-1])
    if filterWidth is not False:
        filterWidth = int(filterWidth/dt)
        fil = np.ones(filterWidth)/filterWidth
        HRRs = np.convolve(HRRs, fil, mode='same')
    
    if targetDt is not False:
        dt = targetDt
    else:
        dt = np.nanmedian(times[1:]-times[:-1])
    tmax = np.round(np.nanmax(times)/dt)*dt
    tmin = np.round(np.nanmin(times)/dt)*dt
    targetTimes = np.linspace(tmin, tmax, int((tmax-tmin)/dt + 1))
    HRRs = np.interp(targetTimes, times, HRRs)
    
    return targetTimes, HRRs


def getMaterials(material=False, dataDirectory="..//data", namespace="*spec_file.csv"):
    files = glob.glob(dataDirectory+os.sep+namespace)
    
    spec_file_dict = dict()
    for file in files:
        specificationFile = pd.read_csv(file)
        
        for i in range(0, specificationFile.shape[0]):
            code = specificationFile.iloc[i]['Code']
            num_id = specificationFile.iloc[i]['Number']
            if code == 's':
                print("Skipping file %s row %d"%(file, i))
                continue
            elif code =='d':
                pass
            else:
                print("Unknown code %s in row %d"%(code, i))
                continue
            
            # Extract specification file data
            m = specificationFile.iloc[i]['Material']
            
            print(m)
            
            if material is not False:
                if m != material:
                    continue
            
            #series = specificationFile.iloc[i]['FYI']
            #materialClass = specificationFile.iloc[i]['MaterialClass']
            referenceExposure = str(specificationFile.iloc[i]['ReferenceExposure'])
            conductivity = specificationFile.iloc[i]['Conductivity']
            specific_heat = specificationFile.iloc[i]['SpecificHeat']
            density = specificationFile.iloc[i]['Density']
            emissivity = specificationFile.iloc[i]['Emissivity']
            thickness = str(specificationFile.iloc[i]['Thickness'])
            #preprocess = specificationFile.iloc[i]['Preprocess']
            nu_char = specificationFile.iloc[i]['CharFraction']
            heat_of_combustion = specificationFile.iloc[i]['HeatOfCombustion']
            materialClass = specificationFile.iloc[i]['MaterialClass']
            
            #resultDir = specificationFile.iloc[i]['ResultDir'].replace('\\\\','\\').replace('"','')
            #if os.path.exists(resultDir) is not True: os.mkdir(resultDir)
            #workingDir = os.path.join(resultDir, 'tmp')
            #if os.path.exists(workingDir) is not True: os.mkdir(workingDir)
            #inputFileDir = specificationFile.iloc[i]['InputFileDir'].replace('\\\\','\\').replace('"','')
            #expFileDir = specificationFile.iloc[i]['ExpFileDir'].replace('\\\\','\\').replace('"','')
            
            referenceTimeColumns = specificationFile.iloc[i]['ReferenceTime']
            referenceHrrpuaColumns = str(specificationFile.iloc[i]['ReferenceHRRPUA'])
            referenceThickness = str(specificationFile.iloc[i]['ReferenceThickness'])
            
            if '|' in referenceTimeColumns:
                referenceTimeColumns = referenceTimeColumns.split('|')
            else:
                referenceTimeColumns = [referenceTimeColumns]
                
            if '|' in referenceHrrpuaColumns:
                referenceHrrpuaColumns = referenceHrrpuaColumns.split('|')
            else:
                referenceHrrpuaColumns = [referenceHrrpuaColumns]
            
            validationTimeColumns = specificationFile.iloc[i]['ValidationTimes'].split('|')
            validationHrrpuaColumns = specificationFile.iloc[i]['ValidationHrrpuaColumns'].split('|')
            validationFluxes = specificationFile.iloc[i]['ValidationFluxes'].split('|')
            
            if '|' in referenceExposure:
                referenceExposures = [float(f) for f in referenceExposure.split('|')]
            else:
                referenceExposures = [float(referenceExposure) for f in referenceTimeColumns]
            
            if '|' in referenceThickness:
                referenceThicknesses = [float(f) for f in referenceThickness.split('|')]
            else:
                referenceThicknesses = [float(referenceThickness) for f in referenceTimeColumns]
            
            fluxes = [float(f) for f in validationFluxes]
            
            if '|' in thickness:
                thicknesses = [float(f) for f in thickness.split('|')]
            else:
                thicknesses = [float(thickness) for f in fluxes]
            
            ignitionTemperature = specificationFile.iloc[i]['IgnitionTemperature']
            if ignitionTemperature == 'Calculate':
                calculateIgnitionTemperature = True
                ignitionTemperatureBasis = specificationFile.iloc[i]['IgnitionTemperatureBasis'].split('|')
                ignitionTemperatureBasis = [float(x) for x in ignitionTemperatureBasis]
                Tign = 1000
            else:
                Tign = float(ignitionTemperature)
                calculateIgnitionTemperature = False
            dataFile = specificationFile.iloc[i]['DataFile'].replace('\\\\','\\').replace('"','')
            headerRows = specificationFile.iloc[i]['HeaderRows']
            if '|' in dataFile:
                dfs = dataFile.split('|')
                hrs = headerRows.split('|')
                exp_data = dict()
                for df, hr in zip(dfs, hrs):
                    fname = df.split(os.sep)[-1]
                    # Read data file, manually due to differing number of header rows
                    with open(df, 'r') as f:
                        d = f.readlines()
                    d = np.array([dd.replace('\n','').replace('/','_').split(',') for dd in d])
                    hr = int(hr)
                    for ii in range(hr, len(d)):
                        for j in range(0, len(d[ii])):
                            try:
                                d[ii,j] = float(d[ii,j])
                            except:
                                d[ii,j] = np.nan
                    columns = [fname + '-' + str(c) for c in d[0]]
                    for ii, c in enumerate(columns):
                        c2 = os.path.abspath(c).split(os.sep)[-1]
                        exp_data[c2] = pd.DataFrame(np.array(d[hr:, ii], dtype=float))
                multipleFiles = True
            else:
                headerRows = int(headerRows)
                # Read data file, manually due to differing number of header rows
                with open(dataFile, 'r') as f:
                    d = f.readlines()
                d = np.array([dd.replace('\n','').split(',') for dd in d])
                
                for ii in range(headerRows, len(d)):
                    for j in range(0, len(d[ii])):
                        try:
                            d[ii,j] = float(d[ii,j])
                        except:
                            d[ii,j] = np.nan
                columns = [str(c) for c in d[0]]
                exp_data = pd.DataFrame(np.array(d[headerRows:, :], dtype=float), columns=columns)
                multipleFiles = False

            cases = dict()
            for ii in range(0, len(validationTimeColumns)):
                casename = 'case-%03d'%(ii)
                cases[casename] = {'Time': validationTimeColumns[ii], 'HRR': validationHrrpuaColumns[ii], 'delta': thicknesses[ii], 'cone': fluxes[ii]}
            
            case_basis = dict()
            for ii in range(0, len(referenceTimeColumns)):
                casename = 'case-1%03d'%(ii)
                case_basis[casename] = {'Time': referenceTimeColumns[ii], 'HRR': referenceHrrpuaColumns[ii], 'delta': referenceThicknesses[ii], 'cone': referenceExposures[ii]}
        
            spec_file_dict[m] = {'density': density, 'conductivity': conductivity, 'specific_heat': specific_heat,
                                        'heat_of_combustion': heat_of_combustion, 'emissivity': emissivity, 'nu_char': nu_char,
                                        'data': exp_data, 'cases': cases, 'case_basis': case_basis, 
                                        'material': material, 'materialClass': materialClass}
    return spec_file_dict
            


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
        heat_of_combustion = 24.450
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
        heat_of_combustion = 38.1
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
        heat_of_combustion = 43.5
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

def getDimensionlessNumbers(Tg, eps, Ts, d1, t, conductivity, density, specific_heat, hc):
    params = getFixedModelParams()
    converged = False
    Ts_old = Ts
    while converged is False:
        Ts = params['Tinit']
        hr = params['sig']*eps*(Ts**2 + Tg**2)*(Ts + Tg)
        #hr = sig*eps*(Tg**2)*Tg
        ht = params['hc'] + hr
        
        Bi = (ht*1000) /(conductivity/d1) # +0.1/0.0127)
        
        Fo = (conductivity / (density*specific_heat)) * t/((d1)**2)
        
        #Ts = np.exp(-Bi*Fo)*(params['Tinit'] - Tg) + Tg
        
        #q_cond = 0.5*(np.pi/Fo)**0.5
        
        #qr = eps*params['sig']*(Tg**4)/1000
        #Ts = (qr/q_cond)*(d1/conductivity) + params['Tinit']
        
        #print(qr, q_cond)
        #
        #eta = Bi*Fo**0.5
        #try:
        #    beta = min([beta, 15])
        #except:
        #    beta[beta > 15] = 15
        #
        #Ts = (1 - np.exp(beta**2)*scipy.special.erfc(beta))*(Tg-params['Tinit']) + params['Tinit']
        #Ts = (1 - np.exp(beta**2)*scipy.special.erfc(beta))*(Tg-Tinit) + Tinit
        if np.max(abs(Ts_old-Ts)/Ts) < 0.01:
            converged = True
        else:
            if np.isnan((np.max(abs(Ts_old-Ts)/Ts))):
                Ts = Ts_old
                return Bi, Fo, Ts, ht
        Ts_old = Ts
    return Bi, Fo, Ts, ht

def processSingleCase(case, data):
    times = data[case['Time']].values
    HRRs = data[case['HRR']].values
    
    if len(HRRs.shape) == 2:
        HRRs = HRRs[:, 0]
        times = times[:, 0]
    targetTimes, HRRs_interp = interpolateExperimentalData(times, HRRs, targetDt=15, filterWidth=False)
    tign, times_trimmed, hrrs_trimmed = findLimits(times, HRRs, 0.001, 0.9)
    
    tmp = (HRRs*0.1016*0.1016)
    tmp[np.isnan(tmp)] = 0
    times[np.isnan(times)] = 0
    totalEnergy = np.trapz(tmp,  times)
    
    case['tign'] = tign
    case['times'] = times
    case['HRRs'] = HRRs
    case['times_trimmed'] = times_trimmed
    case['hrrs_trimmed'] = hrrs_trimmed
    case['totalEnergy'] = totalEnergy
    return case

def processCaseData(mat):
    for c in list(mat['cases'].keys()):
        mat['cases'][c] = processSingleCase(mat['cases'][c], mat['data'])
    
    for c in list(mat['case_basis'].keys()):
        mat['case_basis'][c] = processSingleCase(mat['case_basis'][c], mat['data'])
    return mat






def getMaterialClass(material):
    materialClass = 'Unknown'
    m = material.lower()
    woods = ['balsa', 'composite_deck_board', 'douglas_fir', 'engineered_flooring', 'eucalyptus',
              'hardboard','homasote','luan','masonite','mdf','oak','osb',
              'particle_board','particleboard','pine',
              'spruce','waferboard','wood']
    for w in woods:
        if w in m: materialClass = 'Wood-Based'
    
    polymers = ['acrylic','hdpe','hips','ldpe','nylon','pc','pp','pvc','pmma','pet','plastic','polyester',
                'vinyl']
    for p in polymers:
        if p in m: materialClass = 'Polymers'
    others = ['asphalt', 'cardboard', 'cotton', 'felt','gypsum', 'hemp', 'insulation', 'membrane',
              'rug_pad','window_screen','wool_rug','xps_foam_board']
    for o in others:
        if o in m: materialClass = 'Others'
    
    if materialClass == 'Unknown':
        print(material, m)
        assert False, "Stopped"
    return materialClass



def developRepresentativeCurve(mat, nondimtype='FoBi', plot=False, lw=3, colors=False, labelPlot=False):
    
    params = getFixedModelParams()
    cone_area = params['cone_area']
    cone_diameter = params['cone_diameter']
    xr = params['xr']
    qa = params['qa']
    sig = params['sig']
    Tf = params['Tf']
    xA = params['xA']
    hc = params['hc']
    
    # Initialize maxes
    tmax = -1
    nondim_time_max = -1
    
    # Extract material parameters
    cases = mat['case_basis']
    case_basis = list(cases.keys())
    (density, conductivity, specific_heat) = (mat['density'], mat['conductivity'], mat['specific_heat'])
    (HoC, emissivity, nu_char) = (mat['heat_of_combustion'], mat['emissivity'], mat['nu_char'])
    material = mat['material']
    
    case_outs = dict()
    
    for i, c in enumerate(case_basis):
        tign = cases[c]['tign']
        delta0 = cases[c]['delta'] #/1000
        coneExposure = cases[c]['cone']
        times_trimmed = cases[c]['times_trimmed']
        hrrs_trimmed = cases[c]['hrrs_trimmed']
        totalEnergy = cases[c]['totalEnergy']
        
        delta = np.zeros_like(hrrs_trimmed) + delta0
        energyFraction = np.zeros_like(delta)
        charFraction = np.zeros_like(delta)
        
        mass = np.zeros_like(hrrs_trimmed) + delta0*density
        mass[1:] = delta0*density - np.cumsum(hrrs_trimmed[1:]/(1e3*HoC)*(times_trimmed[1:]-times_trimmed[:-1]))
        
        energy = 0
        warningPrinted = False
        for j in range(1, hrrs_trimmed.shape[0]):
            energy += hrrs_trimmed[j]*cone_area/(times_trimmed[j]-times_trimmed[j-1])
            energyFraction[j] = energy / totalEnergy
            charFraction[j] = energyFraction[j]*nu_char
            
            mix_density = params['char_density']*charFraction[j] + density*(1-charFraction[j])
            delta[j] = mass[j] / mix_density
            if delta[j] < 0:
                delta[j] = 0 
                if warningPrinted is False:
                    print("Warning %s case %s has calculated thickness less than 0"%(material, c))
                warningPrinted = True
        
        t = times_trimmed - tign
        
        hrrpua_qstar_calc = hrrs_trimmed
        hrrpua_qstar_calc[np.isnan(hrrpua_qstar_calc)] = 0
        
        qstar = hrrpua_qstar_calc*cone_area / (1100 * (cone_diameter**2.5))
        lm = -1.02*cone_diameter + 3.7*cone_diameter * (qstar**0.4)
        
        kf = np.log(1-(xr*qa*lm)/(3.6*sig*(Tf**4)*xA))/lm
        
        qr = sig*(Tf**4)*(1-np.exp(kf*lm)) + coneExposure
        #qr = 55*hrrpua_qstar_calc**0.065 - 50 + coneExposure
        qr[qr < 15] = 15
        
        Tg = (qr/(emissivity*sig))**0.25
        
        Ts = np.zeros_like(Tg) + params['Tinit']
        
        mix_density = params['char_density']*charFraction + density*(1-charFraction)
        mix_conductivity = params['char_conductivity']*charFraction + conductivity*(1-charFraction)
        Bi, Fo, Ts, ht = getDimensionlessNumbers(Tg, emissivity, Ts, delta, t, mix_conductivity, mix_density, specific_heat, hc)
        
        #print(Ts[:10])
        
        mass = (delta)*mix_density
        
        if nondimtype == 'Fo':
            nondim_t = Fo
        elif nondimtype == 'FoBi':
            nondim_t = Fo*Bi
        elif nondimtype == 'Time':
            nondim_t = t
        elif nondimtype == 'FoBi_simple':
            flame = 25
            qr = coneExposure + flame
            hr = 0.0154*((qr*1000)**0.75)/1000
            nondim_t = hr*t/(density*specific_heat*delta)
        nondim_time_max = max([nondim_time_max, np.nanmax(nondim_t[np.isfinite(nondim_t)])])
        case_outs[c] = {'t': t, 'qr': qr, 'Fo': Fo, 'Bi': Bi, 'mass': mass, 'delta': delta, 'nondim_t': nondim_t}
        
        tmax = max([tmax, t.max()])
    
    # Initialize Variables
    len_array = 10001
    
    mlr_out = np.zeros((len_array,len(case_basis)))
    qrs_out = np.zeros((len_array,len(case_basis)))
    hogs_out = np.zeros((len_array,len(case_basis)))
    times_out = np.zeros((len_array,len(case_basis)))
    
    nondim_time_out = np.zeros((len_array, ))
    nondim_time_out[1:] = np.logspace(-3,np.ceil(np.log(nondim_time_max)), int(len_array-1))
        
    f = np.zeros((10))+1
    f = f / np.sum(f)
    
    for i, c in enumerate(case_basis):
        t = case_outs[c]['t']
        mass = case_outs[c]['mass']
        qr = case_outs[c]['qr']
        nondim_t = case_outs[c]['nondim_t']
        coneExposure = cases[c]['cone']
        
        mass2 = np.interp(nondim_time_out, nondim_t, mass, right=mass[-1])
        time2 = np.interp(nondim_time_out, nondim_t, t, right=np.nan)
        
        mlr = np.zeros_like(mass2)
        mlr[1:] = (mass2[:-1]-mass2[1:])/(time2[:-1]-time2[1:])
        mlr[np.isnan(mlr)] = 0
        
        hrrpua_out = mlr*-1000*HoC
        qstar = hrrpua_out*cone_area / (1100 * (cone_diameter**2.5))
        lm = -1.02*cone_diameter + 3.7*cone_diameter * (qstar**0.4)
        
        kf = np.log(1-(xr*qa*lm)/(3.6*sig*(Tf**4)*xA))/lm
        
        qrs = sig*(Tf**4)*(1-np.exp(kf*lm)) + coneExposure
        qrs[qrs < 15] = 15
        
        if nondimtype == 'FoBi_simple':
            qrs = np.zeros_like(qrs_out[:, i]) + coneExposure + flame
        
        inds = mlr < 0
        
        mlr_out[:, i] = mlr
        qrs_out[:, i] = qrs
        
        #hogs_out[:, i] = qrs*(1e3*HoC)/(mlr_out[:, i]*-1e3*HoC)
        hogs_out[inds, i] = -qrs[inds]/(mlr_out[inds, i])
        hogs_out[~inds, i] = np.nan
        
        times_out[:, i] = time2
        
        for jj in range(0, len(time2)):
            if time2[-jj] < time2[-jj-1]:
                hogs_out[-jj, i] = np.nan
            #if (hogs_out[-jj, i]-hogs_out[-jj-1, i])/(0.5*(hogs_out[-jj, i]+hogs_out[-jj-1, i])) > 1:
            #    hogs_out[-jj, i] = np.nan
        
        if plot:
            if colors is False:
                colors = getJHcolors()
            #hog = qrs*(1e3*HoC)/(mlr_out[:, i]*-1e3*HoC)
            delta0 = cases[c]['delta'] #/1000
            coneExposure = cases[c]['cone']
            if labelPlot:
                label = "$\Delta=%0.1f \mathrm{mm}, q''_{cone}=%0.0f \mathrm{kW/m^{2}}$"%(delta0*1e3,coneExposure)
                color = colors[i+1]
            else:
                label = None
                color = colors[2]
            if nondimtype == 'Time':
                plt.loglog(time2/60, hogs_out[:, i], '-', linewidth=lw, label=label, color=color) #label=label, color=colors[i])
            else:
                plt.loglog(nondim_time_out, hogs_out[:,i], '-', linewidth=lw, label=label, color=color) #label=label, color=colors[i])
                
                
    #rep_curve = np.nanmean(mlr_out/qrs_out, axis=1)
    qrs_out = np.nanmean(qrs_out, axis=1)
    
    hog_out = np.nanmedian(hogs_out, axis=1) #1e3*HoC/(rep_curve*-1e3*HoC)
    
    try:
        ind = np.where(~np.isnan(hog_out))[0][-1]
    except:
        ind = -1
    return nondim_time_out[:ind], hog_out[:ind], qrs_out[:ind], mlr_out[:ind, :], times_out[:ind, :]



def developRepresentativeCurve_old(mat, nondimtype='FoBi', plot=False, lw=3, collapse=True, colors=False):
    
    params = getFixedModelParams()
    cone_area = params['cone_area']
    cone_diameter = params['cone_diameter']
    xr = params['xr']
    qa = params['qa']
    sig = params['sig']
    Tf = params['Tf']
    xA = params['xA']
    hc = params['hc']
    
    # Initialize Variables
    len_array = 100001
    max_array = 100000
    cases = mat['case_basis']
    case_basis = list(cases.keys())
    mlr_out = np.zeros((len_array,len(case_basis)))
    fo_out = np.linspace(0,max_array,len_array)
    
    mlr_out = np.zeros((len_array,len(case_basis)))
    fo_out = np.linspace(0,max_array,len_array)
    qrs_out = np.zeros((len_array,len(case_basis)))
    tmax = -1
    
    # Extract material parameters
    (density, conductivity, specific_heat) = (mat['density'], mat['conductivity'], mat['specific_heat'])
    (HoC, emissivity, nu_char) = (mat['heat_of_combustion'], mat['emissivity'], mat['nu_char'])
    material = mat['material']
    
    for i, c in enumerate(case_basis):
        tign = cases[c]['tign']
        delta0 = cases[c]['delta'] #/1000
        coneExposure = cases[c]['cone']
        times_trimmed = cases[c]['times_trimmed']
        hrrs_trimmed = cases[c]['hrrs_trimmed']
        totalEnergy = cases[c]['totalEnergy']
        
        delta = np.zeros_like(hrrs_trimmed) + delta0
        energyFraction = np.zeros_like(delta)
        charFraction = np.zeros_like(delta)
        
        mass = np.zeros_like(hrrs_trimmed) + delta0*density
        mass[1:] = delta0*density - np.cumsum(hrrs_trimmed[1:]/(1e3*HoC)*(times_trimmed[1:]-times_trimmed[:-1]))
        

        
        energy = 0
        warningPrinted = False
        for j in range(1, hrrs_trimmed.shape[0]):
            energy += hrrs_trimmed[j]*cone_area/(times_trimmed[j]-times_trimmed[j-1])
            energyFraction[j] = energy / totalEnergy
            charFraction[j] = energyFraction[j]*nu_char
            
            mix_density = params['char_density']*charFraction[j] + density*(1-charFraction[j])
            delta[j] = mass[j] / mix_density
            if delta[j] < 0:
                delta[j] = 0 
                if warningPrinted is False:
                    print("Warning %s case %s has calculated thickness less than 0"%(material, c))
                warningPrinted = True
        
        t = times_trimmed - tign
        
        tmax = max([tmax, t.max()+tign])
        
        hrrpua_qstar_calc = hrrs_trimmed
        hrrpua_qstar_calc[np.isnan(hrrpua_qstar_calc)] = 0
        
        qstar = hrrpua_qstar_calc*cone_area / (1100 * (cone_diameter**2.5))
        lm = -1.02*cone_diameter + 3.7*cone_diameter * (qstar**0.4)
        
        kf = np.log(1-(xr*qa*lm)/(3.6*sig*(Tf**4)*xA))/lm
        
        qr = sig*(Tf**4)*(1-np.exp(kf*lm)) + coneExposure
        #qr = 55*hrrpua_qstar_calc**0.065 - 50 + coneExposure
        qr[qr < 15] = 15
        
        Tg = (qr/(emissivity*sig))**0.25
        
        Ts = np.zeros_like(Tg) + params['Tinit']
        
        mix_density = params['char_density']*charFraction + density*(1-charFraction)
        mix_conductivity = params['char_conductivity']*charFraction + conductivity*(1-charFraction)
        Bi, Fo, Ts, ht = getDimensionlessNumbers(Tg, emissivity, Ts, delta, t, mix_conductivity, mix_density, specific_heat, hc)
        
        mass = (delta)*mix_density
        
        f = np.zeros((10))+1
        f = f / np.sum(f)
        
        if nondimtype == 'Fo':
            fobi_raw = Fo
        else:
            fobi_raw = Fo*Bi**1
        
        mass2 = np.interp(fo_out, fobi_raw, mass, right=np.nan)
        time2 = np.interp(fo_out, fobi_raw, t, right=np.nan)
        
        mlr = np.zeros_like(mass2)
        mlr[1:] = (mass2[:-1]-mass2[1:])/(time2[:-1]-time2[1:])
        mlr[np.isnan(mlr)] = 0
        
        hrrpua_out = mlr*-1000*HoC
        
        qstar = hrrpua_out*cone_area / (1100 * (cone_diameter**2.5))
        lm = -1.02*cone_diameter + 3.7*cone_diameter * (qstar**0.4)
        
        kf = np.log(1-(xr*qa*lm)/(3.6*sig*(Tf**4)*xA))/lm
        
        qrs = sig*(Tf**4)*(1-np.exp(kf*lm)) + coneExposure
        qrs[qrs < 15] = 15
        
        
        mlr_out[:,i] = mlr
        qrs_out[:, i] = qrs
        
        if plot:
            if colors is False:
                colors = getJHcolors()
            hog = qrs*(1e3*HoC)/(mlr_out[:, i]*-1e3*HoC)
            label = "$\Delta=%0.1f \mathrm{mm}, q''_{cone}=%0.0f \mathrm{kW/m^{2}}$"%(delta0*1e3,coneExposure)
            if collapse:
                plt.loglog(fo_out, hog, '-', linewidth=lw, label=label, color=colors[i])
            else:
                for jj in range(0, len(time2)):
                    if time2[-jj] < time2[-jj-1]:
                        hog[-jj] = np.nan
                plt.loglog(time2, hog, '-', linewidth=lw, label=label, color=colors[i])
            
        
    rep_curve = np.nanmean(mlr_out/qrs_out, axis=1)
    qrs_out = np.nanmean(qrs_out, axis=1)
    
    hog_out = 1e3*HoC/(rep_curve*-1e3*HoC)
    
    return fo_out, hog_out, qrs_out, mlr_out














def developRepresentativeCurve_old2(material, cases, case_basis, nondimtype, plot=False, lw=3, collapse=True, colors=False):
    density, conductivity, specific_heat, HoC, emissivity, nu_char, _, _, _ = getMaterial(material)
    params = getFixedModelParams()
    cone_area = params['cone_area']
    cone_diameter = params['cone_diameter']
    xr = params['xr']
    qa = params['qa']
    sig = params['sig']
    Tf = params['Tf']
    xA = params['xA']
    hc = params['hc']
    
    # Initialize Variables
    len_array = 100001
    max_array = 100000
    mlr_out = np.zeros((len_array,len(case_basis)))
    fo_out = np.linspace(0,max_array,len_array)
    
    mlr_out = np.zeros((len_array,len(case_basis)))
    fo_out = np.linspace(0,max_array,len_array)
    qrs_out = np.zeros((len_array,len(case_basis)))
    tmax = -1
    
    for i, c in enumerate(case_basis):
        tign = cases[c]['tign']
        delta0 = cases[c]['delta']/1000
        coneExposure = cases[c]['cone']
        times_trimmed = cases[c]['times_trimmed']
        hrrs_trimmed = cases[c]['hrrs_trimmed']
        totalEnergy = cases[c]['totalEnergy']
        
        delta = np.zeros_like(hrrs_trimmed) + delta0
        energyFraction = np.zeros_like(delta)
        charFraction = np.zeros_like(delta)
        
        mass = np.zeros_like(hrrs_trimmed.values) + delta0*density
        mass[1:] = delta0*density - np.cumsum(hrrs_trimmed.values[1:]/(1e3*HoC)*(times_trimmed.values[1:]-times_trimmed.values[:-1]))
        
        energy = 0
        warningPrinted = False
        for j in range(1, hrrs_trimmed.shape[0]):
            energy += hrrs_trimmed.values[j]*cone_area/(times_trimmed.values[j]-times_trimmed.values[j-1])
            energyFraction[j] = energy / totalEnergy
            charFraction[j] = energyFraction[j]*nu_char
            
            #mass[j] = mass[j-1] - (hrrs_trimmed.values[j]/HoC)/density*(times_trimmed.values[j]-times_trimmed.values[j-1])
            mix_density = params['char_density']*charFraction[j] + density*(1-charFraction[j])
            #delta[j] = delta[j-1] - (hrrs_trimmed.values[j]/HoC)/density*(times_trimmed.values[j]-times_trimmed.values[j-1])/1000
            delta[j] = mass[j] / mix_density
            if delta[j] < 0:
                delta[j] = 0 
                if warningPrinted is False:
                    print("Warning %s case %s has calculated thickness less than 0"%(material, c))
                warningPrinted = True
        
        t = times_trimmed.values - tign
        
        tmax = max([tmax, t.max()+tign])
        
        hrrpua_qstar_calc = hrrs_trimmed.values
        hrrpua_qstar_calc[np.isnan(hrrpua_qstar_calc)] = 0
        
        qstar = hrrpua_qstar_calc*cone_area / (1100 * (cone_diameter**2.5))
        lm = -1.02*cone_diameter + 3.7*cone_diameter * (qstar**0.4)
        
        kf = np.log(1-(xr*qa*lm)/(3.6*sig*(Tf**4)*xA))/lm
        
        qr = sig*(Tf**4)*(1-np.exp(kf*lm)) + coneExposure
        #qr = 55*hrrpua_qstar_calc**0.065 - 50 + coneExposure
        qr[qr < 15] = 15
        
        Tg = (qr/(emissivity*sig))**0.25
        
        Ts = np.zeros_like(Tg) + params['Tinit']
        
        mix_density = params['char_density']*charFraction + density*(1-charFraction)
        mix_conductivity = params['char_conductivity']*charFraction + conductivity*(1-charFraction)
        Bi, Fo, Ts, ht = getDimensionlessNumbers(Tg, emissivity, Ts, delta, t, mix_conductivity, mix_density, specific_heat, hc)
        
        mass = (delta)*mix_density
        
        f = np.zeros((10))+1
        f = f / np.sum(f)
        
        
        if nondimtype == 'Fo':
            fobi_raw = Fo #*Bi**1
        else:
            fobi_raw = Fo*Bi**1
        
        mass2 = np.interp(fo_out, fobi_raw, mass, right=np.nan)
        time2 = np.interp(fo_out, fobi_raw, t, right=np.nan)
        
        #qrs = np.interp(fo_out, fobi_raw, qr_filtered, right=np.nan)
        
        mlr = np.zeros_like(mass2)
        mlr[1:] = (mass2[:-1]-mass2[1:])/(time2[:-1]-time2[1:])
        mlr[np.isnan(mlr)] = 0
        
        hrrpua_out = mlr*-1000*HoC
        
        qstar = hrrpua_out*cone_area / (1100 * (cone_diameter**2.5))
        lm = -1.02*cone_diameter + 3.7*cone_diameter * (qstar**0.4)
        
        kf = np.log(1-(xr*qa*lm)/(3.6*sig*(Tf**4)*xA))/lm
        
        qrs = sig*(Tf**4)*(1-np.exp(kf*lm)) + coneExposure
        qrs[qrs < 15] = 15
        
        
        mlr_out[:,i] = mlr  #np.convolve(mlr/qrs, f, 'same')
        qrs_out[:, i] = qrs
        
        if plot:
            if colors is False:
                colors = getJHcolors()
            hog = qrs*(1e3*HoC)/(mlr_out[:, i]*-1e3*HoC)
            label = "$\Delta=%s \mathrm{mm}, q''_{cone}=%s \mathrm{kW/m^{2}}$"%(c.split('-')[0],c.split('-')[1])
            if collapse:
                plt.loglog(fo_out, hog, '-', linewidth=lw, label=label, color=colors[i])
            else:
                plt.loglog(time2, hog, '-', linewidth=lw, label=label, color=colors[i])
            
        
    rep_curve = np.nanmean(mlr_out/qrs_out, axis=1)
    qrs_out = np.nanmean(qrs_out, axis=1)
    
    hog_out = 1e3*HoC/(rep_curve*-1e3*HoC)
    
    return fo_out, hog_out, qrs_out, mlr_out

def getFixedModelParams():
    # Define Fixed Model Parameters
    Tf = 1200 # K
    xr = 0.30 # 0.35
    xA = 0.95 # 0.85 # Orloff and de Ris = 0.84 for PMMA, used 0.80 before
    sig = 5.67e-11
    qa = 1200
    #Ts = 300+273.15 # Arbitrary
    Tinit = 300 #20+273.15
    cone_side = 0.1016
    cone_area = cone_side**2
    cone_diameter = (cone_area*4/np.pi)**0.5
    hc = 0.015 # cone heat transfer coefficient kW
    bi_min = 1e-15
    d_min = 1e-4
    char_density = 248
    char_conductivity = 0.37
    
    params = {'Tf':Tf, 
              'xr': xr,
              'xA': xA,
              'sig': sig,
              'qa': qa,
              'Tinit': Tinit,
              'cone_side': cone_side,
              'cone_area': cone_area,
              'cone_diameter': cone_diameter,
              'hc': hc,
              'bi_min': bi_min,
              'd_min': d_min,
              'char_density': char_density,
              'char_conductivity': char_conductivity}
    return params

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




def runSimulation(times, mat, delta0, coneExposure, totalEnergy, fobi_out, hog_out, nondimtype='FoBi'):
    # Extract material parameters
    (density, conductivity, specific_heat) = (mat['density'], mat['conductivity'], mat['specific_heat'])
    (HoC, emissivity, nu_char) = (mat['heat_of_combustion'], mat['emissivity'], mat['nu_char'])
    material = mat['material']
    
    # Extract fixed params
    params = getFixedModelParams()

    delta = np.zeros_like(times) + float(delta0) #/1000
    hrrpuas = np.zeros_like(times)
    mass = np.zeros_like(times)
    mass[0] = delta[0]*density
    energyFraction = np.zeros_like(delta)
    charFraction = np.zeros_like(delta)
    #print(times)
    #print("qr\tTg\tht\tBi\tFo\tHRRPUA\tM_old\tM_new\tMLR\tMLRxdt\tdt\tdelta")
    
    relaxation_factor = 0.5 #0.05
    refs = np.zeros_like(delta)
    Fos = np.zeros_like(delta)
    Bios = np.zeros_like(delta)
    mlrs = np.zeros_like(delta)
    qrs = np.zeros_like(delta)
    energy = np.zeros_like(delta)
    Tinit = params['Tinit']
    cone_area = params['cone_area']
    cone_diameter = params['cone_diameter']
    char_conductivity = params['char_conductivity']
    char_density = params['char_density']
    xr = params['xr']
    sig = params['sig']
    xA = params['xA']
    qa = params['qa']
    Tf = params['Tf']
    d_min = params['d_min']
    hc = params['hc']
    

    
    #mlr_final_ind = np.where(np.logical_and(~np.isnan(mlr_out), mlr_out < 0))[0][-1]
    #mlr_out[mlr_final_ind:] = mlr_out[mlr_final_ind]
    counter = 0
    for j in range(1, times.shape[0]):
        t = times[j]
        d = delta[j-1]
        
        mix_conductivity = char_conductivity*charFraction[j-1] + conductivity*(1-charFraction[j-1])
        mix_density = char_density*charFraction[j-1] + density*(1-charFraction[j-1])
        
        qstar = hrrpuas[j-1]*cone_area / (1100 * (cone_diameter**2.5))
        lm = -1.02*cone_diameter + 3.7*cone_diameter * (qstar**0.4)
        
        kf = np.log(1-(xr*qa*lm)/(3.6*sig*(Tf**4)*xA))/lm
        if np.isnan(kf):
            kf = 0
        qr = max([sig*(Tf**4)*(1-np.exp(kf*lm)) + coneExposure, 15])
        
        #qr = max([55*hrrpuas[j-1]**0.065 - 50 + coneExposure,15])
        
        
        if d < d_min:
            d1 = d_min
        else:
            d1 = d/(1-charFraction[j-1])
        Tg = (qr/(emissivity*sig))**0.25
        Ts = Tinit
        
        Bi, Fo, Ts, ht = getDimensionlessNumbers(Tg, emissivity, Ts, d1, t, mix_conductivity, mix_density, specific_heat, hc)
        
        # Extra unused parameters
        dpen = 2*(t*(mix_conductivity/(mix_density*specific_heat)))**0.5
        theta = (Ts-Tinit)/(Tg-Tinit)
        
        
        if nondimtype == 'Fo':
            ref = np.interp(Fo, fobi_out, hog_out)#, right=0.0)#, right=np.nan)
        elif nondimtype == 'FoBi':
            ref = np.interp(Fo*Bi**1, fobi_out, hog_out)#, right=0.0)#, right=np.nan)
        elif nondimtype == 'FoBi_simple':
            flame = 25
            qr = coneExposure + flame
            hr = 0.0154*((qr*1000)**0.75)/1000
            nondim_t = hr*t/(density*specific_heat*d1)
            ref = np.interp(nondim_t, fobi_out, hog_out)
        
        if np.isnan(ref) or ref == 0:
            ref = 0.0
            mlr = 0.0
        else:
            mlr = -qr/ref
            mlr = mlr*relaxation_factor + mlrs[j-1]*(1-relaxation_factor)
            #ref = mlr_final
        
        refs[j] = ref
        Fos[j] = Fo
        Bios[j] = Bi
        qrs[j] = qr
        
        dt = times[j]-times[j-1]
        
        mlrs[j] = mlr
        
        
        if mlr > 0:
            mlr = 0
        mass[j] = mass[j-1] + mlr*dt
        if mass[j] < 0:
            mass[j] = 0
        if np.isnan(mass[j]):
            mass[j] = 0
        hrrpuas[j] = (mass[j-1]-mass[j])/dt * HoC*1000
        if np.isnan(hrrpuas[j]):
            hrrpuas[j] = 0
        energy[j] = energy[j-1]+(hrrpuas[j]*cone_area)*(times[j]-times[j-1])
        mlrs[j] = (mass[j] - mass[j-1])/dt
        
        '''
        if energy[j] > totalEnergy:
            energy[j] = totalEnergy
            hrrpuas[j] = 0
            mass[j] = mass[j-1]
            mlrs[j] = 0
        '''
        
        energyFraction[j] = energy[j] / totalEnergy
        charFraction[j] = energyFraction[j]*nu_char
        mix_density = char_density*charFraction[j-1] + density*(1-charFraction[j-1])
        
        
        delta[j] = mass[j]/mix_density
        
        if mlrs[j] == 0 and np.nanmax(abs(mlrs[j])) > 0:
            counter += 1
        
        if counter > 10:
            energy[j:] = energy[j]
            break
        
        #print('%0.1f\t%0.1f\t%0.1f\t%0.1f\t%0.1f\t%0.1f\t%0.1f\t%0.1f\t%0.4f\t%0.1f\t%0.1f\t%0.4f'%(qr, Tg, ht, Bi, Fo, hrrpuas[j], mass[j-1], mass[j], mlr, mlr*dt,dt,delta[j]))
    if energy[-1] > totalEnergy:
        runSimulation(times, mat, delta0, coneExposure, energy[-1]*1.01, fobi_out, hog_out, nondimtype='FoBi')
    return times, hrrpuas, energy[-1]




















def runSimulation_old(times, material, delta0, coneExposure, totalEnergy, fobi_out, hog_out, nondimtype):
    params = getFixedModelParams()
    density, conductivity, specific_heat, HoC, emissivity, nu_char, _, _, _ = getMaterial(material)

    delta = np.zeros_like(times) + float(delta0)/1000
    hrrpuas = np.zeros_like(times)
    mass = np.zeros_like(times)
    mass[0] = delta[0]*density
    energyFraction = np.zeros_like(delta)
    charFraction = np.zeros_like(delta)
    #print(times)
    #print("qr\tTg\tht\tBi\tFo\tHRRPUA\tM_old\tM_new\tMLR\tMLRxdt\tdt\tdelta")
    
    relaxation_factor = 0.05
    refs = np.zeros_like(delta)
    Fos = np.zeros_like(delta)
    Bios = np.zeros_like(delta)
    mlrs = np.zeros_like(delta)
    qrs = np.zeros_like(delta)
    energy = np.zeros_like(delta)
    Tinit = params['Tinit']
    cone_area = params['cone_area']
    cone_diameter = params['cone_diameter']
    char_conductivity = params['char_conductivity']
    char_density = params['char_density']
    xr = params['xr']
    sig = params['sig']
    xA = params['xA']
    qa = params['qa']
    Tf = params['Tf']
    d_min = params['d_min']
    hc = params['hc']
    
    #mlr_final_ind = np.where(np.logical_and(~np.isnan(mlr_out), mlr_out < 0))[0][-1]
    #mlr_out[mlr_final_ind:] = mlr_out[mlr_final_ind]
    for j in range(1, times.shape[0]):
        t = times[j]
        d = delta[j-1]
        
        mix_conductivity = char_conductivity*charFraction[j-1] + conductivity*(1-charFraction[j-1])
        mix_density = char_density*charFraction[j-1] + density*(1-charFraction[j-1])
        
        qstar = hrrpuas[j-1]*cone_area / (1100 * (cone_diameter**2.5))
        lm = -1.02*cone_diameter + 3.7*cone_diameter * (qstar**0.4)
        
        kf = np.log(1-(xr*qa*lm)/(3.6*sig*(Tf**4)*xA))/lm
        if np.isnan(kf):
            kf = 0
        qr = max([sig*(Tf**4)*(1-np.exp(kf*lm)) + coneExposure, 15])
        
        #qr = max([55*hrrpuas[j-1]**0.065 - 50 + coneExposure,15])
        
        
        if d < d_min:
            d1 = d_min
        else:
            d1 = d/(1-charFraction[j-1])
        Tg = (qr/(emissivity*sig))**0.25
        Ts = Tinit
        
        Bi, Fo, Ts, ht = getDimensionlessNumbers(Tg, emissivity, Ts, d1, t, mix_conductivity, mix_density, specific_heat, hc)
        
        # Extra unused parameters
        dpen = 2*(t*(mix_conductivity/(mix_density*specific_heat)))**0.5
        theta = (Ts-Tinit)/(Tg-Tinit)
        
        
        if nondimtype == 'Fo':
            ref = np.interp(Fo, fobi_out, hog_out)#, right=np.nan)
        else:
            ref = np.interp(Fo*Bi**1, fobi_out, hog_out)#, right=np.nan)
        
        if np.isnan(ref) or ref == 0:
            ref = 0.0
            #ref = mlr_final
        
        refs[j] = ref
        Fos[j] = Fo
        Bios[j] = Bi
        qrs[j] = qr
        
        dt = times[j]-times[j-1]
        
        mlr = -qr/ref
        mlr = mlr*relaxation_factor + mlrs[j-1]*(1-relaxation_factor)
        mlrs[j] = mlr
        
        
        if mlr > 0:
            mlr = 0
        mass[j] = mass[j-1] + mlr*dt
        if mass[j] < 0:
            mass[j] = 0
        if np.isnan(mass[j]):
            mass[j] = 0
        hrrpuas[j] = (mass[j-1]-mass[j])/dt * HoC*1000
        if np.isnan(hrrpuas[j]):
            hrrpuas[j] = 0
        energy[j] = energy[j-1]+(hrrpuas[j]*cone_area)*(times[j]-times[j-1])
        mlrs[j] = (mass[j] - mass[j-1])/dt
        
        '''
        if energy[j] > totalEnergy:
            energy[j] = totalEnergy
            hrrpuas[j] = 0
            mass[j] = mass[j-1]
            mlrs[j] = 0
        '''
        
        energyFraction[j] = energy[j] / totalEnergy
        charFraction[j] = energyFraction[j]*nu_char
        mix_density = char_density*charFraction[j-1] + density*(1-charFraction[j-1])
        
        
        delta[j] = mass[j]/mix_density
        
        #print('%0.1f\t%0.1f\t%0.1f\t%0.1f\t%0.1f\t%0.1f\t%0.1f\t%0.1f\t%0.4f\t%0.1f\t%0.1f\t%0.4f'%(qr, Tg, ht, Bi, Fo, hrrpuas[j], mass[j-1], mass[j], mlr, mlr*dt,dt,delta[j]))
    if energy[-1] > totalEnergy:
        times, hrrpuas, totalEnergy = runSimulation_old(times, material, delta0, coneExposure, energy[-1]*1.01, fobi_out, hog_out, nondimtype)
    return times, hrrpuas, energy[-1]
    
def calculateUncertainty(x, y):
    sigma_e = 0.075
    mask = np.logical_and(~np.isnan(np.array(x, dtype=float)),
                          ~np.isnan(np.array(y, dtype=float)))
    if np.var(np.log(y[mask] / x[mask])) - (sigma_e**2) < 0:
        sigma_m = sigma_e
    else:
        sigma_m = (np.var(np.log(y[mask] / x[mask])) - (sigma_e**2))**0.5
    #sigma_m2 = np.var(np.log(y / x)) / 2
    sigma_m = np.nanmax([sigma_m, sigma_e])
    delta = np.exp(np.mean(np.log(y[mask] / x[mask])) + (sigma_m**2)/2 - (sigma_e**2)/2)
    return delta, sigma_m, sigma_e, np.log(y/x)

def calculateUncertaintyBounds(flatx, flaty, flatFlux, split=False):
    d = pd.DataFrame(np.array([flatx, flaty, flatFlux]).T, columns=['exp','mod','flux'])
    d[d == 0] = np.nan
    #d2[d2 < 0] = np.nan
    mask = np.logical_and(~np.isnan(np.array(d.values[:,0], dtype=float)),
                          ~np.isnan(np.array(d.values[:,1], dtype=float)))
    d2 = d.loc[mask]
    if split:
        uniqueFluxes = np.unique(flatFlux)
        delta = dict()
        sigma_m = dict()
        num_points = dict()
        points = dict()
        for flux in uniqueFluxes:
            x = np.array(d.loc[d['flux'] == flux, 'exp'].values, dtype=float)
            y = np.array(d.loc[d['flux'] == flux, 'mod'].values, dtype=float)
            delta[flux], sigma_m[flux], sigma_e, points[flux] = calculateUncertainty(x, y)
            num_points[flux] = x.shape[0]
    else:
        (x, y) = (np.array(d['exp'].values, dtype=float), np.array(d['mod'].values, dtype=float))
        delta, sigma_m, sigma_e, points = calculateUncertainty(x, y)
        num_points = d2.shape[0]
    return delta, sigma_m, sigma_e, num_points, points

def plotMaterialExtraction(x, y, f, label, diff=None, axmin=None, axmax=None, loglog=False, labelName=None, mask=None):
    
    axmin2 = min([np.min(x), np.min(y)])
    axmax2 = min([np.max(x), np.max(y)])
    if mask is not None:
        xx = x[mask]
        yy = y[mask]
    else:
        xx = x
        yy = y
    delta, sigma_m, sigma_e, num_points, points = calculateUncertaintyBounds(xx, yy, f, split=False)
    
    #print(label, delta, sigma_m, sigma_e)
    
    if axmin is not None:
        axmin2 = axmin
    if axmax is not None:
        axmax2 = axmax
        
    fig, ax = plt.subplots(figsize=(12, 10))
    if loglog:
        ax.set_yscale('log')
        ax.set_xscale('log')
    fs=24
    
    xcoords = np.array([axmin2, axmax2])
    ycoords = np.array([axmin2, axmax2])
    dashes=(10, 10)
    ax.plot(xcoords, ycoords, 'k', linewidth=2)
    ax.plot(xcoords, ycoords*(1+2*sigma_e), '--k', linewidth=2, dashes=dashes)
    ax.plot(xcoords, ycoords/(1+2*sigma_e), '--k', linewidth=2, dashes=dashes)
    
    ax.plot(xcoords, ycoords*delta, 'r', linewidth=2)
    ax.plot(xcoords, ycoords*delta*(1+2*sigma_m), '--r', linewidth=2, dashes=dashes)
    ax.plot(xcoords, ycoords*delta/(1+2*sigma_m), '--r', linewidth=2, dashes=dashes)
    
    markers = ['o', 's', 'd', '>', '<', '^']
    cmap = plt.cm.viridis
    
    colors2 = getNewColors()
    
    cinds = {25:0, 50:1, 75:2}
    
    
    mew = 3
    if diff is not None:
        cases = np.array(list(set(diff)))
        cases.sort()
        for j in range(0, len(f)):
            caseInd = np.where(cases == diff[j])[0][0]
            #c = 0 if diff[j] > 0 else 1
            ax.scatter(x[j], y[j], marker=markers[caseInd], s=100, facecolors='none', edgecolors=colors2[caseInd], linewidths=mew)
        customMarkers = []
        for caseInd, case in enumerate(cases):
            if labelName is None:
                customMarkers.append(Line2D([0],[0],marker=markers[caseInd], color='w', markeredgecolor=colors2[caseInd], markerfacecolor='w', label=case, markersize=15, markeredgewidth=mew))
            else:
                customMarkers.append(Line2D([0],[0],marker=markers[caseInd], color='w', markeredgecolor=colors2[caseInd], markerfacecolor='w', label=labelName[case], markersize=15, markeredgewidth=mew))
        
        #minFlux = -50
        #maxFlux = 50
        #for j in range(0, len(f)):
        #    c = (diff[j]-minFlux)/(maxFlux-minFlux)
        #    ax.scatter(x[j], y[j], s=100, color=cmap(c))
        #customMarkers = []
        #for i, f in enumerate([-50,-25,0,25,50]):
        #    v = (f-minFlux)/(maxFlux-minFlux)
        #    customMarkers.append(Line2D([0],[0],marker=markers[0], color='w', markerfacecolor=cmap(v), label='%0.0f $\mathrm{kW/m^{2}}$'%(f), markersize=15))
        
        ax.legend(handles=customMarkers, fontsize=fs)
    else:
        ax.scatter(x, y, s=100)
    '''
    for j in range(0, len(f)):
        try:
            ind = np.where(f[j] == uniqueFluxes)[0][0]
            c = (plotFlux[j]-minFlux)/(maxFlux-minFlux)
            print(i, j, c, plotFlux[j], x[j], y[j])
            #ax.scatter(x[j], y[j], marker=markers[ind], label=material, s=100, color=cmap(c))
            ax.scatter(x[j], y[j], marker=markers[ind], label=material, s=100, color=colors2[cinds[plotFlux[j]]])
        except:
            pass
    
    customMarkers = []
    for i, f in enumerate(uniqueFluxes):
        v = (f-minFlux)/(maxFlux-minFlux)
        customMarkers.append(Line2D([0],[0],marker=markers[i], color='w', markerfacecolor=colors2[cinds[f]], label='%0.0f $\mathrm{kW/m^{2}}$'%(f), markersize=15))
    
    ax.legend(handles=customMarkers, fontsize=fs)
    '''
    plt.xlabel('Measured %s'%(label), fontsize=fs)
    plt.ylabel('Predicted %s'%(label), fontsize=fs)
    
    plt.xlim([axmin2, axmax2])
    plt.ylim([axmin2, axmax2])
    plt.tick_params(labelsize=fs)
    plt.tick_params(which='major', length=16, width=1, direction='in', top=True,right=True)
    plt.tick_params(which='minor', length=8, width=1, direction='in', top=True,right=True)
    
    #annotation = '%s\n'%(label)
    annotation = ''
    annotation = '%s%s %0.2f\n'%(annotation, 'Exp. Rel. Std. Dev.:', sigma_e)
    annotation = '%s%s %0.2f\n'%(annotation, 'Model Rel. Std. Dev.:', sigma_m)
    annotation = '%s%s %0.2f\n'%(annotation, 'Model Bias Factor:', delta)
    plt.annotate(annotation, (0.5, 0.1), size=fs, xycoords='figure fraction', textcoords='figure fraction', xytext=(0.56,0.1))
    plt.tight_layout()
    return fig, sigma_m, delta

def getNewColors():
    colors2 = np.array([[216, 27, 96],
                        [30, 136, 229],
                        [0, 77, 64],
                        [255, 193, 7],
                        [216, 27, 216],
                        [27, 216, 27],
                        ]) / 255
    return colors2
