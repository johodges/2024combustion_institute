# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 18:18:31 2023

@author: jhodges
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from matplotlib.lines import Line2D
import glob

from plotting import getJHcolors, getNewColors

def buildFdsFile(chid, cone_hf_ref, cone_d_ref, emissivity, conductivity, density, 
                 specific_heat, Tign, time, hrrpua, tend, deltas, fluxes, front_h,
                 case_tigns=False, ignitionMode='Temperature', outputTemperature=False,
                 calculateDevcDt=True, devc_dt=1.,
                 qflame_method='Froude', qflame_fixed=25):
    ''' Generate a solid phase only FDS input file representing cone
    experimens at different exposures given a reference curve and
    material properties. The configuration can support up to 9
    thermal exposures, configured in a 3x3 grid.
    
    Notes:
        1. Zero heat transfer coefficient at the surface. This is
           intentional because the calculated flame heat flux which
           is included in the IGNITION-RAMPs includes convection.
        2. The estimated flame heat flux currently assumes a surface
           heat transfer coefficient of 15 W/m2-K and a gas phase
           radiative fraction of 0.35.
        3. If the ignition temperature is unknown, it can be calculated
           by comparing with experimental times to ignition. Changing
           the input of Tign to 'Calculated' will tell FDS to save
           out the WALL TEMPERATURE data needed to extract this
           information.
        4. All samples are assumed to have 0.5in / 12.7 mm of ceramic
           fiber insulation behind them.
    '''
    hrrpua_ref = getRepresentativeHrrpua(hrrpua, time)
    qref = estimateExposureFlux(cone_hf_ref, hrrpua_ref, qflame_method, qflame_fixed)
    
    tempOutput = '.TRUE.' if outputTemperature else '.FALSE.'
    DT_DEVC = devc_dt
    if calculateDevcDt:
        NFRAMES = 1200/1.
        DT_DEVC = tend/NFRAMES
    if ignitionMode == 'Time': Tign = 20
    txt = "&HEAD CHID='%s', /\n"%(chid)
    txt = txt+"&TIME DT=1., T_END=%0.1f /\n"%(tend)
    txt = txt+"&DUMP DT_CTRL=%0.1f, DT_DEVC=%0.1f, DT_HRR=%0.1f, SIG_FIGS=4, SIG_FIGS_EXP=2, /\n"%(DT_DEVC, DT_DEVC, DT_DEVC)
    txt = txt+"&MISC SOLID_PHASE_ONLY=.TRUE., TMPA=27., /\n"
    txt = txt+"&MESH ID='MESH', IJK=3,3,3, XB=0.,0.3,0.,0.3,0.,0.3, /\n"
    txt = txt+"&REAC ID='PROPANE', FUEL='PROPANE', /\n"
    txt = txt+"&MATL ID='BACKING', CONDUCTIVITY=0.10, DENSITY=65., EMISSIVITY=0.9, SPECIFIC_HEAT=1.14, /\n"
    #txt = txt+"&MATL ID='BACKING', CONDUCTIVITY=0.2, DENSITY=585., EMISSIVITY=1., SPECIFIC_HEAT=0.8, /\n"
    txt = txt+"&MATL ID='SAMPLE', CONDUCTIVITY=%0.4f, DENSITY=%0.1f, EMISSIVITY=%0.4f, SPECIFIC_HEAT=%0.4f, /\n"%(conductivity, density, emissivity, specific_heat)
    
    prevTime=-1e6
    for i in range(0, len(time)):
        if (time[i]-prevTime) < 0.0001:
            #txt = txt+"&RAMP ID='CONE-RAMP', T=%0.4f, F=%0.1f, /\n"%(time[i]-time[0]+0.0001, hrrpua[i])
            pass
        else:
            txt = txt+"&RAMP ID='CONE-RAMP', T=%0.4f, F=%0.1f, /\n"%(time[i]-time[0], hrrpua[i])
        prevTime = time[i]
    y = -0.05
    for i, hf in enumerate(fluxes):
        hf_ign = estimateHrrpua(cone_hf_ref, hrrpua_ref, hf, 'Froude', qflame_fixed)
        
        delta = deltas[i]
        if i%3 == 0: y = y + 0.1
        XYZ = [((i % 3))*0.1+0.05, y, 0.0]
        XB = [XYZ[0]-0.05, XYZ[0]+0.05, XYZ[1]-0.05, XYZ[1]+0.05, 0.0,0.0]
        
        namespace = '%02d-%03d'%(hf, delta*1e3)
        
        txt = txt+"&SURF ID='SAMPLE-%s', EXTERNAL_FLUX=1., "%(namespace)
        txt = txt+"HEAT_TRANSFER_COEFFICIENT=%0.4f, HEAT_TRANSFER_COEFFICIENT_BACK=10., "%(front_h)
        txt = txt+"HRRPUA=1., IGNITION_TEMPERATURE=%0.1f, MATL_ID(1:2,1)='SAMPLE','BACKING', "%(Tign)
        txt = txt+"RAMP_EF='IGNITION_RAMP-%s', RAMP_Q='CONE-RAMP', "%(namespace)
        txt = txt+"REFERENCE_HEAT_FLUX=%0.4f, REFERENCE_HEAT_FLUX_TIME_INTERVAL=1., REFERENCE_CONE_THICKNESS=%0.8f, "%(qref, cone_d_ref)
        txt = txt+'THICKNESS(1:2)=%0.8f,%0.8f, /\n'%(delta, 0.0254/2)
        
        if ignitionMode == 'Temperature':
            txt = txt+"&RAMP ID='IGNITION_RAMP-%s', T=%0.1f, F=%0.4f, DEVC_ID='IGNITION_DEVC-%s', /\n"%(namespace, 0.0, hf, namespace)
            txt = txt+"&RAMP ID='IGNITION_RAMP-%s', T=%0.1f, F=%0.4f, /\n"%(namespace, 1.0, hf_ign)
        else:
            txt = txt+"&RAMP ID='IGNITION_RAMP-%s', T=%0.1f, F=%0.4f, /\n"%(namespace, 0.0, hf_ign)
            txt = txt+"&RAMP ID='IGNITION_RAMP-%s', T=%0.1f, F=%0.4f, /\n"%(namespace, 1.0, hf_ign)
        
        txt = txt+"&OBST ID='SAMPLE-%s', SURF_ID='SAMPLE-%s', XB="%(namespace, namespace)
        for x in XB:
            txt = txt+"%0.4f,"%(x)
        if ignitionMode == 'Time':
            txt = txt+"DEVC_ID='TIGN-%s'"%(namespace)
        txt = txt+', /\n'
        
        txt = txt+"&DEVC ID='WALL TEMPERATURE-%s', INITIAL_STATE=.FALSE., IOR=3, OUTPUT=%s, "%(namespace, tempOutput)
        txt = txt+"QUANTITY='WALL TEMPERATURE', SETPOINT=%0.1f, XYZ=%0.4f,%0.4f,%0.4f, /\n"%(Tign, XYZ[0], XYZ[1], XYZ[2])
        
        txt = txt+"&CTRL ID='IGNITION-CTRL-%s', FUNCTION_TYPE='ANY', INPUT_ID='WALL TEMPERATURE-%s', /\n"%(namespace, namespace)
        if ignitionMode == 'Time':
            txt = txt+"&DEVC ID='TIGN-%s', XYZ=0,0,0, SETPOINT=%0.4f, QUANTITY='TIME', INITIAL_STATE=.FALSE., /\n"%(namespace, case_tigns[i])
            
        txt = txt+"&DEVC ID='IGNITION_DEVC-%s', CTRL_ID='IGNITION-CTRL-%s', IOR=3, OUTPUT=.FALSE., QUANTITY='CONTROL', "%(namespace,namespace)
        txt = txt+"XYZ=%0.4f,%0.4f,%0.4f, /\n"%(XYZ[0], XYZ[1], XYZ[2])
        
        txt = txt+"&DEVC ID='HRRPUA-%s', IOR=3, QUANTITY='HRRPUA', SPEC_ID='PROPANE', "%(namespace)
        txt = txt+"XYZ=%0.4f,%0.4f,%0.4f, /\n"%(XYZ[0], XYZ[1], XYZ[2])
        
        txt = txt+"&DEVC ID='IGNITION-TIME-%s', NO_UPDATE_DEVC_ID='IGNITION_DEVC-%s', OUTPUT=.FALSE., "%(namespace,namespace)
        txt = txt+"QUANTITY='TIME', XYZ=%0.4f,%0.4f,%0.4f, /\n"%(XYZ[0], XYZ[1], XYZ[2])
        
                        
    return txt


def runModel(outdir, outfile, mpiProcesses, fdsdir, fdscmd, printLiveOutput=False):
    ''' This function will run fds with an input file
    '''
    my_env = os.environ.copy()
    my_env['I_MPI_ROOT'] = fdsdir+"\\mpi"
    my_env['PATH'] = fdsdir + ';' + my_env['I_MPI_ROOT'] + ';' + my_env["PATH"]
    my_env['OMP_NUM_THREADS'] = '1'
    
    process = subprocess.Popen([fdscmd, outfile, ">&", "log.err"], cwd=r'%s'%(outdir), env=my_env, shell=False, stdout=subprocess.DEVNULL)
    
    out, err = process.communicate()
    errcode = process.returncode   
    return out, err, errcode

def findHeaderLength(lines):
    ''' This is a helper function to dynamically find the
    length of a header in csv data
    '''
    counter = 0
    headerCheck = True
    while headerCheck and counter < 100:
        line = (lines[counter].decode('utf-8')).replace('\r\n','')
        while line[-1] == ',': line = line[:-1]
        try:
            [float(y) for y in line.split(',')]
            counter = counter - 1
            headerCheck = False
        except:
            counter = counter + 1
    if counter < 100:
        return counter
    else:
        print("Unable to find header length, returning 0")
        return 0

def cleanDataLines(lines2, headerLines):
    ''' This is a helper function to clean data rows
    '''
    lines = lines2[headerLines+1:]
    for i in range(0, len(lines)):
        line = (lines[i].decode('utf-8')).replace('\r\n','')
        while line[-1] == ',': line = line[:-1]
        lines[i] = [float(y) for y in line.split(',')]
    return lines

def load_csv(modeldir, chid, suffix='_devc', labelRow=-1):
    ''' This function imports a csv output by FDS
    '''
    file = "%s%s%s%s.csv"%(modeldir, os.sep, chid, suffix)
    f = open(file, 'rb')
    lines = f.readlines()
    f.close()
    headerLines = findHeaderLength(lines)
    if labelRow == -1:
        header = (lines[headerLines].decode('utf-8')).replace('\r\n','').replace('\n','').split(',')
    else:
        header = (lines[labelRow].decode('utf-8')).replace('\r\n','').replace('\n','').split(',')
    dataLines = cleanDataLines(lines, headerLines)
    data = pd.DataFrame(dataLines, columns=header,)
    return data

def estimateExposureFlux(coneExposure, representativeHRRPUA, method, fixed_qflame=25):
    ''' Estimates the exposure flux at a cone exposure by using
    an empirical estimate of the flame heat flux. The empirical
    estimate of the flame heat flux was developed by running
    steady-state FDS simulations at fixed HRRPUAs in a cone
    calorimeter configuration. The simulations used a fixed
    surface heat transfer coefficient of 15 W/m2-K and a fixed
    gas phase radiative fraction of 0.35.
    '''
    
    qflame = calculateFlameHeatFlux(representativeHRRPUA, method, fixed_qflame=fixed_qflame)
    return qflame + coneExposure

def getRepresentativeHrrpua(HRRPUA, time, factor=0.5):
    ''' This function calculates a representative HRRPUA for use
    in estimating the flame heat flux. HRR contains a time series
    of HRRPUA data from a cone calorimeter experiment. All data
    points above a threshold percentile (default 50%) are used
    to calculate the average HRRPUA. The threshold is intended
    to remove leading and trailing parts of the curve which
    would artificially lower the HRRPUA. The threshold of 50%
    is arbitrary but provides reasonable agreement on the cases
    evaluated here.
    '''
    dts = time[1:]-time[:-1]
    dt = np.min(dts[np.nonzero(dts)])
    tmin = time.min()
    tmax = time.max()
    time_i = np.linspace(tmin, tmax, int((tmax-tmin)/dt + 1))
    hrrpua_i = np.interp(time_i, time, HRRPUA)
    representativeHRRPUA = hrrpua_i[hrrpua_i > HRRPUA.max()*factor].mean()
    return representativeHRRPUA

def estimateHrrpua(cone_ref, hrrpua_ref, cone_flux, method, fixed_qflame=25):
    ''' Calculate the scaled heat flux based on the reference hrrpua,
    reference cone exposure, and scaled cone exposure. An iterative
    process is used since the flame heat flux depends on the scaled
    hrrpua.
    '''
    
    exposureFlux = estimateExposureFlux(cone_ref, hrrpua_ref, method, fixed_qflame)
    scaledFlux = exposureFlux - cone_ref + cone_flux
    
    prevFlux = scaledFlux
    scaled_hrrpua = (scaledFlux/exposureFlux)*hrrpua_ref
    scaledFlux = estimateExposureFlux(cone_flux, scaled_hrrpua, method, fixed_qflame)
    
    while abs(prevFlux - scaledFlux) > 0.01:
        prevFlux = scaledFlux
        scaled_hrrpua = (scaledFlux/exposureFlux)*hrrpua_ref
        scaledFlux = estimateExposureFlux(cone_flux, scaled_hrrpua, method, fixed_qflame)
    return scaledFlux

def cleanTrailingTimes(time, value):
    ''' Removes trailing data from a time series, typically nan from
    import on a shared csv file.
    '''
    tmax = np.nanmax(time)
    while time[-1] < tmax:
        time = time[:-1]
        value = value[:-1]
    return time, value

def interpolateTimeSeriesToDt(time, value, dt):
    ''' Interpolates time series to a fixed dt
    '''
    t, v = cleanTrailingTimes(time, value)
    tmin = np.floor(np.nanmin(t)/dt)*dt
    tmax = np.ceil(np.nanmax(t)/dt)*dt
    t1 = np.linspace(tmin, tmax, int((tmax-tmin)/dt + 1))
    v1 = np.interp(t1, t, v)
    return t1, v1   

def timeAverage(time, value, window):
    ''' Calculate the time averaged value for a quantity.
    
    Checks to make sure the data does not have nans or trailing zeros
    at the end.
    
    Interpolates to a finer time interval first to ensure that the
    data has a constant dt.
    
    Averaging is computed through a convolution with a uniform distribution.
    '''
    tmax = np.nanmax(time)
    tmin = np.nanmin(time)
    if window > (tmax-tmin):
        print("Warning, windowSize > time interval. Not time averaging.")
        return time, value
    
    time, value = cleanTrailingTimes(time, value)
    dt = np.nanmin(time[1:] - time[:-1])
    time, value = interpolateTimeSeriesToDt(time, value, dt)
    
    N = int(np.round(window/dt))
    f = np.zeros((N)) + 1
    f = f/f.sum()
    value = np.convolve(value, f, mode='same')
    return time, value

def getTimeAveragedPeak(times, values, windowSize):
    ''' Returns peak of a time averaged curve
    '''
    t, v = timeAverage(times, values, windowSize)
    return np.nanmax(v)

def truncateTrailingHrrpua(times, hrrpuas, truncateHRRPUA):
    ''' Truncate trailing hrrpua which is less than threshold
    '''
    try:
        ind = np.where(hrrpuas > truncateHRRPUA)[0][-1]
    except:
        ind = -1
    times = times[:ind]
    hrrpuas = hrrpuas[:ind]
    return times, hrrpuas

def getTimeAveragedEnergy(times, hrrpuas, windowSize, percentile, truncateHRRPUA=10):
    ''' Calculates the time at which a percentile of energy is
    consumed.
    
    Applies truncation if requested by truncateHRRPUA.
    Time averages data to common interval.
    
    '''
    times, hrrpuas = truncateTrailingHrrpua(times, hrrpuas, truncateHRRPUA)
    t, v = timeAverage(times, hrrpuas, windowSize)
    energy_with_time = np.zeros_like(t)
    energy_with_time[1:] = np.cumsum(v[1:]*(t[1:]-t[:-1]))
    energyThreshold = energy_with_time[-1]*percentile/100
    ind = np.argwhere(energy_with_time > energyThreshold)[0][0]
    
    return energyThreshold, t[ind]

'''
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
'''



def getTimeAveraged_timeToEnergy(times, hrrpuas, windowSize, energyThreshold,  truncateHRRPUA=10):
    t, v = truncateTrailingHrrpua(times, hrrpuas, truncateHRRPUA)
    t, v = timeAverage(t, v, windowSize)
    energy_with_time = np.zeros_like(t)
    energy_with_time[1:] = np.cumsum(v[1:]*(t[1:]-t[:-1]))
    try:
        ind = np.argwhere(energy_with_time > energyThreshold)[0][0]
    except:
        ind = -1
    
    return t[ind], timeAverage


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
    ''' Interpolate experimental data to a common dt. Will use the minimum dt
    in the dataset unless a target dt is specified. Will apply time averaging
    on a fixed number of dt if filterWidth is specified.
    '''
    if targetDt is not False:
        dt = targetDt
    else:
        dt = np.nanmin(times[1:]-times[:-1])
    t, v = interpolateTimeSeriesToDt(times, HRRs, dt)
    
    if filterWidth is not False:
        window = filterWidth*dt
        t, v = timeAverage(t, v, window)
        
    tmax = np.ceil(np.nanmax(times)/dt)*dt
    tmin = np.floor(np.nanmin(times)/dt)*dt
    targetTimes = np.linspace(tmin, tmax, int((tmax-tmin)/dt + 1))
    HRRs = np.interp(targetTimes, t, v)
    return targetTimes, HRRs


def getMaterials(material=False, dataDirectory="..//data", namespace="*spec_file.csv"):
    ''' Builds a dictionary of all available materials based on specification
    files located in dataDirectory.
    '''
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

            cases = dict()
            for ii in range(0, len(validationTimeColumns)):
                casename = 'case-%03d'%(ii)
                cases[casename] = {'Time': validationTimeColumns[ii], 'HRR': validationHrrpuaColumns[ii], 'delta': thicknesses[ii], 'cone': fluxes[ii], 'case': casename}
            
            case_basis = dict()
            for ii in range(0, len(referenceTimeColumns)):
                casename = 'case-1%03d'%(ii)
                case_basis[casename] = {'Time': referenceTimeColumns[ii], 'HRR': referenceHrrpuaColumns[ii], 'delta': referenceThicknesses[ii], 'cone': referenceExposures[ii], 'case': casename}
        
            spec_file_dict[m] = {'density': density, 'conductivity': conductivity, 'specific_heat': specific_heat,
                                        'heat_of_combustion': heat_of_combustion, 'emissivity': emissivity, 'nu_char': nu_char,
                                        'data': exp_data, 'cases': cases, 'case_basis': case_basis, 
                                        'material': m, 'materialClass': materialClass}
    return spec_file_dict
            

def getDimensionlessNumbers(qr, eps, d1, t, conductivity, density, specific_heat):
    params = getFixedModelParams()
    d_min = params['d_min']
    if isinstance(d1, np.ndarray):
        d1[d1 < d_min] = d_min
    else:
        if d1 < d_min: d1 = d_min
    
    Tg = (qr/(eps*params['sig']))**0.25
    Ts = params['Tinit']
    hr = params['sig']*eps*(Ts**2 + Tg**2)*(Ts + Tg)
    #hr = sig*eps*(Tg**2)*Tg
    ht = params['hc'] + hr
    
    Bi = (ht*1000) /(conductivity/d1) # +0.1/0.0127)
    Fo = (conductivity / (density*specific_heat)) * t/((d1)**2)
    
    return Bi, Fo, Ts, ht

def processSingleCase(c, data):
    ''' Cleans up experimental data for a single case.
    '''
    times = data[c['Time']].values
    HRRs = data[c['HRR']].values
    
    if len(HRRs.shape) == 2:
        HRRs = HRRs[:, 0]
        times = times[:, 0]
    targetTimes, HRRs_interp = interpolateExperimentalData(times, HRRs, targetDt=15, filterWidth=False)
    tign, times_trimmed, hrrs_trimmed = findLimits(times, HRRs, 0.001, 0.99)
    
    '''
    tmp = (HRRs*0.1016*0.1016)
    tmp[np.isnan(tmp)] = 0
    times[np.isnan(times)] = 0
    totalEnergy = np.trapz(tmp,  times)
    '''
    tmp = (hrrs_trimmed*0.1016*0.1016)
    tmp[np.isnan(tmp)] = 0
    times_trimmed[np.isnan(times_trimmed)] = 0
    totalEnergy = np.trapz(tmp,  times_trimmed)
    
    c['tign'] = tign
    c['times'] = times
    c['HRRs'] = HRRs
    c['times_trimmed'] = times_trimmed
    c['hrrs_trimmed'] = hrrs_trimmed
    c['totalEnergy'] = totalEnergy
    return c

def processCaseData(mat):
    ''' Cleans up all experimental data for a single material.
    '''
    for c in list(mat['cases'].keys()):
        mat['cases'][c] = processSingleCase(mat['cases'][c], mat['data'])
    
    for c in list(mat['case_basis'].keys()):
        mat['case_basis'][c] = processSingleCase(mat['case_basis'][c], mat['data'])
    return mat






def getMaterialClass(material):
    ''' Establishes material class based on a simple set of text search rules.
    '''
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

def calculateThicknessFromHrr(hrrs_trimmed, times_trimmed, mat, c):
    ''' Calculates the char fraction based on the fraction of total energy
    consumed with time and the specified char fraction. Uses estimated char
    fraction to calculate a mixture density and return a thickness.
    '''
    (nu_char, density, HoC, material) = (mat['nu_char'], mat['density'], mat['heat_of_combustion'], mat['material'])
    (delta0, totalEnergy) = (c['delta'], c['totalEnergy'])
    
    params = getFixedModelParams()
    cone_area = params['cone_area']
    char_density = params['char_density']
    d_min = params['d_min']
    delta = np.zeros_like(hrrs_trimmed) + delta0
    energyFraction = np.zeros_like(delta)
    charFraction = np.zeros_like(delta)
    
    mass = np.zeros_like(hrrs_trimmed) + delta0*density
    mass[1:] = delta0*density - np.cumsum(hrrs_trimmed[1:]/(1e3*HoC)*(times_trimmed[1:]-times_trimmed[:-1]))
    
    energy = 0
    warningPrinted = True
    for j in range(1, hrrs_trimmed.shape[0]):
        energy += hrrs_trimmed[j]*cone_area/(times_trimmed[j]-times_trimmed[j-1])
        energyFraction[j] = energy / totalEnergy
        charFraction[j] = energyFraction[j]*nu_char
        
        mix_density = char_density*charFraction[j] + density*(1-charFraction[j])
        delta[j] = mass[j] / mix_density
        if delta[j] < d_min:
            delta[j] = params['d_min']
            if warningPrinted is False:
                print("Warning %s case %s has calculated thickness less than 0"%(material, c['case']))
            warningPrinted = True
    return delta, charFraction, mass

def calculateFlameHeatFlux(hrrpua, method, fixed_qflame=25):
    ''' Returns the flame heat flux in a cone calorimeter based on hrrpua.
    '''
    params = getFixedModelParams()
    cone_area = params['cone_area']
    cone_diameter = params['cone_diameter']
    xr = params['xr']
    qa = params['qa']
    sig = params['sig']
    Tf = params['Tf']
    xA = params['xA']
    if method == 'Froude':
        if isinstance(hrrpua, np.ndarray): hrrpua[np.isnan(hrrpua)] = 0
        qstar = hrrpua*cone_area / (1100 * (cone_diameter**2.5))
        lm = -1.02*cone_diameter + 3.7*cone_diameter * (qstar**0.4)
        
        kf = np.log(1-(xr*qa*lm)/(3.6*sig*(Tf**4)*xA))/lm
        
        qflame = sig*(Tf**4)*(1-np.exp(kf*lm))
    elif method == 'Empirical':
        qflame = 55*(hrrpua**0.065) - 50
    elif method == 'Fixed':
        qflame = np.zeros_like(hrrpua) + fixed_qflame
    return qflame

def getFlameMethodFromNonDimType(nondimtype):
    if nondimtype in ['Fo','FoBi','Time']:
        flame_method = 'Froude'
    elif nondimtype in ['FoBi_simple', 'FoBi_simple_fixed_d']:
        flame_method = 'Fixed'
    return flame_method

def analyzeExpCase(mat, c, nondimtype):
    ''' Analyze a single experimental case. This includes calculating the
    thickness and mass from the measured hrrpua curve and heat of combustion,
    and calculating the non dimensional time scale.
    '''
    hrrs_trimmed, times_trimmed = c['hrrs_trimmed'], c['times_trimmed']
    delta, charFraction, mass = calculateThicknessFromHrr(hrrs_trimmed, times_trimmed, mat, c)
    
    flame_method = getFlameMethodFromNonDimType(nondimtype)
    
    coneExposure = c['cone']
    qflame = calculateFlameHeatFlux(hrrs_trimmed, flame_method, fixed_qflame=25)
    qr = qflame + coneExposure
    nondim_t, t = getExpNonDimensionalTime(qr, charFraction, delta, mat, c, nondimtype)
    return qr, mass, delta, nondim_t, t
    

def getExpNonDimensionalTime(qr, charFraction, delta, mat, c, nondimtype):
    ''' Calculates the non-dimensional time for a specific case.
    '''
    (emissivity) = (mat['emissivity'])
    (density, conductivity, specific_heat) = (mat['density'], mat['conductivity'], mat['specific_heat'])
    params = getFixedModelParams()
    d_min = params['d_min']
    (char_density, char_conductivity) = (params['char_density'], params['char_conductivity'])
    
    (times_trimmed, tign, delta0) = (c['times_trimmed'], c['tign'], c['delta'])
    
    Ts = np.zeros_like(qr) + params['Tinit']
    t = times_trimmed - tign
    
    if nondimtype in ['Fo','FoBi','Time']:
        mix_conductivity = char_conductivity*charFraction + conductivity*(1-charFraction)
        mix_density = char_density*charFraction + density*(1-charFraction)
        Bi, Fo, Ts, ht = getDimensionlessNumbers(qr, emissivity, delta, t, mix_conductivity, mix_density, specific_heat)
        if nondimtype == 'Fo': nondim_t = Fo
        if nondimtype == 'FoBi': nondim_t = Fo*Bi
        if nondimtype == 'Time': nondim_t = t
    elif nondimtype in ['FoBi_simple', 'FoBi_simple_fixed_d']:
        #hr = 0.0154*((qr*1000)**0.75)/1000
        hr = 0.0154*((qr*1000))/1000
        d = delta if (nondimtype == 'FoBi_simple') else np.zeros_like(delta) + delta0
        d[d < d_min] = d_min
        nondim_t = hr*t/(density*specific_heat*d)
    return nondim_t, t
    

def analyzeBasisCases(mat, nondimtype='FoBi'):
    ''' Analyzes all experimental basis cases.
    '''
    cases = mat['case_basis']
    case_basis = list(cases.keys())
    (qr, mass, delta, nondim_t, t) = (dict(), dict(), dict(), dict(), dict())
    
    for i, c in enumerate(case_basis):
        qr[c], mass[c], delta[c], nondim_t[c], t[c] = analyzeExpCase(mat, cases[c], nondimtype)
    return qr, mass, delta, nondim_t, t


def interpolateBasisCases(mat, qr1, mass1, delta1, nondim_t1, t1, nondimtype):
    ''' Interpolates all basis cases to a common non-dimensional time interval.
    '''
    # Find time limits
    (tmax, nondim_time_max, nondim_time_min) = (-1, -1, 1e15)
    for c in list(t1.keys()):
        nondim_time_max = max([nondim_time_max, np.nanmax(nondim_t1[c][np.isfinite(nondim_t1[c])])])
        nondim_time_min = min([nondim_time_min, np.nanmin(nondim_t1[c][np.isfinite(nondim_t1[c])][1:])])
        tmax = max([tmax, t1[c].max()])
    
    # Initialize arrays
    case_basis = list(qr1.keys())
    len_array = 10001
    nondim_time_out = np.zeros((len_array, ))
    
    nondim_time_out[1:] = np.logspace(np.log10(nondim_time_min),np.log10(nondim_time_max), int(len_array-1))
    #nondim_time_out[1:] = np.logspace(-3,np.log(nondim_time_max), int(len_array-1))
    mlr_out = np.zeros((len_array,len(case_basis)))
    qrs_out = np.zeros((len_array,len(case_basis)))
    hogs_out = np.zeros((len_array,len(case_basis)))
    times_out = np.zeros((len_array,len(case_basis)))
    f = np.zeros((10))+1
    f = f / np.sum(f)
    
    # Get parameters
    flame_method = getFlameMethodFromNonDimType(nondimtype)
    HoC = mat['heat_of_combustion']
    
    nonnan_max = -1
    for i, c in enumerate(case_basis):
        t = t1[c]
        mass = mass1[c]
        nondim_t = nondim_t1[c]
        
        coneExposure = mat['case_basis'][c]['cone']
        
        mass2 = np.interp(nondim_time_out, nondim_t, mass, right=mass[-1])
        time2 = np.interp(nondim_time_out, nondim_t, t, right=np.nan)
        mlr = np.zeros_like(mass2)
        for j in range(0, time2.shape[0]-1):
            if np.isnan(time2[j+1]):
                break
            dt = time2[j+1]-time2[j]
            if dt <= 0.0:
                break
            mlr[j+1] = (mass2[j+1]-mass2[j])/dt
        mlr[np.isnan(mlr)] = 0
        
        hrrpua_out = mlr*-1000*HoC
        
        inds = mlr < 0
        
        qrs_out[:, i] = calculateFlameHeatFlux(hrrpua_out, flame_method) + coneExposure
        mlr_out[:, i] = mlr
        
        times_out[:, i] = time2
        hogs_out[inds, i] = -qrs_out[inds, i]/(mlr_out[inds, i])
        hogs_out[~inds, i] = np.nan
        
        for jj in range(0, len(time2)):
            if time2[-jj] < time2[-jj-1]:
                hogs_out[-jj, i] = np.nan
        
        try:
            ind = np.where(~np.isnan(hogs_out[:, i]))[0][-1]
            nonnan_max = max([nonnan_max, nondim_time_out[ind]])
        except:
            pass
    inds = np.where(nondim_time_out <= nonnan_max)[0]
    inds = inds[1:]
    return nondim_time_out[inds], hogs_out[inds, :], qrs_out[inds, :], mlr_out[inds, :], times_out[inds, :]

def plotBasisCases(mat, times, hogs, nondim_t, nondimtype, lw, colors, labelPlot):
    case_basis = list(mat['case_basis'].keys())
    if colors is False: colors = getJHcolors()
    for i, c in enumerate(case_basis):
        delta0 = mat['case_basis'][c]['delta']
        coneExposure = mat['case_basis'][c]['cone']
        
        if labelPlot:
            label = r"$\Delta=%0.1f \mathrm{mm}, q''_{cone}=%0.0f \mathrm{kW/m^{2}}$"%(delta0*1e3,coneExposure)
            color = colors[i+1]
        else:
            label = None
            color = colors[2]
        if nondimtype == 'Time':
            plt.loglog(times[:, i]/60, hogs[:, i], '-', linewidth=lw, label=label, color=color) #label=label, color=colors[i])
        else:
            plt.loglog(nondim_t, hogs[:,i], '-', linewidth=lw, label=label, color=color) #label=label, color=colors[i])

def developRepresentativeCurve(mat, nondimtype='FoBi', plot=False, lw=3, colors=False, labelPlot=False):
    
    # Analyze basis cases
    qr1, mass1, delta1, nondim_t1, t1 = analyzeBasisCases(mat, nondimtype)
    
    # Interpolate basis cases to common time interval
    nondim_t, hogs, qrs, mlrs, times = interpolateBasisCases(mat, qr1, mass1, delta1, nondim_t1, t1, nondimtype)
    #(qr1, mass1, delta1, nondim_t1, t1) = (qr, mass, delta, nondim_t, t)
    
    # Plot basis cases if required
    if plot: plotBasisCases(mat, times, hogs, nondim_t, nondimtype, lw, colors, labelPlot)
    
    # Determine representative HoG curve
    qrs_out = np.nanmean(qrs, axis=1)
    hog_out = np.nanmedian(hogs, axis=1)
    
    try:
        ind = np.where(~np.isnan(hog_out))[0][-1]
    except:
        ind = -1
    return nondim_t[:ind], hog_out[:ind], qrs_out[:ind], mlrs[:ind, :], times[:ind, :]

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
    d_min = 1e-15
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


def runSimulation(times, mat, delta0, coneExposure, totalEnergy, fobi_out, hog_out, nondimtype='FoBi'):
    # Extract material parameters
    (density, conductivity, specific_heat) = (mat['density'], mat['conductivity'], mat['specific_heat'])
    (HoC, emissivity, nu_char) = (mat['heat_of_combustion'], mat['emissivity'], mat['nu_char'])
    
    # Extract fixed params
    params = getFixedModelParams()
    
    delta = np.zeros_like(times) + float(delta0) #/1000
    hrrpuas = np.zeros_like(times)
    mass = np.zeros_like(times)
    mass[0] = delta[0]*density
    energyFraction = np.zeros_like(delta)
    charFraction = np.zeros_like(delta)
    
    relaxation_factor = 0.5
    refs = np.zeros_like(delta)
    Fos = np.zeros_like(delta)
    Bios = np.zeros_like(delta)
    mlrs = np.zeros_like(delta)
    qrs = np.zeros_like(delta)
    energy = np.zeros_like(delta)
    cone_area = params['cone_area']
    char_conductivity = params['char_conductivity']
    char_density = params['char_density']
    d_min = params['d_min']
    
    flame_method = getFlameMethodFromNonDimType(nondimtype)
    
    for j in range(1, times.shape[0]):
        t = times[j]
        d1 = delta[j-1]
        
        mix_conductivity = char_conductivity*charFraction[j-1] + conductivity*(1-charFraction[j-1])
        mix_density = char_density*charFraction[j-1] + density*(1-charFraction[j-1])
        
        qr = calculateFlameHeatFlux(hrrpuas[j-1], flame_method) + coneExposure
        
        Bi, Fo, Ts, ht = getDimensionlessNumbers(qr, emissivity, d1, t, mix_conductivity, mix_density, specific_heat)
        
        if nondimtype == 'Fo': nondim_t = Fo
        if nondimtype == 'FoBi': nondim_t = Fo*Bi #*1.2
        if nondimtype == 'Time': nondim_t = t
        if nondimtype in ['FoBi_simple', 'FoBi_simple_fixed_d']:
            #hr = 0.0154*((qr*1000)**0.75)/1000
            hr = 0.0154*((qr*1000))/1000
            if nondimtype == 'FoBi_simple_fixed_d': d = delta0
            if d < d_min: d = d_min
            nondim_t = hr*t/(density*specific_heat*d)
        ref = np.interp(nondim_t, fobi_out, hog_out)
        
        if np.isnan(ref) or ref == 0:
            (ref, mlr) = (0.0, 0.0)
        else:
            mlr = -qr/ref
            mlr = mlr*relaxation_factor + mlrs[j-1]*(1-relaxation_factor)
        if mlr > 0: mlr = 0
        
        refs[j] = ref
        Fos[j] = Fo
        Bios[j] = Bi
        qrs[j] = qr
        
        dt = times[j]-times[j-1]
        
        mlrs[j] = mlr
        
        hrrpuas[j] = mlr*HoC*-1000
        energy[j] = energy[j-1]+(hrrpuas[j]*cone_area)*dt
        #energy[j] = np.trapz(hrrpuas*cone_area, times) ##energy[j-1]+(hrrpuas[j]*cone_area)*dt
        
        if mlr > 0: mlr = 0
        mass[j] = mass[j-1] + mlr*dt
        
        if mass[j] < 0:
            mass[j] = 0
        if np.isnan(mass[j]):
            mass[j] = 0
        hrrpuas[j] = (mass[j-1]-mass[j])/dt * HoC*1000
        if np.isnan(hrrpuas[j]):
            hrrpuas[j] = 0
        mlrs[j] = (mass[j] - mass[j-1])/dt
        
        if energy[j] > totalEnergy: 
            hrrpuas[j] = 0
            mlrs[j] = 0
            mass[j] = mass[j-1]
            energy[j:] = totalEnergy
            hrrpuas[j:] = 0
            mlrs[j:] = 0
            mass[j:] = mass[j-1]
            charFraction[j:] = 1.
            
        energyFraction[j] = energy[j] / totalEnergy
        charFraction[j] = energyFraction[j]*nu_char
        #mix_density = char_density*charFraction[j-1] + density*(1-charFraction[j-1])
        
        
        delta[j] = mass[j]/mix_density
        
        '''
        if mlrs[j] == 0 and np.nanmax(abs(mlrs[j])) > 0:
            counter += 1
        
        if counter > 10:
            energy[j:] = energy[j]
            break
        '''
        #print('%0.1f\t%0.1f\t%0.1f\t%0.1f\t%0.1f\t%0.1f\t%0.1f\t%0.1f\t%0.4f\t%0.1f\t%0.1f\t%0.4f'%(qr, Tg, ht, Bi, Fo, hrrpuas[j], mass[j-1], mass[j], mlr, mlr*dt,dt,delta[j]))
    if energy[-1] > totalEnergy:
        runSimulation(times, mat, delta0, coneExposure, energy[-1]*1.01, fobi_out, hog_out, nondimtype='FoBi')
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
        ff = f[mask]
    else:
        xx = x
        yy = y
        ff = f
    delta, sigma_m, sigma_e, num_points, points = calculateUncertaintyBounds(xx, yy, ff, split=False)
    
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
    colors2 = getNewColors()
    
    mew = 3
    if diff is not None:
        cases = np.array(list(set(diff)))
        cases.sort()
        for j in range(0, len(ff)):
            caseInd = np.where(cases == diff[j])[0][0]
            #c = 0 if diff[j] > 0 else 1
            ax.scatter(x[j], y[j], marker=markers[caseInd], s=100, facecolors='none', edgecolors=colors2[caseInd], linewidths=mew)
        customMarkers = []
        for caseInd, case in enumerate(cases):
            if labelName is None:
                customMarkers.append(Line2D([0],[0],marker=markers[caseInd], color='w', markeredgecolor=colors2[caseInd], markerfacecolor='w', label=case, markersize=15, markeredgewidth=mew))
            else:
                customMarkers.append(Line2D([0],[0],marker=markers[caseInd], color='w', markeredgecolor=colors2[caseInd], markerfacecolor='w', label=labelName[case], markersize=15, markeredgewidth=mew))
        
        ax.legend(handles=customMarkers, fontsize=fs)
    else:
        ax.scatter(x, y, s=100)
    plt.xlabel(r'Measured %s'%(label), fontsize=fs)
    plt.ylabel(r'Predicted %s'%(label), fontsize=fs)
    
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

def spyroScaling(ref_time, ref_hrrpua, ref_delta, ref_cone, delta, cone, tign=0, qflame='Empirical', referenceTimes=False, averageWindow=False):
    if qflame == 'Empirical':
        h_ref = getRepresentativeHrrpua(ref_hrrpua, ref_time)
        q1 = estimateExposureFlux(ref_cone, h_ref)
        q2 = estimateHrrpua(ref_cone, h_ref, cone)
    else:
        q1 = ref_cone + qflame
        q2 = cone + qflame
    scaled_t = np.zeros((ref_time.shape[0]+4))
    scaled_hrrpua = np.zeros_like(scaled_t)
    scaled_t[2:-2] = ref_time*(delta/ref_delta) * (q1/q2) + tign
    scaled_t[1] = scaled_t[2]*0.99
    scaled_t[-2] = scaled_t[-3]*1.01
    scaled_t[-1] = scaled_t[-2]*2
    scaled_hrrpua[2:-2] = ref_hrrpua*(q2/q1)
    if referenceTimes is not False:
        scaled_hrrpua = np.interp(referenceTimes, scaled_t, scaled_hrrpua)
        scaled_t = referenceTimes
    if averageWindow is not False:
        scaled_t, scaled_hrrpua = timeAverage(scaled_t, scaled_hrrpua, averageWindow)
    return scaled_t, scaled_hrrpua
