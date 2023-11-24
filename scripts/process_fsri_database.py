import os
import glob
import numpy as np
import pandas as pd

from algorithms import getMaterialClass

def extractConeAnalysisData(values, scaled_times, tigns, uniqueFluxes):
    coneAnalysisData = dict()
    for i, flux in enumerate(uniqueFluxes):
        
        v = values[i]
        t = scaled_times[i]
        tign = tigns[i]
        coneAnalysisData[flux] = dict()
        coneAnalysisData[flux]['peakHRRPUA'] = np.max(v)
        coneAnalysisData[flux]['timeToPeak'] = t[np.argmax(v)]-tign
        
        tMax = int(np.ceil(t.max()))
        tMax = max([tMax, 60])
        t2 = np.linspace(0, tMax, tMax+1)
        v2 = np.interp(t2, t, v)
        
        dt = t2[1] - t2[0]
        
        filters = [int(60/dt), int(180/dt), int(300/dt)]
        filteredValues = []
        for f in filters:
            fil = np.ones(f)/float(f)
            vfilt = np.convolve(v2, fil, mode='same')
            filteredValues.append(np.max(vfilt))
        
        coneAnalysisData[flux]['avg60s'] = filteredValues[0]
        coneAnalysisData[flux]['avg180s'] = filteredValues[1]
        coneAnalysisData[flux]['avg300s'] = filteredValues[2]
    return coneAnalysisData

def getConeData(material, directory):
    p = os.path.abspath(directory)
    files = glob.glob(p+os.sep+'cone_H*')
    files = [x for x in files if "Scalar" not in x]
    
    cad_prop = pd.read_csv(p+os.sep+material+'_Cone_Analysis_Data.csv', index_col=0)
    try:
        tmp_prop = pd.read_csv(p+os.sep+material+'_Ignition_Temp_Properties.csv', header=None, index_col=0).T
    except:
        tmp_prop = pd.read_csv(p+os.sep+'Ignition_Temp_Properties.csv', header=None, index_col=0).T
    density = tmp_prop['Density (kg/m3)'].values[0]
    coneData = dict()
    for j, file in enumerate(files):
        dc = pd.read_csv(file)
        rev = file.split('_')[-1].split('.csv')[0]
        HF = file.split('HF')[1].split('Scalar')[0].split('_')[0]
        f = material + '_Cone_HF' + HF + 'Scalar_*_' + rev + '.csv'
        d = glob.glob(os.path.join(directory, f))[0]
        ds = pd.read_csv(d, header=None, index_col=0)
        
        namespace = file.split('cone_')[-1].split('.csv')[0]
        
        coneData[j] = dict()
        coneData[j]['flux'] = float(ds.loc['HEAT FLUX', 1]) # kW/m2
        coneData[j]['mass'] = float(ds.loc['SPECIMEN MASS', 1]) / 1000 # kg
        coneData[j]['area'] = float(ds.loc['SURF AREA', 1]) # m2
        coneData[j]['thickness'] = coneData[j]['mass'] / (density*coneData[j]['area'])
        coneData[j]['time'] = dc['Time']
        coneData[j]['hrrpua'] = dc['HRRPUA']
        coneData[j]['timeMax'] = np.nanmax(dc['Time'])
        coneData[j]['timeInt'] = float(ds.loc['SCAN TIME', 1])
        coneData[j]['timeIgn'] = float(ds.loc['TIME TO IGN', 1]) # m2
        coneData[j]['peakHRRPUA'] = cad_prop.loc['Peak HRRPUA (kW/m2)', namespace]
        coneData[j]['timeToPeak'] = cad_prop.loc['Time to Peak HRRPUA (s)', namespace]
        coneData[j]['HRRPUA, 60s average'] = cad_prop.loc['Average HRRPUA over 60 seconds (kW/m2)', namespace]
        coneData[j]['HRRPUA, 180s average'] = cad_prop.loc['Average HRRPUA over 180 seconds (kW/m2)', namespace]
        coneData[j]['HRRPUA, 300s average'] = cad_prop.loc['Average HRRPUA over 300 seconds (kW/m2)', namespace]
        coneData[j]['HeatOfCombustion'] = cad_prop.loc['Avg. Effective Heat of Combustion (MJ/kg)', namespace]
        coneData[j]['InitialMass'] = cad_prop.loc['Initial Mass (g)', namespace]
        coneData[j]['FinalMass'] = cad_prop.loc['Final Mass (g)', namespace]
    
    coneData = pd.DataFrame(coneData).T
    
    return coneData

def timeAverageFsriConeData(coneData, flux, outInt):
    times = coneData.loc[coneData['flux'] == flux, 'time'].values
    hrrpuas = coneData.loc[coneData['flux'] == flux, 'hrrpua'].values
    
    tMax = np.max(coneData.loc[coneData['flux'] == flux, 'timeMax'].values)
    dt = np.min(coneData.loc[coneData['flux'] == flux, 'timeInt'].values)
    

def importFsriMaterial(p, material, outInt, uniqueFluxes, filterWidth=11):
    material_dict = dict()
    try:
        tmp_prop = pd.read_csv(p+os.sep+material+'_Ignition_Temp_Properties.csv', header=None, index_col=0).T
    except:
        tmp_prop = pd.read_csv(p+os.sep+'Ignition_Temp_Properties.csv', header=None, index_col=0).T
    
    conductivity = tmp_prop['Thermal Conductivity (W/m-K)'].values[0]
    heatCapacity = tmp_prop['Heat Capacity (J/kg-K)'].values[0]
    density = tmp_prop['Density (kg/m3)'].values[0]
    
    material_dict['conductivity'] = conductivity
    material_dict['heatCapacity'] = heatCapacity
    material_dict['density'] = density
    material_dict['directory'] = p
    material_dict['materialClass'] = getMaterialClass(material)
    coneData = getConeData(material, p)
    
    #uniqueFluxes = np.unique(coneData['flux'].values)
    fil = np.ones(filterWidth)/filterWidth
    
    for flux in uniqueFluxes:
        outInt2 = outInt
        dt = np.min(coneData.loc[coneData['flux'] == flux, 'timeInt'].values)
        tMax = np.max(coneData.loc[coneData['flux'] == flux, 'timeMax'].values)
        
        thickness = np.median(coneData.loc[coneData['flux'] == flux, 'thickness'].values)
        mass = np.median(coneData.loc[coneData['flux'] == flux, 'mass'].values)
        area = np.median(coneData.loc[coneData['flux'] == flux, 'area'].values)
        tign = np.median(coneData.loc[coneData['flux'] == flux, 'timeIgn'].values)
        peakHRRPUA = np.median(coneData.loc[coneData['flux'] == flux, 'peakHRRPUA'].values)
        timeToPeak = np.median(coneData.loc[coneData['flux'] == flux, 'timeToPeak'].values)
        avg60s = np.nanmedian(coneData.loc[coneData['flux'] == flux, 'HRRPUA, 60s average'].values)
        avg180s = np.nanmedian(coneData.loc[coneData['flux'] == flux, 'HRRPUA, 180s average'].values)
        avg300s = np.nanmedian(coneData.loc[coneData['flux'] == flux, 'HRRPUA, 300s average'].values)
        avgHoC = np.nanmedian(coneData.loc[coneData['flux'] == flux, 'HeatOfCombustion'].values)
        initial_mass = np.nanmedian(coneData.loc[coneData['flux'] == flux, 'InitialMass'].values)
        final_mass = np.nanmedian(coneData.loc[coneData['flux'] == flux, 'FinalMass'].values)
        
        times = coneData.loc[coneData['flux'] == flux, 'time'].values
        hrrpuas = coneData.loc[coneData['flux'] == flux, 'hrrpua'].values
        
        # Interpolate and filter raw data
        time = np.linspace(0, tMax, int(tMax/dt)+1)
        hrrpua = np.zeros_like(time)
        hrrpuas_interp = np.zeros((time.shape[0], len(times)))
        
        filterWidth=41
        fil = np.ones(filterWidth)/filterWidth
        
        for j in range(0, len(times)):
            hrrpuas_interp[:, j] = np.convolve(np.interp(time, times[j], hrrpuas[j]), fil, mode='same')
        
        # Find time to ignitions
        tIgns = []
        for j in range(0, len(times)):
            totalEnergy = np.trapz(hrrpuas_interp[:, j], time)
            totalEnergy2 = 0
            ind2 = 1
            while totalEnergy2 < 0.0001 * totalEnergy:
                ind2 = ind2 + 1
                totalEnergy2 = np.trapz(hrrpuas_interp[:ind2, j], time[:ind2])
            hrrpuas_interp[:, j] = np.interp(time, time, hrrpuas_interp[:, j])
            tIgns.append(time[ind2-1])
        tign = np.median(tIgns)
        
        # Average neglecting time to ignition
        hrrpuas_interp_notign = np.zeros_like(hrrpuas_interp)
        for j in range(0, len(times)):
            hrrpuas_interp_notign[:, j] = np.interp(time, time-tIgns[j], hrrpuas_interp[:, j])
        
        hrrpua = np.mean(hrrpuas_interp_notign, axis=1)
        
        #_, times_trim, hrrpuas_trim = findLimits(time+tign,hrrpua, 0.001, 0.99)
        
        #times_trim = np.append(np.array([0, tign*0.9]), times_trim)
        #hrrpuas_trim = np.append(np.array([0, 0]), hrrpuas_trim)
        
        #tMax = times_trim.max()+tign
        #targetTimes, HRRs_interp = interpolateExperimentalData(times, HR
        
        
        hrrpua_mn = np.min(hrrpuas_interp_notign, axis=1)
        hrrpua_mx = np.max(hrrpuas_interp_notign, axis=1)
        hrrpua_std = np.std(hrrpuas_interp_notign, axis=1)
        
        tMax = np.ceil((tMax-tign)/outInt2)*outInt2
        outTime = np.linspace(0, tMax, int(tMax/outInt2)+1)
        outHrrpua = np.interp(outTime, time, hrrpua)
        
        totalEnergy = np.trapz(hrrpua, time)
        totalEnergy2 = np.trapz(outHrrpua, outTime)
        
        while abs(totalEnergy2 - totalEnergy)/totalEnergy > 0.00001:
            #print(abs(totalEnergy2 - totalEnergy)/totalEnergy)
            outInt2 = outInt2*0.9
            outTime = np.linspace(0, tMax, int(tMax/outInt2)+1)
            outHrrpua = np.interp(outTime, time, hrrpua)
            totalEnergy2 = np.trapz(outHrrpua, outTime)
            if outInt2 <= 1:
                totalEnergy2 = totalEnergy
        
        outTime = np.append(np.array([0, tign*0.9]), outTime+tign)
        outHrrpua = np.append(np.array([0, 0]), outHrrpua)
        
        material_dict[flux] = dict()
        material_dict[flux]['time'] = outTime
        material_dict[flux]['hrrpua'] = outHrrpua
        material_dict[flux]['hrrpua_full_mean'] = hrrpua
        material_dict[flux]['hrrpua_full_std'] = hrrpua_std
        material_dict[flux]['hrrpua_full_min'] = hrrpua_mn
        material_dict[flux]['hrrpua_full_max'] = hrrpua_mx
        material_dict[flux]['time_full'] = time #-tign #-tStart
        material_dict[flux]['tIgn'] = tign
        material_dict[flux]['peakHRRPUA'] = peakHRRPUA
        material_dict[flux]['timeToPeak'] = timeToPeak
        material_dict[flux]['avg60s'] = avg60s
        material_dict[flux]['avg180s'] = avg180s
        material_dict[flux]['avg300s'] = avg300s
        material_dict[flux]['thickness'] = thickness
        material_dict[flux]['HeatOfCombustion'] = avgHoC
        material_dict[flux]['hrrpuas_interp'] = hrrpuas_interp
        material_dict[flux]['hrrpuas_interp_notign'] = hrrpuas_interp_notign
        material_dict[flux]['time_interp'] = time
        material_dict[flux]['initial_mass'] = initial_mass
        material_dict[flux]['final_mass'] = final_mass
        
    return material_dict

def checkMaterial(p, material, ignores):
    files = glob.glob(p+os.sep+'cone_H*')
    complete = True
    #if os.path.exists(p+os.sep+'ignition_temp.csv') is False: complete = False
    if os.path.exists(p+os.sep+material+'_Ignition_Temp_Properties.csv') is False:
        if os.path.exists(p+os.sep+'Ignition_Temp_Properties.csv') is False:
            complete = False
    if len(files) == 0: complete = False
    if material in ignores: complete = False
    return complete

def importFsriDatabase(data_dir, outInt, Tinfty=300, ignores=['']):
    material_directories = glob.glob(os.path.join(os.path.abspath(data_dir),'*'))
    possible_materials = [d.split(os.sep)[-1] for d in material_directories]
    
    complete_materials = dict()
    materials = []
    for i in range(0, len(material_directories)):
        p = os.path.abspath(material_directories[i])
        check = checkMaterial(p, possible_materials[i], ignores)
        if check: materials.append(possible_materials[i])
    
    uniqueFluxes = [25, 50, 75]
    for i in range(0, len(materials)):
        material = materials[i]
        p = os.path.join(os.path.abspath(data_dir), material)
        material_dict = importFsriMaterial(p, material, outInt, uniqueFluxes)
        complete_materials[material] = material_dict
        
    return complete_materials

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
    tmax = np.round(times.max()/dt)*dt
    tmin = np.round(times.min()/dt)*dt
    targetTimes = np.linspace(tmin, tmax, int((tmax-tmin)/dt + 1))
    HRRs = np.interp(targetTimes, times, HRRs)
    
    return targetTimes, HRRs

def findLimits(times, HRRs, energyCutoff1, energyCutoff2):
    v = np.cumsum(HRRs)
    ind1 = 0 
    while ind1 == 0:
        try:
            ind1 = np.where(v < np.nanmax(v)*energyCutoff1)[0][-1]
        except:
            energyCutoff1 = energyCutoff1*2
            #print(energyCutoff1)
    ind2 = v.shape[0]
    while ind2 == v.shape[0]:
        try:
            ind2 = np.where(v > np.nanmax(v)*energyCutoff2)[0][0]
        except:
            energyCutoff2 = energyCutoff2*0.99
            #print(energyCutoff2)
    times_trimmed = times[ind1:ind2]
    hrrs_trimmed = HRRs[ind1:ind2]
    tign = times[ind1]
    return tign, times_trimmed, hrrs_trimmed

if __name__ == "__main__":
    
    data_dir = '../data/fsri_materials_processed/'
    material_database = importFsriDatabase(data_dir, 15)
    
    resultDir = data_dir
    inputFileDir = data_dir
    expFileDir = data_dir
    emissivity = 1
    txt = 'Code,Number,Material,MaterialClass,DataFile,ResultDir,InputFileDir,ExpFileDir,'
    txt = txt + 'ReferenceExposure,ReferenceThickness,ReferenceTime,ReferenceHRRPUA,'
    txt = txt + 'ValidationTimes,ValidationHrrpuaColumns,ValidationFluxes,'
    txt = txt + 'Density,Conductivity,SpecificHeat,Emissivity,Thickness,'
    txt = txt + 'CharFraction,HeatOfCombustion,'
    txt = txt + 'IgnitionTemperature,IgnitionTemperatureBasis,HeaderRows,FYI'
    fluxes = [25, 50, 75]
    for material in list(material_database.keys()):
        conductivity = material_database[material]['conductivity']
        specific_heat = material_database[material]['heatCapacity']
        density = material_database[material]['density']
        thickness = material_database[material][50]['thickness']
        heat_of_combustion = material_database[material][50]['HeatOfCombustion']
        initial_mass = material_database[material][50]['initial_mass']
        final_mass = material_database[material][50]['final_mass']
        matClass = material_database[material]['materialClass']
        
        code ='d'
        number = 1
        mat = 'FSRI_%s'%(material)
        dataFiles = ''
        for flux in fluxes:
            dataFile = '../data/fsri_materials_processed/scaling_pyrolysis/%s-%02d.csv'%(mat, flux)
            dataFiles = dataFiles + dataFile + '|'
        dataFiles = dataFiles[:-1]
        
        txt = txt + "\n" + "%s,%s,%s,%s,%s,%s,"%(code, number, mat, matClass, dataFiles, resultDir)
        txt = txt + "%s,%s,50,%0.8f,%s-50.csv-Time,%s-50.csv-HRRPUA,"%(inputFileDir, expFileDir, thickness, mat, mat)
        txt = txt + '%s-25.csv-Time|%s-50.csv-Time|%s-75.csv-Time,'%(mat, mat, mat)
        txt = txt + '%s-25.csv-HRRPUA|%s-50.csv-HRRPUA|%s-75.csv-HRRPUA,'%(mat, mat, mat)
        txt = txt + '25|50|75,'
        txt = txt + '%0.1f,%0.4f,%0.4f,%0.4f,%0.8f,'%(density, conductivity, specific_heat, emissivity, thickness)
        txt = txt + '%0.4f,%0.4f,'%(max([final_mass/initial_mass,0]), heat_of_combustion)
        txt = txt + 'Calculate,25|50,1|1|1,FSRI_materials'
        
        for flux in fluxes:
            tign = material_database[material][flux]['tIgn']
            
            times = material_database[material][flux]['time']
            hrrpuas = material_database[material][flux]['hrrpua']
            
            d = pd.DataFrame(np.array([times, hrrpuas]).T, columns=['Time','HRRPUA'])
            if os.path.exists('..\\data\\fsri_materials_processed\\scaling_pyrolysis\\') is False:
                os.mkdir('..\\data\\fsri_materials_processed\\scaling_pyrolysis\\')
            dataFile = os.path.abspath('..\\data\\fsri_materials_processed\\scaling_pyrolysis\\%s-%02d.csv'%(mat, flux))
            d.to_csv(dataFile, index=False)
            
    with open('../data/fsri_spec_file.csv', 'w') as f:
        f.write(txt)
