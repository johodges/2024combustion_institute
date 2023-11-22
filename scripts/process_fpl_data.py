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

def importFplMaterial(material, outInt=15, filterWidth=11):
    files = glob.glob(material + os.sep + '*.xls')
    exposures = []
    file_data = dict()
    for file in files:
        data = pd.read_excel(file, sheet_name='data')
        
        data_processed = dict()
        for j in range(0, data.shape[0]):
            d = data.loc[j, 'Unnamed: 0']
            d2 = data.loc[j, 'Unnamed: 1']
            
            if type(data.loc[j, 'Unnamed: 0']) != str: continue
            #if np.isnan(data.loc[j, 'Unnamed: 0']): continue
            if 'Radiant Heat Flux' in d:
                data_processed['flux'] = [float(x) for x in d.split(' ') if checkForFloat(x)][0]
            elif ('thick' in d):
                data_processed['thickness'] = float(d.split('thick')[0].replace('mm',''))
            elif  ('thk' in d):
                data_processed['thickness'] = float(d.split('thk')[0].replace('mm',''))
            elif 'Sample Surface Area' in d:
                data_processed['area'] = [float(x) for x in d.split(' ') if checkForFloat(x)][0]
            elif 'Initial Specimen Mass' in d:
                data_processed['initial_mass'] = [float(x) for x in d.split(' ') if checkForFloat(x)][0]
            elif 'Final Specimen Mass' in d:
                data_processed['final_mass'] = [float(x) for x in d.split(' ') if checkForFloat(x)][0]
            elif 'Time to Sustained Ignition' in d:
                data_processed['timeIgn'] = [float(x) for x in d.split(' ') if checkForFloat(x)][0]
            elif 'Average Effective HOC' in d:
                data_processed['HeatOfCombustion'] = [float(x) for x in d.split(' ') if checkForFloat(x)][0]
            elif 'Peak Heat Release Rate' in d:
                data_processed['peakHRRPUA'] = [float(x) for x in d.split(' ') if checkForFloat(x)][0]
                data_processed['timeToPeak'] = [float(x) for x in d.split(' ') if checkForFloat(x)][1]
            elif ('Average Heat Release Rate' in d) and ('T60' in d):
                data_processed['HRRPUA, 60s average'] = [float(x) for x in d.split(' ') if checkForFloat(x)][0]
            elif ('Average Heat Release Rate' in d) and ('T180' in d):
                data_processed['HRRPUA, 180s average'] = [float(x) for x in d.split(' ') if checkForFloat(x)][0]
            elif ('Average Heat Release Rate' in d) and ('T300' in d):
                data_processed['HRRPUA, 300s average'] = [float(x) for x in d.split(' ') if checkForFloat(x)][0]
                #print(d)
                #print(data_processed['peakHRRPUA'])
                #assert False, "Stopped"
            elif type(d2) == str:
                d3 = data.values[j+2:, :2]
                delta = abs(d3[1:, 1] - d3[:-1, 1])
                end_ind = np.where(delta > 1000)[0][0]
                d3 = data.values[j+2:end_ind+j+3, :2]
                d3[d3[:, 1] < 0, 1] = 0
                d3[d3[:, 0] < data_processed['timeIgn'], 1] = 0
                data_processed['time'] = np.array(d3[:, 0], dtype=float)
                data_processed['hrrpua'] = np.array(d3[:, 1], dtype=float)
                data_processed['timeInt'] = np.median(abs(d3[1:, 0] - d3[:-1, 0]))
                data_processed['timeMax'] = d3[:, 0].max()
        #print(file)
        volume = data_processed['area'] * data_processed['thickness']
        initial_mass = data_processed['initial_mass']
        
        data_processed['density'] = initial_mass/volume
        
        file_data[file] = data_processed
        exposures.append(data_processed['flux'])
    fil = np.ones(filterWidth)/filterWidth
    uniqueFluxes = np.unique(exposures)
    
    coneData = file_data
    coneData = pd.DataFrame(file_data).T
    material_dict = dict()
    for flux in uniqueFluxes:
        outInt2 = outInt
        dt = np.min(coneData.loc[coneData['flux'] == flux, 'timeInt'].values)
        tMax = np.max(coneData.loc[coneData['flux'] == flux, 'timeMax'].values)
        
        thickness = np.median(coneData.loc[coneData['flux'] == flux, 'thickness'].values)
        initial_mass = np.median(coneData.loc[coneData['flux'] == flux, 'initial_mass'].values)
        final_mass = np.median(coneData.loc[coneData['flux'] == flux, 'final_mass'].values)
        area = np.median(coneData.loc[coneData['flux'] == flux, 'area'].values)
        density = np.median(coneData.loc[coneData['flux'] == flux, 'density'].values)
        tign = np.median(coneData.loc[coneData['flux'] == flux, 'timeIgn'].values)
        peakHRRPUA = np.median(coneData.loc[coneData['flux'] == flux, 'peakHRRPUA'].values)
        timeToPeak = np.median(coneData.loc[coneData['flux'] == flux, 'timeToPeak'].values)
        avg60s = np.nanmedian(coneData.loc[coneData['flux'] == flux, 'HRRPUA, 60s average'].values)
        avg180s = np.nanmedian(coneData.loc[coneData['flux'] == flux, 'HRRPUA, 180s average'].values)
        avg300s = np.nanmedian(coneData.loc[coneData['flux'] == flux, 'HRRPUA, 300s average'].values)
        avgHoC = np.nanmedian(coneData.loc[coneData['flux'] == flux, 'HeatOfCombustion'].values)
        
        times = coneData.loc[coneData['flux'] == flux, 'time'].values
        hrrpuas = coneData.loc[coneData['flux'] == flux, 'hrrpua'].values
        
        # Interpolate and filter
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
        hrrpua_mn = np.min(hrrpuas_interp_notign, axis=1)
        hrrpua_mx = np.max(hrrpuas_interp_notign, axis=1)
        hrrpua_std = np.std(hrrpuas_interp_notign, axis=1)
        
        '''
        tMax = np.ceil((tMax-tign)/outInt2)*outInt2
        outTime = np.linspace(0, tMax, int(tMax/outInt2)+1)
        outHrrpua = np.interp(outTime, time, hrrpua)
        
        totalEnergy = np.trapz(hrrpua, time)
        totalEnergy2 = np.trapz(outHrrpua, outTime)
        
        while abs(totalEnergy2 - totalEnergy)/totalEnergy > 0.001:
            print(abs(totalEnergy2 - totalEnergy)/totalEnergy)
            outInt2 = outInt2 / 2
            outTime = np.linspace(0, tMax, int(tMax/outInt2)+1)
            outHrrpua = np.interp(outTime, time, hrrpua)
            totalEnergy2 = np.trapz(outHrrpua, outTime)
            if outInt2 <= 1:
                totalEnergy2 = totalEnergy
        '''
        
        tMax = time.max()
        tMax = np.ceil((tMax)/outInt2)*outInt2
        outTime = np.linspace(0, tMax, int(tMax/outInt2)+1)
        outHrrpua = np.interp(outTime, time, hrrpua)
        
        totalEnergy = np.trapz(hrrpua, time)
        totalEnergy2 = np.trapz(outHrrpua, outTime)
        
        while abs(totalEnergy2 - totalEnergy)/totalEnergy > 0.01:
            #print(abs(totalEnergy2 - totalEnergy)/totalEnergy)
            outInt2 = outInt2*0.9
            outTime = np.linspace(0, tMax, int(tMax/outInt2)+1)
            outHrrpua = np.interp(outTime, time, hrrpua)
            totalEnergy2 = np.trapz(outHrrpua, outTime)
            if outInt2 <= 0.1:
                totalEnergy2 = totalEnergy
        
        if tign > 0:
            outTime = np.append(np.array([0, tign*0.9]), outTime+tign)
            outHrrpua = np.append(np.array([0, 0]), outHrrpua)
        
        outHrrpua[outHrrpua < 0] = 0
        
        print(material, outInt, outInt2, outTime.shape)
        
        
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
        material_dict[flux]['density'] = density
        material_dict[flux]['initial_mass'] = initial_mass
        material_dict[flux]['final_mass'] = final_mass
        material_dict['materialClass'] = getMaterialClass(material)
    
    return material_dict
    
    

def checkMaterial(p, material, ignores):
    files = glob.glob(p+os.sep+'cone_H*')
    complete = True
    #if os.path.exists(p+os.sep+'ignition_temp.csv') is False: complete = False
    if os.path.exists(p+os.sep+material+'_Ignition_Temp_Properties.csv') is False: complete = False
    if len(files) == 0: complete = False
    if material in ignores: complete = False
    return complete

def importFplDatabase(data_dir, outInt, Tinfty=300, ignores=['']):
    materials = glob.glob(data_dir+"*/")
    complete_materials = dict()
    for i in range(0, len(materials)):
        material = materials[i]
        material_dict = importFplMaterial(material, outInt, filterWidth=11)
        complete_materials[material.split(os.sep)[-2]] = material_dict
    return complete_materials

def checkForFloat(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

if __name__ == "__main__":
    data_dir ="..\\data\\fpl_materials\\"
    material_database = importFplDatabase(data_dir, 15)
    
    import matplotlib.pyplot as plt
    
    resultDir = ""
    inputFileDir = ""
    expFileDir = "../data/fpl_materials_processed/"
    emissivity = 1
    txt = 'Code,Number,Material,MaterialClass,DataFile,ResultDir,InputFileDir,ExpFileDir,'
    txt = txt + 'ReferenceExposure,ReferenceThickness,ReferenceTime,ReferenceHRRPUA,'
    txt = txt + 'ValidationTimes,ValidationHrrpuaColumns,ValidationFluxes,'
    txt = txt + 'Density,Conductivity,SpecificHeat,Emissivity,Thickness,'
    txt = txt + 'CharFraction,HeatOfCombustion,'
    txt = txt + 'IgnitionTemperature,IgnitionTemperatureBasis,HeaderRows,FYI'
    for material in list(material_database.keys()):
        conductivity = 0.4 #material_database[material]['conductivity']
        specific_heat = 1. #material_database[material]['heatCapacity']
        density = material_database[material][50.0]['density']
        thickness = material_database[material][50.0]['thickness']/1000
        initial_mass = material_database[material][50.0]['initial_mass']
        final_mass = material_database[material][50.0]['final_mass']
        heat_of_combustion = material_database[material][50.0]['HeatOfCombustion']
        matClass = material_database[material]['materialClass']
        
        fluxes = list(material_database[material].keys())
        fluxes.remove('materialClass')
        
        code ='d'
        number = 1
        mat = 'FPL_%s'%(material)
        dataFiles = ''
        for flux in fluxes:
            dataFile = os.path.join(expFileDir, '%s-%02d.csv'%(mat, flux))
            dataFiles = dataFiles + dataFile + '|'
        dataFiles = dataFiles[:-1]
        
        txt = txt + "\n" + "%s,%s,%s,%s,%s,%s,"%(code, number, mat, matClass, dataFiles, resultDir)
        txt = txt + "%s,%s,50,%0.8f,%s-50.csv-Time,%s-50.csv-HRRPUA,"%(inputFileDir, expFileDir, thickness, mat, mat)
        
        for flux in fluxes:
            txt = txt + '%s-%02d.csv-Time|'%(mat, flux)
        txt = txt[:-1] + ','
        for flux in fluxes:
            txt = txt + '%s-%02d.csv-HRRPUA|'%(mat, flux)
        txt = txt[:-1] + ','
        for flux in fluxes:
            txt = txt + '%02d|'%(flux)
        txt = txt[:-1] + ','
        #txt = txt + '%s-25.csv-Time|%s-50.csv-Time|%s-75.csv-Time,'%(mat, mat, mat)
        #txt = txt + '%s-25.csv-HRRPUA|%s-50.csv-HRRPUA|%s-75.csv-HRRPUA,'%(mat, mat, mat)
        #txt = txt + '25|50|75,'
        txt = txt + '%0.1f,%0.4f,%0.4f,%0.4f,%0.8f,'%(density, conductivity, specific_heat, emissivity, thickness)
        txt = txt + '%0.4f,%0.4f,'%(max([final_mass/initial_mass,0]), heat_of_combustion)
        txt = txt + 'Calculate,'
        for flux in fluxes:
            txt = txt + '%02d|'%(flux)
        txt = txt[:-1] + ','
        for flux in fluxes:
            txt = txt + '1|'
        txt = txt[:-1] + ','
        txt = txt + 'FPL_materials'
        
        for flux in fluxes:
            tign = material_database[material][flux]['tIgn']
            out_times = material_database[material][flux]['time']
            out_hrrpuas = material_database[material][flux]['hrrpua']
            #dt = np.median(times[1:]-times[:-1])
            #out_times = np.linspace(0, times.max()+tign, int((times.max()+tign)/dt + 1))
            #out_hrrpuas = np.interp(out_times, times+tign, hrrpuas)
            #out_hrrpuas[out_times < tign] = 0
            d = pd.DataFrame(np.array([out_times, out_hrrpuas]).T, columns=['Time','HRRPUA'])
            #dataFile = os.path.abspath('fpl_materials//%s-%02d.csv'%(mat, flux))
            dataFile = os.path.abspath('..\\data\\fpl_materials_processed\\%s-%02d.csv'%(mat, flux))
            d.to_csv(dataFile, index=False)
            
    with open('..\\data\\fpl_spec_file.csv', 'w') as f:
        f.write(txt)