# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 18:18:31 2023

@author: jhodges
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, shutil, subprocess

from plotting import getJHcolors, getPlotLimits
from algorithms import getMaterials, processCaseData
from algorithms import developRepresentativeCurve, getFixedModelParams
from algorithms import runSimulation
from algorithms import calculateUncertainty, plotMaterialExtraction, calculateUncertaintyBounds
from algorithms import getTimeAveragedPeak
from algorithms import getTimeAveragedEnergy, getTimeAveraged_timeToEnergy
from algorithms import getRepresentativeHrrpua, estimateExposureFlux, estimateHrrpua

def findFds():
    ''' First check if FDSDIR environmental variable is defined. If not, print
    warning then use which to look for a checklist. If not found anywhere
    error out.
    '''
    fdsDir = os.getenv('FDSDIR')
    if fdsDir is not None: return fdsDir, 'fds'
    print("Warning FDSDIR environmental variable not set. Trying to find FDS in path.")
    checklist = ['fds', 'fds_ompi_gnu_linux']
    for check in checklist:
        fdsPath = shutil.which(check)
        if fdsPath is not None:
            fdsDir = os.sep.join(fdsPath.split(os.sep)[:-1]) + os.sep
            print("FDS found in %s"%(fdsDir))
            return fdsDir, check
    print("Warning FDS not found")

def buildFdsFile(chid, cone_hf_ref, cone_d_ref, emissivity, conductivity, density, 
                 specific_heat, Tign, time, hrrpua, tend, deltas, fluxes, front_h,
                 case_tigns=False, ignitionMode='Temperature', outputTemperature=False,
                 calculateDevcDt=True, devc_dt=1.):
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
    qref = estimateExposureFlux(cone_hf_ref, hrrpua_ref)
    
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
        hf_ign = estimateHrrpua(cone_hf_ref, hrrpua_ref, hf)
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

if __name__ == "__main__":
    
    fdsdir, fdscmd = findFds()
    fileDir = os.path.dirname(os.path.abspath(__file__))
    
    spec_file_dict = getMaterials()
    materials = list(spec_file_dict.keys())
    
    materials = ['FAA_PMMA', 'FAA_HDPE', 'FAA_HIPS', 'FAA_PC', 'FAA_PVC']
    nondimtype = 'FoBi_simple_fixed_d'
    figoutdir = "figures"
    runSimulations = False
    plotResults = False
    lineStyles = ['--','-.',':']
    
    if figoutdir is not None:
        if os.path.exists(figoutdir) is not True: os.mkdir(figoutdir)
        import matplotlib.pyplot as plt
    
    output_statistics = dict()
    params = getFixedModelParams()
    colors = getJHcolors()
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
        tend = np.nanmax([cases[c]['times_trimmed'].max()+cases[c]['tign'] for c in cases]*2)
        
        deltas = [cases[c]['delta'] for c in cases]
        fluxes = [cases[c]['cone'] for c in cases]
        tigns = [cases[c]['tign'] for c in cases]
        
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
            data = load_csv(workingDir, chid)
            # Plot results
            if figoutdir is not None:
                fig = plt.figure(figsize=(24,18))
                (fs, lw, s, exp_num_points) = (24, 6, 100, 25)
                exp_int = 5
                case_names = list(cases.keys())
                for i in range(0, len(case_names)):
                    c = case_names[i]
                    namespace = '%02d-%03d'%(fluxes[i], deltas[i]*1e3)
                    label = r'%s'%(namespace) #'$\mathrm{kW/m^{2}}$'%(coneExposure)
                    plt.scatter(cases[c]['times'][::exp_int]/60,cases[c]['HRRs'][::exp_int], s=s, label=label, linewidth=lw, color=colors[i])
                    times = data['Time']
                    hrrpuas = data['"HRRPUA-'+namespace+'"']
                    plt.plot(times/60, hrrpuas, lineStyles[1], linewidth=lw, color=colors[i])
            plt.xlabel("Time (min)", fontsize=fs)
            plt.ylabel(r'HRRPUA ($\mathrm{kW/m^{2}}$)', fontsize=fs)
            #plt.ylim(0, np.ceil(1.1*ymax/100)*100)
            #plt.xlim(0, np.ceil(exp_tmax/60))
            plt.grid()
            plt.tick_params(labelsize=fs)
            plt.legend(fontsize=fs, bbox_to_anchor=(1.05,0.6))
            plt.tight_layout(rect=(0, 0, 1, 0.95))
            
            #if savefigure: plt.savefig(namespace, dpi=300)
            #if closefigure: plt.close()
            
