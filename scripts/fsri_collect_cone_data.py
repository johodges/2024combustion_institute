# Cone calorimeter data processing script
#   by: ULRI's Fire Safety Research Institute
#   Questions? Submit them here: https://github.com/ulfsri/fsri_materials_database/issues

# ***************************** Usage Notes *************************** #
# - Script outputs as a function of heat flux                           #
#   -  PDF Graphs dir: /03_Charts/{Material}/Cone                       #
#      Graphs: Extinction_Coefficient, Heat Release Rate Per Unit Area, #
#      Mass Loss Rate, Specific Extinction Area, Smoke Production Rate  #
#                                                                       #
#      CSV Tables dir: /01_Data/{Material}/Cone                         #
#      Tables: Cone Notes, Analysis Data                                #
# ********************************************************************* #

# --------------- #
# Import Packages #
# --------------- #
import os
import glob
import numpy as np
import pandas as pd
import math
from scipy.signal import savgol_filter
import git
import shutil

# Define variables #
data_dir = '../data/fsri_materials_database/01_Data/'
save_dir = '../data/fsri_materials_processed/'
if not os.path.exists(save_dir): os.makedirs(save_dir)

hf_list_default = ['25', '50', '75']
quant_list = ['HRRPUA', 'MLR', 'SPR', 'Extinction Coefficient'] #'EHC','SEA'

y_max_dict = {'HRRPUA':500, 'MLR':1, 'SPR':5, 'Extinction Coefficient':2} #'EHC':50,'SEA':1000,
y_inc_dict = {'HRRPUA':100, 'MLR':0.2, 'SPR':1, 'Extinction Coefficient':0.5} #'EHC':10,'SEA':200

label_size = 20
tick_size = 18
line_width = 2
legend_font = 10
fig_width = 10
fig_height = 6

### Fuel Properties ###
e = 13100 # [kJ/kg O2] del_hc/r_0
laser_wl = 632.8/10e9 # m
smoke_density = 1100 # kg/m3
c = 7 # average coefficient of smoke extinction
avg_ext_coeff = 8700 # m2/kg  from Mullholland

def apply_savgol_filter(raw_data):
    # raw_data.drop('Baseline', axis = 'index', inplace = True)
    raw_data = raw_data.dropna()
    converted_data = savgol_filter(raw_data,31,3)
    filtered_data = pd.Series(converted_data, index=raw_data.index.values)
    return(filtered_data.iloc[0:])


def air_density(temperature):
    # returns density in kg/m3 given a temperature in C
    rho = 1.2883 - 4.327e-3*temperature + 8.78e-6*temperature**2
    return rho


for d in sorted((f for f in os.listdir(data_dir) if not f.startswith(".")), key=str.lower):
    df_dict = {}
    material = d
    output_df = pd.DataFrame()
    co_df = pd.DataFrame()
    soot_df = pd.DataFrame()
    notes_df = pd.DataFrame()
    if os.path.isdir(f'{data_dir}{d}/Cone/'):
        print(material + ' Cone')
        data_df = pd.DataFrame()
        reduced_df = pd.DataFrame()
        if os.path.isfile(f'{data_dir}{d}/Cone/hf_list.csv'):
            hf_list =  pd.read_csv(f'{data_dir}{d}/Cone/hf_list.csv') # for parsing hf outside of base set of ranges
        else:
            hf_list = hf_list_default
        for f in sorted(glob.iglob(f'{data_dir}{d}/Cone/*.csv')):
            if 'scan' in f.lower():
                label_list = f.split('.csv')[0].split('_')
                label = label_list[-3].split('Scan')[0] + '_' + label_list[-1]
                data_temp_df = pd.read_csv(f, header = 0, skiprows = [1, 2, 3, 4], index_col = 'Names')

                scalar_data_fid = f.replace('Scan','Scalar')
                scalar_data_series = pd.read_csv(scalar_data_fid, index_col = 0).squeeze()

                # Test Notes #
                try:
                    pretest_notes = scalar_data_series.at['PRE TEST CMT']
                except:
                    pretest_notes = ' '
                surf_area_mm2 = 10000
                dims = 'not specified'
                frame = False
                for notes in pretest_notes.split(';'):
                    if 'Dimensions' in notes:
                        dims = []
                        for i in notes.split(' '):
                            try:
                                    dims.append(float(i))
                            except: continue
                        surf_area_mm2 = dims[0] * dims[1]
                    elif 'frame' in notes:
                        frame = True
                if frame or '-Frame' in f:
                    surf_area_mm2 = 8836

                surf_area_m2 = surf_area_mm2 / 1000000.0

                notes_df.at[label, 'Surface Area (mm^2)'] = surf_area_mm2
                notes_df.at[label, 'Pretest'] = pretest_notes
                try:
                    notes_df.at[label, 'Posttest'] = scalar_data_series.at['POST TEST CMT']
                except:
                    notes_df.at[label, 'Posttest'] = ' '


                c_factor = float(scalar_data_series.at['C FACTOR'])

                data_temp_df['O2 Meter'] = data_temp_df['O2 Meter']/100
                data_temp_df['CO2 Meter'] = data_temp_df['CO2 Meter']/100
                data_temp_df['CO Meter'] = data_temp_df['CO Meter']/100

                data_temp_df.loc[:,'EDF'] = ((data_temp_df.loc[:,'Exh Press']/(data_temp_df.loc[:,'Stack TC']+273.15)).apply(np.sqrt)).multiply(c_factor) # Exhaust Duct Flow (m_e_dot)
                data_temp_df.loc[:,'Volumetric Flow'] = data_temp_df.loc[:,'EDF']*air_density(data_temp_df.loc[:,'Smoke TC']) # Exhaust Duct Flow (m_e_dot)
                # O2_offset = 0.2095 - data_temp_df.at['Baseline', 'O2 Meter']
                # data_temp_df.loc[:,'ODF'] = (0.2095 - data_temp_df.loc[:,'O2 Meter'] + O2_offset) / (1.105 - (1.5*(data_temp_df.loc[:,'O2 Meter'] + O2_offset))) # Oxygen depletion factor with only O2
                data_temp_df.loc[:,'ODF'] = (data_temp_df.at['Baseline', 'O2 Meter'] - data_temp_df.loc[:,'O2 Meter']) / (1.105 - (1.5*(data_temp_df.loc[:,'O2 Meter']))) # Oxygen depletion factor with only O2
                data_temp_df.loc[:,'ODF_ext'] = (data_temp_df.at['Baseline', 'O2 Meter']*(1-data_temp_df.loc[:, 'CO2 Meter'] - data_temp_df.loc[:, 'CO Meter']) - data_temp_df.loc[:, 'O2 Meter']*(1-data_temp_df.at['Baseline', 'CO2 Meter']))/(data_temp_df.at['Baseline', 'O2 Meter']*(1-data_temp_df.loc[:, 'CO2 Meter']-data_temp_df.loc[:, 'CO Meter']-data_temp_df.loc[:, 'O2 Meter'])) # Oxygen Depletion Factor with O2, CO, and CO2
                data_temp_df.loc[:,'HRR'] = 1.10*(e)*data_temp_df.loc[:,'EDF']*data_temp_df.loc[:,'ODF']
                data_temp_df.loc[:,'HRR_ext'] = 1.10*(e)*data_temp_df.loc[:,'EDF']*data_temp_df.at['Baseline', 'O2 Meter']*((data_temp_df.loc[:,'ODF_ext']-0.172*(1-data_temp_df.loc[:,'ODF'])*(data_temp_df.loc[:, 'CO2 Meter']/data_temp_df.loc[:, 'O2 Meter']))/((1-data_temp_df.loc[:,'ODF'])+1.105*data_temp_df.loc[:,'ODF']))
                data_temp_df.loc[:,'HRRPUA'] = data_temp_df.loc[:,'HRR']/surf_area_m2
                data_temp_df['THR'] = 0.25*data_temp_df['HRRPUA'].cumsum()/1000
                data_temp_df['MLR_grad'] = -np.gradient(data_temp_df['Sample Mass'], 0.25)
                data_temp_df['MLR'] = apply_savgol_filter(data_temp_df['MLR_grad'])
                data_temp_df['MLR'][data_temp_df['MLR'] > 5] = 0

                data_temp_df['EHC'] = data_temp_df['HRR']/data_temp_df['MLR'] # kW/(g/s) -> MJ/kg
                data_temp_df['Extinction Coefficient'] = data_temp_df['Ext Coeff'] - data_temp_df.at['Baseline','Ext Coeff']
                data_temp_df['SPR'] = (data_temp_df.loc[:,'Extinction Coefficient'] * data_temp_df.loc[:,'Volumetric Flow'])/surf_area_m2
                data_temp_df['SPR'][data_temp_df['SPR'] < 0] = 0
                data_temp_df['SEA'] = (1000*data_temp_df.loc[:,'Volumetric Flow']*data_temp_df.loc[:,'Extinction Coefficient'])/data_temp_df['MLR']
                # data_temp_df['SEA'][np.isinf(data_temp_df['SEA'])] = np.nan

                df_dict[label] = data_temp_df[['Time', 'HRRPUA', 'MLR', 'EHC', 'SPR', 'SEA', 'Extinction Coefficient']].copy()
                df_dict[label].set_index(df_dict[label].loc[:,'Time'], inplace = True)
                df_dict[label] = df_dict[label][df_dict[label].index.notnull()]
                df_dict[label].drop('Time', axis = 1, inplace = True)
                end_time = float(scalar_data_series.at['END OF TEST TIME'])
                num_intervals = (max(df_dict[label].index)-end_time)/0.25
                drop_list = list(np.linspace(end_time, max(df_dict[label].index), int(num_intervals+1)))
                df_dict[label].drop(labels = drop_list, axis = 0, inplace = True)

                output_df.at['Time to Sustained Ignition (s)', label] = scalar_data_series.at['TIME TO IGN']
                output_df.at['Peak HRRPUA (kW/m2)', label] = float("{:.2f}".format(max(data_temp_df['HRRPUA'])))
                output_df.at['Time to Peak HRRPUA (s)', label] = data_temp_df.loc[data_temp_df['HRRPUA'].idxmax(), 'Time'] - float(scalar_data_series.at['TIME TO IGN'])
                ign_index = data_temp_df.index[data_temp_df['Time'] == float(scalar_data_series.at['TIME TO IGN'])][0]
                t60 = str(int(ign_index) + 240)
                t180 = str(int(ign_index) + 720)
                t300 = str(int(ign_index) + 1200)

                try: output_df.at['Average HRRPUA over 60 seconds (kW/m2)', label] = float("{:.2f}".format(np.mean(data_temp_df.loc[ign_index:t60,'HRRPUA'])))
                except: output_df.at['Average HRRPUA over 60 seconds (kW/m2)', label] = math.nan

                try: output_df.at['Average HRRPUA over 180 seconds (kW/m2)', label] = float("{:.2f}".format(np.mean(data_temp_df.loc[ign_index:t180,'HRRPUA'])))
                except: output_df.at['Average HRRPUA over 180 seconds (kW/m2)', label] = math.nan

                try: output_df.at['Average HRRPUA over 300 seconds (kW/m2)', label] = float("{:.2f}".format(np.mean(data_temp_df.loc[ign_index:t300,'HRRPUA'])))
                except: output_df.at['Average HRRPUA over 300 seconds (kW/m2)', label] = math.nan

                output_df.at['Total Heat Released (MJ/m2)', label] = float("{:.2f}".format(data_temp_df.at[scalar_data_series.at['END OF TEST SCAN'],'THR']))
                total_mass_lost = data_temp_df.at['1','Sample Mass'] - data_temp_df.at[scalar_data_series.at['END OF TEST SCAN'],'Sample Mass']
                holder_mass = data_temp_df.at['1','Sample Mass'] - float(scalar_data_series.at['SPECIMEN MASS'])
                output_df.at['Avg. Effective Heat of Combustion (MJ/kg)', label] = float("{:.2f}".format(((data_temp_df.at[scalar_data_series.at['END OF TEST SCAN'],'THR'])*surf_area_m2)/(total_mass_lost/1000)))
                output_df.at['Initial Mass (g)', label] = scalar_data_series.at['SPECIMEN MASS']
                output_df.at['Final Mass (g)', label] = float("{:.2f}".format(data_temp_df.at[scalar_data_series.at['END OF TEST SCAN'],'Sample Mass'] - holder_mass))
                output_df.at['Mass at Ignition (g)', label] = float("{:.2f}".format(data_temp_df.at[ign_index,'Sample Mass'] - holder_mass))

                t10 = data_temp_df['Sample Mass'].sub(data_temp_df.at['1','Sample Mass'] - 0.1*total_mass_lost).abs().idxmin()
                t90 = data_temp_df['Sample Mass'].sub(data_temp_df.at['1','Sample Mass'] - 0.9*total_mass_lost).abs().idxmin()

                output_df.at['Avg. Mass Loss Rate [10% to 90%] (g/m2s)', label] = float("{:.2f}".format(np.mean(data_temp_df.loc[t10:t90,'MLR']/surf_area_m2)))
                save_dir2 = f'{save_dir}/{material}/'
                if not os.path.exists(save_dir2): os.makedirs(save_dir2)
                fname = 'cone_%s.csv'%(label)
                tmp = list(data_temp_df.index)
                try:
                        tmp.remove('Baseline')
                except:
                        pass
                tign = float(scalar_data_series['TIME TO IGN'])
                time = data_temp_df.loc[tmp, 'Time'].values
                hrrpua = data_temp_df.loc[tmp, 'HRRPUA'].values
                hrrpua[time < tign] = 0
                hrrpua[hrrpua < 0] = 0
                hrrpua[np.isnan(hrrpua)] = 0
                timeResolvedOutput = pd.DataFrame(np.array([time, hrrpua]).T, columns=['Time','HRRPUA'])
                timeResolvedOutput.to_csv(os.path.join(save_dir2, fname), index=False)
                src = os.path.abspath(f.replace('Scan','Scalar'))
                basename = os.path.basename(src).split("_HF")[1]
                basename = f'{material}_Cone_HF{basename}'
                dst = f'{save_dir}/{material}/{basename}'
                shutil.copy(src, dst)
        output_df.sort_index(axis=1, inplace=True)
        output_df.to_csv(f'{save_dir}/{material}/{material}_Cone_Analysis_Data.csv', float_format='%.2f')

        notes_df.sort_index(axis=0, inplace=True)
        notes_df.to_csv(f'{save_dir}/{material}/{material}_Cone_Notes.csv', float_format='%.2f')
    else:
        continue


