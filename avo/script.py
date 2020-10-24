import configparser
import argparse


import pandas as pd
import numpy as np
import pylab as plt
import os
import sys
import seaborn as sns
import scipy.signal as scp

import avoFunc as AF

sys.path.append('../libs/')
from SeismicData import SeismicData
from Impulse  import impulse



def run_avo_script():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--settings', default='settings.ini')
    args = parser.parse_args()

    config = configparser.ConfigParser()

    config.read(args.settings)

    segy_directory = config.get('Paths', 'segy')
    well_directory = config.get('Paths', 'las')
    pulse_directory = config.get('Paths', 'impulse')



    # Interval time analysis
    int_time1 = config.getint('Arguments', 'int_time1')
    int_time2 = config.getint('Arguments', 'int_time2')

    approx = config.get('Arguments','approx')

    # HT0
    vsp = config.get('Arguments','vsp')

    # smooth Vrms
    win_len_smooth = config.getint('Arguments', 'win_len_smooth')

    
    # Muting parameters for synthetic gather (mute = a1*offset+b1)
    on = config.get('Arguments','on')
    a1 = config.getfloat('Arguments', 'a1')
    b1 = config.getfloat('Arguments', 'b1')

    # Coef.corr.estimate parameters
    win = config.getint('Arguments', 'win')

    # Metrics (window for analysis amplitude)
    time1 = config.getint('Arguments', 'time1')
    time2 = config.getint('Arguments', 'time2')
    param = config.get('Arguments','param')
    threshold = config.getint('Arguments', 'threshold')

    # Upper bound for offset
    offset_range = config.getint('Arguments', 'offset_range')

    #!/usr/bin/env python
    # coding: utf-8




#     pictures_path = os.path.join(directory,'OUT_pics')
#     metrics_path = os.path.join(directory,'OUT_metrics')

    pictures_path = '.\\OUT_pics\\'
    metrics_path = '.\\OUT_metrics\\'

    qc_path = os.path.join(pictures_path,'QC')
    impulse_path = os.path.join(pictures_path,'Impulse')
    seismic_path = os.path.join(pictures_path,'Seismic')

    segy_names = os.listdir( segy_directory )
    well_names = os.listdir( well_directory )
    pulse_names = os.listdir( pulse_directory )

    segy_names.sort()

    name_segy=[]
    for ii in range(len(segy_names)):
        name_segy.append(os.path.splitext(segy_names[ii])[0])

    name_well=[]
    for ii in range(len(well_names)):
        name_well.append(os.path.splitext(well_names[ii])[0])
    name_well = name_well[0]

    name_pulse=[]
    for ii in range(len(pulse_names)):
        name_pulse.append(os.path.splitext(pulse_names[ii])[0])
    name_pulse = name_pulse[0]


    # In[5]:


    print('Start AVO...')
    if vsp == 'on':
        for i in well_names:
            for j in i.split("_"):
                if j == 'VSP':
                    well_name2 = os.path.join(well_directory,i)
                    tmp_str = i
        for i in well_names:
            if i != tmp_str:
                well_name = os.path.join(well_directory,i)

        print('Read '+well_name+' ...')
        well_X, well_Y, df_well = AF.read_well(well_name)
        print('Done!')
        print(' ')
        print('Read '+well_name2+' ...')
        df_well_VSP = AF.read_well_VSP(well_name2, df_well)
        print('Done!')
    else:
        for i in well_names:
            well_name = os.path.join(well_directory,i)
        print('Read '+well_name+' ...')
        well_X, well_Y, df_well = AF.read_well(well_name)
        print('Done!')


    print(' ')
    print('Read '+pulse_names[0]+' ...')
    pulse_name = os.path.join(pulse_directory,pulse_names[0]) 
    dt_well = np.round(np.mean(np.diff(df_well['HT0'].values)[1:]),decimals=1)
    imp = impulse.Impulse(pulse_name,-0.2,0.2,dt_well*1e-3)
    t_int = imp.T_int2 
    Sig_int = imp.Sig
    print('Done!')


    df_metrics = pd.DataFrame()
    oo = -1
    for start_names in segy_names:
        oo += 1
        print('---------------------')
        print('  Read '+start_names+' ...')
        sgyname = start_names
        sgy_filename = os.path.join(segy_directory,sgyname)
        seismicData=SeismicData.SeismicData(sgy_filename)
        sgy_data = seismicData.data[:, seismicData.head.index]
        seismicData.head['well_X'] = np.float(well_X)
        seismicData.head['well_Y'] = np.float(well_Y)
        seismicData.head['well_offset'] = np.sqrt(np.array(((seismicData.head[['CDP_X', 'CDP_Y']].values - seismicData.head[['well_X', 'well_Y']].values) ** 2).sum(axis=1),dtype=float))
        indexies = seismicData.head[seismicData.head['well_offset']==seismicData.head['well_offset'].min()].index
        heads = seismicData.head.iloc[indexies].reset_index(drop=True)
        sgy_data = sgy_data[:,indexies]
        T_new = np.arange(int_time1, int_time2, seismicData.dt)

        print('  Done!')


        T_well = np.arange(df_well['HT0'].values[0],df_well['HT0'].values[-1],dt_well)

        RHOB_int, Vp_int, Vs_int, DH_int = AF.interpolate_parameters(T_well, df_well, 'HT0')

        if vsp =='on':
            df_tmp = df_well_VSP.loc[:,['Vrms','HT0']].dropna()
            Vp_int_VSP=np.interp(T_well,df_tmp['HT0'],df_tmp['Vrms'])

            df_tmp = df_well_VSP.loc[:,['H','HT0']].dropna()
            DH_int_VSP=np.interp(T_well,df_tmp['HT0'],df_tmp['H'])

            Vrms = AF.vrms(DH_int_VSP,Vp_int_VSP, T_well)
            df_well_int = pd.DataFrame({'RHOB':RHOB_int,'Vp':Vp_int,'Vs':Vs_int,
                                    'H':DH_int,'Vrms':Vrms,'T':T_well})
        else:
            Vrms = AF.vrms(DH_int, Vp_int, T_well)
            df_well_int = pd.DataFrame({'RHOB':RHOB_int[:-1],'Vp':Vp_int[:-1],'Vs':Vs_int[:-1],
                                    'H':DH_int[:-1],'Vrms':Vrms,'T':T_well[:-1]})



        data_short = sgy_data[int(int_time1/seismicData.dt):int(int_time2/seismicData.dt),:]
        df_well_int_cut=df_well_int[(df_well_int['T'] >= int_time1) & (df_well_int['T'] <= int_time2)].reset_index(drop=True)
        Vp_int_filt = scp.savgol_filter(df_well_int_cut['Vp'],win_len_smooth,2)

        # reflectivity
        X =seismicData.head['offset'].unique().astype(float)# heads['offset'].values.astype(float)
        if X[0] == 0:
            X[0] = 0.01

        T_well = df_well_int_cut['T'].values*1e-3
        Vp = df_well_int_cut['Vp'].values
        Vs = df_well_int_cut['Vs'].values
        pho = df_well_int_cut['RHOB'].values
        Vrms = df_well_int_cut['Vrms'].values
        DH = df_well_int_cut['H'].values
        #df_well_tmp = df_well_int_cut
        print(' ')
        print('  Calculate reflectivity ...')
        df_R, matr_theta = AF.reflectivity_calculation(Vp, Vs, pho, Vp_int_filt, X, Vrms, T_well,DH,approx)
        data_synt_ar = AF.reflectivity_gather(df_R)
        t_refl = AF.refl_times(data_synt_ar, T_well, X, Vrms)
        data_synt_ar_n, T_well_n = AF.trace_synt_int1(T_well, t_refl, dt_well, data_synt_ar)
        trace_synt_ar = AF.conv_refl_imp(data_synt_ar_n, Sig_int)

        Vrms_int = np.interp(T_well_n,T_well,Vrms)
        print('  Done!')
        print(' ')
        print('  Apply NMO ...')
        nmo = AF.nmo_correction(trace_synt_ar.T, dt_well*1e-3, X, Vrms_int, T_well[0])
        print('  Done!')
        if on == 'y':
            print(' ')
            print('  Apply muting ...')
            nmo_tmp = AF.mute_synth(nmo, a1, b1, X, dt_well)
            nmo_tmp = nmo_tmp[:len(T_well),:]
            print('  Done!')
        else:
            nmo_tmp = nmo[:len(T_well),:]

        trace_synt_ar_int, T = AF.trace_synt_int2(int_time1, int_time2, seismicData.dt, T_well*1e3, nmo_tmp)
        data_KK = AF.win_coef_corr(win, trace_synt_ar_int, data_short)

        new_offset_index = np.where(X<=offset_range)
        offsets = X[new_offset_index]
        doff = trace_synt_ar.shape[0]-len(offsets)


        tmp_time_max = []
        a = np.ravel(np.where(T==time1))[0]
        b = np.ravel(np.where(T==time2))[0]
        T_tmp = np.arange(time1,time2,seismicData.dt)
        for z in range(trace_synt_ar.shape[0]-doff+1):
            tmp_time_max.append(np.argmax(trace_synt_ar_int[z,a:b]))
        tmp_time_max = np.stack(tmp_time_max,axis=0)*seismicData.dt+time1

        hor_kk = data_KK[a:b,:len(offsets)]
        hor_kk = np.ravel(hor_kk)
        hor_kk = hor_kk[~np.isnan(hor_kk)]
        mean_kk_hor = np.mean(hor_kk)

        interval1 = 0
        interval2 = len(offsets)
        #int_time1 = time1
        #int_time2 = time2
        print(' ')
        print('  Calculate metrics ...')
        head_new = AF.metrics_theor_real_data(data_short.T, trace_synt_ar_int,T, 
                                           df_R['offset'].unique(), threshold, 
                                           time1, time2, interval1, interval2)

        disp = np.std(head_new['diff_'+param])*np.std(head_new['diff_'+param])
        kmean = np.mean(head_new['diff_'+param])
        print('  Done!')

        print('  Create pictures ...')
        AF.plot_pulse(imp)
        plt.savefig(os.path.join(impulse_path,'Impulse.png'), dpi = 300)
        AF.plot_interval_seismic(seismicData, sgy_data, heads, int_time1, int_time2, sgyname)
        plt.savefig(os.path.join(seismic_path,'interval_'+name_segy[oo]+'.png'), dpi = 300)
        AF.plot_well(df_well_int_cut)
        plt.savefig(os.path.join(qc_path,'well.png'), dpi = 300)
        if vsp == 'on':
            AF.plot_well_with_VSP(df_well_int_cut)
            plt.savefig(os.path.join(qc_path,'VSP_vel.png'), dpi = 300)
        AF.plot_angle_matr(X, T_well, matr_theta)
        plt.savefig(os.path.join(pictures_path,'angles.png'), dpi = 300)
        AF.plot_seismic_vs_synth(nmo_tmp, T_well, data_short, int_time2, int_time1, df_R, sgyname)
        plt.savefig(os.path.join(pictures_path,'Seism_vs_synth('+name_segy[oo]+').png'), dpi = 300)
        AF.plot_coef_corr_seismic(offsets, trace_synt_ar_int, seismicData, T, 
                                  tmp_time_max, data_KK, int_time1, int_time2, data_short.T,sgyname)
        plt.savefig(os.path.join(pictures_path,'Coef_cor('+name_segy[oo]+').png'), dpi = 300)
        AF.plot_metrics(head_new, df_R['offset'].unique()[interval1:interval2], threshold, param)
        plt.savefig(os.path.join(pictures_path,'metrics1('+name_segy[oo]+').png'), dpi = 300)
        AF.plot_metrics_distribution(interval1, interval2, head_new, param, kmean, disp, T_well, tmp_time_max, df_R, offsets)
        plt.savefig(os.path.join(pictures_path,'metrics2('+name_segy[oo]+').png'), dpi = 300)

        print('  Done!')
        print(' ')
        print('  Save metrics and create table ...')
        disper = np.round(disp*10000)/10000
        means = np.round(kmean*10000)/10000
        procent_trace = int(np.round(len(head_new[head_new['diff_'+param+'_%']>threshold])/len(head_new)*100))
        df_temp = pd.DataFrame({
                                'filename':name_segy[oo],'Mean_kk_on_horizon':[mean_kk_hor],
                                'Mean_diff_ampl':[means],'Mean_disp_ampl':[disper],
                                'Percentage (threshold >'+str(threshold)+'%)':[procent_trace],
                                   })
        df_metrics = df_metrics.append(df_temp, ignore_index = True)
        df_metrics.to_csv(os.path.join(metrics_path,'metrics.csv'),index=False)
        AF.create_dataframe_img(df_metrics,os.path.join(pictures_path,'metrics_table.png'))
        print('  Done!') 


