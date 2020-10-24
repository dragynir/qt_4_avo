import re
import pandas as pd
import numpy as np
import pylab as plt
#import os
import sys
#from sklearn.linear_model import LinearRegression
import seaborn as sns
#from SeismicData import SeismicData

sys.path.append('../libs/')
from WellData import WellData 
#import SegRead as s
#from time import time
from LAS import Converter
from seiscm import seismic
from PlotData import PlotData as pltd
from scipy.interpolate import CubicSpline
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

from string import Template
from matplotlib import colors

def reflection_time(t0, x, vnmo):
    """
    Function of calculating the arrival time of the reflected wave
    
    Parameters
    ----------
    Input:
        t0 - zero-offset travel time
        x - array of offsets
        vnmo - stacking velocity
    Output:
        t - time of reflective wave
    """
    t = np.sqrt(t0**2 + x**2/vnmo**2)
    return t

def sample_trace(trace, time, dt):
    """
    interpolate sample of traces
    
    Parameters
    ----------
    Input:
        trace - seismic trace
        time - time axs for seismic
        dt - time step for seismic
    Output:
        renew amplitude
    """
    before = int(np.floor(time/dt))
    N = trace.size
    # Use the 4 samples around time to interpolate
    samples = np.arange(before - 1, before + 3)
    if any(samples < 0) or any(samples >= N):
        amplitude = None
    else:
        times = dt*samples
        amps = trace[samples]
        interpolator = CubicSpline(times, amps)
        amplitude = interpolator(time)
    return amplitude

def nmo_correction(cmp, dt, offsets, velocities, start_t):
    """
    NMO correction on synthetic gather
    
    Parameters
    ----------
    Input:
        cmp - synthetics gather
        dt - time step for well
        offsets - array of offsets(m)
        velocities - RMS velocities
        start_t - start time for synthetic gather

    Output:
        nmo - synthetic gather with nmo correction
    """
    nmo = np.zeros_like(cmp)
    nsamples = cmp.shape[0]
    times = np.linspace(start_t,nsamples*dt+start_t,nsamples)
    for i, t0 in enumerate(times):
        for j, x in enumerate(offsets):
            t = reflection_time(t0, x, velocities[i])-start_t
            amplitude = sample_trace(cmp[:, j], t, dt)
            # If the time t is outside of the CMP time range,
            # amplitude will be None.
            if amplitude is not None:
                nmo[i, j] = amplitude
    return nmo    


def metrics_theor_real_data(data_short, synt_data, T, offsets, threshold, int_time1, int_time2, interval1, interval2):  
    """
    Calculate metrics for theorethic and real data
    
    Parameters
    ----------
    Input:
        data_short - interval real data
        synt_data - synthetic data
        T - time axs for seismic
        offsets - offsets
        threshold - percentage of bounce of relative amplitudes
        int_time1 - upper time near explored horizon
        int_time2 - lower time near explored horizon
        interval1 - min offset number for analysis
        interval2 - max offset number for analysis

    Output:
        head_tmp - dataframe with metrics and amplitudes
        
    """

    ampl_max_teor = []
    ampl_rms_teor = []
    a = np.ravel(np.where(T==int_time1))[0]
    b = np.ravel(np.where(T==int_time2))[0]
    for z in range(synt_data.shape[0]):
        ampl_max_teor.append(np.max(synt_data[z,a:b]))
        ampl_rms_teor.append(np.sqrt(sum(synt_data[z,a:b]*synt_data[z,a:b])/synt_data.shape[1]))
        
    ampl_max_real = []
    ampl_rms_real = []
    a = np.ravel(np.where(T==int_time1))[0]
    b = np.ravel(np.where(T==int_time2))[0]
    for z in range(data_short.shape[0]):
        ampl_max_real.append(np.max(data_short[z,a:b]))
        ampl_rms_real.append(np.sqrt(sum(data_short[z,a:b]*data_short[z,a:b])/data_short.shape[1]))
        
    head_tmp = pd.DataFrame()
    head_tmp['ampl_max_real_n'] = np.asarray(ampl_max_real[interval1:interval2])
    head_tmp['ampl_rms_real_n'] = np.asarray(ampl_rms_real[interval1:interval2])
    head_tmp['ampl_max_theor_n'] = np.asarray(ampl_max_teor[interval1:interval2])
    head_tmp['ampl_rms_theor_n'] = np.asarray(ampl_rms_teor[interval1:interval2])
    
    ampl_max_teor = np.asarray(ampl_max_teor[interval1:interval2])/np.mean(np.asarray(ampl_max_teor[interval1:interval2]))
    ampl_max_real = np.asarray(ampl_max_real[interval1:interval2])/np.mean(np.asarray(ampl_max_real[interval1:interval2]))
    
    ampl_rms_teor = np.asarray(ampl_rms_teor[interval1:interval2])/np.mean(np.asarray(ampl_rms_teor[interval1:interval2]))
    ampl_rms_real = np.asarray(ampl_rms_real[interval1:interval2])/np.mean(np.asarray(ampl_rms_real[interval1:interval2]))
    
    diff = abs(ampl_rms_teor-ampl_rms_real)
    
    head_tmp['diff_rms_%'] = diff*100
    head_tmp['diff_rms'] = diff
    
    diff = abs(ampl_max_teor-ampl_max_real)
    head_tmp['diff_max_%'] = diff*100
    head_tmp['diff_max'] = diff
   

    head_tmp['colors_max'] = 'g'
    head_tmp['colors_max'][head_tmp[(abs( head_tmp['diff_max_%']) >= threshold)].index] = 'r'
    head_tmp['ampl_max_real'] = ampl_max_real
    head_tmp['ampl_max_theor'] = ampl_max_teor

    head_tmp['colors_rms'] = 'g'
    head_tmp['colors_rms'][head_tmp[(abs( head_tmp['diff_rms_%']) >= threshold)].index] = 'r'
    head_tmp['ampl_rms_real'] = ampl_rms_real
    head_tmp['ampl_rms_theor'] = ampl_rms_teor
    
    
    return head_tmp

def plot_metrics(head, offsets, threshold, param):
    """
    Function draws metrics calculated in metrics_theor_real_data function
    
    Parameters
    ----------
    Input:
        head - dataset with headers and metrics 
        offsets - array of offsets 
        threshold - percentage of bounce of relative amplitudes
        
    Output:  
        figure with difference of AVO distributions
    """
    sns.set()
    fig = plt.figure(figsize = (10,5))
    plt.scatter(offsets,head['diff_'+param+'_%'].values,c = head['colors_'+param],s = 10)
    plt.ylim(0,100)
    plt.title('Difference between real and theor. distibution, threshold = '+str(threshold)+'%')
    plt.xlabel('Offset, m')
    plt.ylabel('diff Ampl %')
    return 

def zoeprittz(i, alf1, alf2, bet1, bet2, dens1, dens2):
    """
    Calculate zoeprittz approximation
    
    Parameters
    ----------
    Input:
        i - current incidence angle
        alf1 - upper primary velocity
        alf2 - lower primary velocity
        bet1 - upper secondary velocity
        bet2 - lower secondary velocity
        dens1 - upper density
        dens2 - lower density
    Output:
        Rp - reflectivity coefficient
        
    """
    p  = np.sin(i) / alf1
    I1 = i
    I2 = np.arcsin(p * alf2)
    J1 = np.arcsin(p * bet1)
    J2 = np.arcsin(p * bet2)
    a  = dens2*(1-2*bet2*bet2*p*p) - dens1*(1-2*bet1*bet1*p*p)
    b  = dens2*(1-2*bet2*bet2*p*p) + 2*dens1*bet1*bet1*p*p
    c  = dens1*(1-2*bet1*bet1*p*p) + 2*dens2*bet2*bet2*p*p
    d  = 2*( dens2*bet2*bet2 - dens1*bet1*bet1 )
    E  = b*np.cos(I1)/alf1 + c*np.cos(I2)/alf2
    F  = b*np.cos(J1)/bet1 + c*np.cos(J2)/bet2
    G  = a - d*np.cos(I1)/alf1*np.cos(J2)/bet2
    H  = a - d*np.cos(I2)/alf2*np.cos(J1)/bet1
    D  = E*F + G*H*p*p;
        
    Rp  =   ((b*np.cos(I1)/alf1-c*np.cos(I2)/alf2)*F - (a + d*np.cos(I1)/alf1*np.cos(J2)/bet2 )*H*(p*p))/D
    return Rp


def richards(theta_i, Vp1, Vp2, Vs1, Vs2, pho1, pho2):
    """
    Aki-Richards approximation calculation function
    
    Parameters
    ----------
    Input:
        theta_i - current incidence angle
        Vp1 - upper primary velocity
        Vp2 - lower primary velocity
        Vs1 - upper secondary velocity
        Vs2 - lower secondary velocity
        pho1 - upper density
        pho2 - lower density
    Output:
        Rp - reflectivity coefficient
        
    """
    theta_t = np.arcsin((Vp2/Vp1)*np.sin(theta_i))
    theta = (theta_i+theta_t)/2
    Vp = (Vp1+Vp2)/2
    Vs = (Vs1+Vs2)/2
    pho = (pho1+pho2)/2
    dVp = Vp2-Vp1
    dVs = Vs2-Vs1
    dpho = pho2-pho1
    Abar = 1/2/(np.cos(theta)*np.cos(theta))
    Bbar = -4*((Vs*Vs)/(Vp*Vp))*np.sin(theta)*np.sin(theta)
    Cbar = 0.5+Bbar/2
    Rp = Abar*(dVp/Vp)+Bbar*(dVs/Vs)+Cbar*(dpho/pho)
    return Rp

def interpolate_parameters(param, df_well, column_name):
    """
    Function interpolates parameters from well to the same grid
    
    Parameters
    ----------
    Input:
        param - regular grid (time or depth)
        df_well - dataframe with well_date
        column_name - nonregular grid (HT0 or H)

    Output:
        RHOB_int - interpolate densities
        Vp_int - interpolate Vp
        Vs_int - interpolate Vs
        tmp - interpolate HT0 or H
    """
    if column_name == 'HT0':
        df_tmp = df_well.loc[:,['PHO','HT0']].dropna()
        RHOB_int=np.interp(param,df_tmp['HT0'],df_tmp['PHO'])

        df_tmp = df_well.loc[:,['Vp','HT0']].dropna()
        Vp_int=np.interp(param,df_tmp['HT0'],df_tmp['Vp'])

        df_tmp = df_well.loc[:,['Vs','HT0']].dropna()
        Vs_int=np.interp(param,df_tmp['HT0'],df_tmp['Vs'])

        df_tmp = df_well.loc[:,['H','HT0']].dropna()
        tmp = np.interp(param,df_tmp['HT0'],df_tmp['H'])
    elif column_name == 'H':
        df_tmp = df_well.loc[:,['PHO','H']].dropna()
        RHOB_int=np.interp(param,df_tmp['H'],df_tmp['PHO'])

        df_tmp = df_well.loc[:,['Vp','H']].dropna()
        Vp_int=np.interp(param,df_tmp['H'],df_tmp['Vp'])

        df_tmp = df_well.loc[:,['Vs','H']].dropna()
        Vs_int=np.interp(param,df_tmp['H'],df_tmp['Vs'])

        df_tmp = df_well.loc[:,['HT0','H']].dropna()
        tmp = np.interp(param,df_tmp['H'],df_tmp['HT0'])
    
    return RHOB_int, Vp_int, Vs_int, tmp

def vrms(DH, Vp_int_filt, T_well):
    """
    Function calculates RMS velocities
    
    Parameters
    ----------
    Input:
    All input parameters must be interpolated to the same grid
        DH - array of H(m) from well
        Vp_int_filt - array of Vp from well,
        T_well - array of T(s) from well

    Output:
        Vrms - array of RMS velocities    
    """
    Vrms=[]
    
    for j in range(T_well.shape[0]):
        Vrms.append(np.sqrt(np.sum(Vp_int_filt[0:j+1]*Vp_int_filt[0:j+1])/len(Vp_int_filt[0:j+1])))
    return Vrms


def reflectivity_calculation(Vp, Vs, pho, Vp_int_filt, X, Vrms, T_well, DH, approx):
    """
    Calculate reflectivity by approximation
    
    Parameters
    ----------
    Input:
        Vp - primary velocities from well
        Vs - secondary velocities from well
        pho - densities from well
        Vp_int_filt - smoothed primary velocities
        X - offsets
        Vrms - RMS velocities
        T_well - time axs for well
        DH - well depths
        approx - type of approximation (richards or zoeprittz)

    Output:
        Vrms - array of RMS velocities    
    """
    
    R = []
    offset = []
    th = []
    Z = []
    Times = []
    c = []
    b = []
    a = []
    th1 = []
    Vint_temp = []
    for i in range(len(T_well)-1):
        Vint = (Vp_int_filt[i+1]+Vp_int_filt[i])/2
        Vint_temp.append(Vint)
        chislitel = X*X*Vint*Vint
        znamenatel = Vrms[i]*Vrms[i]*(Vrms[i]*Vrms[i]*T_well[i]*T_well[i]+X*X)
        theta = np.arcsin(np.sqrt(chislitel/znamenatel))
        theta = np.where(np.isnan(theta), 0, theta)
        dVp = Vp[i+1]-Vp[i]
        dVs = Vs[i+1]-Vs[i]
        dpho = pho[i+1]-pho[i]
        Vp_m = (Vp[i+1]+Vp[i])/2
        Vs_m = (Vs[i+1]+Vs[i])/2
        Pho_m = (pho[i+1]+pho[i])/2
        #k = (Vs[i]/Vp[i])*(Vs[i]/Vp[i])
        th1.append(theta)
        #theta = np.arctan(X/2/DH[i])
        #th1.append(theta)
        A = 1/2*(dVp/Vp_m+dpho/Pho_m)
        B = (1/2*dVp/Vp_m-4*(Vs_m/Vp_m)*(Vs_m/Vp_m)*(dVs/Vs_m)-2*(Vs_m/Vp_m)*(Vs_m/Vp_m)*dpho/Pho_m)
        C = 1/2*dVp/Vp_m
        for j in range(len(theta)):
            if approx == 'richards':
                M = A + B*np.sin(theta[j])*np.sin(theta[j])+C*(np.tan(theta[j])*np.tan(theta[j])-np.sin(theta[j])*np.sin(theta[j])) 
            else:
            #M = richards(theta[j], Vp[i], Vp[i+1], Vs[i], Vs[i+1], pho[i], pho[i+1])
                M = zoeprittz(theta[j], Vp_int_filt[i], Vp_int_filt[i+1], Vs[i], Vs[i+1], pho[i], pho[i+1])
            R.append(M)
            offset.append(X[j])
            Times.append(T_well[i])
            th.append(theta[j])
            Z.append(DH[i])

        a.append(A)
        b.append(B)
        c.append(C)     
    df_R = pd.DataFrame({'Depth':Z,'R':R,'offset':offset,'theta':th,'time':Times})
    df_R['theta'].fillna(0, inplace=True)
    df_R['Phi_'] = df_R['theta'].values/(np.pi/180)
    matr_theta=np.stack(th1,axis = 0)
    return df_R, matr_theta

def win_coef_corr(win, trace_synt_ar_int, data_short):
    """
    Calculate coefficient correlation between two seimic gathers
    
    Parameters
    ----------
    Input:
        win - sliding window size
        trace_synt_ar_int - synthetic data interpolate on seismic grid
        data_short - interval real data

    Output:
        data_KK - array with correlation coefficients   
    """

    data_KK=np.zeros((trace_synt_ar_int.shape[1],trace_synt_ar_int.shape[0]))
    sig1=[]    
    sig2=[]
    corr12=[]                

    for jj in range(0,trace_synt_ar_int.shape[0]):
        for ii in range(win,trace_synt_ar_int.shape[1]-win):
            sig1=data_short[(ii-win):(ii+win),jj]
            sig2=trace_synt_ar_int.T[(ii-win):(ii+win),jj]
            data_KK[ii,jj]=(np.mean(sig1*sig2)-np.mean(sig1)*np.mean(sig2))/(np.std(sig2)*np.std(sig1))
    data_KK = np.where(data_KK > 0.05, data_KK, data_KK*np.nan)
    return data_KK

def plot_coef_corr_seismic(offsets, trace_synt_ar_int, seismicData, T, tmp_time_max, data_KK, int_time1, int_time2, data_short, sgyname):
    """
    Draws coef. correlation betweem real and synthetic seismic data
    
    Parameters
    ----------
    Input:
        offsets - array of offsets (m)
        trace_synt_ar_int - synthetic data interpolate on seismic grid
        seismicData - seismic data class
        T - time axs for seismic
        tmp_time_max - picking of max amplitude near explored horizon
        data_KK - array with correlation coefficients
        int_time1 - upper time near explored horizon
        int_time2 - lower time near explored horizon
        data_short - interval real data
        sgy_name - name of current segy

    Output:
        subplot figure with synthetic and real data, color shows the correlation coefficient between them
    """
    #допилить граничные значения
    #T_plot = np.arange(int_time1, int_time2, seismicData.dt)
    fig,axs = plt.subplots(nrows = 1, ncols = 2,figsize=(35,12))

    plt.sca(axs[0])
    
    for ii in range(trace_synt_ar_int.shape[0]):
        plt.plot(trace_synt_ar_int[ii,:]/np.max(trace_synt_ar_int[ii,:]).T+ii+1,T,'k',alpha=1,zorder=1,lw=1)
        x = trace_synt_ar_int[ii,:]/np.max(trace_synt_ar_int[ii,:]).T+ii+1
        T_tmp_plot = np.arange(int_time1,int_time2,seismicData.dt/4)
        y = T_tmp_plot
        x2 = np.interp(T_tmp_plot,T,x)
        plt.fill_betweenx(y,ii+1,x2,where=(x2>ii+1),color='k')
        #plt.plot(ii+1,pick[ii],'c.',alpha=1,zorder=1,lw=1,markersize = 10)
    x1 = np.arange(1,ii+1)
    plt.xticks(x1,(np.round(offsets/1000,decimals = 1)),fontsize=10,)
    plt.gca().invert_yaxis()
    plt.xlabel('Offset,km')
    plt.ylabel('T, ms')
    plt.title('Synthetic') 
    plt.plot(tmp_time_max,'.r',markersize = 25)
    plt.imshow(np.zeros_like(data_KK[:,:len(offsets)]),
               alpha=1,extent = [1,len(offsets)+1,int_time2,int_time1],aspect='auto',cmap = 'Pastel1_r', vmin =0, vmax =0,)



    plt.sca(axs[1])
    
    for ii in range(data_short.shape[0]):
        plt.plot(data_short[ii,:]/np.max(data_short[ii,:]).T+ii+1,T,'k',alpha=1,zorder=1,lw=1)
        x = data_short[ii,:]/np.max(data_short[ii,:]).T+ii+1
        T_tmp_plot = np.arange(int_time1,int_time2,seismicData.dt/4)
        y = T_tmp_plot
        x2 = np.interp(T_tmp_plot,T,x)
        plt.fill_betweenx(y,ii+1,x2,where=(x2>ii+1),color='k')
        #plt.plot(ii+1,pick[ii],'c.',alpha=1,zorder=1,lw=1,markersize = 10)

    x1 = np.arange(1,ii+1)
    plt.xticks(x1,(np.round(offsets[:-2]/1000,decimals = 1)),fontsize=10,)
    plt.gca().invert_yaxis()
    plt.xlabel('Offset,km')
    plt.ylabel('T, ms')
    plt.title(sgyname)

    im = plt.imshow(data_KK[:,:len(offsets)+1],alpha=1,extent = [1,len(offsets)+1,int_time2,int_time1],
                    aspect='auto',cmap = 'rainbow', vmin =0, vmax =1,)
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Coef. correlation', rotation = 270)
    return

def plot_angle_matr(X, T_well, matr_theta):
    """
    Imshow of angle matrix
    
    Parameters
    ----------
    Input:
        X - array of offsets(m)
        T_well - time axs for well
        matr_theta - array with angles

    Output:
        figure with angles   
    """
    
    plt.figure(figsize = (15,7))
    plt.imshow(matr_theta/(np.pi/180),cmap='Set2',aspect = 'auto',extent = [X[0],X[-1],T_well[-1], T_well[0]])
    plt.xlabel('Offset')
    plt.ylabel('T')
    plt.colorbar()
    return

def reflectivity_gather(df_R):
    """
    
    Reflection coefficient gathering function
    
    Parameters
    ----------
    Input:
        df_R - dataframe with reflectivity, angles and offsets

    Output:
        data_synt_ar - array of syntethic reflectivity coefficients    
    """

    data_synt = []
    for i in range(len(df_R['offset'].unique())):
        r = df_R.loc[df_R['offset']==df_R['offset'].unique()[i]]['R']
        phi = df_R.loc[df_R['offset']==df_R['offset'].unique()[i]]['Phi_']
        data_synt.append(r)
    data_synt_ar = np.asarray(data_synt)
    return data_synt_ar


def conv_refl_imp(data_synt_ar, Sig_int):
    """
    Convolve reflictivity with seismic impulse
    
    Parameters
    ----------
    Input:
        data_synt_ar - array of syntethic reflectivity coefficients 
        Sig_int - seismic impulse

    Output:
        trace_synt_ar - array of synthetic traces 
    """
    trace_synt = []
    for i in range(len(data_synt_ar[:,0])):
        #Sig2=np.interp(T_int,time,data_synt_ar[i,:])
        trace_synt.append(np.convolve(Sig_int,data_synt_ar[i,:],mode='same'))
    trace_synt_ar = np.asarray(trace_synt)
    return trace_synt_ar


def plot_seismic_vs_synth(nmo_tmp, T_well, data_short, int_time2, int_time1, df_R, sgyname):
    """
    Plot two seismic gathers - real and synthetic
    
    Parameters
    ----------
    Input:
        nmo_tmp - synthetic gather with nmo correction and muting(optional)
        T_well - time axs for well
        data_short - interval real data
        int_time2 - lower time for analysis interval
        int_time1 - upper time for analysis interval
        df_R - dataframe with reflectivity, angels and offsets
        sgyname - name of current segy

    Output:
        figure with two seismic gathers    
    """
    fig,axs = plt.subplots(1,2, figsize = (15,5))
    im0 = axs[0].imshow(nmo_tmp,aspect='auto',cmap =seismic(),extent = [df_R['offset'].min(),df_R['offset'].max(),T_well[-1]*1e3,T_well[0]*1e3])
    axs[0].set_title('Synthetics')
    axs[0].set_xlabel('Offset,m')
    axs[0].set_ylabel('T,ms')
    im1 = axs[1].imshow(data_short,aspect='auto',cmap =seismic(),extent = [df_R['offset'].min(),df_R['offset'].max(),int_time2,int_time1])
    axs[1].set_title(sgyname)
    axs[1].set_xlabel('Offset,m')
    axs[1].set_ylabel('T,ms')
    fig.colorbar(im0,ax = axs[0])
    fig.colorbar(im1,ax = axs[1])
    return


def mute_synth(nmo, a1, b1, X, dt_well):
    """
    Mute sythetic gather
    
    Parameters
    ----------
    Input:
        nmo - synthetic gather with nmo correction
        a1 - linear coefficient for muting
        b1 - constant coefficient for muting
        X - array of offsets(m)
        dt_well - time step for well

    Output:
        nmo_tmp - synthetic gather with nmo correction and muting(optional)
    """
    
    nmo_tmp = nmo.copy()
    tmut = a1*abs(X)+b1
    #tmut[0:35]=0
    for ii in range(nmo_tmp.shape[1]):
        for jj in range(nmo_tmp.shape[0]):
            if (jj*dt_well) < tmut[ii]:
                nmo_tmp[jj][ii] = 0
    return nmo_tmp

def plot_well(well):
    """
    Plot well data
    
    Parameters
    ----------
    Input:
        well - dataframe with well data

    Output:
        figure with Vp, Vs, RHOB   
    """
    
    sns.set()
    Tmin=well['T'].min()
    Tmax=well['T'].max()

    plotdata_1= pltd.PlotData(well['RHOB'],#data x
                          well['T'],#data y
                          linewidth = 2,
                          label = 'RHOB',
                          color = 'orange'
                         )
    plotdata_2= pltd.PlotData(well['Vp'],
                              well['T'],
                              linewidth = 2,
                              label='Vp',
                              color='b'
                             )

    plotdata_3= pltd.PlotData(well['Vs'],
                              well['T'],
                              linewidth = 2,
                              label='Vs',
                              color='g'
                             )



    layers=[]
    layers.append(pltd.SubFig([plotdata_1],#data
                          y_lim=[Tmax,Tmin],#limits
                          x_label="RHOB, kg/m^3",#x label
                          y_label="Time, ms",#y label
                          title="RHO",#title
    #                       x_lim=[1200,5900]
                         ))
    layers.append(pltd.SubFig([plotdata_2],
                              y_lim=[Tmax,Tmin],
                              title="Vp",
                              x_label="Velocity, m/s",
                              y_label="Time, ms"))

    layers.append(pltd.SubFig([plotdata_3],
                              y_lim=[Tmax,Tmin],
                              title="Vs",
                              x_label="Velocity, m/s",
                              y_label="Time, ms"))

    pltd.paint_subplots(1,3,figsize = (10,14),layers=layers,wspace=0.45)
    plt.style.use('seaborn-white')
    return


def plot_well_with_VSP(df_well_int_cut):
    """
    Plot VSP data with well data
    
    Parameters
    ----------
    Input:
        df_well_int_cut - dataframe with inteprolate well and VSP data

    Output:
        figure with VSP and well data
    """
    
    plt.figure(figsize=(10,15))
    plt.plot(df_well_int_cut['Vp'],df_well_int_cut['T'], label = 'Vp')
    plt.plot(df_well_int_cut['Vs'],df_well_int_cut['T'], label = 'Vs')
    plt.plot(df_well_int_cut['Vrms'],df_well_int_cut['T'], label = 'VSP')
    plt.xlabel('Velocity, m/s')
    plt.ylabel('T, ms')
    plt.legend()
    plt.gca().invert_yaxis()
    return

def plot_pulse(imp):
    """
    Plot seimsic impulse
    
    Parameters
    ----------
    Input:
        imp - impulse class data

    Output:
        figure with impulse    
    """
    
    sns.set()
    layers=[]
    plotdata_1 = pltd.PlotData(imp.T_int2,imp.Sig,linewidth = 2)
    layers.append(pltd.SubFig([plotdata_1],x_label="Time, s",y_label="Amplitude",title="Signal") )
    pltd.paint_subplots(figsize = (10,5),layers=layers,wspace=1)
    return

def plot_interval_seismic(seismicData, sgy_data, heads, int_time1, int_time2, sgyname):
    """
    Plot interval seismic data
    
    Parameters
    ----------
    Input:
        seismicData - seismic data class
        sgy_data - array of real seismic data
        heads - headers for real seismic data
        int_time1 - upper time for analysis interval
        int_time2 - lower time for analysis interval
        sgyname - current sgyname

    Output:
        figure with seismic data in analysis interval
    """
    
    plt.style.use('seaborn-white')
    plt.figure(figsize = (10,5))
    plt.imshow(sgy_data[int(int_time1/seismicData.dt):int(int_time2/seismicData.dt),:],aspect = 'auto',
               cmap = seismic(), extent = [heads['offset'].min(),heads['offset'].max(),int_time2,int_time1,])
    plt.xlabel('Offset, m')
    plt.ylabel('T, ms')
    plt.title(sgyname)
    return


def read_well_VSP(well_name2, df_well):
    """
    Read VSP data from las-file
    
    Parameters
    ----------
    Input:
        well_name2 - name of las-file with VSP data
        df_well - dataframe with well data

    Output:
        df_well_VSP - dataframe with VSP data
    """
    
    wellData=WellData.WellData(path= well_name2,
                    params={},
                    data={"depth":"H","dt":"DT"})
    df_well_VSP= wellData.convert_to_df_data()
    df_well_VSP['Vrms'] = df_well_VSP['DT'].values
    df_well_VSP['dt'] = (df_well_VSP['H'][2]- df_well_VSP['H'][1])/df_well_VSP['Vrms']*1e3
    df_well_VSP['dt'][0] = 0
    ht0 = []
    for i in range(len(df_well_VSP)):
        ht0.append(np.sum(df_well_VSP['dt'][0:i+1].values)*2)
    ht0 = np.stack(ht0,axis = 0)
    ht0 = (ht0 + df_well['HT0'][0])
    df_well_VSP['HT0'] = ht0
    return df_well_VSP


def read_well(well_name):
    """
    Read VSP data from las-file
    
    Parameters
    ----------
    Input:
        well_name - name of las-file with well data
        
    Output:
        well_X - X coordinate of well
        well_Y - Y coordinate of well
        df_well - dataframe with well data
    """
    
    wellData=WellData.WellData(path= well_name,
                    params={},
                    data={"depth":"H","dt":"DT","dptm":"HT0","rhob":"PHO",'sdt':"Vs"})
    
    well_tmp=wellData.convert_to_df_data()
 
    df_well = pd.DataFrame()
    df_well['H'] = well_tmp['H']
    df_well['HT0'] = well_tmp['HT0']
    df_well = df_well.dropna()
    well_tmp = well_tmp.dropna()
 
    df_well['PHO'] = np.interp(df_well['HT0'], well_tmp['HT0'][~well_tmp['PHO'].isna()], well_tmp['PHO'][~well_tmp['PHO'].isna()])
    df_well['Vs'] = np.interp(df_well['HT0'], well_tmp['HT0'][~well_tmp['Vs'].isna()], well_tmp['Vs'][~well_tmp['Vs'].isna()])
    df_well['DT'] = np.interp(df_well['HT0'], well_tmp['HT0'][~well_tmp['DT'].isna()], well_tmp['DT'][~well_tmp['DT'].isna()])
    
    
    df_well['Vp'] = (1/df_well['DT'])*1000000
    well_X,well_Y=re.findall(r'\d+\.\d+', wellData.log.well["LOC"]["value"])
    
    return well_X, well_Y, df_well

def refl_times(data_synt_ar, T_well, X, Vrms):
    """
    Calculate reflection times
    
    Parameters
    ----------
    Input:
        data_synt_ar - array of synthetic reflectivity
        T_well - time axs for well
        X - array of offsets
        Vrms - RMS velocities

    Output:
        t_refl - array of reflection times    
    """
    
    t_refl = np.zeros(data_synt_ar.shape)
    for i in range(data_synt_ar.shape[0]):
        for j in range(data_synt_ar.shape[1]):
            t_refl[i,j] = np.sqrt(T_well[j]*T_well[j]+(X[i]*X[i])/(Vrms[j]*Vrms[j]))
    return t_refl

def trace_synt_int1(T_well, t_refl, dt_well, data_synt_ar):
    """
    Shift of reflection coefficients by reflection times
    
    Parameters
    ----------
    Input:
        T_well - time axs for well
        t_refl - reflection times
        dt_well - time step for well
        data_synt_ar - array of synthetic reflectivity

    Output:
        data_synt_ar_n - array of synthetic reflectivity with reflection times
        T_well_n - tmp time axs for well
    """
    
    T_well_n = np.arange(T_well[0],max(np.ravel(t_refl)),dt_well*1e-3)
    data_synt_n = []
    for i in range(data_synt_ar.shape[0]):
        data_synt_n.append(np.interp(T_well_n,t_refl[i,:],data_synt_ar[i,:],right = 0,left = 0))
    data_synt_ar_n = np.stack(data_synt_n,axis = 0)
    data_synt_ar_n= np.where(np.isnan(data_synt_ar_n), 0, data_synt_ar_n)
    return data_synt_ar_n, T_well_n

def trace_synt_int2(int_time1, int_time2, dt, T_well, data):
    """
    Interpolate synthetic data to time seismic dimension
    
    Parameters
    ----------
    Input:
        int_time1 - time axs for well
        int_time2 - reflection times
        dt - time step for seismic
        T_well - time axs for well
        data - synth data on well time grid 

    Output:
        trace_synt_ar_int - synthetic data interpolate on seismic grid
        T - interval time on seismic time grid
    """
    
    T = np.arange(int_time1,int_time2, dt)
    synt_tmp = []
    for i in range(data.shape[1]):
        synt_tmp.append(np.interp(T,T_well,data.T[i,:]))
    trace_synt_ar_int = np.stack(synt_tmp,axis = 0)
    return trace_synt_ar_int, T

def regression_aki_richards(df_R, T_well, offsets, head_new, param, tmp_time_index, key):
    """
    Calculation of regression by approximation of aki-richards
    
    Parameters
    ----------
    Input:
        df_R - dataframe with reflectivity, angels and offsets
        Vp_int_filt - array of Vp from well,
        T_well - time axs for well
        offsets - array of offsets
        head_new - dataframe with metrics and amplitudes
        param - estimation of amplitudes ('max' or 'rms')
        tmp_time_index - horizon position (time index)
        key - type of data ('real' or 'teor')

    Output:
        z - regression for amplitude distribution    
    """
    
    df_R_tmp = df_R[(df_R['time']==T_well[tmp_time_index[0]]) & (df_R['offset'] <= offsets[-1])].reset_index(drop = True)
    xi = np.sin(df_R_tmp['theta'])*np.sin(df_R_tmp['theta'])
    psi = np.sin(df_R_tmp['theta'])*np.sin(df_R_tmp['theta'])*np.tan(df_R_tmp['theta'])*np.tan(df_R_tmp['theta'])
    one = np.ones(np.size(xi))
    B_matr = head_new['ampl_'+param+'_'+key]
    A_matr = np.vstack((one,xi,psi)).T
    D = A_matr.T@A_matr
    y = A_matr.T@B_matr
    x_theor = np.linalg.pinv(D)@y
    theta_tmp = df_R_tmp['theta'].values
    z = x_theor[0]+x_theor[1]*np.sin(theta_tmp)*np.sin(theta_tmp)+x_theor[2]*np.sin(theta_tmp)*np.sin(theta_tmp)*np.tan(theta_tmp)*np.tan(theta_tmp)
    return z

def plot_metrics_distribution(interval1, interval2, head_new, param, kmean, disp, T_well, tmp_time_max, df_R, offsets):
    """
    Plot metrics and distributions
    
    Parameters
    ----------
    Input:
        interval1 - minimal offset number
        interval2 - maximum offset number
        head_new - dataframe with metrics and amplitudes
        param - estimation of amplitudes ('max' or 'rms')
        kmean - mean of difference of amplitude
        disp - dispersion of difference of amplitude
        T_well - time axs for well
        tmp_time_max - pick max amplitudes
        df_R - dataframe with reflectivity, angles and offsets
        offsets - array of offsets

    Output:
        figure with metrics and AVO distributions   
    """
    
    fig = plt.figure(figsize = (10,5))
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    #ax = plt.subplot(111)
    ax.plot(offsets[interval1:interval2],head_new['ampl_'+param+'_real'],'b.', label = 'Real ampl')
    ax.plot(offsets[interval1:interval2],head_new['ampl_'+param+'_theor'],'g.-',label = 'Synth ampl')
    ax.set_xlabel('Offset, m')
    ax.set_ylabel('norm(Ampl_'+param+')')

    ax.set_title('mean = '+str(np.round(kmean*10000)/10000)+', disp = '+str(np.round(disp*10000)/10000))

    a1 = head_new['ampl_'+param+'_theor'].values
    b1 = offsets[interval1:interval2]
    x = b1
    polynomial_features= PolynomialFeatures(degree=2)
    x_poly = polynomial_features.fit_transform(x.reshape(-1, 1))
    model = LinearRegression()
    model.fit(x_poly, a1)
    y_poly_pred = model.predict(x_poly)


    a2 = head_new['ampl_'+param+'_real'].values
    b2 = offsets[interval1:interval2]
    reg2 = LinearRegression().fit(b2.reshape(-1, 1), a2.reshape(-1, 1))


    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])


    tmp_time_index = np.where(np.round(T_well, decimals=4)==tmp_time_max[0]*1e-3)[0]

    z = regression_aki_richards(df_R, T_well, offsets, head_new, param, tmp_time_index,'real')
    ax.plot(b2,z,'r',label = 'Regression for real')


    z = regression_aki_richards(df_R, T_well, offsets, head_new, param, tmp_time_index,'theor')
    ax.plot(b2,z,'y',label = 'Regression for synth.')

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        if ((feature_name=='Percentage (threshold >30%)')| (feature_name=='Mean_diff_ampl') | (feature_name=='Mean_disp_ampl')):
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            if(max_value==min_value):
                if(max_value==-1):
                    result[feature_name] = np.NaN
                    continue
                result[feature_name]=1
                continue
            result[feature_name] = -(df[feature_name] - min_value) / (max_value - min_value)+1
            
        else:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            if(max_value==min_value):
                if(max_value==-1):
                    result[feature_name] = np.NaN
                    continue
                result[feature_name]=1
                continue
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

def create_dataframe_img(df,path):
    cmap = colors.LinearSegmentedColormap.from_list("", ["#f8696b","#fdec82", "#63be7b"])
    fig, ax = plt.subplots(figsize=(21, 2))
    cube = df["filename"]
    df["filename"] =-1
    normalized_df=normalize(df)
    df["filename"]=cube
    idx = df.columns.get_loc("filename")
    df=df.to_numpy()
    heatmap = sns.heatmap(normalized_df,ax=ax,annot=df, cmap=cmap,linewidths=1,linecolor="black",cbar=False)
    plt.tick_params(axis='y', which='major',labelrotation=0)

    plt.tick_params(axis='x', which='major', labelbottom = False, bottom=False, top = False, labeltop=True,labelrotation =0)
    for x in range(df.shape[0]):
        if(type(df[x,idx]) ==str):
            plt.text(idx + 0.5, x + 0.5, '%s' % df[x,idx],
                     horizontalalignment='center',
                     verticalalignment='center',
                     color="black",
                     )
    plt.savefig(path)
    return