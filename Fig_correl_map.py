#!/usr/bin/python

__author__ = "Mengyuan Mu"
__email__  = "mu.mengyuan815@gmail.com"

'''
Functions:
1. process multi-year dataset and calculate a few metrics
'''

from netCDF4 import Dataset
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import scipy.stats as stats
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.cm import get_cmap
from sklearn.metrics import mean_squared_error
from scipy.interpolate import griddata, interp1d
from cartopy.feature import NaturalEarthFeature, OCEAN
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from wrf import (getvar, interplevel, get_cartopy, cartopy_xlim,
                 cartopy_ylim, to_np, latlon_coords)
from common_utils import *

def read_spatial_data(land_ctl_path, land_sen_path, var_name, time_ss=None,
                      time_es=None, lat_names="lat", lon_names="lon",loc_lat=None,
                      loc_lon=None, wrf_path=None):

    '''
    Read ctl and sen data
    '''

    print("var_name= "+var_name)

    if var_name in ["Rnet"]:
        land_ctl_files= [land_ctl_path+'Lwnet_tavg/LIS.CABLE.201701-202002.nc']
        land_sen_files= [land_sen_path+'Lwnet_tavg/LIS.CABLE.201701-202002.nc']
        time, Ctl_Lwnet_tmp = read_var_multi_file(land_ctl_files, 'Lwnet_tavg', loc_lat, loc_lon, lat_names, lon_names)
        time, Sen_Lwnet_tmp = read_var_multi_file(land_sen_files, 'Lwnet_tavg', loc_lat, loc_lon, lat_names, lon_names)
        land_ctl_files= [land_ctl_path+'Swnet_tavg/LIS.CABLE.201701-202002.nc']
        land_sen_files= [land_sen_path+'Swnet_tavg/LIS.CABLE.201701-202002.nc']
        time, Ctl_Swnet_tmp = read_var_multi_file(land_ctl_files, 'Swnet_tavg', loc_lat, loc_lon, lat_names, lon_names)
        time, Sen_Swnet_tmp = read_var_multi_file(land_sen_files, 'Swnet_tavg', loc_lat, loc_lon, lat_names, lon_names)
        Ctl_tmp = Ctl_Lwnet_tmp+Ctl_Swnet_tmp
        Sen_tmp = Sen_Lwnet_tmp+Sen_Swnet_tmp
    elif var_name in ['VPD']:
        tair_ctl_files = [land_ctl_path+'Tair_f_inst/LIS.CABLE.201701-202002.nc']
        tair_sen_files = [land_sen_path+'Tair_f_inst/LIS.CABLE.201701-202002.nc']
        qair_ctl_files = [land_ctl_path+'Qair_f_inst/LIS.CABLE.201701-202002.nc']
        qair_sen_files = [land_sen_path+'Qair_f_inst/LIS.CABLE.201701-202002.nc']
        time, Tair_ctl = read_var_multi_file(tair_ctl_files, "Tair_f_inst", loc_lat, loc_lon, lat_names, lon_names)
        time, Tair_sen = read_var_multi_file(tair_sen_files, "Tair_f_inst", loc_lat, loc_lon, lat_names, lon_names)
        time, Qair_ctl = read_var_multi_file(qair_ctl_files, "Qair_f_inst", loc_lat, loc_lon, lat_names, lon_names)
        time, Qair_sen = read_var_multi_file(qair_sen_files, "Qair_f_inst", loc_lat, loc_lon, lat_names, lon_names)

        pres_ctl_files = ['/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/'
                            +'WRF_output/slp/wrfout_201701-202002.nc']
        pres_sen_files = ['/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2_obs_LAI_ALB/'
                            +'WRF_output/slp/wrfout_201701-202002.nc']
        time_wrf, Pres_ctl_tmp = read_var_multi_file(pres_ctl_files, "slp", loc_lat, loc_lon, lat_names, lon_names)
        time_wrf, Pres_sen_tmp = read_var_multi_file(pres_sen_files, "slp", loc_lat, loc_lon, lat_names, lon_names)

        time_in = []
        time_out= []
        for t in time_wrf:
            time_in.append(t.total_seconds())
        for t in time:
            time_out.append(t.total_seconds())

        f_ctl          = interp1d(np.array(time_in), Pres_ctl_tmp[:], kind='linear',fill_value='extrapolate', axis=0)
        f_sen          = interp1d(np.array(time_in), Pres_sen_tmp[:],kind='linear', fill_value='extrapolate', axis=0)
        Pres_ctl       = f_ctl(np.array(time_out))
        Pres_sen       = f_sen(np.array(time_out))
        Ctl_tmp        = qair_to_vpd(Qair_ctl, Tair_ctl, Pres_ctl)
        Sen_tmp        = qair_to_vpd(Qair_sen, Tair_sen, Pres_sen)
    else:
        if var_name in ["Tmax","Tmin",]:
            vname = 'Tair_f_inst'
        elif var_name in ["VegTmax","VegTmin"]:
            vname = 'VegT_tavg'
        elif var_name in ["SurfTmax","SurfTmin"]:
            vname = 'AvgSurfT_tavg'
        elif var_name in ["SM_top50cm",]:
            vname = 'SoilMoist_inst'
        else:
            vname = var_name
        land_ctl_files= [land_ctl_path+vname+'/LIS.CABLE.201701-202002.nc']
        land_sen_files= [land_sen_path+vname+'/LIS.CABLE.201701-202002.nc']
        time, Ctl_tmp = read_var_multi_file(land_ctl_files, vname, loc_lat, loc_lon, lat_names, lon_names)
        time, Sen_tmp = read_var_multi_file(land_sen_files, vname, loc_lat, loc_lon, lat_names, lon_names)

    # Read particular periods
    ctl_in = []
    sen_in = []

    for i in np.arange(len(time_ss)):

        # time-step into daily
        if var_name in ["SurfTmax","Tmax","VegTmax"]:
            # average of daily max
            ctl_tmp  = time_clip_to_day_max(time,Ctl_tmp,time_ss[i],time_es[i])
            sen_tmp  = time_clip_to_day_max(time,Sen_tmp,time_ss[i],time_es[i])
        elif var_name in ["SurfTmin","Tmin","VegTmin"]:
            # average of daily min
            ctl_tmp  = time_clip_to_day_min(time,Ctl_tmp,time_ss[i],time_es[i])
            sen_tmp  = time_clip_to_day_min(time,Sen_tmp,time_ss[i],time_es[i])
        elif var_name in ["SM",]:
            ctl_tmp  = time_clip_to_day(time,c_tmp,Ctl_tmp[i,0,:,:],time_es[i])
            sen_tmp  = time_clip_to_day(time,s_tmp,Sen_tmp[i,0,:,:],time_es[i])
        elif var_name in ["SM_top50cm",]:
            # top 1m soil moisture [.022, .058, .154, .409, 1.085, 2.872]
            c_tmp    = Ctl_tmp[:,0,:,:]*0.022 + Ctl_tmp[:,1,:,:]*0.058 + Ctl_tmp[:,2,:,:]*0.154 + Ctl_tmp[:,3,:,:]*0.266
            s_tmp    = Sen_tmp[:,0,:,:]*0.022 + Sen_tmp[:,1,:,:]*0.058 + Sen_tmp[:,2,:,:]*0.154 + Sen_tmp[:,3,:,:]*0.266
            ctl_tmp  = time_clip_to_day(time,c_tmp,time_ss[i],time_es[i])
            sen_tmp  = time_clip_to_day(time,s_tmp,time_ss[i],time_es[i])
        else:
            ctl_tmp  = time_clip_to_day(time,Ctl_tmp,time_ss[i],time_es[i])
            sen_tmp  = time_clip_to_day(time,Sen_tmp,time_ss[i],time_es[i])
        if i == 0:
            ctl_in   = ctl_tmp
            sen_in   = sen_tmp
        else:
            ctl_in   = np.append(ctl_in,ctl_tmp[:],axis=0)
            sen_in   = np.append(sen_in,sen_tmp[:],axis=0)

    print('np.shape(ctl_in)',np.shape(ctl_in))
    print('np.shape(sen_in)',np.shape(sen_in))

    if var_name in ['WaterTableD_tavg','WatTable']:
        ctl_in     = ctl_in/1000.
        sen_in     = sen_in/1000.
    if var_name in ['ESoil_tavg','Evap_tavg',"ECanop_tavg",'TVeg_tavg',"Rainf_tavg","Snowf_tavg","Qs_tavg","Qsb_tavg"]:
        ctl_in     = ctl_in*3600*24
        sen_in     = sen_in*3600*24
    if var_name in ['Qair_f_inst']:
        ctl_in     = ctl_in*1000
        sen_in     = sen_in*1000
    if var_name in ['GPP_tavg','NPP_tavg']:
        s2d        = 3600*24.          # s-1 to d-1
        GPP_scale  = -0.000001*12*s2d   # umol s-1 to g d-1
        ctl_in     = ctl_in*GPP_scale
        sen_in     = sen_in*GPP_scale

    return ctl_in, sen_in

def plot_correl_map(land_ctl_path, land_sen_path, var_names, summer_ss=None,summer_es=None, winter_ss=None,winter_es=None,
                    lat_names="lat", lon_names="lon", loc_lat=None, loc_lon=None, wrf_path=None, message=None,method='spearman'):

    ctl_one_summer, sen_one_summer = read_spatial_data(land_ctl_path, land_sen_path, var_names[0], summer_ss, summer_es, lat_names, lon_names,loc_lat, loc_lon, wrf_path)
    ctl_two_summer, sen_two_summer = read_spatial_data(land_ctl_path, land_sen_path, var_names[1], summer_ss, summer_es, lat_names, lon_names,loc_lat, loc_lon, wrf_path)
    one_diff_summer = sen_one_summer - ctl_one_summer
    two_diff_summer = sen_two_summer - ctl_two_summer

    ctl_one_winter, sen_one_winter = read_spatial_data(land_ctl_path, land_sen_path, var_names[0], winter_ss, winter_es, lat_names, lon_names,loc_lat, loc_lon, wrf_path)
    ctl_two_winter, sen_two_winter = read_spatial_data(land_ctl_path, land_sen_path, var_names[1], winter_ss, winter_es, lat_names, lon_names,loc_lat, loc_lon, wrf_path)
    one_diff_winter = sen_one_winter - ctl_one_winter
    two_diff_winter = sen_two_winter - ctl_two_winter

    land_ctl         = land_ctl_path+"LIS.CABLE.201701-201701.d01.nc"
    time, lats       = read_var(land_ctl, lat_names, loc_lat, loc_lon, lat_names, lon_names)
    time, lons       = read_var(land_ctl, lon_names, loc_lat, loc_lon, lat_names, lon_names)

    nlat             = np.shape(ctl_one_summer)[1]
    nlon             = np.shape(ctl_one_summer)[2]

    # ======== calcualte metrics =========
    r_summer    = np.zeros((nlat,nlon))
    p_summer    = np.zeros((nlat,nlon))

    r_winter    = np.zeros((nlat,nlon))
    p_winter    = np.zeros((nlat,nlon))

    for x in np.arange(nlat):
        for y in np.arange(nlon):

            one_tmp_summer = one_diff_summer[:,x,y]
            two_tmp_summer = two_diff_summer[:,x,y]
            one_tmp_winter = one_diff_winter[:,x,y]
            two_tmp_winter = two_diff_winter[:,x,y]

            if np.any(np.isnan(one_tmp_summer)) or np.any(np.isnan(two_tmp_summer)):
                r_summer[x,y]    = np.nan
                p_summer[x,y]    = np.nan
                r_winter[x,y]    = np.nan
                p_winter[x,y]    = np.nan
            else:
                if method == "pearson":
                    r_summer[x,y],p_summer[x,y]    = stats.pearsonr(one_tmp_summer, two_tmp_summer)
                    r_winter[x,y],p_winter[x,y]    = stats.pearsonr(one_tmp_winter, two_tmp_winter)
                elif method == 'spearman':
                    r_summer[x,y],p_summer[x,y]    = stats.spearmanr(one_tmp_summer, two_tmp_summer)
                    r_winter[x,y],p_winter[x,y]    = stats.spearmanr(one_tmp_winter, two_tmp_winter)

    # ================== Start Plotting =================
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=[6,4],sharex=False,
                sharey=False, squeeze=True, subplot_kw={'projection': ccrs.PlateCarree()})

    plt.subplots_adjust(wspace=0.12, hspace=0.)

    plt.rcParams['text.usetex']     = False
    plt.rcParams['font.family']     = "sans-serif"
    plt.rcParams['font.serif']      = "Helvetica"
    plt.rcParams['axes.linewidth']  = 1.5
    plt.rcParams['axes.labelsize']  = 12
    plt.rcParams['font.size']       = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    almost_black                    = '#262626'
    # change the tick colors also to the almost black
    plt.rcParams['ytick.color']     = almost_black
    plt.rcParams['xtick.color']     = almost_black

    # change the text colors also to the almost black
    plt.rcParams['text.color']      = almost_black

    # Change the default axis colors from black to a slightly lighter black,
    # and a little thinner (0.5 instead of 1)
    plt.rcParams['axes.edgecolor']  = almost_black
    plt.rcParams['axes.labelcolor'] = almost_black

    # set the box type of sequence number
    props = dict(boxstyle="round", facecolor='white', alpha=0.0, ec='white')

    states= NaturalEarthFeature(category="cultural", scale="50m",
                                        facecolor="none",
                                        name="admin_1_states_provinces_shp")

    # =============== CHANGE HERE ===============
    cmap  = plt.cm.BrBG# seismic_r
    clevs = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

    for i in np.arange(2):

        # start plotting
        axs[i].coastlines(resolution="50m",linewidth=1)
        axs[i].set_extent([135,155,-39,-24])
        axs[i].add_feature(states, linewidth=.5, edgecolor="black")

        # Set the ticks on the x-axis and y-axis
        axs[i].tick_params(axis='x', direction='out')
        axs[i].tick_params(axis='y', direction='out')
        x_ticks = np.arange(135, 156, 5)
        y_ticks = np.arange(-40, -20, 5)
        axs[i].set_xticks(x_ticks)
        axs[i].set_yticks(y_ticks)
        axs[i].set_facecolor('lightgray')
        axs[i].add_feature(OCEAN,edgecolor='none', facecolor="white")

    axs[0].set_xticklabels(['135$\mathregular{^{o}}$E','140$\mathregular{^{o}}$E','145$\mathregular{^{o}}$E',
                                    '150$\mathregular{^{o}}$E','155$\mathregular{^{o}}$E'],rotation=25)

    axs[1].set_xticklabels(['135$\mathregular{^{o}}$E','140$\mathregular{^{o}}$E','145$\mathregular{^{o}}$E',
                                    '150$\mathregular{^{o}}$E','155$\mathregular{^{o}}$E'],rotation=25)

    axs[0].set_yticklabels(['40$\mathregular{^{o}}$S','35$\mathregular{^{o}}$S',
                                    '30$\mathregular{^{o}}$S','25$\mathregular{^{o}}$S'])
    axs[1].set_yticklabels([])

    r_summer              = np.where(p_summer<0.05, r_summer, np.nan)
    plot1 = axs[0].contourf(lons, lats, r_summer, clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend='both') #

    r_winter              = np.where(p_winter<0.05, r_winter, np.nan)
    plot2 = axs[1].contourf(lons, lats, r_winter, clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend='both') #

    cbar = plt.colorbar(plot1, ax=axs, ticklocation="right", pad=0.13, orientation="horizontal",
            aspect=40, shrink=1.) # cax=cax,
    # cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])  # Adjust the position and size of the colorbar axes
    # cb = plt.colorbar(cax=cbar_ax, orientation="horizontal")
    cbar.set_label('r', loc='center', size=12)
    cbar.ax.tick_params(labelsize=12)

    axs[0].set_title("Winter", fontsize=12)
    axs[1].set_title("Summer", fontsize=12)

    plt.savefig('./plots/correl_map_'+method+'_'+message+'.png',dpi=300)

if __name__ == "__main__":

    # ======================= Option =======================
    region = "SE Aus" #"SE Aus" #"CORDEX" #"SE Aus"

    if region == "Aus":
        loc_lat    = [-44,-10]
        loc_lon    = [112,154]
    elif region == "SE Aus":
        loc_lat    = [-40,-23]
        loc_lon    = [135,155]
    elif region == "CORDEX":
        loc_lat    = [-52.36,3.87]
        loc_lon    = [89.25,180]

    # #################################
    # Plot WRF-CABLE vs AWAP temperal metrics
    # #################################
    if 1:
        '''
        Test WRF-CABLE output
        '''

        case_ctl       = "drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2"
        case_sen       = "drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2_obs_LAI_ALB"

        wrf_path       = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_ctl+"/WRF_output/wrfout_d01_2019-12-01_01:00:00"
        land_sen_path  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_sen+"/LIS_output/"
        land_ctl_path  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_ctl+"/LIS_output/"

        if 1:

            # Calculate correlation coefficent between Tmax and LAI_inst
            summer_ss    = [datetime(2017,1,1,0,0,0,0), datetime(2018,12,1,0,0,0,0),datetime(2019,12,1,0,0,0,0)]
            summer_es    = [datetime(2020,3,1,0,0,0,0), datetime(2019,3,1,0,0,0,0), datetime(2020,3,1,0,0,0,0)]

            winter_ss    = [datetime(2017,6,1,0,0,0,0), datetime(2018,6,1,0,0,0,0), datetime(2019,6,1,0,0,0,0)]
            winter_es    = [datetime(2017,9,1,0,0,0,0), datetime(2018,9,1,0,0,0,0), datetime(2019,9,1,0,0,0,0)]

            var_names  = ["Albedo_inst", "LAI_inst"]

            message    = "Correl_map_"+var_names[0]+"_vs_"+var_names[1]
            plot_correl_map(land_ctl_path, land_sen_path, var_names, summer_ss=summer_ss,summer_es=summer_es, winter_ss=winter_ss,winter_es=winter_es,
                            lat_names="lat", lon_names="lon", loc_lat=loc_lat, loc_lon=loc_lon, wrf_path=wrf_path, message=message,method='spearman')
