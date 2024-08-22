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
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import scipy.stats as stats
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.cm import get_cmap
from sklearn.metrics import mean_squared_error
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from wrf import (getvar, interplevel, get_cartopy, cartopy_xlim,
                 cartopy_ylim, to_np, latlon_coords)
from common_utils import *

def plot_spital_map(file_paths, var_names, time_s, time_e, loc_lat=None, loc_lon=None, lat_names=None, lon_names=None, message=None, diff=False):

    print("======== In plot_spital_map =========")
    print(var_names)
    if diff == False:
        # Open the NetCDF4 file (add a directory path if necessary) for reading:
        time1, Var1  = read_var(file_paths[0], var_names[0], loc_lat, loc_lon, lat_names[0], lon_names[0])
        print(time1)
        print("The number of pixels are Nan: ", np.count_nonzero(np.isnan(Var1)))
        time1, lats1 = read_var(file_paths[0], lat_names[0], loc_lat, loc_lon, lat_names[0], lon_names[0])
        time1, lons1 = read_var(file_paths[0], lon_names[0], loc_lat, loc_lon, lat_names[0], lon_names[0])

        if var_names[0] in ['tas','Tair','Tair_f_inst']:
            var1         = spatial_var(time1,Var1,time_s,time_e)-273.15
            print(var1)
        elif var_names[0] in ['tp']:
            scale        = get_scale(var_names[0])
            var1         = spital_ERAI_tp(time1,Var1,time_s,time_e)*scale
        elif var_names[0] in ['Rainf','Rainf_tavg']:
            var1         = spatial_var(time1,Var1,time_s,time_e)*24*60*60.
            print(var1)
        elif var_names[0] in ['Wind']:
            # !!!!!!!!! Note that !!!!!!!!!!!
            # Wind speeds is at 2 meter height in AWAP while 10 meters in WRF-CABLE
            # So here converting AWAP 2m wind speed to 10m wind speed by multipling 2
            var1         = spatial_var(time1,Var1,time_s,time_e)*2.
            print(var1)
        else:
            scale        = get_scale(var_names[0])
            var1         = spatial_var(time1,Var1,time_s,time_e)*scale

        if len(file_paths) > 1:
            time2, Var2  = read_var(file_paths[1], var_names[1], loc_lat, loc_lon, lat_names[1], lon_names[1])
            time2, lats2 = read_var(file_paths[1], lat_names[1], loc_lat, loc_lon, lat_names[1], lon_names[1])
            time2, lons2 = read_var(file_paths[1], lon_names[1], loc_lat, loc_lon, lat_names[1], lon_names[1])
            scale        = get_scale(var_names[1])
            var2         = spatial_var(time2,Var2,time_s,time_e)*scale

        if len(file_paths) > 2:
            time3, Var3  = read_var(file_paths[2], var_names[2], loc_lat, loc_lon, lat_names[2], lon_names[2])
            time3, lats3 = read_var(file_paths[2], lat_names[2], loc_lat, loc_lon, lat_names[2], lon_names[2])
            time3, lons3 = read_var(file_paths[2], lon_names[2], loc_lat, loc_lon, lat_names[2], lon_names[2])
            scale        = get_scale(var_names[2])
            var3         = spatial_var(time3,Var3,time_s,time_e)*scale

        if len(file_paths) > 3:
            time4, Var4  = read_var(file_paths[3], var_names[3], loc_lat, loc_lon, lat_names[3], lon_names[3])
            time4, lats4 = read_var(file_paths[3], lat_names[3], loc_lat, loc_lon, lat_names[3], lon_names[3])
            time4, lons4 = read_var(file_paths[3], lon_names[3], loc_lat, loc_lon, lat_names[3], lon_names[3])
            scale        = get_scale(var_names[3])
            var4         = spatial_var(time4,Var4,time_s,time_e)*scale

    elif diff == True:
        time1, Var1  = read_var(file_paths[0], var_names[0], loc_lat, loc_lon, lat_names[0], lon_names[0])
        time1, lats1 = read_var(file_paths[0], lat_names[0], loc_lat, loc_lon, lat_names[0], lon_names[0])
        time1, lons1 = read_var(file_paths[0], lon_names[0], loc_lat, loc_lon, lat_names[0], lon_names[0])

        time2, Var2  = read_var(file_paths[1], var_names[1], loc_lat, loc_lon, lat_names[1], lon_names[1])
        time2, lats2 = read_var(file_paths[1], lat_names[1], loc_lat, loc_lon, lat_names[1], lon_names[1])
        time2, lons2 = read_var(file_paths[1], lon_names[1], loc_lat, loc_lon, lat_names[1], lon_names[1])

        print("np.shape(Var1) ", np.shape(Var1))
        print("np.shape(Var2) ", np.shape(Var2))

        if var_names[0] in ['tas','Tair','Tair_f_inst']:
            var1         = spatial_var(time1,Var1,time_s,time_e)-273.15 # AWAP
            var2         = spatial_var(time2,Var2,time_s,time_e)-273.15 # WRF
            # regrid_data(lat_in, lon_in, lat_out, lon_out, input_data)
            var1_regrid  = regrid_data(lats1, lons1, lats2, lons2, var1)
            var1         = var2 - var1_regrid
        elif var_names[0] in ['tp']:
            scale        = get_scale(var_names[0])
            var1         = spital_ERAI_tp(time1,Var1,time_s,time_e)*scale
            var2         = spital_ERAI_tp(time2,Var2,time_s,time_e)*scale
            # regrid_data(lat_in, lon_in, lat_out, lon_out, input_data)
            var1_regrid  = regrid_data(lats1, lons1, lats2, lons2, var1)
            var1         = var2 - var1_regrid
        elif var_names[0] in ['Rainf','Rainf_tavg']:
            var1         = spatial_var(time1,Var1,time_s,time_e)*24*60*60.
            var2         = spatial_var(time2,Var2,time_s,time_e)*24*60*60.
            # regrid_data(lat_in, lon_in, lat_out, lon_out, input_data)
            var1_regrid  = regrid_data(lats1, lons1, lats2, lons2, var1)
            var1         = var2 - var1_regrid
        else:
            scale        = get_scale(var_names[0])
            var1         = spatial_var(time1,Var1,time_s,time_e)*scale
            var2         = spatial_var(time2,Var2,time_s,time_e)*scale
            # regrid_data(lat_in, lon_in, lat_out, lon_out, input_data)
            var1_regrid  = regrid_data(lats1, lons1, lats2, lons2, var1)
            var1         = var2 - var1_regrid

    # ================== Start Plotting =================
    fig = plt.figure(figsize=(6,5))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # start plotting
    if loc_lat == None:
        # ax.set_extent([140,154,-40,-28])
        ax.set_extent([135,155,-40,-25])
    else:
        ax.set_extent([loc_lon[0],loc_lon[1],loc_lat[0],loc_lat[1]])

    ax.coastlines(resolution="50m",linewidth=1)

    # Add gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=1, color='black', linestyle='--')
    gl.xlabels_top   = False
    gl.ylabels_right = False
    gl.xlines        = True

    if loc_lat == None:
        # gl.xlocator = mticker.FixedLocator([140,145,150])
        # gl.ylocator = mticker.FixedLocator([-40,-35,-30])
        gl.xlocator     = mticker.FixedLocator([135,140,145,150,155])
        gl.ylocator     = mticker.FixedLocator([-40,-35,-30,-25])
    else:
        gl.xlocator = mticker.FixedLocator(loc_lon)
        gl.ylocator = mticker.FixedLocator(loc_lat)

    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size':10, 'color':'black'}
    gl.ylabel_style = {'size':10, 'color':'black'}


    # plot Var1
    if var_names[0] in ['tas','Tair','Tair_f_inst']:
        clevs = np.arange( 15.,40.,1.) #np.linspace( 15.,45., num=31)
        cmap  = plt.cm.RdYlBu_r
    elif var_names[0] in ['Rainf','Rainf_tavg']:
        clevs = np.arange( 0.,22.,2.) #np.linspace( 15.,45., num=31)
        cmap  = plt.cm.Blues
    elif var_names[0] in ['Wind','Wind_f_inst']:
        clevs = np.arange( 0,10.,0.5) #np.linspace( 15.,45., num=31)
        cmap  = plt.cm.Blues
    elif var_names[0] in ['LWdown','LWdown_f_inst','SWdown','SWdown_f_inst']:
        clevs = np.arange( 80.,500.,20.) #np.linspace( 15.,45., num=31)
        cmap  = plt.cm.RdYlBu_r
    elif var_names[0] in ['Qair','Qair_f_inst']:
        # kg kg-1
        clevs = np.arange( 0.,0.02, 0.001) #np.linspace( 15.,45., num=31)
        cmap  = plt.cm.RdYlBu_r
    else:
        # clevs = np.linspace( 0.,120., num=13)
        clevs = np.linspace( 0.,5., num=11)
        cmap  = plt.cm.GnBu # BrBG
    # print(var1)
    # np.savetxt("./"+message,var1)
    plt.contourf(lons1, lats1, var1, clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend='both') #,#bwr)#coolwarm)#cm.BrBG) # clevs,

    plt.title(var_names[0], size=16)
    cb = plt.colorbar(ax=ax, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
    # cb.set_label(units,size=14,rotation=270,labelpad=15)
    cb.ax.tick_params(labelsize=10)

    # plot Var2
    if len(file_paths) > 1 and var_names[1] == 'ps':
        clevs = np.arange( -100.,100., 20.)
        cs = plt.contour(lons2, lats2, var2-1010., clevs, transform=ccrs.PlateCarree(),linewidths=0.8,colors="darkgray") #, ,cmap=plt.cm.hot_r)#bwr)#coolwarm)#cm.BrBG) # clevs,
        cl = plt.clabel(cs, inline=True, fmt="%4d",fontsize=6) #manual=True)

    # plot Var3, Var4
    if len(file_paths) > 3 and var_names[2] == 'uas' and var_names[3] == 'vas':
        qv = plt.quiver(lons1[::3,::3], lats1[::3,::3], var3[::3,::3], var4[::3,::3], scale=300, color='k')

    if message == None:
        message = var_names[0]
    else:
        message = message + "_" + var_names[0]
    if diff:
        message = message + "_diff"
    plt.savefig('./plots/weather/spatial_map_weather_analysis_'+message+'.png',dpi=300)

def plot_spital_map_multi(wrf_path, names, file_paths, var_names, time_s, time_e, loc_lat=None, loc_lon=None, lat_names=None, lon_names=None, message=None, metric=False, month=None):

    print("======== In plot_spital_map_multi =========")

    # ================== Plot setting ==================
    case_sum = np.shape(file_paths)[0]
    print(case_sum//2)
    fig, ax = plt.subplots( nrows=(case_sum // 2), ncols=2, figsize=[12,10],sharex=True, sharey=True, squeeze=True,
                            subplot_kw={'projection': ccrs.PlateCarree()})
    plt.subplots_adjust(wspace=0, hspace=0.2) # left=0.15,right=0.95,top=0.85,bottom=0.05,

    plt.rcParams['text.usetex']     = False
    plt.rcParams['font.family']     = "sans-serif"
    plt.rcParams['font.serif']      = "Helvetica"
    plt.rcParams['axes.linewidth']  = 1.5
    plt.rcParams['axes.labelsize']  = 14
    plt.rcParams['font.size']       = 14
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    almost_black = '#262626'
    # change the tick colors also to the almost black
    plt.rcParams['ytick.color']     = almost_black
    plt.rcParams['xtick.color']     = almost_black

    # change the text colors also to the almost black
    plt.rcParams['text.color']      = almost_black

    # Change the default axis colors from black to a slightly lighter black,
    # and a little thinner (0.5 instead of 1)
    plt.rcParams['axes.edgecolor']  = almost_black
    plt.rcParams['axes.labelcolor'] = almost_black

    props = dict(boxstyle="round", facecolor='white', alpha=0.0, ec='white')

    for i, file_path in enumerate(file_paths):

        row = i // 2
        col = i % 2
        print("row & col ", row, col)

        # ================== Reading data =================
        time, Var  = read_var(file_path, var_names[i], loc_lat, loc_lon, lat_names[i], lon_names[i])
        time, lats = read_var(file_path, lat_names[i], loc_lat, loc_lon, lat_names[i], lon_names[i])
        time, lons = read_var(file_path, lon_names[i], loc_lat, loc_lon, lat_names[i], lon_names[i])

        if var_names[i] in ['t2m','tas','Tair','Tair_f_inst']:
            if i == 0:
                clevs    = np.linspace( 15.,45., num=31)
                cmap     = plt.cm.coolwarm
            else:
                clevs    = [-5,-4,-3,-2,-1,-0.5,0.5,1,2,3,4,5] # np.linspace( -5., 5., num=11)
                cmap     = plt.cm.coolwarm
            var      = spatial_var(time,Var,time_s,time_e)-273.15
        elif var_names[i] in ['Rainf','Rainf_tavg','tp']:
            if i == 0:
                clevs    = np.linspace( 0., 300., num=16)
                cmap     = plt.cm.BrBG
            else:
                clevs    = [-180,-160,-140,-120,-100,-80,-60,-40,-20,20,40,60,80,100,120,140,160,180]
                cmap     = plt.cm.BrBG #RdYlBu_r

            if month == "01":
                var      = spatial_var(time,Var,time_s,time_e)*24*60*60.*30
            elif month == "02":
                var      = spatial_var(time,Var,time_s,time_e)*24*60*60.*28
            elif month == "03":
                var      = spatial_var(time,Var,time_s,time_e)*24*60*60.*31
            else:
                var      = spatial_var(time,Var,time_s,time_e)*24*60*60.

        elif var_names[i] in ['LWdown','LWdown_f_inst','SWdown','SWdown_f_inst']:
            if i == 0:
                clevs = np.arange( 80.,520.,20.) #np.linspace( 15.,45., num=31)
            else:
                clevs = np.arange( -90.,100.,10.)
            cmap  = plt.cm.BrBG_r
            scale = get_scale(var_names[i])
            var   = spatial_var(time,Var,time_s,time_e)*scale
        elif var_names[i] in ['Wind','Wind_f_inst']:
            if i == 0:
                clevs = np.arange( 0,10.5,0.5) #np.linspace( 15.,45., num=31)
                var   = spatial_var(time,Var,time_s,time_e)*2
            else:
                clevs = np.arange( -5,5.5,0.5)
                var   = spatial_var(time,Var,time_s,time_e)
            cmap  = plt.cm.BrBG
        elif var_names[i] in ['Qair','Qair_f_inst']:
            # kg kg-1
            if i == 0:
                clevs = np.arange( 0.,0.02, 0.001) #np.linspace( 15.,45., num=31)
            else:
                clevs = np.arange( -0.006,0.007, 0.001)
            cmap  = plt.cm.BrBG
            scale = get_scale(var_names[i])
            var   = spatial_var(time,Var,time_s,time_e)*scale
        else:
            if i == 0:
                clevs = np.linspace( 0.,5., num=11)
            else:
                clevs = np.linspace( -5.,5., num=11)
            cmap  = plt.cm.GnBu # BrBG
            scale = get_scale(var_names[i])
            var   = spatial_var(time,Var,time_s,time_e)*scale

        if i == 0:
            # save AWAP dataset
            lat_AWAP   = lats
            lon_AWAP   = lons
            var_AWAP   = var
            print("lat_AWAP ", lat_AWAP)
            print("lon_AWAP ", lon_AWAP)
            print("var_AWAP ", var_AWAP)
            wrf        = Dataset(wrf_path,  mode='r')
            lons_out   = wrf.variables['XLONG'][0,:,:]
            lats_out   = wrf.variables['XLAT'][0,:,:]
            AWAP_regrid= regrid_data(lat_AWAP, lon_AWAP, lats_out, lons_out, var_AWAP)
            var        = AWAP_regrid
            lats       = lats_out
            lons       = lons_out
        elif i == 1:
            var        = var - AWAP_regrid
        else:
            var        = var - AWAP_regrid

        # =============== setting plots ===============
        if loc_lat == None:
            # ax.set_extent([140,154,-40,-28])
            ax[row,col].set_extent([135,155,-40,-25])
        else:
            ax[row,col].set_extent([loc_lon[0],loc_lon[1],loc_lat[0],loc_lat[1]])

        ax[row,col].coastlines(resolution="50m",linewidth=1)

        # Add gridlines
        gl               = ax[row,col].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=1, color='black', linestyle='--')

        if loc_lat == None:
            gl.xlocator = mticker.FixedLocator([135,140,145,150,155])
            gl.ylocator = mticker.FixedLocator([-40,-35,-30,-25])
        else:
            gl.xlocator = mticker.FixedLocator([135,140,144.2,150,155])
            gl.ylocator = mticker.FixedLocator([-40,-35,-31.8,-25])

        gl.xformatter   = LONGITUDE_FORMATTER
        gl.yformatter   = LATITUDE_FORMATTER
        gl.xlabel_style = {'size':10, 'color':'black'}
        gl.ylabel_style = {'size':10, 'color':'black'}
        gl.xlabels_bottom= True
        gl.xlabels_top   = False
        gl.ylabels_left  = True
        gl.ylabels_right = False
        gl.xlines        = True
        gl.ylines        = True
        plot1 = ax[row, col].contourf(lons, lats, var, clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend='both') #,#bwr)#coolwarm)#cm.BrBG) # clevs,
        cb    = plt.colorbar(plot1, ax=ax[row, col], orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
        cb.ax.tick_params(labelsize=7)
        ax[row, col].set_title(names[i], size=12)
    # cb.set_label(units,size=14,rotation=270,labelpad=15)

        # ============ metrics ============
        if i > 0 and metric:
            AWAP_regrid_1D = np.reshape(AWAP_regrid,-1)
            var_1D         = np.reshape(var,-1)
            print("first")
            print(AWAP_regrid_1D)
            print(var_1D)
            var_1D         = np.where(np.isnan(AWAP_regrid_1D), np.nan, var_1D)
            AWAP_regrid_1D = np.where(np.isnan(var_1D), np.nan, AWAP_regrid_1D)
            var_1D         = np.where(np.isnan(AWAP_regrid_1D), np.nan, var_1D)
            print("second")
            print(AWAP_regrid_1D)
            print(var_1D)
            var_1D         = var_1D[~np.isnan(var_1D)]
            AWAP_regrid_1D = AWAP_regrid_1D[~np.isnan(AWAP_regrid_1D)]
            print("third")
            print(AWAP_regrid_1D)
            print(var_1D)

            r    = stats.pearsonr(AWAP_regrid_1D, var_1D)[0]
            RMSE = np.sqrt(mean_squared_error(AWAP_regrid_1D, var_1D))
            MBE  = np.mean(var_1D - AWAP_regrid_1D)
            # p5   = np.percentile(cable, 5) - np.percentile(obs, 5)
            # p95  = np.percentile(cable, 95) - np.percentile(obs, 95)
            text = "r=" + "%.2f" % r + ", RMSE=" + "%.2f" % RMSE + ", MBE=" + "%.2f" % MBE
            ax[row, col].text(0.02, 0.95, text, transform=ax[row, col].transAxes, verticalalignment='top', bbox=props, fontsize=10)

    if message == None:
        message = var_names[0]
    else:
        message = message + "_" + var_names[0]

    plt.savefig('./plots/weather/spatial_map_weather_analysis_'+message+'_multi.png',dpi=300)

def spital_map_temperal_metrics(wrf_path, obs_path, obs_name, file_paths, var_name, time_s, time_e,
                                loc_lat=None, loc_lon=None, lat_obs_name=None, lon_obs_name=None,
                                lat_var_name=None, lon_var_name=None):

    '''
    file_paths : the multi year files
    '''

    print("======== In spital_map_temperal_metrics =========")

    # ================== Reading data =================
    time_var, Var_tmp = read_var_multi_file(file_paths, var_name, loc_lat, loc_lon, lat_var_name, lon_var_name)
    time_obs, Obs_tmp = read_var_multi_file(obs_path, obs_name, loc_lat, loc_lon, lat_obs_name, lon_obs_name)

    time, lats_obs    = read_var(obs_path[0], lat_obs_name, loc_lat, loc_lon, lat_obs_name, lon_obs_name)
    time, lons_obs    = read_var(obs_path[0], lon_obs_name, loc_lat, loc_lon, lat_obs_name, lon_obs_name)

    # clip and resample sims to daily data
    # if var_name in ["WaterTableD_tavg","Rainf_tavg","Evap_tavg","TVeg_tavg","ESoil_tavg","ECanop_tavg","Qs_tavg","Qsb_tavg"]:
    #     Var_daily = time_clip_to_day(time_var, Var_tmp, time_s, time_e, seconds=None)
    # else:
    #     Var_daily = time_clip_to_day(time_var, Var_tmp, time_s, time_e, seconds=None)
    Var_daily = time_clip_to_day(time_var, Var_tmp, time_s, time_e, seconds=None)

    ntime     = len(Var_daily[:,0,0])
    nlat      = len(Var_daily[0,:,0])
    nlon      = len(Var_daily[0,0,:])
    print("var dimension is ", np.shape(Var_daily))

    # clip time coordinate
    # if obs_name in ["E"]:
    #     Obs_daily = time_clip_to_day_sum(time_obs, Obs_tmp, time_s, time_e)
    # else:
    Obs_daily = time_clip_to_day(time_obs, Obs_tmp, time_s, time_e, seconds=None)
    print("obs dimension is ", np.shape(Obs_daily))

    # read no-nan wrf-cable lat and lon
    wrf        = Dataset(wrf_path,  mode='r')
    lons_out   = wrf.variables['XLONG'][0,:,:]
    lats_out   = wrf.variables['XLAT'][0,:,:]

    # interpolate obs
    Obs_regrid = np.zeros((ntime,nlat,nlon))
    for t in np.arange(ntime):
        print("t=",t)
        Obs_regrid[t, :, :] = regrid_data(lats_obs, lons_obs, lats_out, lons_out, Obs_daily[t,:,:])

    if var_name in ['t2m','tas','Tair','Tair_f_inst']:
        var    = Var_daily - 273.15
        obs    = Obs_regrid - 273.15
    elif var_name in ['Rainf','Rainf_tavg','tp']:
        var    = Var_daily*60*60.*24
        obs    = Obs_regrid*60*60.*24 # for AWAP 3 hourly data
    elif var_name in ['Evap_tavg','TVeg_tavg','ESoil_tavg','ECanop_tavg','Qs_tavg','Qsb_tavg']:
        var    = Var_daily*60*60.*24
        obs    = Obs_regrid
    elif var_name in ['Wind','Wind_f_inst']:
        var    = Var_daily
        obs    = Obs_regrid*2
    else:
        var    = Var_daily
        obs    = Obs_regrid

    # ============ metrics ============
    print("======== calcualte metrics =========")
    r    = np.zeros((nlat,nlon))
    RMSE = np.zeros((nlat,nlon))
    for x in np.arange(nlat):
        for y in np.arange(nlon):
            obs_tmp = obs[:,x,y]
            var_tmp = var[:,x,y]
            if np.any(np.isnan(obs_tmp)) or np.any(np.isnan(var_tmp)):
                r[x,y]    = np.nan
                RMSE[x,y] = np.nan
            else:
                r[x,y]    = stats.pearsonr(obs_tmp, var_tmp)[0]
                RMSE[x,y] = np.sqrt(mean_squared_error(obs_tmp, var_tmp))
    if var_name in ['Rainf','Rainf_tavg','tp','Evap_tavg','Qs_tavg','Qsb_tavg']:
        MBE  = np.sum((var - obs),axis=0)/3
    else:
        MBE  = np.mean((var - obs),axis=0)
    p5   = np.percentile(var, 5, axis=0) - np.percentile(obs, 5, axis=0)
    p95  = np.percentile(var, 95, axis=0) - np.percentile(obs, 95, axis=0)

    print("r=",r)
    print("RMSE=",RMSE)
    print("MBE=",MBE)
    print("p5=",p5)
    print("p95=",p95)

    if var_name in ['t2m','tas','Tair','Tair_f_inst','Wind','Wind_f_inst']:
        var    = np.mean(var,axis=0)
        obs    = np.mean(obs,axis=0)
    elif var_name in ['Rainf','Rainf_tavg','tp','Evap_tavg','Qs_tavg','Qsb_tavg']:
        var    = np.sum(var,axis=0)
        obs    = np.sum(obs,axis=0)
        var    = var/3.
        obs    = obs/3.
    else:
        var    = np.mean(var,axis=0)
        obs    = np.mean(obs,axis=0)

    return lats_out, lons_out, var, obs, r, RMSE, MBE, p5, p95

def plot_map_temperal_metrics(wrf_path, obs_path, obs_name, file_paths, var_name, time_s, time_e,
                              loc_lat=None, loc_lon=None, lat_obs_name=None, lon_obs_name=None,
                              lat_var_name=None, lon_var_name=None, message=None):

    print("======== In plot_map_temperal_metrics =========")

    # ================== Reading data =================
    lats, lons, var, obs, r, RMSE, MBE, p5, p95 = \
                        spital_map_temperal_metrics(wrf_path, obs_path, obs_name,
                        file_paths, var_name, time_s, time_e, loc_lat, loc_lon,
                        lat_obs_name, lon_obs_name, lat_var_name, lon_var_name)


    # ================== Plot setting ==================
    fig, ax = plt.subplots( nrows=3, ncols=2, figsize=[12,10],sharex=True, sharey=True, squeeze=True,
                            subplot_kw={'projection': ccrs.PlateCarree()})
    plt.subplots_adjust(wspace=0, hspace=0.2) # left=0.15,right=0.95,top=0.85,bottom=0.05,

    plt.rcParams['text.usetex']     = False
    plt.rcParams['font.family']     = "sans-serif"
    plt.rcParams['font.serif']      = "Helvetica"
    plt.rcParams['axes.linewidth']  = 1.5
    plt.rcParams['axes.labelsize']  = 14
    plt.rcParams['font.size']       = 14
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    almost_black = '#262626'
    # change the tick colors also to the almost black
    plt.rcParams['ytick.color']     = almost_black
    plt.rcParams['xtick.color']     = almost_black

    # change the text colors also to the almost black
    plt.rcParams['text.color']      = almost_black

    # Change the default axis colors from black to a slightly lighter black,
    # and a little thinner (0.5 instead of 1)
    plt.rcParams['axes.edgecolor']  = almost_black
    plt.rcParams['axes.labelcolor'] = almost_black

    props = dict(boxstyle="round", facecolor='white', alpha=0.0, ec='white')

    # =============== setting plots ===============
    for row in np.arange(0,3):
        for col in np.arange(0,2):
            if loc_lat == None:
                ax[row,col].set_extent([135,155,-40,-25])
            else:
                ax[row,col].set_extent([loc_lon[0],loc_lon[1],loc_lat[0],loc_lat[1]])
            ax[row,col].coastlines(resolution="50m",linewidth=1)

            # Add gridlines
            gl  = ax[row,col].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=1, color='black', linestyle='--')

            if loc_lat == None:
                gl.xlocator = mticker.FixedLocator([135,140,145,150,155])
                gl.ylocator = mticker.FixedLocator([-40,-35,-30,-25])
            else:
                gl.xlocator = mticker.FixedLocator([135,140,144.2,150,155])
                gl.ylocator = mticker.FixedLocator([-40,-35,-31.8,-25])

            gl.xformatter   = LONGITUDE_FORMATTER
            gl.yformatter   = LATITUDE_FORMATTER
            gl.xlabel_style = {'size':10, 'color':'black'}
            gl.ylabel_style = {'size':10, 'color':'black'}
            gl.xlabels_bottom= True
            gl.xlabels_top   = False
            gl.ylabels_left  = True
            gl.ylabels_right = False
            gl.xlines        = False
            gl.ylines        = False

    if var_name in ['t2m','tas','Tair','Tair_f_inst']:
        cmap1  = plt.cm.coolwarm
        cmap2  = plt.cm.coolwarm
        cmap3  = plt.cm.coolwarm
        cmap4  = plt.cm.coolwarm
        cmap5  = plt.cm.coolwarm
        cmap6  = plt.cm.coolwarm
        clevs1 = np.arange( 0, 42, 2)
        clevs2 = np.arange( 0, 42, 2)
        clevs3 = np.arange( 0.,5.5,0.5)
        clevs4 = [-5,-4,-3,-2,-1,-0.5,0.5,1,2,3,4,5]
        clevs5 = [-5,-4,-3,-2,-1,-0.5,0.5,1,2,3,4,5]
        clevs6 = [-5,-4,-3,-2,-1,-0.5,0.5,1,2,3,4,5]
    elif var_name in ['Rainf','Rainf_tavg','tp']:
        cmap1  = plt.cm.BrBG
        cmap2  = plt.cm.BrBG
        cmap3  = plt.cm.BrBG
        cmap4  = plt.cm.BrBG
        cmap5  = plt.cm.BrBG
        cmap6  = plt.cm.BrBG
        clevs1 = np.arange( 0.,1050.,50.)
        clevs2 = np.arange( 0.,1050.,50.)
        clevs3 = np.arange( 0.0,11, 1.)
        clevs4 = [-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,10,20,30,40,50,60,70,80,90,100]
        clevs5 = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9,10]
        #[-1,-0.8,-0.6,-0.4,-0.2,0.2,0.4,0.6,0.8,1.]
        clevs6 = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9,10]
        # [-10,-8,-6,-4,-2,-1,1,2,4,6,8,10]
    elif var_name in ['E','Evap_tavg']:
        cmap1  = plt.cm.BrBG
        cmap2  = plt.cm.BrBG
        cmap3  = plt.cm.BrBG
        cmap4  = plt.cm.BrBG
        cmap5  = plt.cm.BrBG
        cmap6  = plt.cm.BrBG
        clevs1 = np.arange( 0.,1050.,50.)
        clevs2 = np.arange( 0.,1050.,50.)
        clevs3 = np.arange( 0.,5.5,0.5)
        clevs4 = [-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,10,20,30,40,50,60,70,80,90,100]
        clevs5 = [-5.,-4.5,-4.,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0.5,1,1.5,2.,2.5,3,3.5,4.,4.5,5.]
        #[-1,-0.8,-0.6,-0.4,-0.2,0.2,0.4,0.6,0.8,1.]
        clevs6 = [-5.,-4.5,-4.,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0.5,1,1.5,2.,2.5,3,3.5,4.,4.5,5.]
        #[-10,-8,-6,-4,-2,-1,1,2,4,6,8,10]
    # elif var_name in ['hfls','Qle_tavg']:
    #     cmap1  = plt.cm.YlGnBu
    #     cmap2  = plt.cm.YlGnBu
    #     cmap3  = plt.cm.coolwarm_r
    #     cmap4  = plt.cm.coolwarm_r
    #     cmap5  = plt.cm.coolwarm_r
    #     cmap6  = plt.cm.coolwarm_r
    #     clevs1 = np.arange( 0.0,55.,5)
    #     clevs2 = np.arange( 0.0,55.,5)
    #     clevs3 = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9,10]
    #     clevs4 = [-1,-0.8,-0.6,-0.4,-0.2,0.2,0.4,0.6,0.8,1.]
    #     clevs5 = [-30,-25,-20,-15,-10,-5,-2,2,5,10,15,20,25,30]

    # var
    plot1 = ax[0, 0].contourf(lons, lats, var, clevs1, transform=ccrs.PlateCarree(), cmap=cmap1, extend='both')
    cb    = plt.colorbar(plot1, ax=ax[0, 0], orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
    ax[0, 0].text(0.02, 0.95, "Sim", transform=ax[0, 0].transAxes, verticalalignment='top', bbox=props, fontsize=12)

    # obs
    plot1 = ax[0, 1].contourf(lons, lats, obs, clevs2, transform=ccrs.PlateCarree(), cmap=cmap2, extend='both')
    cb    = plt.colorbar(plot1, ax=ax[0, 1], orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
    ax[0, 1].text(0.02, 0.95, "Obs", transform=ax[0, 1].transAxes, verticalalignment='top', bbox=props, fontsize=12)

    # RMSE
    plot1 = ax[1, 0].contourf(lons, lats, RMSE, clevs3, transform=ccrs.PlateCarree(), cmap=cmap3, extend='both')
    cb    = plt.colorbar(plot1, ax=ax[1, 0], orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
    ax[1, 0].text(0.02, 0.95, "RMSE=" + "%.2f" % np.nanmean(RMSE), transform=ax[1, 0].transAxes, verticalalignment='top', bbox=props, fontsize=12)

    # MBE
    plot1 = ax[1, 1].contourf(lons, lats, MBE, clevs4, transform=ccrs.PlateCarree(), cmap=cmap4, extend='both')
    cb    = plt.colorbar(plot1, ax=ax[1, 1], orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
    ax[1, 1].text(0.02, 0.95, "MBE=" + "%.2f" % np.nanmean(MBE), transform=ax[1, 1].transAxes, verticalalignment='top', bbox=props, fontsize=12)

    # p5
    plot1 = ax[2, 0].contourf(lons, lats, p5, clevs5, transform=ccrs.PlateCarree(), cmap=cmap5, extend='both')
    cb    = plt.colorbar(plot1, ax=ax[2, 0], orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
    ax[2, 0].text(0.02, 0.95, "p5=" + "%.2f" % np.nanmean(p5), transform=ax[2, 0].transAxes, verticalalignment='top', bbox=props, fontsize=12)

    # p95
    plot1 = ax[2, 1].contourf(lons, lats, p95, clevs6, transform=ccrs.PlateCarree(), cmap=cmap6, extend='both')
    cb    = plt.colorbar(plot1, ax=ax[2, 1], orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
    ax[2, 1].text(0.02, 0.95, "p95=" + "%.2f" % np.nanmean(p95), transform=ax[2, 1].transAxes, verticalalignment='top', bbox=props, fontsize=12)

    # cb.ax.tick_params(labelsize=7)

    if message == None:
        message = var_name
    else:
        message = message + "_" + var_name

    plt.savefig('./plots/model_evaluation/spatial_map_temperal_metrics_'+message+'.png',dpi=300)

if __name__ == "__main__":

    # ======================= Option =======================
    region = "SE Aus" #"SE Aus" #"CORDEX" #"SE Aus"

    # ====================== Pre-load =======================
    AWAP_path    = '/g/data/w97/Shared_data/AWAP_3h_v1'
    AWAP_T_file  = AWAP_path + '/Tair/AWAP.Tair.3hr.2017.nc'     # air temperature
    AWAP_R_file  = AWAP_path + '/Rainf/AWAP.Rainf.3hr.2017.nc'   # Daily rainfall
    AWAP_LW_file = AWAP_path + '/LWdown/AWAP.LWdown.3hr.2017.nc'   # Downward Longwave Radiation
    AWAP_SW_file = AWAP_path + '/SWdown/AWAP.SWdown.3hr.2017.nc'  # Downward Shortwave Radiation
    AWAP_W_file  = AWAP_path + '/Wind/AWAP.Wind.3hr.2017.nc'     # Near surface wind speed
    AWAP_Q_file  = AWAP_path + '/Qair/AWAP.Qair.3hr.2017.nc'    # Near surface specific humidity

    DOLCE_path   = "/g/data/w97/mm3972/data/DOLCE/v3/"
    DOLCE_file   = DOLCE_path+"DOLCE_v3_2000-2018.nc"
    GLEAM_path   = "/g/data/ua8/GLEAM_v3-5/v3-6a/daily/"
    GRACE_path   = "/g/data/w97/mm3972/data/GRACE/GRACE_JPL_RL06/GRACE_JPLRL06M_MASCON/"
    GRACE_file   = GRACE_path + "GRCTellus.JPL.200204_202004.GLO.RL06M.MSCNv02CRI.nc"

    if region == "Aus":
        loc_lat    = [-44,-10]
        loc_lon    = [112,154]
    elif region == "SE Aus":
        loc_lat    = [-40,-25]
        loc_lon    = [135,155]
    elif region == "CORDEX":
        loc_lat    = [-52.36,3.87]
        loc_lon    = [89.25,180]

    # #################################
    # Plot WRF-CABLE vs AWAP temperal metrics
    # #################################
    if 1:
        case_names =[ "drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2",
                      "drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2_obs_LAI_ALB",]   
        time_s     = datetime(2017,1,1,0,0,0,0)
        # time_e     = datetime(2019,12,31,23,59,0,0)
        time_e     = datetime(2019,1,1,0,0,0,0)
        for case_name in case_names:
            message    = "WRF_vs_AWAP_" + case_name + "_201701-201912"
            wrf_path   = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/WRF_output/wrfout_d01_2017-02-01_06:00:00"
            file_paths = ["/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_name+"/LIS_output/LIS.CABLE.201701-201701.d01.nc",
                          "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_name+"/LIS_output/LIS.CABLE.201702-201702.d01.nc",
                          "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_name+"/LIS_output/LIS.CABLE.201703-201703.d01.nc",
                          "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_name+"/LIS_output/LIS.CABLE.201704-201704.d01.nc",
                          "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_name+"/LIS_output/LIS.CABLE.201705-201705.d01.nc",
                          "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_name+"/LIS_output/LIS.CABLE.201706-201706.d01.nc",
                          "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_name+"/LIS_output/LIS.CABLE.201707-201707.d01.nc",
                          "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_name+"/LIS_output/LIS.CABLE.201708-201708.d01.nc",
                          "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_name+"/LIS_output/LIS.CABLE.201709-201709.d01.nc",
                          "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_name+"/LIS_output/LIS.CABLE.201710-201710.d01.nc",
                          "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_name+"/LIS_output/LIS.CABLE.201711-201711.d01.nc",
                          "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_name+"/LIS_output/LIS.CABLE.201712-201712.d01.nc",
                          "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_name+"/LIS_output/LIS.CABLE.201801-201801.d01.nc",
                          "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_name+"/LIS_output/LIS.CABLE.201802-201802.d01.nc",
                          "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_name+"/LIS_output/LIS.CABLE.201803-201803.d01.nc",
                          "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_name+"/LIS_output/LIS.CABLE.201804-201804.d01.nc",
                          "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_name+"/LIS_output/LIS.CABLE.201805-201805.d01.nc",
                          "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_name+"/LIS_output/LIS.CABLE.201806-201806.d01.nc",
                          "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_name+"/LIS_output/LIS.CABLE.201807-201807.d01.nc",
                          "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_name+"/LIS_output/LIS.CABLE.201808-201808.d01.nc",
                          "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_name+"/LIS_output/LIS.CABLE.201809-201809.d01.nc",
                          "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_name+"/LIS_output/LIS.CABLE.201810-201810.d01.nc",
                          "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_name+"/LIS_output/LIS.CABLE.201811-201811.d01.nc",
                          "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_name+"/LIS_output/LIS.CABLE.201812-201812.d01.nc",]
                        #   "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/"+case_name+"/LIS_output/LIS.CABLE.201901-201901.d01.nc",
                        #   "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/"+case_name+"/LIS_output/LIS.CABLE.201902-201902.d01.nc",
                        #   "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/"+case_name+"/LIS_output/LIS.CABLE.201903-201903.d01.nc",
                        #   "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/"+case_name+"/LIS_output/LIS.CABLE.201904-201904.d01.nc",
                        #   "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/"+case_name+"/LIS_output/LIS.CABLE.201905-201905.d01.nc",
                        #   "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/"+case_name+"/LIS_output/LIS.CABLE.201906-201906.d01.nc",
                        #   "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/"+case_name+"/LIS_output/LIS.CABLE.201907-201907.d01.nc",
                        #   "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/"+case_name+"/LIS_output/LIS.CABLE.201908-201908.d01.nc",
                        #   "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/"+case_name+"/LIS_output/LIS.CABLE.201909-201909.d01.nc",
                        #   "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/"+case_name+"/LIS_output/LIS.CABLE.201910-201910.d01.nc",
                        #   "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/"+case_name+"/LIS_output/LIS.CABLE.201911-201911.d01.nc",
                        #   "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/"+case_name+"/LIS_output/LIS.CABLE.201912-201912.d01.nc",]
                          # "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/"+case_name+"/LIS_output/LIS.CABLE.202001-202001.d01.nc",
                          # "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/"+case_name+"/LIS_output/LIS.CABLE.202002-202002.d01.nc",
                          # "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/"+case_name+"/LIS_output/LIS.CABLE.202003-202003.d01.nc",
                          # "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/"+case_name+"/LIS_output/LIS.CABLE.202004-202004.d01.nc",
                          # "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/"+case_name+"/LIS_output/LIS.CABLE.202005-202005.d01.nc",
                          # "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/"+case_name+"/LIS_output/LIS.CABLE.202006-202006.d01.nc", ]

            print(file_paths)


            # 'plot Evap'
            if 1:
                obs_path       = [ GLEAM_path + "2017/E_2017_GLEAM_v3.6a.nc",
                                   GLEAM_path + "2018/E_2018_GLEAM_v3.6a.nc",]
                                #    GLEAM_path + "2019/E_2019_GLEAM_v3.6a.nc",]
                                #  GLEAM_path + "2019/E_2020_GLEAM_v3.6a.nc"    ]
                obs_name       = 'E'
                var_name       = 'Evap_tavg'

                lat_var_name   = 'lat'
                lon_var_name   = 'lon'
                lat_obs_name   = 'lat'
                lon_obs_name   = 'lon'

                plot_map_temperal_metrics(wrf_path, obs_path, obs_name, file_paths, var_name, time_s, time_e,
                                    loc_lat=loc_lat, loc_lon=loc_lon, lat_obs_name=lat_obs_name, lon_obs_name=lon_obs_name,
                                    lat_var_name=lat_var_name, lon_var_name=lon_var_name, message=message)


            # 'plot Tair'
            if 1:
                obs_path       = [ AWAP_path+'/Tair/AWAP.Tair.3hr.2017.nc',
                                   AWAP_path+'/Tair/AWAP.Tair.3hr.2018.nc',]
                                #    AWAP_path+'/Tair/AWAP.Tair.3hr.2019.nc' ] #[AWAP_R_file] #
                obs_name       = 'Tair' # 'Rainf'        #
                var_name       = 'Tair_f_inst' # 'Rainf_tavg'   #

                lat_var_name   = 'lat'
                lon_var_name   = 'lon'
                lat_obs_name   = 'lat'
                lon_obs_name   = 'lon'

                plot_map_temperal_metrics(wrf_path, obs_path, obs_name, file_paths, var_name, time_s, time_e,
                                    loc_lat=loc_lat, loc_lon=loc_lon, lat_obs_name=lat_obs_name, lon_obs_name=lon_obs_name,
                                    lat_var_name=lat_var_name, lon_var_name=lon_var_name, message=message)

            # 'plot Rainf'
            if 1:
                obs_path       = [ AWAP_path+'/Rainf/AWAP.Rainf.3hr.2017.nc',
                                   AWAP_path+'/Rainf/AWAP.Rainf.3hr.2018.nc',]
                                #    AWAP_path+'/Rainf/AWAP.Rainf.3hr.2019.nc' ] #[AWAP_R_file] #
                obs_name       = 'Rainf' # 'Rainf'        #
                var_name       = 'Rainf_tavg' # 'Rainf_tavg'   #

                lat_var_name   = 'lat'
                lon_var_name   = 'lon'
                lat_obs_name   = 'lat'
                lon_obs_name   = 'lon'

                plot_map_temperal_metrics(wrf_path, obs_path, obs_name, file_paths, var_name, time_s, time_e,
                                    loc_lat=loc_lat, loc_lon=loc_lon, lat_obs_name=lat_obs_name, lon_obs_name=lon_obs_name,
                                    lat_var_name=lat_var_name, lon_var_name=lon_var_name, message=message)
