#!/usr/bin/env python

__author__ = "Mengyuan Mu"
__email__  = "mu.mengyuan815@gmail.com"

import os
import sys
import cartopy
import pandas as pd
import numpy as np
import shapefile as shp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import scipy.ndimage as ndimage
from scipy.interpolate import griddata, interp1d
from netCDF4 import Dataset,num2date
from datetime import datetime, timedelta
from matplotlib.cm import get_cmap
from matplotlib.patches import Polygon
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.feature import NaturalEarthFeature
from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim,
                        cartopy_ylim, latlon_coords, ALL_TIMES)
from common_utils import *

def regrid_to_fire_map_resolution(fire_path, var_in, lat_in, lon_in, loc_lat=None, loc_lon=None, burn=1):

    # =========== Read in fire data ============
    print("burn=",burn)
    fire_file  = Dataset(fire_path, mode='r')
    Burn_Date  = fire_file.variables['Burn_Date'][0:8,:,:]  # 2019-07 - 2020-02
    lat_out    = fire_file.variables['lat'][:]
    lon_out    = fire_file.variables['lon'][:]

    var_in     = np.where(var_in<-1,np.nan,var_in)
    var_in     = np.where(var_in>1000,np.nan,var_in)

    var_regrid = regrid_data(lat_in, lon_in, lat_out, lon_out, var_in, method='nearest')

    # burnt region from 2019-07 to 2020-02
    burn_area  = np.where( Burn_Date[0,:,:] + Burn_Date[1,:,:] + Burn_Date[2,:,:] + Burn_Date[3,:,:] +
                           Burn_Date[4,:,:] + Burn_Date[5,:,:] + Burn_Date[6,:,:] + Burn_Date[7,:,:] > 0, 1, Burn_Date[0,:,:])

    print("np.unique(burn_area)",np.unique(burn_area))

    if burn == 1:
        print("burn=",burn)
        # burnt region
        var_regrid = np.where(burn_area==1, var_regrid, np.nan )
    elif burn == 0:
        # all region
        var_regrid = var_regrid
    elif burn == -1:
        # unburnt region
        var_regrid = np.where(burn_area==0, var_regrid, np.nan )

    print("np.unique(var_regrid)",np.unique(var_regrid))

    if loc_lat !=None:
        lons_2D, lats_2D = np.meshgrid(lon_out, lat_out)
        var_regrid = np.where(np.all(( lats_2D>loc_lat[0],
                                       lats_2D<loc_lat[1],
                                       lons_2D>loc_lon[0],
                                       lons_2D<loc_lon[1]), axis=0),
                                       var_regrid, np.nan)
        lat_out    = lats_2D
        lon_out    = lons_2D

    return var_regrid, lat_out, lon_out

def plot_time_series_FFDI_burn_region(file_out, file_time_series_out, wrf_path, fire_path, loc_lats=None, loc_lons=None, time_s=None, time_e=None, burn=1):

    # Set lat and lon input
    wrf     = Dataset(wrf_path,  mode='r')
    lon_in  = wrf.variables['XLONG'][0,:,:]
    lat_in  = wrf.variables['XLAT'][0,:,:]

    # Read in FFDI index
    FFDI_file  = Dataset(file_out, mode='r')
    Time       = FFDI_file.variables['time'][:]
    FFDI_ctl   = FFDI_file.variables['FFDI_ctl'][:]
    FFDI_sen   = FFDI_file.variables['FFDI_sen'][:]

    ntime      = np.shape(FFDI_ctl)[0]
    print("ntime =",ntime)

    # for i in np.arange(ntime):
    for i in np.arange(3):
        print("i=",i)

        # regrid to burn map resolution ~ 400m
        if i == 0:
            FFDI_ctl_regrid_tmp, lats, lons  = regrid_to_fire_map_resolution(fire_path, FFDI_ctl[i,:,:], lat_in, lon_in, loc_lat=None, loc_lon=None, burn=burn)
            FFDI_sen_regrid_tmp, lats, lons  = regrid_to_fire_map_resolution(fire_path, FFDI_sen[i,:,:], lat_in, lon_in, loc_lat=None, loc_lon=None, burn=burn)

            # Set up array
            nlat = np.shape(FFDI_ctl_regrid_tmp)[0]
            nlon = np.shape(FFDI_ctl_regrid_tmp)[1]

            FFDI_ctl_regrid = np.zeros((ntime, nlat, nlon))
            FFDI_sen_regrid = np.zeros((ntime, nlat, nlon))

            # Assign the first time step value
            FFDI_ctl_regrid[i,:,:]  = FFDI_ctl_regrid_tmp
            FFDI_sen_regrid[i,:,:]  = FFDI_sen_regrid_tmp

        else:
            FFDI_ctl_regrid[i,:,:], lats, lons = regrid_to_fire_map_resolution(fire_path, FFDI_ctl[i,:,:], lat_in, lon_in, loc_lat=None, loc_lon=None, burn=burn)
            FFDI_sen_regrid[i,:,:], lats, lons = regrid_to_fire_map_resolution(fire_path, FFDI_sen[i,:,:], lat_in, lon_in, loc_lat=None, loc_lon=None, burn=burn)
    print('np.unique(FFDI_ctl_regrid)',np.unique(FFDI_ctl_regrid))
    print('np.unique(FFDI_sen_regrid)',np.unique(FFDI_sen_regrid))

    # ===== Make masks for three regions =====
    # make fire lats and lons into 2 D
    lons_2D, lats_2D = np.meshgrid(lons, lats)
    mask_val         = np.zeros((3,np.shape(lons_2D)[0],np.shape(lons_2D)[1]),dtype=bool)

    for i in np.arange(3):
        mask_val[i,:,:]  = np.all(( lats_2D>loc_lats[i][0],lats_2D<loc_lats[i][1],
                                    lons_2D>loc_lons[i][0],lons_2D<loc_lons[i][1]), axis=0)

    # Extend the 3D mask (nreg, nlat, nlon) to 4D (nreg, ntime, nlat, nlon)
    mask_val_4D      = np.expand_dims(mask_val,axis=1).repeat(ntime,axis=1)

    # Set up the output variables
    nreg            = 3
    FFDI_ctl_mean   = np.zeros((nreg,ntime))
    FFDI_ctl_std    = np.zeros((nreg,ntime))
    FFDI_sen_mean   = np.zeros((nreg,ntime))
    FFDI_sen_std    = np.zeros((nreg,ntime))

    if 1:
        fig1, ax1 = plt.subplots(nrows=1, ncols=2, figsize=[5,4],sharex=True, sharey=True, squeeze=True,
                                    subplot_kw={'projection': ccrs.PlateCarree()})

        states    = NaturalEarthFeature(category="cultural", scale="50m",
                                            facecolor="none",
                                            name="admin_1_states_provinces_shp")

        # ======================= Set colormap =======================
        cmap    = plt.cm.BrBG
        cmap.set_bad(color='lightgrey')
        for i in np.arange(2):
            ax1[i].coastlines(resolution="50m",linewidth=1)
            ax1[i].set_extent([135,155,-39,-23])
            ax1[i].add_feature(states, linewidth=.5, edgecolor="black")

        plot1  = ax1[0].contourf( lons, lats, np.nanmean(FFDI_ctl_regrid,axis=0), transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
        plot2  = ax1[1].contourf( lons, lats, np.nanmean(FFDI_sen_regrid,axis=0), transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
        cbar1  = plt.colorbar(plot1, ax=ax1, ticklocation="right", pad=0.08, orientation="horizontal",
                            aspect=40, shrink=0.6)
        cbar1.ax.tick_params(labelsize=8, labelrotation=45)

        plt.savefig('./plots/spatial_map_check_burn_region.png',dpi=300)

    # Mask out three regions
    for i in np.arange(3):
        print("process reg",i)

        var_masked_ctl  = np.where( mask_val_4D[i,:,:,:], FFDI_ctl_regrid, np.nan)
        var_masked_sen  = np.where( mask_val_4D[i,:,:,:], FFDI_sen_regrid, np.nan)

        FFDI_ctl_mean[i,:] = np.nanmean(var_masked_ctl,axis=(1,2))
        FFDI_ctl_std[i,:]  = np.nanstd(var_masked_ctl, axis=(1,2))
        FFDI_sen_mean[i,:] = np.nanmean(var_masked_sen,axis=(1,2))
        FFDI_sen_std[i,:]  = np.nanstd(var_masked_sen, axis=(1,2))


    # ================== make output file ==================

    # create file and write global attributes
    f = nc.Dataset(file_time_series_out, 'w', format='NETCDF4')

    ### Create nc file ###
    f.history           = "Created by: %s" % (os.path.basename(__file__))
    f.creation_date     = "%s" % (datetime.now())
    f.description       = '201909-202002 FFDI in three burnt regions, made by MU Mengyuan'
    f.Conventions       = "CF-1.0"

    # set dimensions
    f.createDimension('region', 3)
    f.createDimension('time',  ntime)

    # Set cooridiates
    region               = f.createVariable('region', 'S7', ('region'))
    region.standard_name = "Burnt regions"
    region.long_name     = "Name of the burnt regions"
    region[:]            = np.array(['North  ', 'Central', 'South  '], dtype='S7')

    time                 = f.createVariable('time', 'f4', ('time'))
    time.standard_name   = "time"
    time.units           = "days since 2000-01-01 00:00:00"
    time[:]              = Time

    Var_mean_ctl               = f.createVariable( 'FFDI_ctl_mean', 'f4', ('region','time'))
    Var_mean_ctl.standard_name = "FFDI in ctl"
    Var_mean_ctl.units         = "-"
    Var_mean_ctl[:]            = FFDI_ctl_mean

    Var_std_ctl               = f.createVariable('FFDI_ctl_std', 'f4', ('region','time'))
    Var_std_ctl.standard_name = "standard deviation of FFDI in burnt region in ctl"
    Var_std_ctl.units         = "-"
    Var_std_ctl[:]            = FFDI_ctl_std

    Var_mean_sen               = f.createVariable('FFDI_sen_mean', 'f4', ('region','time'))
    Var_mean_sen.standard_name = "FFDI in sen"
    Var_mean_sen.units         = "-"
    Var_mean_sen[:]            = FFDI_sen_mean

    Var_std_sen               = f.createVariable('FFDI_sen_std', 'f4', ('region','time'))
    Var_std_sen.standard_name = "standard deviation of FFDI in burnt region in sen"
    Var_std_sen.units         = "-"
    Var_std_sen[:]            = FFDI_sen_std

    f.close()

    # ================= Plotting =================
    f_FFDI          = Dataset(file_time_series_out, mode='r')
    time            = f_FFDI.variables['time'][:]

    FFDI_ctl_mean   = f_FFDI.variables['FFDI_ctl_mean'][:]
    FFDI_ctl_std    = f_FFDI.variables['FFDI_ctl_std'][:]
    FFDI_sen_mean   = f_FFDI.variables['FFDI_sen_mean'][:]
    FFDI_sen_std    = f_FFDI.variables['FFDI_sen_std'][:]
    f_FFDI.close()

    df_reg1                   = pd.DataFrame({'FFDI_ctl_mean': FFDI_ctl_mean[0,:]})
    df_reg1['FFDI_sen_mean']  = FFDI_sen_mean[0,:]
    df_reg1['FFDI_ctl_low']   = FFDI_ctl_mean[0,:] - FFDI_ctl_std[0,:]
    df_reg1['FFDI_ctl_high']  = FFDI_ctl_mean[0,:] + FFDI_ctl_std[0,:]
    df_reg1['FFDI_sen_low']   = FFDI_sen_mean[0,:] - FFDI_sen_std[0,:]
    df_reg1['FFDI_sen_high']  = FFDI_sen_mean[0,:] + FFDI_sen_std[0,:]

    print("df_reg1", df_reg1)

    df_reg2                   = pd.DataFrame({'FFDI_ctl_mean': FFDI_ctl_mean[1,:]})
    df_reg2['FFDI_sen_mean']  = FFDI_sen_mean[1,:]
    df_reg2['FFDI_ctl_low']   = FFDI_ctl_mean[1,:] - FFDI_ctl_std[1,:]
    df_reg2['FFDI_ctl_high']  = FFDI_ctl_mean[1,:] + FFDI_ctl_std[1,:]
    df_reg2['FFDI_sen_low']   = FFDI_sen_mean[1,:] - FFDI_sen_std[1,:]
    df_reg2['FFDI_sen_high']  = FFDI_sen_mean[1,:] + FFDI_sen_std[1,:]


    df_reg3                   = pd.DataFrame({'FFDI_ctl_mean': FFDI_ctl_mean[2,:]})
    df_reg3['FFDI_sen_mean']  = FFDI_sen_mean[2,:]
    df_reg3['FFDI_ctl_low']   = FFDI_ctl_mean[2,:] - FFDI_ctl_std[2,:]
    df_reg3['FFDI_ctl_high']  = FFDI_ctl_mean[2,:] + FFDI_ctl_std[2,:]
    df_reg3['FFDI_sen_low']   = FFDI_sen_mean[2,:] - FFDI_sen_std[2,:]
    df_reg3['FFDI_sen_high']  = FFDI_sen_mean[2,:] + FFDI_sen_std[2,:]


    if 0:

        # =========== Fire date ===========
        fire_file         = Dataset(fire_path, mode='r')
        Burn_Date_tmp     = fire_file.variables['Burn_Date'][2:8,::-1,:]  # 2019-09 - 2020-02
        lat_fire          = fire_file.variables['lat'][::-1]
        lon_fire          = fire_file.variables['lon'][:]

        Burn_Date         = Burn_Date_tmp.astype(float)
        Burn_Date         = np.where(Burn_Date<=0, 99999, Burn_Date)

        Burn_Date[4:,:,:] = Burn_Date[4:,:,:]+365 # Add 365 to Jan-Feb 2020

        Burn_Date_min     = np.nanmin(Burn_Date, axis=0)

        Burn_Date_min     = np.where(Burn_Date_min>=99999, np.nan, Burn_Date_min)
        Burn_Date_min     = Burn_Date_min - 243 # start from Sep 2019

        lons_2D, lats_2D = np.meshgrid(lon_fire, lat_fire)

        mask_val         = np.zeros((3,np.shape(lons_2D)[0],np.shape(lons_2D)[1]),dtype=bool)

        for i in np.arange(3):
            mask_val[i,:,:]  = np.all(( lats_2D>loc_lats[i][0],lats_2D<loc_lats[i][1],
                                        lons_2D>loc_lons[i][0],lons_2D<loc_lons[i][1]), axis=0)

        Burn_Date_min_reg1 = np.where( mask_val[0,:,:], Burn_Date_min, np.nan)
        Burn_Date_min_reg2 = np.where( mask_val[1,:,:], Burn_Date_min, np.nan)
        Burn_Date_min_reg3 = np.where( mask_val[2,:,:], Burn_Date_min, np.nan)

        Burn_reg1_10th = np.nanpercentile(Burn_Date_min_reg1, 10)
        Burn_reg1_50th = np.nanpercentile(Burn_Date_min_reg1, 50)
        Burn_reg1_90th = np.nanpercentile(Burn_Date_min_reg1, 90)

        Burn_reg2_10th = np.nanpercentile(Burn_Date_min_reg2, 10)
        Burn_reg2_50th = np.nanpercentile(Burn_Date_min_reg2, 50)
        Burn_reg2_90th = np.nanpercentile(Burn_Date_min_reg2, 90)

        Burn_reg3_10th = np.nanpercentile(Burn_Date_min_reg3, 10)
        Burn_reg3_50th = np.nanpercentile(Burn_Date_min_reg3, 50)
        Burn_reg3_90th = np.nanpercentile(Burn_Date_min_reg3, 90)

        print('Burn_reg1_10th',Burn_reg1_10th)
        print('Burn_reg1_50th',Burn_reg1_50th)
        print('Burn_reg1_90th',Burn_reg1_90th)
        print('Burn_reg2_10th',Burn_reg2_10th)
        print('Burn_reg2_50th',Burn_reg2_50th)
        print('Burn_reg2_90th',Burn_reg2_90th)
        print('Burn_reg3_10th',Burn_reg3_10th)
        print('Burn_reg3_50th',Burn_reg3_50th)
        print('Burn_reg3_90th',Burn_reg3_90th)

    cleaner_dates = ["Dec 2019", "Jan 2020", "Feb 2020",       ""]
    xtickslocs    = [         -1,       30,         61,       90 ]

    # ===================== Plotting =====================
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=[10,4], sharex=False,
                sharey=False, squeeze=True)
    plt.subplots_adjust(wspace=0.08, hspace=0.08)

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
    props      = dict(boxstyle="round", facecolor='white', alpha=0.0, ec='white')

    time_steps = np.arange(len(time))

    # FFDI
    axs[0].fill_between(time_steps, df_reg1['FFDI_ctl_low'].rolling(window=5).mean(), df_reg1['FFDI_ctl_high'].rolling(window=5).mean(),
                    color="green", edgecolor="none", alpha=0.3)
    axs[0].fill_between(time_steps, df_reg1['FFDI_sen_low'].rolling(window=5).mean(), df_reg1['FFDI_sen_high'].rolling(window=5).mean(),
                    color="orange", edgecolor="none", alpha=0.3)
    axs[0].plot(df_reg1['FFDI_ctl_mean'].rolling(window=5).mean(), label="ctl", c = "green", lw=0.5, alpha=1)
    axs[0].plot(df_reg1['FFDI_sen_mean'].rolling(window=5).mean(), label="exp", c = "orange", lw=0.5, alpha=1)

    axs[1].fill_between(time_steps, df_reg2['FFDI_ctl_low'].rolling(window=5).mean(), df_reg2['FFDI_ctl_high'].rolling(window=5).mean(),
                    color="green", edgecolor="none", alpha=0.3)
    axs[1].fill_between(time_steps, df_reg2['FFDI_sen_low'].rolling(window=5).mean(), df_reg2['FFDI_sen_high'].rolling(window=5).mean(),
                    color="orange", edgecolor="none", alpha=0.3)
    axs[1].plot(df_reg2['FFDI_ctl_mean'].rolling(window=5).mean(), label="ctl", c = "green", lw=0.5, alpha=1)
    axs[1].plot(df_reg2['FFDI_sen_mean'].rolling(window=5).mean(), label="exp", c = "orange", lw=0.5, alpha=1)

    axs[2].fill_between(time_steps, df_reg3['FFDI_ctl_low'].rolling(window=5).mean(), df_reg3['FFDI_ctl_high'].rolling(window=5).mean(),
                    color="green", edgecolor="none", alpha=0.3)
    axs[2].fill_between(time_steps, df_reg3['FFDI_sen_low'].rolling(window=5).mean(), df_reg3['FFDI_sen_high'].rolling(window=5).mean(),
                    color="orange", edgecolor="none", alpha=0.3)
    axs[2].plot(df_reg3['FFDI_ctl_mean'].rolling(window=5).mean(), label="ctl", c = "green", lw=0.5, alpha=1)
    axs[2].plot(df_reg3['FFDI_sen_mean'].rolling(window=5).mean(), label="exp", c = "orange", lw=0.5, alpha=1)

    # Set top titles
    axs[0].set_title("North")
    axs[1].set_title("Central")
    axs[2].set_title("South")

    plt.savefig('./plots/FFDI_burnt_reg_time_series.png',dpi=300)

    return

def output_spatial_FFDI(atmo_ctl_path, atmo_sen_path, land_ctl_path, land_sen_path, file_out, time_s=None, time_e=None,
                     lat_names="lat", lon_names="lon",loc_lat=None,loc_lon=None, wrf_path=None, message=None):

    '''
    plot a single spatial map
    '''

    # ============== Reading the Tmax, rh, wind ================
    # read lat and lon infomation
    wrf            = Dataset(wrf_path,  mode='r')
    lons           = wrf.variables['XLONG'][0,:,:]
    lats           = wrf.variables['XLAT'][0,:,:]
    nlat           = np.shape(lons)[0]
    nlon           = np.shape(lons)[1]

    # read Tmax
    land_ctl_files = [land_ctl_path+'Tair_f_inst/LIS.CABLE.201912-202002.nc']
    land_sen_files = [land_sen_path+'Tair_f_inst/LIS.CABLE.201912-202002.nc']
    time, T_Ctl    = read_var_multi_file(land_ctl_files, 'Tair_f_inst', loc_lat, loc_lon, lat_names, lon_names)
    time, T_Sen    = read_var_multi_file(land_sen_files, 'Tair_f_inst', loc_lat, loc_lon, lat_names, lon_names)
    T_Ctl          = T_Ctl -273.15
    T_Sen          = T_Sen -273.15

    # read Wind
    land_ctl_files = [land_ctl_path+'Wind_f_inst/LIS.CABLE.201912-202002.nc']
    land_sen_files = [land_sen_path+'Wind_f_inst/LIS.CABLE.201912-202002.nc']
    time, Wind_Ctl = read_var_multi_file(land_ctl_files, 'Wind_f_inst', loc_lat, loc_lon, lat_names, lon_names)
    time, Wind_Sen = read_var_multi_file(land_sen_files, 'Wind_f_inst', loc_lat, loc_lon, lat_names, lon_names)

    # convert from m/s to km/h
    Wind_Ctl = Wind_Ctl*3.6
    Wind_Sen = Wind_Sen*3.6

    # read RH
    atmo_ctl_files = [atmo_ctl_path+'rh2/wrfout_201912-202002_hourly.nc']
    atmo_sen_files = [atmo_sen_path+'rh2/wrfout_201912-202002_hourly.nc']
    time, RH_Ctl   = read_var_multi_file(atmo_ctl_files, 'rh2', loc_lat, loc_lon, lat_names, lon_names)
    time, RH_Sen   = read_var_multi_file(atmo_sen_files, 'rh2', loc_lat, loc_lon, lat_names, lon_names)

    # mask out the time - Note that time_s is the second day in the nc file
    # since in AEST time, nc file doesn't cover all hours in the first day
    time_cood = time_mask(time, time_s, time_e, seconds=None)

    T_ctl     = T_Ctl[time_cood,:,:]
    T_sen     = T_Sen[time_cood,:,:]
    wind_ctl  = Wind_Ctl[time_cood,:,:]
    wind_sen  = Wind_Sen[time_cood,:,:]
    rh_ctl    = RH_Ctl[time_cood,:,:]
    rh_sen    = RH_Sen[time_cood,:,:]

    # Calcuate day dimension
    t_s          = time_s - datetime(2000,1,1,0,0,0,0)
    t_e          = time_e - datetime(2000,1,1,0,0,0,0)
    nday         = t_e.days - t_s.days

    # reshape [time,lat,lon] to [day,hour,lat,lon]
    nhour        = 24
    T_ctl_4D     = np.zeros((nday,nhour,nlat,nlon))
    T_sen_4D     = np.zeros((nday,nhour,nlat,nlon))
    wind_ctl_4D  = np.zeros((nday,nhour,nlat,nlon))
    wind_sen_4D  = np.zeros((nday,nhour,nlat,nlon))
    rh_ctl_4D    = np.zeros((nday,nhour,nlat,nlon))
    rh_sen_4D    = np.zeros((nday,nhour,nlat,nlon))

    for d in np.arange(nday):
        d_s = d*24
        d_e = (d+1)*24
        T_ctl_4D[d,:,:,:]     = T_ctl[d_s:d_e,:,:]
        T_sen_4D[d,:,:,:]     = T_sen[d_s:d_e,:,:]
        wind_ctl_4D[d,:,:,:]  = wind_ctl[d_s:d_e,:,:]
        wind_sen_4D[d,:,:,:]  = wind_sen[d_s:d_e,:,:]
        rh_ctl_4D[d,:,:,:]    = rh_ctl[d_s:d_e,:,:]
        rh_sen_4D[d,:,:,:]    = rh_sen[d_s:d_e,:,:]

    # sort out data, after sorting, it will be from small to large
    T_ctl_sort = np.sort(T_ctl_4D,axis=1)
    T_sen_sort = np.sort(T_sen_4D,axis=1)

    # get the index after sort
    T_ctl_sort_index = np.argsort(T_ctl_4D,axis=1)
    T_sen_sort_index = np.argsort(T_sen_4D,axis=1)
    print('T_ctl_sort_index',T_ctl_sort_index)

    # sorted out variables by temperature ranking
    wind_ctl_sort  = np.take_along_axis(wind_ctl_4D, T_ctl_sort_index, axis=1)
    wind_sen_sort  = np.take_along_axis(wind_sen_4D, T_sen_sort_index, axis=1)

    rh_ctl_sort    = np.take_along_axis(rh_ctl_4D, T_ctl_sort_index, axis=1)
    rh_sen_sort    = np.take_along_axis(rh_sen_4D, T_sen_sort_index, axis=1)

    # T, rh and wind when Tmax occur
    T_ctl_max     = T_ctl_sort[:,23,:,:]
    T_sen_max     = T_sen_sort[:,23,:,:]
    wind_ctl_max  = wind_ctl_sort[:,23,:,:]
    wind_sen_max  = wind_sen_sort[:,23,:,:]
    rh_ctl_max    = rh_ctl_sort[:,23,:,:]
    rh_sen_max    = rh_sen_sort[:,23,:,:]

    # ========== Calculate FFDI ==========
    DF = 20

    FFDI_ctl = np.exp(0.0338*T_ctl_max - 0.0345*rh_ctl_max + 0.0234*wind_ctl_max + 0.243147) * DF**0.987
    FFDI_sen = np.exp(0.0338*T_sen_max - 0.0345*rh_sen_max + 0.0234*wind_sen_max + 0.243147) * DF**0.987

    if 1:
        # ================== Start Plotting =================
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=[10,6],sharex=True,
                    sharey=True, squeeze=True, subplot_kw={'projection': ccrs.PlateCarree()})

        plt.rcParams['text.usetex']     = False
        plt.rcParams['font.family']     = "sans-serif"
        plt.rcParams['font.serif']      = "Helvetica"
        plt.rcParams['axes.linewidth']  = 1.5
        plt.rcParams['axes.labelsize']  = 14
        plt.rcParams['font.size']       = 14
        plt.rcParams['legend.fontsize'] = 14
        plt.rcParams['xtick.labelsize'] = 14
        plt.rcParams['ytick.labelsize'] = 14

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

        cmap  = plt.cm.BrBG
        # =============== CHANGE HERE ===============
        for i in np.arange(2):

            axs[i].coastlines(resolution="50m",linewidth=1)
            axs[i].set_extent([135,155,-39,-23])
            axs[i].add_feature(states, linewidth=.5, edgecolor="black")

            # Add gridlines
            gl = axs[i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color=almost_black, linestyle='--')
            gl.xlabels_top  = False
            gl.ylabels_right= False
            gl.xlines       = False
            gl.ylines       = False
            gl.xlocator     = mticker.FixedLocator([130,135,140,145,150,155,160])
            gl.ylocator     = mticker.FixedLocator([-40,-35,-30,-25,-20])
            gl.xformatter   = LONGITUDE_FORMATTER
            gl.yformatter   = LATITUDE_FORMATTER
            gl.xlabel_style = {'size':12, 'color':almost_black}#,'rotation': 90}
            gl.ylabel_style = {'size':12, 'color':almost_black}

            gl.xlabels_bottom = True
            gl.ylabels_left   = True

        # print("any(not np.isnan(var_diff))",any(not np.isnan(var_diff)))
        plot1 = axs[0].contourf(lons, lats, np.nanmean(FFDI_sen-FFDI_ctl,axis=0), transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
        cbar = plt.colorbar(plot1, ax=axs[0], ticklocation="right", pad=0.08, orientation="horizontal",
                aspect=40, shrink=1) # cax=cax,
        cbar.ax.tick_params(labelsize=10,labelrotation=45)

        plot2 = axs[1].contourf(lons, lats, np.nanmean(FFDI_sen,axis=0), transform=ccrs.PlateCarree(), cmap=cmap, extend='both') # clevs_percentage
        cbar  = plt.colorbar(plot2, ax=axs[1], ticklocation="right", pad=0.08, orientation="horizontal",
                aspect=40, shrink=1) # cax=cax,
        cbar.ax.tick_params(labelsize=10,labelrotation=45)

        plt.savefig('./plots/spatial_map_FFDI.png',dpi=300)


    # ========== Make nc file ==========
    # create file and write global attributes
    f                   = Dataset(file_out, 'w', format='NETCDF4')
    f.history           = "Created by: %s" % (os.path.basename(__file__))
    f.creation_date     = "%s" % (datetime.now())
    f.description       = 'FFDI for ctl and sen simulations for 201912-202002, made by MU Mengyuan'

    # set dimensions
    f.createDimension('time', nday)
    f.createDimension('north_south', nlat)
    f.createDimension('east_west', nlon)
    f.Conventions        = "CF-1.0"

    time                 = f.createVariable('time', 'f4', ('time'))
    time.units           = "days since 2000-01-01"
    time[:]              = np.arange(t_s.days,t_e.days)

    latitude             = f.createVariable('lat', 'f4', ('north_south', 'east_west'))
    latitude.long_name   = "latitude"
    latitude.units       = "degree_north"
    latitude._CoordinateAxisType = "Lat"
    latitude[:]          = lats

    longitude            = f.createVariable('lon', 'f4', ('north_south', 'east_west'))
    longitude.long_name  = "longitude"
    longitude.units      = "degree_east"
    longitude._CoordinateAxisType = "Lon"
    longitude[:]         = lons

    ffdi_ctl               = f.createVariable('FFDI_ctl', 'f4', ('time', 'north_south', 'east_west'))
    ffdi_ctl.standard_name = 'FFDI for ctl when Tmax occurs'
    ffdi_ctl.long_name     = 'McArthur forest ﬁre danger index (FFDI) in ctl at the time step when maximum temperature occurs in the day'
    ffdi_ctl.units         = '-'
    ffdi_ctl[:]            = FFDI_ctl

    ffdi_sen               = f.createVariable('FFDI_sen', 'f4', ('time', 'north_south', 'east_west'))
    ffdi_sen.standard_name = 'FFDI for sen when Tmax occurs'
    ffdi_sen.long_name     = 'McArthur forest ﬁre danger index (FFDI) in sen at the time step when maximum temperature occurs in the day'
    ffdi_sen.units         = '-'
    ffdi_sen[:]            = FFDI_sen

    f.close()

def output_spatial_max_FMI(atmo_ctl_path, atmo_sen_path, land_ctl_path, land_sen_path, max_FMI_out, time_s=None, time_e=None,
                     lat_names="lat", lon_names="lon",loc_lat=None,loc_lon=None, wrf_path=None):

    '''
    plot a single spatial map
    '''

    # ============== Reading the Tmax, rh, wind ================
    # read lat and lon infomation
    wrf            = Dataset(wrf_path,  mode='r')
    lons           = wrf.variables['XLONG'][0,:,:]
    lats           = wrf.variables['XLAT'][0,:,:]
    nlat           = np.shape(lons)[0]
    nlon           = np.shape(lons)[1]

    # read Tmax
    land_ctl_files = [land_ctl_path+'Tair_f_inst/LIS.CABLE.201912-202002.nc']
    land_sen_files = [land_sen_path+'Tair_f_inst/LIS.CABLE.201912-202002.nc']
    time, T_Ctl    = read_var_multi_file(land_ctl_files, 'Tair_f_inst', loc_lat, loc_lon, lat_names, lon_names)
    time, T_Sen    = read_var_multi_file(land_sen_files, 'Tair_f_inst', loc_lat, loc_lon, lat_names, lon_names)
    T_Ctl          = T_Ctl -273.15
    T_Sen          = T_Sen -273.15

    # read RH
    atmo_ctl_files = [atmo_ctl_path+'rh2/wrfout_201912-202002_hourly.nc']
    atmo_sen_files = [atmo_sen_path+'rh2/wrfout_201912-202002_hourly.nc']
    time, RH_Ctl   = read_var_multi_file(atmo_ctl_files, 'rh2', loc_lat, loc_lon, lat_names, lon_names)
    time, RH_Sen   = read_var_multi_file(atmo_sen_files, 'rh2', loc_lat, loc_lon, lat_names, lon_names)

    # mask out the time - Note that time_s is the second day in the nc file
    # since in AEST time, nc file doesn't cover all hours in the first day
    time_cood = time_mask(time, time_s, time_e, seconds=None)

    T_ctl     = T_Ctl[time_cood,:,:]
    T_sen     = T_Sen[time_cood,:,:]
    rh_ctl    = RH_Ctl[time_cood,:,:]
    rh_sen    = RH_Sen[time_cood,:,:]

    # Calcuate day dimension
    t_s          = time_s - datetime(2000,1,1,0,0,0,0)
    t_e          = time_e - datetime(2000,1,1,0,0,0,0)
    nday         = t_e.days - t_s.days

    # reshape [time,lat,lon] to [day,hour,lat,lon]
    nhour        = 24
    T_ctl_4D     = np.zeros((nday,nhour,nlat,nlon))
    T_sen_4D     = np.zeros((nday,nhour,nlat,nlon))
    rh_ctl_4D    = np.zeros((nday,nhour,nlat,nlon))
    rh_sen_4D    = np.zeros((nday,nhour,nlat,nlon))

    for d in np.arange(nday):
        d_s = d*24
        d_e = (d+1)*24
        T_ctl_4D[d,:,:,:]     = T_ctl[d_s:d_e,:,:]
        T_sen_4D[d,:,:,:]     = T_sen[d_s:d_e,:,:]
        rh_ctl_4D[d,:,:,:]    = rh_ctl[d_s:d_e,:,:]
        rh_sen_4D[d,:,:,:]    = rh_sen[d_s:d_e,:,:]

    # sort out data, after sorting, it will be from small to large
    T_ctl_sort = np.sort(T_ctl_4D,axis=1)
    T_sen_sort = np.sort(T_sen_4D,axis=1)

    print('T_ctl_sort[12,:,245,300]',T_ctl_sort[12,:,245,300])
    # get the index after sort
    T_ctl_sort_index = np.argsort(T_ctl_4D,axis=1)
    T_sen_sort_index = np.argsort(T_sen_4D,axis=1)
    print('T_ctl_sort_index',T_ctl_sort_index)

    # sorted out variables by temperature ranking
    rh_ctl_sort    = np.take_along_axis(rh_ctl_4D, T_ctl_sort_index, axis=1)
    rh_sen_sort    = np.take_along_axis(rh_sen_4D, T_sen_sort_index, axis=1)
    print('rh_ctl_sort[12,:,245,300]',rh_ctl_sort[12,:,245,300])

    # T, rh and wind when Tmax occur
    T_ctl_max     = T_ctl_sort[:,23,:,:]
    T_sen_max     = T_sen_sort[:,23,:,:]
    rh_ctl_max    = rh_ctl_sort[:,23,:,:]
    rh_sen_max    = rh_sen_sort[:,23,:,:]

    # Calculate FMI
    FMI_Ctl        = 10-0.25*(T_ctl_max-rh_ctl_max)
    FMI_Sen        = 10-0.25*(T_sen_max-rh_sen_max)

    # ========== Make nc file ==========
    # create file and write global attributes
    f                   = Dataset(max_FMI_out, 'w', format='NETCDF4')
    f.history           = "Created by: %s" % (os.path.basename(__file__))
    f.creation_date     = "%s" % (datetime.now())
    f.description       = 'FMI at Tmax time step in 201912-202002, made by MU Mengyuan'

    # set dimensions
    f.createDimension('time', nday)
    f.createDimension('north_south', nlat)
    f.createDimension('east_west', nlon)
    f.Conventions        = "CF-1.0"

    time                 = f.createVariable('time', 'f4', ('time'))
    time.units           = "days since 2000-01-01"
    time[:]              = np.arange(t_s.days,t_e.days)

    latitude             = f.createVariable('lat', 'f4', ('north_south', 'east_west'))
    latitude.long_name   = "latitude"
    latitude.units       = "degree_north"
    latitude._CoordinateAxisType = "Lat"
    latitude[:]          = lats

    longitude            = f.createVariable('lon', 'f4', ('north_south', 'east_west'))
    longitude.long_name  = "longitude"
    longitude.units      = "degree_east"
    longitude._CoordinateAxisType = "Lon"
    longitude[:]         = lons

    fmi_ctl               = f.createVariable('max_FMI_ctl', 'f4', ('time', 'north_south', 'east_west'))
    fmi_ctl.standard_name = 'FMI in ctl'
    fmi_ctl.long_name     = 'Fuel Moisture Index (FMI) at Tmax time step in ctl'
    fmi_ctl.units         = '-'
    fmi_ctl[:]            = FMI_Ctl

    fmi_sen               = f.createVariable('max_FMI_sen', 'f4', ('time', 'north_south', 'east_west'))
    fmi_sen.standard_name = 'FMI in sen'
    fmi_sen.long_name     = 'Fuel Moisture Index (FMI) at Tmax time step in sen'
    fmi_sen.units         = '-'
    fmi_sen[:]            = FMI_Sen

    f.close()

    return

def output_spatial_FMI(atmo_ctl_path, atmo_sen_path, land_ctl_path, land_sen_path, FMI_out, time_s=None, time_e=None,
                     lat_names="lat", lon_names="lon",loc_lat=None,loc_lon=None, wrf_path=None, seconds=None, message=None):

    '''
    plot a single spatial map
    '''

    # ============== Reading the Tmax, rh, wind ================
    # read lat and lon infomation
    wrf            = Dataset(wrf_path,  mode='r')
    lons           = wrf.variables['XLONG'][0,:,:]
    lats           = wrf.variables['XLAT'][0,:,:]
    nlat           = np.shape(lons)[0]
    nlon           = np.shape(lons)[1]

    # read Tmax
    land_ctl_files = [land_ctl_path+'Tair_f_inst/LIS.CABLE.201701-202002.nc']
    land_sen_files = [land_sen_path+'Tair_f_inst/LIS.CABLE.201701-202002.nc']
    time_T, T_Ctl  = read_var_multi_file(land_ctl_files, 'Tair_f_inst', loc_lat, loc_lon, lat_names, lon_names)
    time_T, T_Sen  = read_var_multi_file(land_sen_files, 'Tair_f_inst', loc_lat, loc_lon, lat_names, lon_names)
    T_Ctl          = T_Ctl -273.15
    T_Sen          = T_Sen -273.15

    # read RH
    atmo_ctl_files = [atmo_ctl_path+'rh2/wrfout_201701-202006.nc']
    atmo_sen_files = [atmo_sen_path+'rh2/wrfout_201701-202006.nc']
    time_RH, RH_Ctl= read_var_multi_file(atmo_ctl_files, 'rh2', loc_lat, loc_lon, lat_names, lon_names)
    time_RH, RH_Sen= read_var_multi_file(atmo_sen_files, 'rh2', loc_lat, loc_lon, lat_names, lon_names)

    T_Ctl_daily    = time_clip_to_day(time_T, T_Ctl,   time_s, time_e, seconds)
    T_Sen_daily    = time_clip_to_day(time_T, T_Sen,   time_s, time_e, seconds)
    RH_Ctl_daily   = time_clip_to_day(time_RH, RH_Ctl, time_s, time_e, seconds)
    RH_Sen_daily   = time_clip_to_day(time_RH, RH_Sen, time_s, time_e, seconds)

    # Calculate FMI
    FMI_Ctl        = 10-0.25*(T_Ctl_daily-RH_Ctl_daily)
    FMI_Sen        = 10-0.25*(T_Sen_daily-RH_Sen_daily)

    t_s            = time_s - datetime(2000,1,1,0,0,0,0)
    t_e            = time_e - datetime(2000,1,1,0,0,0,0)
    nday           = t_e.days - t_s.days

    # ========== Make nc file ==========
    # create file and write global attributes
    f                   = Dataset(FMI_out, 'w', format='NETCDF4')
    f.history           = "Created by: %s" % (os.path.basename(__file__))
    f.creation_date     = "%s" % (datetime.now())
    f.description       = 'FMI for ctl and sen simulations for 201909-202002, made by MU Mengyuan'

    # set dimensions
    f.createDimension('time', nday)
    f.createDimension('north_south', nlat)
    f.createDimension('east_west', nlon)
    f.Conventions        = "CF-1.0"

    time                 = f.createVariable('time', 'f4', ('time'))
    time.units           = "days since 2000-01-01"
    time[:]              = np.arange(t_s.days,t_e.days)

    latitude             = f.createVariable('lat', 'f4', ('north_south', 'east_west'))
    latitude.long_name   = "latitude"
    latitude.units       = "degree_north"
    latitude._CoordinateAxisType = "Lat"
    latitude[:]          = lats

    longitude            = f.createVariable('lon', 'f4', ('north_south', 'east_west'))
    longitude.long_name  = "longitude"
    longitude.units      = "degree_east"
    longitude._CoordinateAxisType = "Lon"
    longitude[:]         = lons

    fmi_ctl               = f.createVariable('FMI_ctl', 'f4', ('time', 'north_south', 'east_west'))
    fmi_ctl.standard_name = 'FMI for ctl'
    fmi_ctl.long_name     = 'Fuel Moisture Index (FMI) in ctl'
    fmi_ctl.units         = '-'
    fmi_ctl[:]            = FMI_Ctl

    fmi_sen               = f.createVariable('FMI_sen', 'f4', ('time', 'north_south', 'east_west'))
    fmi_sen.standard_name = 'FMI for sen'
    fmi_sen.long_name     = 'Fuel Moisture Index (FMI) in sen'
    fmi_sen.units         = '-'
    fmi_sen[:]            = FMI_Sen

    f.close()

    return

def output_FMI_burnt_region(FMI_out, FMI_time_series_out, wrf_path, fire_path, loc_lats=None, loc_lons=None, time_s=None, time_e=None, burn=1):

    # Set lat and lon input
    wrf       = Dataset(wrf_path,  mode='r')
    lon_in    = wrf.variables['XLONG'][0,:,:]
    lat_in    = wrf.variables['XLAT'][0,:,:]

    # Read in FMI index
    FMI_file  = Dataset(FMI_out, mode='r')
    Time      = FMI_file.variables['time'][:]
    try:
        FMI_ctl   = FMI_file.variables['max_FMI_ctl'][:]
        FMI_sen   = FMI_file.variables['max_FMI_sen'][:]
    except:
        FMI_ctl   = FMI_file.variables['FMI_ctl'][:]
        FMI_sen   = FMI_file.variables['FMI_sen'][:]

    ntime     = np.shape(FMI_ctl)[0]
    print("ntime =",ntime)

    for i in np.arange(ntime):
    # for i in np.arange(3):
        print("i=",i)

        # regrid to burn map resolution ~ 400m
        if i == 0:
            FMI_ctl_regrid_tmp, lats, lons  = regrid_to_fire_map_resolution(fire_path, FMI_ctl[i,:,:], lat_in, lon_in, loc_lat=None, loc_lon=None, burn=burn)
            FMI_sen_regrid_tmp, lats, lons  = regrid_to_fire_map_resolution(fire_path, FMI_sen[i,:,:], lat_in, lon_in, loc_lat=None, loc_lon=None, burn=burn)

            # Set up array
            nlat = np.shape(FMI_ctl_regrid_tmp)[0]
            nlon = np.shape(FMI_ctl_regrid_tmp)[1]

            FMI_ctl_regrid = np.zeros((ntime, nlat, nlon))
            FMI_sen_regrid = np.zeros((ntime, nlat, nlon))

            # Assign the first time step value
            FMI_ctl_regrid[i,:,:]  = FMI_ctl_regrid_tmp
            FMI_sen_regrid[i,:,:]  = FMI_sen_regrid_tmp

        else:
            FMI_ctl_regrid[i,:,:], lats, lons = regrid_to_fire_map_resolution(fire_path, FMI_ctl[i,:,:], lat_in, lon_in, loc_lat=None, loc_lon=None, burn=burn)
            FMI_sen_regrid[i,:,:], lats, lons = regrid_to_fire_map_resolution(fire_path, FMI_sen[i,:,:], lat_in, lon_in, loc_lat=None, loc_lon=None, burn=burn)

    print('np.unique(FMI_ctl_regrid)',np.unique(FMI_ctl_regrid))
    print('np.unique(FMI_sen_regrid)',np.unique(FMI_sen_regrid))

    # ===== Make masks for three regions =====
    # make fire lats and lons into 2D
    lons_2D, lats_2D = np.meshgrid(lons, lats)
    mask_val         = np.zeros((3,np.shape(lons_2D)[0],np.shape(lons_2D)[1]),dtype=bool)

    for i in np.arange(3):
        mask_val[i,:,:]  = np.all(( lats_2D>loc_lats[i][0],lats_2D<loc_lats[i][1],
                                    lons_2D>loc_lons[i][0],lons_2D<loc_lons[i][1]), axis=0)

    # Extend the 3D mask (nreg, nlat, nlon) to 4D (nreg, ntime, nlat, nlon)
    mask_val_4D    = np.expand_dims(mask_val,axis=1).repeat(ntime,axis=1)

    # Set up the output variables
    nreg           = 3
    FMI_ctl_mean   = np.zeros((nreg,ntime))
    FMI_ctl_std    = np.zeros((nreg,ntime))
    FMI_sen_mean   = np.zeros((nreg,ntime))
    FMI_sen_std    = np.zeros((nreg,ntime))

    if 1:
        fig1, ax1 = plt.subplots(nrows=1, ncols=2, figsize=[5,4],sharex=True, sharey=True, squeeze=True,
                                    subplot_kw={'projection': ccrs.PlateCarree()})

        states    = NaturalEarthFeature(category="cultural", scale="50m",
                                            facecolor="none",
                                            name="admin_1_states_provinces_shp")

        # ======================= Set colormap =======================
        cmap    = plt.cm.BrBG
        cmap.set_bad(color='lightgrey')
        for i in np.arange(2):
            ax1[i].coastlines(resolution="50m",linewidth=1)
            ax1[i].set_extent([135,155,-39,-23])
            ax1[i].add_feature(states, linewidth=.5, edgecolor="black")

        plot1  = ax1[0].contourf( lons, lats, np.nanmean(FMI_ctl_regrid,axis=0), transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
        plot2  = ax1[1].contourf( lons, lats, np.nanmean(FMI_sen_regrid,axis=0), transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
        cbar1  = plt.colorbar(plot1, ax=ax1, ticklocation="right", pad=0.08, orientation="horizontal",
                            aspect=40, shrink=0.6)
        cbar1.ax.tick_params(labelsize=8, labelrotation=45)

        plt.savefig('./plots/spatial_map_check_burn_region.png',dpi=300)

    # Mask out three regions
    for i in np.arange(3):
        print("process reg",i)

        var_masked_ctl  = np.where( mask_val_4D[i,:,:,:], FMI_ctl_regrid, np.nan)
        var_masked_sen  = np.where( mask_val_4D[i,:,:,:], FMI_sen_regrid, np.nan)

        FMI_ctl_mean[i,:] = np.nanmean(var_masked_ctl,axis=(1,2))
        FMI_ctl_std[i,:]  = np.nanstd(var_masked_ctl, axis=(1,2))
        FMI_sen_mean[i,:] = np.nanmean(var_masked_sen,axis=(1,2))
        FMI_sen_std[i,:]  = np.nanstd(var_masked_sen, axis=(1,2))


    # ================== make output file ==================

    # create file and write global attributes
    f = nc.Dataset(FMI_time_series_out, 'w', format='NETCDF4')

    ### Create nc file ###
    f.history           = "Created by: %s" % (os.path.basename(__file__))
    f.creation_date     = "%s" % (datetime.now())
    if burn == -1:
        f.description       = '201909-202002 FMI in three unburnt regions, made by MU Mengyuan'
    elif burn == 1:
        f.description       = '201909-202002 FMI in three burnt regions, made by MU Mengyuan'
    f.Conventions       = "CF-1.0"

    # set dimensions
    f.createDimension('region', 3)
    f.createDimension('time',  ntime)

    # Set cooridiates
    region               = f.createVariable('region', 'S7', ('region'))
    if burn == -1:
        region.standard_name = "Unburnt regions"
    elif burn == 1:
        region.standard_name = "Burnt regions"
    region.long_name     = "Name of the burnt regions"
    region[:]            = np.array(['North  ', 'Central', 'South  '], dtype='S7')

    time                 = f.createVariable('time', 'f4', ('time'))
    time.standard_name   = "time"
    time.units           = "days since 2000-01-01 00:00:00"
    time[:]              = Time

    Var_mean_ctl               = f.createVariable( 'FMI_ctl_mean', 'f4', ('region','time'))
    Var_mean_ctl.standard_name = "FMI in ctl"
    Var_mean_ctl.units         = "-"
    Var_mean_ctl[:]            = FMI_ctl_mean

    Var_std_ctl               = f.createVariable('FMI_ctl_std', 'f4', ('region','time'))
    Var_std_ctl.standard_name = "standard deviation of FMI in burnt region in ctl"
    Var_std_ctl.units         = "-"
    Var_std_ctl[:]            = FMI_ctl_std

    Var_mean_sen               = f.createVariable('FMI_sen_mean', 'f4', ('region','time'))
    Var_mean_sen.standard_name = "FMI in sen"
    Var_mean_sen.units         = "-"
    Var_mean_sen[:]            = FMI_sen_mean

    Var_std_sen                = f.createVariable('FMI_sen_std', 'f4', ('region','time'))
    Var_std_sen.standard_name  = "standard deviation of FMI in burnt region in sen"
    Var_std_sen.units          = "-"
    Var_std_sen[:]             = FMI_sen_std

    f.close()

def plot_time_series_FMI_burn_region(FMI_time_series_out, wrf_path, fire_path, loc_lats=None, loc_lons=None, time_s=None, time_e=None, burn=1):

    # Set lat and lon input
    wrf     = Dataset(wrf_path,  mode='r')
    lon_in  = wrf.variables['XLONG'][0,:,:]
    lat_in  = wrf.variables['XLAT'][0,:,:]

    # Read in FFDI index
    FMI_file     = Dataset(FMI_time_series_out, mode='r')
    Time         = FMI_file.variables['time'][:]
    FMI_ctl_mean = FMI_file.variables['FMI_ctl_mean'][:]
    FMI_sen_mean = FMI_file.variables['FMI_sen_mean'][:]
    FMI_ctl_std  = FMI_file.variables['FMI_ctl_std'][:]
    FMI_sen_std  = FMI_file.variables['FMI_sen_std'][:]
    FMI_file.close()

    ntime      = np.shape(FMI_ctl_mean)[0]
    print("ntime =",ntime)

    df_reg1                  = pd.DataFrame({'FMI_ctl_mean': FMI_ctl_mean[0,:]})
    df_reg1['FMI_sen_mean']  = FMI_sen_mean[0,:]
    df_reg1['FMI_ctl_low']   = FMI_ctl_mean[0,:] - FMI_ctl_std[0,:]
    df_reg1['FMI_ctl_high']  = FMI_ctl_mean[0,:] + FMI_ctl_std[0,:]
    df_reg1['FMI_sen_low']   = FMI_sen_mean[0,:] - FMI_sen_std[0,:]
    df_reg1['FMI_sen_high']  = FMI_sen_mean[0,:] + FMI_sen_std[0,:]

    print("df_reg1", df_reg1)

    df_reg2                   = pd.DataFrame({'FMI_ctl_mean': FMI_ctl_mean[1,:]})
    df_reg2['FMI_sen_mean']  = FMI_sen_mean[1,:]
    df_reg2['FMI_ctl_low']   = FMI_ctl_mean[1,:] - FMI_ctl_std[1,:]
    df_reg2['FMI_ctl_high']  = FMI_ctl_mean[1,:] + FMI_ctl_std[1,:]
    df_reg2['FMI_sen_low']   = FMI_sen_mean[1,:] - FMI_sen_std[1,:]
    df_reg2['FMI_sen_high']  = FMI_sen_mean[1,:] + FMI_sen_std[1,:]


    df_reg3                   = pd.DataFrame({'FMI_ctl_mean': FMI_ctl_mean[2,:]})
    df_reg3['FMI_sen_mean']  = FMI_sen_mean[2,:]
    df_reg3['FMI_ctl_low']   = FMI_ctl_mean[2,:] - FMI_ctl_std[2,:]
    df_reg3['FMI_ctl_high']  = FMI_ctl_mean[2,:] + FMI_ctl_std[2,:]
    df_reg3['FMI_sen_low']   = FMI_sen_mean[2,:] - FMI_sen_std[2,:]
    df_reg3['FMI_sen_high']  = FMI_sen_mean[2,:] + FMI_sen_std[2,:]


    if 0:

        # =========== Fire date ===========
        fire_file         = Dataset(fire_path, mode='r')
        Burn_Date_tmp     = fire_file.variables['Burn_Date'][2:8,::-1,:]  # 2019-09 - 2020-02
        lat_fire          = fire_file.variables['lat'][::-1]
        lon_fire          = fire_file.variables['lon'][:]

        Burn_Date         = Burn_Date_tmp.astype(float)
        Burn_Date         = np.where(Burn_Date<=0, 99999, Burn_Date)

        Burn_Date[4:,:,:] = Burn_Date[4:,:,:]+365 # Add 365 to Jan-Feb 2020

        Burn_Date_min     = np.nanmin(Burn_Date, axis=0)

        Burn_Date_min     = np.where(Burn_Date_min>=99999, np.nan, Burn_Date_min)
        Burn_Date_min     = Burn_Date_min - 243 # start from Sep 2019

        lons_2D, lats_2D = np.meshgrid(lon_fire, lat_fire)

        mask_val         = np.zeros((3,np.shape(lons_2D)[0],np.shape(lons_2D)[1]),dtype=bool)

        for i in np.arange(3):
            mask_val[i,:,:]  = np.all(( lats_2D>loc_lats[i][0],lats_2D<loc_lats[i][1],
                                        lons_2D>loc_lons[i][0],lons_2D<loc_lons[i][1]), axis=0)

        Burn_Date_min_reg1 = np.where( mask_val[0,:,:], Burn_Date_min, np.nan)
        Burn_Date_min_reg2 = np.where( mask_val[1,:,:], Burn_Date_min, np.nan)
        Burn_Date_min_reg3 = np.where( mask_val[2,:,:], Burn_Date_min, np.nan)

        Burn_reg1_10th = np.nanpercentile(Burn_Date_min_reg1, 10)
        Burn_reg1_50th = np.nanpercentile(Burn_Date_min_reg1, 50)
        Burn_reg1_90th = np.nanpercentile(Burn_Date_min_reg1, 90)

        Burn_reg2_10th = np.nanpercentile(Burn_Date_min_reg2, 10)
        Burn_reg2_50th = np.nanpercentile(Burn_Date_min_reg2, 50)
        Burn_reg2_90th = np.nanpercentile(Burn_Date_min_reg2, 90)

        Burn_reg3_10th = np.nanpercentile(Burn_Date_min_reg3, 10)
        Burn_reg3_50th = np.nanpercentile(Burn_Date_min_reg3, 50)
        Burn_reg3_90th = np.nanpercentile(Burn_Date_min_reg3, 90)

        print('Burn_reg1_10th',Burn_reg1_10th)
        print('Burn_reg1_50th',Burn_reg1_50th)
        print('Burn_reg1_90th',Burn_reg1_90th)
        print('Burn_reg2_10th',Burn_reg2_10th)
        print('Burn_reg2_50th',Burn_reg2_50th)
        print('Burn_reg2_90th',Burn_reg2_90th)
        print('Burn_reg3_10th',Burn_reg3_10th)
        print('Burn_reg3_50th',Burn_reg3_50th)
        print('Burn_reg3_90th',Burn_reg3_90th)


    cleaner_dates = ["Sep 2019", "Oct 2019", "Nov 2019", "Dec 2019", "Jan 2020", "Feb 2020",       ""]
    xtickslocs    = [         0,         30,         61,         91,       122,         153,     182 ]

    # ===================== Plotting =====================
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=[10,4], sharex=False,
                sharey=False, squeeze=True)
    plt.subplots_adjust(wspace=0.08, hspace=0.08)

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
    props      = dict(boxstyle="round", facecolor='white', alpha=0.0, ec='white')

    time_steps = np.arange(92,92+len(Time),1)

    # FMI
    axs[0].fill_between(time_steps, df_reg1['FMI_ctl_low'].rolling(window=5).mean(), df_reg1['FMI_ctl_high'].rolling(window=5).mean(),
                    color="green", edgecolor="none", alpha=0.3)
    axs[0].fill_between(time_steps, df_reg1['FMI_sen_low'].rolling(window=5).mean(), df_reg1['FMI_sen_high'].rolling(window=5).mean(),
                    color="orange", edgecolor="none", alpha=0.3)
    axs[0].plot(time_steps, df_reg1['FMI_ctl_mean'].rolling(window=5).mean(), label="ctl", c = "green", lw=0.5, alpha=1)
    axs[0].plot(time_steps, df_reg1['FMI_sen_mean'].rolling(window=5).mean(), label="exp", c = "orange", lw=0.5, alpha=1)

    axs[1].fill_between(time_steps, df_reg2['FMI_ctl_low'].rolling(window=5).mean(), df_reg2['FMI_ctl_high'].rolling(window=5).mean(),
                    color="green", edgecolor="none", alpha=0.3)
    axs[1].fill_between(time_steps, df_reg2['FMI_sen_low'].rolling(window=5).mean(), df_reg2['FMI_sen_high'].rolling(window=5).mean(),
                    color="orange", edgecolor="none", alpha=0.3)
    axs[1].plot(time_steps, df_reg2['FMI_ctl_mean'].rolling(window=5).mean(), label="ctl", c = "green", lw=0.5, alpha=1)
    axs[1].plot(time_steps, df_reg2['FMI_sen_mean'].rolling(window=5).mean(), label="exp", c = "orange", lw=0.5, alpha=1)

    axs[2].fill_between(time_steps, df_reg3['FMI_ctl_low'].rolling(window=5).mean(), df_reg3['FMI_ctl_high'].rolling(window=5).mean(),
                    color="green", edgecolor="none", alpha=0.3)
    axs[2].fill_between(time_steps, df_reg3['FMI_sen_low'].rolling(window=5).mean(), df_reg3['FMI_sen_high'].rolling(window=5).mean(),
                    color="orange", edgecolor="none", alpha=0.3)
    axs[2].plot(time_steps, df_reg3['FMI_ctl_mean'].rolling(window=5).mean(), label="ctl", c = "green", lw=0.5, alpha=1)
    axs[2].plot(time_steps, df_reg3['FMI_sen_mean'].rolling(window=5).mean(), label="exp", c = "orange", lw=0.5, alpha=1)

    # Set top titles
    axs[0].set_title("North")
    axs[1].set_title("Central")
    axs[2].set_title("South")

    plt.savefig('./plots/max_FMI_burnt_reg_time_series.png',dpi=300)

    return

if __name__ == "__main__":

    # ======================= Option =======================
    region = "SE Aus" #"SE Aus" #"CORDEX" #"SE Aus"

    if region == "Aus":
        loc_lat    = [-44,-10]
        loc_lon    = [112,154]
    elif region == "SE Aus":
        loc_lat    = [-40,-23.6]
        loc_lon    = [134,155]
    elif region == "CORDEX":
        loc_lat    = [-52.36,3.87]
        loc_lon    = [89.25,180]

    #######################################################
    # Decks to run:
    #    plot a single map
    #######################################################
    if 1:
        '''
        Test WRF-CABLE LIS output
        '''

        case_ctl       = "drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2"
        case_sen       = "drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2_obs_LAI_ALB"

        fire_path      = '/g/data/w97/mm3972/data/MODIS/MODIS_fire/MCD64A1.061_500m_aid0001.nc'
        wrf_path       = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_ctl+"/WRF_output/wrfout_d01_2020-02-01_01:00:00"
        land_sen_path  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_sen+"/LIS_output/"
        land_ctl_path  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_ctl+"/LIS_output/"
        atmo_sen_path  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_sen+"/WRF_output/"
        atmo_ctl_path  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_ctl+"/WRF_output/"
        FFDI_out       = "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/FFDI_Dec2019_Feb2020.nc"
        FMI_out        = "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/FMI_201909_202002.nc"
        FFDI_time_series_out = "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/FFDI_time_series_Dec2019_Feb2020.nc"
        FMI_time_series_out  = "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/FMI_time_series_201909_202002.nc"

        if 1:
            '''
            use 6-hourly data to calcuate the FMI at Tmax time step in 201909-202002
            '''
            message        = "Sep_2019_Feb_2020"
            time_s         = datetime(2019,9,2,0,0,0,0)
            time_e         = datetime(2020,3,1,0,0,0,0)
            seconds        = [6.*60.*60.,18.*60.*60.]

            #                   North ,        Central,       South
            loc_lats       = [[-32,-28.5],   [-34.5,-32.5], [-38,-34.5]]
            loc_lons       = [[151.5,153.5], [149.5,151.5], [146.5,151]]

            # Step 1 calculate FMI
            FMI_out        = "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/daytime_FMI_201909_202002.nc"
            # output_spatial_FMI(atmo_ctl_path, atmo_sen_path, land_ctl_path, land_sen_path, FMI_out, time_s=time_s, time_e=time_e,
            #                    lat_names="lat", lon_names="lon", loc_lat=loc_lat, loc_lon=loc_lon, wrf_path=wrf_path, seconds=seconds)

            # Step 2 calculate FMI in burnt region
            FMI_time_series_out    = "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/daytime_FMI_time_series_201909_202002_unburnt.nc"
            output_FMI_burnt_region(FMI_out, FMI_time_series_out, wrf_path, fire_path, loc_lats=loc_lats, 
                                    loc_lons=loc_lons, time_s=time_s, time_e=time_e, burn=-1)

            # Step 3 plot max FMI time series in burnt region
            # FMI_time_series_out    = "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/daytime_FMI_time_series_201909_202002.nc"
            # plot_time_series_FMI_burn_region(FMI_time_series_out, wrf_path, fire_path, loc_lats=loc_lats, 
            #                                  loc_lons=loc_lons, time_s=time_s, time_e=time_e, burn=1)

            # spatial_map_FFDI(atmo_ctl_path, atmo_sen_path, land_ctl_path, land_sen_path, file_out, time_s=time_s, time_e=time_e, lat_names="lat",
            #                     lon_names="lon",loc_lat=loc_lat, loc_lon=loc_lon, wrf_path=wrf_path, message=message)

            # spatial_map_FMI(atmo_ctl_path, atmo_sen_path, land_ctl_path, land_sen_path, FMI_out, time_s=time_s, time_e=time_e,
            #                     lat_names="lat", lon_names="lon",loc_lat=loc_lat,loc_lon=loc_lon, wrf_path=wrf_path, message=message)

            # FMI_burnt_region(FMI_out, FMI_time_series_out, wrf_path, fire_path, loc_lats=None, loc_lons=None, time_s=None, time_e=None, burn=1)
            # output_spatial_FMI(atmo_ctl_path, atmo_sen_path, land_ctl_path, land_sen_path, FMI_out, time_s=time_s, time_e=time_e,
            #                     lat_names="lat", lon_names="lon",loc_lat=loc_lat,loc_lon=loc_lon, wrf_path=wrf_path, message=message)


        if 0:
            '''
            use hourly data to calcuate the FMI at Tmax time step in 201912-202002
            '''
            time_s         = datetime(2019,12,2,0,0,0,0)
            time_e         = datetime(2020,3,1,0,0,0,0)

            #                   North ,        Central,       South
            loc_lats       = [[-32,-28.5],   [-34.5,-32.5], [-38,-34.5]]
            loc_lons       = [[151.5,153.5], [149.5,151.5], [146.5,151]]

            # Step 1 calculate FMI
            max_FMI_out    = "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/max_FMI_201912_202002.nc"
            # output_spatial_max_FMI(atmo_ctl_path, atmo_sen_path, land_ctl_path, land_sen_path, max_FMI_out, time_s=time_s, time_e=time_e,
            #          lat_names="lat", lon_names="lon",loc_lat=loc_lat,loc_lon=loc_lon, wrf_path=wrf_path)

            # Step 2 calculate FMI in burnt region
            max_FMI_time_series_out    = "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/max_FMI_time_series_201912_202002.nc"
            output_FMI_burnt_region(max_FMI_out, max_FMI_time_series_out, wrf_path, fire_path, loc_lats=loc_lats, loc_lons=loc_lons, time_s=time_s, time_e=time_e, burn=1)

            # Step 3 plot max FMI time series in burnt region
            max_FMI_time_series_out    = "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/max_FMI_time_series_201912_202002.nc"
            plot_time_series_FMI_burn_region(max_FMI_time_series_out, wrf_path, fire_path, loc_lats=loc_lats, loc_lons=loc_lons, time_s=time_s, time_e=time_e, burn=1)

        if 0:
            #                   North ,        Central,       South
            loc_lats       = [[-32,-28.5],   [-34.5,-32.5], [-38,-34.5]]
            loc_lons       = [[151.5,153.5], [149.5,151.5], [146.5,151]]
            # output_FMI_burnt_region(FMI_out, FMI_time_series_out, wrf_path, fire_path, loc_lats=loc_lats, loc_lons=loc_lons, time_s=time_s, time_e=time_e, burn=1)
            plot_time_series_FMI_burn_region(FMI_time_series_out, wrf_path, fire_path, loc_lats=loc_lats, loc_lons=loc_lons, time_s=time_s, time_e=time_e, burn=1)
