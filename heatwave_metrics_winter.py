#!/usr/bin/python

__author__ = "Mengyuan Mu"
__email__  = "mu.mengyuan815@gmail.com"

'''
Functions:
1. Climdex indices: https://www.climdex.org/learn/indices/
'''

import os
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

def regrid_EHF_4_WRF_domain(EHF_path,wrf_path,EHF_out):

    # regridding EHF index
    time_in, Var  = read_var(EHF_path, 'event', loc_lat=None, loc_lon=None, lat_name='lat', lon_name='lon')
    time_in, lats = read_var(EHF_path, 'lat', loc_lat=None, loc_lon=None, lat_name='lat', lon_name='lon')
    time_in, lons = read_var(EHF_path, 'lon', loc_lat=None, loc_lon=None, lat_name='lat', lon_name='lon')
    ntime         = len(time_in)
    Time          = [ ]

    print('lats',lats)

    wrf        = Dataset(wrf_path,  mode='r')
    lats_out   = wrf.variables['XLAT'][0,:,:]
    lons_out   = wrf.variables['XLONG'][0,:,:]
    nlat       = np.shape(lats_out)[0]
    nlon       = np.shape(lats_out)[1]
    var_regrid = np.zeros([ntime,nlat,nlon])
    for i in np.arange(ntime):
        print("regridding ",time_in[i].days)
        Time.append(time_in[i].days+12) # Since EHF doesn't consider leaf years so add 12 leap years back
        var_regrid[i,:,:]= regrid_data(lats, lons, lats_out, lons_out, Var[i,:,:],method="nearest")
    print('MMY Time',Time)

    # create file and write global attributes
    f                   = nc.Dataset(EHF_out, 'w', format='NETCDF4')
    f.history           = "Created by: %s written by MU Mengyuan" % (os.path.basename(__file__))
    f.creation_date     = "%s" % (datetime.now())

    # Copy global attributes from old file
    f_org               = nc.Dataset(EHF_path, 'r', format='NETCDF4')
    for attr_name in f_org.ncattrs():
        attr_value = getattr(f_org, attr_name)
        setattr(f, attr_name, attr_value)

    f.Conventions       = "CF-1.0"

    # set dimensions
    f.createDimension('time', None)
    f.createDimension('lat', nlat)
    f.createDimension('lon', nlon)

    time                = f.createVariable('time', 'f4', ('time'))
    time.standard_name  = "time"
    time.units          = "days since 2000-01-01"
    time[:]             = Time[:]

    lat                = f.createVariable('lat', 'f4', ('lat','lon'))
    lat.standard_name  = "latitude"
    lat.long_name      = "Latitude"
    lat.units          = "degrees_north"
    lat[:]             = lats_out

    lon                = f.createVariable('lon', 'f4', ('lat','lon'))
    lon.standard_name  = "longitude"
    lon.long_name      = "Longitude"
    lon.units          = "degrees_east"
    lon[:]             = lons_out

    event               = f.createVariable('event', 'f4', ('time','lat','lon'))
    event.FillValue     = 9.96921e+36
    event.missing_value = -999.99
    event.long_name     = "Event indicator"
    event.description   = "Indicates whether a summer heatwave is happening on that day"
    event[:]            = var_regrid

    f.close()
    f_org.close()

def calc_heatwave_magnitude(EHF_out, land_ctl_path, land_sen_path, var_name, time_s=None, time_e=None,
                            lat_names="lat", lon_names="lon",loc_lat=None, loc_lon=None):

    time_ehf, hw_event = read_var_multi_file(EHF_out, 'event', loc_lat, loc_lon, lat_names, lon_names)
    # time_ehf = time_ehf + timedelta(days=12) # to adjust time

    print('time_ehf',time_ehf)
    print('hw_event',hw_event)

    if var_name in ["Tmax","Tmin"]:
        land_ctl_files= [land_ctl_path+'Tair_f_inst/LIS.CABLE.201701-202002.nc']
        land_sen_files= [land_sen_path+'Tair_f_inst/LIS.CABLE.201701-202002.nc']
        time, Ctl_tmp = read_var_multi_file(land_ctl_files, 'Tair_f_inst', loc_lat, loc_lon, lat_names, lon_names)
        time, Sen_tmp = read_var_multi_file(land_sen_files, 'Tair_f_inst', loc_lat, loc_lon, lat_names, lon_names)
    elif var_name in ["VegTmax","VegTmin"]:
        land_ctl_files= [land_ctl_path+'VegT_tavg/LIS.CABLE.201701-202002.nc']
        land_sen_files= [land_sen_path+'VegT_tavg/LIS.CABLE.201701-202002.nc']
        time, Ctl_tmp = read_var_multi_file(land_ctl_files, 'VegT_tavg', loc_lat, loc_lon, lat_names, lon_names)
        time, Sen_tmp = read_var_multi_file(land_sen_files, 'VegT_tavg', loc_lat, loc_lon, lat_names, lon_names)
    elif var_name in ["SurfTmax","SurfTmin"]:
        land_ctl_files= [land_ctl_path+'AvgSurfT_tavg/LIS.CABLE.201701-202002.nc']
        land_sen_files= [land_sen_path+'AvgSurfT_tavg/LIS.CABLE.201701-202002.nc']
        time, Ctl_tmp = read_var_multi_file(land_ctl_files, 'AvgSurfT_tavg', loc_lat, loc_lon, lat_names, lon_names)
        time, Sen_tmp = read_var_multi_file(land_sen_files, 'AvgSurfT_tavg', loc_lat, loc_lon, lat_names, lon_names)

    # time-step into daily
    if var_name in ["SurfTmax","Tmax","VegTmax"]:
        # average of daily max
        ctl_in       = time_clip_to_day_max(time,Ctl_tmp,time_s,time_e)
        sen_in       = time_clip_to_day_max(time,Sen_tmp,time_s,time_e)
    elif var_name in ["SurfTmin","Tmin","VegTmin"]:
        # average of daily min
        ctl_in       = time_clip_to_day_min(time,Ctl_tmp,time_s,time_e)
        sen_in       = time_clip_to_day_min(time,Sen_tmp,time_s,time_e)

    print('np.shape(ctl_in)',np.shape(ctl_in))
    Var_diff = sen_in-ctl_in

    # calculate HW periods values
    time_cood    = time_mask(time_ehf, time_s, time_e, seconds=None)
    hw_event_new = hw_event[time_cood,:,:]

    print('np.shape(hw_event_new)',np.shape(hw_event_new))

    if time_s == datetime(2017,1,1,0,0,0,0):
        # since LIS-CABLE miss 2017-01-01, to match Var_diff, change to hw_event_new[1:,:,:]
        Var_diff     = np.where(hw_event_new[1:,:,:]==1, Var_diff, np.nan)
    else:
        Var_diff     = np.where(hw_event_new==1, Var_diff, np.nan)
    var_diff     = np.nanmean(Var_diff,axis=0)

    return var_diff

def plot_heatwave_days(EHF_out, time_s=None, time_e=None, lat_names="lat", lon_names="lon", loc_lat=None, loc_lon=None,message=None):

    # ==== calculate HW days ====
    # read in HW events
    time_ehf, hw_event = read_var_multi_file(EHF_out, 'event', loc_lat, loc_lon, lat_names, lon_names)
    time_cood          = time_mask(time_ehf, time_s, time_e, seconds=None)
    hw_event_new       = hw_event[time_cood,:,:]

    time, lats          = read_var(EHF_out[0], 'lat', loc_lat, loc_lon, lat_name='lat', lon_name='lon')
    time, lons          = read_var(EHF_out[0], 'lon', loc_lat, loc_lon, lat_name='lat', lon_name='lon')

    if time_s == datetime(2017,1,1,0,0,0,0):
        # since LIS-CABLE miss 2017-01-01, to match Var_diff, change to hw_event_new[1:,:,:]
        days     = np.where(hw_event_new[1:,:,:]==1, 1, 0)
    else:
        days     = np.where(hw_event_new==1, 1, 0)
    days_total   = np.sum(days,axis=0)


    # ================== Start Plotting =================
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=[5,5],sharex=True,
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
    # =============== CHANGE HERE ===============
    cmap  = plt.cm.rainbow #seismic

    axs.coastlines(resolution="50m",linewidth=1)
    axs.set_extent([135,155,-39,-23])
    axs.add_feature(states, linewidth=.5, edgecolor="black")

    # Add gridlines
    gl = axs.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color=almost_black, linestyle='--')
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

    clevs = [0,5,10,15,20,25,30,35,40,45,50,55,60]

    plot1 = axs.contourf(lons, lats, days_total, clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend='both') #
    cb = plt.colorbar(plot1, ax=axs, ticklocation="right", pad=0.08, orientation="horizontal",aspect=40, shrink=1)
    cb.ax.tick_params(labelsize=10,labelrotation=45)
    # plt.title(message, size=16)

    plt.savefig('./plots/spatial_map_total_hw_days_'+message+'.png',dpi=300)



def plot_spatial_map_hw_magnitude(EHF_out, land_ctl_path, land_sen_path, var_names,
                                  time_s=None,time_e=None, lat_names="lat", lon_names="lon",
                                  loc_lat=None, loc_lon=None, message=None):

    hw_tmax_diff = calc_heatwave_magnitude(EHF_out, land_ctl_path, land_sen_path, var_names[0], time_s, time_e,
                                            lat_names, lon_names, loc_lat, loc_lon)
    hw_tmin_diff = calc_heatwave_magnitude(EHF_out, land_ctl_path, land_sen_path, var_names[1], time_s, time_e,
                                            lat_names, lon_names, loc_lat, loc_lon)
    hw_dr_diff   = hw_tmax_diff - hw_tmin_diff

    print("hw_tmax_diff",hw_tmax_diff)

    land_ctl_file = land_ctl_path+"Tair_f_inst/LIS.CABLE.201701-202002.nc"
    time, lats   = read_var(land_ctl_file, 'lat', loc_lat, loc_lon, lat_name='lat', lon_name='lon')
    time, lons   = read_var(land_ctl_file, 'lon', loc_lat, loc_lon, lat_name='lat', lon_name='lon')

    # ================== Start Plotting =================
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=[12,6],sharex=True,
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
    # =============== CHANGE HERE ===============
    cmap  = plt.cm.seismic
    clevs = [-1.2,-1.1,-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.,1.1,1.2]

    for i in np.arange(3):

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


    plot1 = axs[0].contourf(lons, lats, hw_tmax_diff, clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend='both') #
    cb = plt.colorbar(plot1, ax=axs[0], ticklocation="right", pad=0.08, orientation="horizontal",aspect=40, shrink=1)
    cb.ax.tick_params(labelsize=10,labelrotation=45)
    # plt.title(message, size=16)

    plot1 = axs[1].contourf(lons, lats, hw_tmin_diff, clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend='both') #
    cb = plt.colorbar(plot1, ax=axs[1], ticklocation="right", pad=0.08, orientation="horizontal",aspect=40, shrink=1)
    cb.ax.tick_params(labelsize=10,labelrotation=45)
    # plt.title(message, size=16)

    plot1 = axs[2].contourf(lons, lats, hw_dr_diff, clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend='both') #
    cb = plt.colorbar(plot1, ax=axs[2], ticklocation="right", pad=0.08, orientation="horizontal",aspect=40, shrink=1)
    cb.ax.tick_params(labelsize=10,labelrotation=45)
    # plt.title(message, size=16)

    plt.savefig('./plots/spatial_map_hw_'+message+'.png',dpi=300)

if __name__ == "__main__":

    # ======================= Option =======================
    region = "Aus" #"SE Aus" #"CORDEX" #"SE Aus"

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

    if 0:
        # regrid EHF event

        EHF_path   = '/g/data/w97/mm3972/scripts/ehfheatwaves/nc_file/AUS_1970_2022/EHF_heatwaves_201701-202002_daily_flip.nc'
        wrf_path   = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/WRF_output/wrfout_d01_2017-02-01_06:00:00"
        EHF_out    = '/g/data/w97/mm3972/scripts/ehfheatwaves/nc_file/AUS_1970_2022/HW_Event_Indicator_201701-202002.nc'
        regrid_EHF_4_WRF_domain(EHF_path,wrf_path,EHF_out)

    if 0:
        '''
        Plot Tmax Tmin & TDR during HW
        '''

        case_ctl       = "drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2"
        case_sen       = "drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2_obs_LAI_ALB"

        land_ctl_path  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_ctl+"/LIS_output/"
        land_sen_path  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_sen+"/LIS_output/"

        EHF_out        = ['/g/data/w97/mm3972/scripts/ehfheatwaves/nc_file/AUS_1970_2022/HW_Event_Indicator_201701-202002.nc']


        var_names      = ['Tmax','Tmin'] # "SurfTmax","SurfTmin",

        time_s         = datetime(2017,6,1,0,0,0,0)
        time_e         = datetime(2017,9,1,0,0,0,0)
        message        = "2017_Winter_Tmax_Tmin_TDR"
        plot_spatial_map_hw_magnitude(EHF_out, land_ctl_path, land_sen_path, var_names,
                                      time_s=time_s,time_e=time_e, lat_names="lat", lon_names="lon",
                                      loc_lat=loc_lat, loc_lon=loc_lon, message=message)

        time_s         = datetime(2018,6,1,0,0,0,0)
        time_e         = datetime(2018,9,1,0,0,0,0)
        message        = "2018_Winter_Tmax_Tmin_TDR"
        plot_spatial_map_hw_magnitude(EHF_out, land_ctl_path, land_sen_path, var_names,
                                      time_s=time_s,time_e=time_e, lat_names="lat", lon_names="lon",
                                      loc_lat=loc_lat, loc_lon=loc_lon, message=message)

        time_s         = datetime(2019,6,1,0,0,0,0)
        time_e         = datetime(2019,9,1,0,0,0,0)
        message        = "2019_Winter_Tmax_Tmin_TDR"
        plot_spatial_map_hw_magnitude(EHF_out, land_ctl_path, land_sen_path, var_names,
                                      time_s=time_s,time_e=time_e, lat_names="lat", lon_names="lon",
                                      loc_lat=loc_lat, loc_lon=loc_lon, message=message)


    if 1:

        case_name      = "ALB-CTL_new" #"bl_pbl2_mp4_sf_sfclay2" 
        EHF_out        = ['/g/data/w97/mm3972/scripts/ehfheatwaves/nc_file/AUS_1970_2022/HW_Event_Indicator_201701-202002.nc']
        
        time_s         = datetime(2017,6,1,0,0,0,0)
        time_e         = datetime(2017,9,1,0,0,0,0)
        message        = "2017_Winter"
        plot_heatwave_days(EHF_out, time_s=time_s, time_e=time_e, lat_names="lat", lon_names="lon", loc_lat=loc_lat, loc_lon=loc_lon,message=message)


        time_s         = datetime(2018,6,1,0,0,0,0)
        time_e         = datetime(2018,9,1,0,0,0,0)
        message        = "2018_Winter"
        plot_heatwave_days(EHF_out, time_s=time_s, time_e=time_e, lat_names="lat", lon_names="lon", loc_lat=loc_lat, loc_lon=loc_lon,message=message)

        time_s         = datetime(2019,6,1,0,0,0,0)
        time_e         = datetime(2019,9,1,0,0,0,0)
        message        = "2019_Winter"
        plot_heatwave_days(EHF_out, time_s=time_s, time_e=time_e, lat_names="lat", lon_names="lon", loc_lat=loc_lat, loc_lon=loc_lon,message=message)




