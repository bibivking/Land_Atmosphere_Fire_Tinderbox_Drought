#!/usr/bin/env python
"""
Produce the Australia map with the label of EucFACE site - Figure 1
"""

__author__ = "Mengyuan Mu"
__email__  = "mu.mengyuan815@gmail.com"


#!/usr/bin/python

import sys
import cartopy
import numpy as np
import pandas as pd
import shapefile as shp
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import scipy.ndimage as ndimage
import multiprocessing as mp
from sklearn.metrics import mean_squared_error
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


def read_LIS_obs_var(file_name, land_ctl_path, land_sen_path, AWAP_T_file, nc_path,
                     var_name, loc_lat, loc_lon, lat_names, lon_names, time_s, time_e, message=None):

    # Read simulations
    land_ctl_files    = [land_ctl_path+'Tair_f_inst/'+file_name]
    land_sen_files    = [land_sen_path+'Tair_f_inst/'+file_name]
    time, Ctl_tmp     = read_var_multi_file(land_ctl_files, 'Tair_f_inst', loc_lat, loc_lon, lat_names, lon_names)
    time, Sen_tmp     = read_var_multi_file(land_sen_files, 'Tair_f_inst', loc_lat, loc_lon, lat_names, lon_names)
    Ctl_tmp           = Ctl_tmp - 273.15
    Sen_tmp           = Sen_tmp - 273.15
    print('time',time)
    print('is any Ctl_tmp not nan ',np.any(~np.isnan(Ctl_tmp)))

    # read lat and lon in simulations
    wrf_path   = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/WRF_output/wrfout_d01_2019-12-01_01:00:00"
    wrf        = Dataset(wrf_path,  mode='r')
    lon_out    = wrf.variables['XLONG'][0,:,:]
    lat_out    = wrf.variables['XLAT'][0,:,:]

    # Read observations
    if 'max' in var_name:
        time_obs, Obs_tmp = read_var_multi_file([AWAP_T_file], 'tmax', loc_lat, loc_lon, 'lat', 'lon')
    elif 'mean' in var_name:
        AWAP_Tmax_file    = '/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/agcd_v1-0-1_tmax_mean_r005_daily_2017-2020.nc'     # air temperature
        AWAP_Tmin_file    = '/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/agcd_v1-0-1_tmin_mean_r005_daily_2017-2020.nc'     # air temperature
        time_obs, Obs_tmax = read_var_multi_file([AWAP_Tmax_file], 'tmax', loc_lat, loc_lon, 'lat', 'lon')
        time_obs, Obs_tmin = read_var_multi_file([AWAP_Tmin_file], 'tmin', loc_lat, loc_lon, 'lat', 'lon')
        Obs_tmp           = (Obs_tmax+Obs_tmin)/2.

    awap              = Dataset(AWAP_T_file,  mode='r')
    lat_in            = awap.variables['lat'][:]
    lon_in            = awap.variables['lon'][:]

    # Read ERA5
    current_month = time_s.month
    current_year  = time_s.year

    if current_year == 2017:
        time_s_winter = datetime(2017,6,1,0,0,0,0)
        time_e_winter = datetime(2017,9,1,0,0,0,0)
    elif current_year == 2018:
        time_s_winter = datetime(2018,6,1,0,0,0,0)
        time_e_winter = datetime(2018,9,1,0,0,0,0)
    elif current_year == 2019:
        time_s_winter = datetime(2019,6,1,0,0,0,0)
        time_e_winter = datetime(2019,9,1,0,0,0,0)

    if message == "last_summer":
        print('current_year',current_year,'current_month',current_month)
        ERA5_T_files      = [ f'{nc_path}2t_era5_oper_sfc_201912_day_max.nc',
                              f'{nc_path}2t_era5_oper_sfc_202001_day_max.nc',
                              f'{nc_path}2t_era5_oper_sfc_202002_day_max.nc',]
    elif current_month == 6:
        print('current_year',current_year,'current_month',current_month)
        ERA5_T_files      = [ f'{nc_path}2t_era5_oper_sfc_{current_year}06_day_max.nc',
                              f'{nc_path}2t_era5_oper_sfc_{current_year}07_day_max.nc',
                              f'{nc_path}2t_era5_oper_sfc_{current_year}08_day_max.nc',]
    elif current_month == 12:
        print('current_year',current_year,'current_month',current_month)
        ERA5_T_files      = [ f'{nc_path}2t_era5_oper_sfc_{current_year}12_day_max.nc',
                              f'{nc_path}2t_era5_oper_sfc_{current_year+1}01_day_max.nc',
                              f'{nc_path}2t_era5_oper_sfc_{current_year+1}02_day_max.nc',]

    time_ERA, ERA_tmp = read_var_multi_file(ERA5_T_files, 't2m', loc_lat, loc_lon, 'latitude', 'longitude')
    era5              = Dataset(ERA5_T_files[0],  mode='r')
    ERA_tmp           = ERA_tmp - 273.15
    lat_ERA_in        = era5.variables['latitude'][:]
    lon_ERA_in        = era5.variables['longitude'][:]

    # ctl and sen temporal mean
    if 'max' in var_name:
        # average of daily max
        ctl_in       = spatial_var_max(time,Ctl_tmp,time_s,time_e)
        sen_in       = spatial_var_max(time,Sen_tmp,time_s,time_e)
    elif 'min' in var_name:
        # average of daily min
        ctl_in       = spatial_var_min(time,Ctl_tmp,time_s,time_e)
        sen_in       = spatial_var_min(time,Sen_tmp,time_s,time_e)
    elif 'TDR' in var_name:
        # average of daily min
        ctl_in_max   = spatial_var_max(time,Ctl_tmp,time_s,time_e)
        sen_in_max   = spatial_var_max(time,Sen_tmp,time_s,time_e)
        ctl_in_min   = spatial_var_min(time,Ctl_tmp,time_s,time_e)
        sen_in_min   = spatial_var_min(time,Sen_tmp,time_s,time_e)
        ctl_in       = ctl_in_max - ctl_in_min
        sen_in       = sen_in_max - sen_in_min
    else:
        ctl_in       = spatial_var(time,Ctl_tmp,time_s,time_e)
        sen_in       = spatial_var(time,Sen_tmp,time_s,time_e)

    # obs temporal mean
    obs_in           = spatial_var_mean(time_obs, Obs_tmp, time_s, time_e)
    #obs_in           = spatial_var_mean(time_obs, Obs_tmp, time_s_winter, time_e_winter)

    # ERA5 temporal mean
    ERA_in           = spatial_var_mean(time_ERA, ERA_tmp, time_s, time_e)

    # regrid
    obs_regrid       = regrid_data(lat_in, lon_in, lat_out, lon_out, obs_in)
    ERA_regrid       = regrid_data(lat_ERA_in, lon_ERA_in, lat_out, lon_out, ERA_in)

    ERA_regrid       = np.where(ctl_in>-9999., ERA_regrid, np.nan)
    obs_regrid       = np.where(ctl_in>-9999., obs_regrid, np.nan)

    print('is any Obs_tmp not nan ',np.any(~np.isnan(Obs_tmp)))

    # =============== CHANGE HERE ===============
    # Define the RGB values as a 2D array
    rgb_17colors= np.array([
                        [0.338024, 0.193310, 0.020377],
                        [0.458593, 0.264360, 0.031142],
                        [0.576471, 0.343483, 0.058055],
                        [0.686275, 0.446828, 0.133410],
                        [0.778547, 0.565859, 0.250288],
                        [0.847443, 0.705805, 0.422530],
                        [0.932872, 0.857209, 0.667820],
                        [0.964091, 0.917801, 0.795463],
                        [0.955517, 0.959016, 0.9570165],
                        [0.808689, 0.924414, 0.907882],
                        [0.627528, 0.855210, 0.820531],
                        [0.426990, 0.749942, 0.706882],
                        [0.265513, 0.633679, 0.599231],
                        [0.135871, 0.524337, 0.492964],
                        [0.023914, 0.418839, 0.387466],
                        [0.002153, 0.325721, 0.287274],
                        [0.000000, 0.235294, 0.188235]
                    ])

    rgb_21colors = np.array([
                [0.338024, 0.193310, 0.020377],
                [0.441369, 0.254210, 0.029604],
                [0.544714, 0.315110, 0.038831],
                [0.631373, 0.395156, 0.095732],
                [0.733333, 0.491119, 0.165706],
                [0.793310, 0.595848, 0.287197],
                [0.857286, 0.725798, 0.447136],
                [0.904575, 0.810458, 0.581699],
                [0.947020, 0.880584, 0.710880],
                [0.963629, 0.923799, 0.818531],
                [0.955517, 0.959016, 0.9570165],
                [0.822837, 0.927797, 0.912803],
                [0.714879, 0.890888, 0.864821],
                [0.583852, 0.837370, 0.798385],
                [0.461592, 0.774856, 0.729950],
                [0.311649, 0.666897, 0.629988],
                [0.183852, 0.569550, 0.538178],
                [0.087889, 0.479123, 0.447751],
                [0.003691, 0.390311, 0.358016],
                [0.001845, 0.312803, 0.273126],
                [0.000000, 0.235294, 0.188235]
            ])


    # Create a colormap from the RGB values
    cmap17 = plt.cm.colors.ListedColormap(rgb_17colors)
    cmap21 = plt.cm.colors.ListedColormap(rgb_21colors)

    cmap  = plt.cm.BrBG
    if var_name in ['SoilMoist_inst','SoilMoist',"SM_top50cm"]:
        clevs = [-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0.05,0.1,0.15,0.2,0.25,0.3]
    elif var_name in ["Swnet_tavg","Lwnet_tavg","SWdown_f_inst","LWdown_f_inst","Qle_tavg","Qh_tavg","Qg_tavg"]:
        clevs = [-22,-18,-14,-10,-6,-2,2,6,10,14,18,22]
    elif var_name in ["Rnet"]:
        clevs = [-22,-18,-14,-10,-6,-2,2,6,10,14,18,22]
        cmap  = plt.cm.BrBG_r
    elif var_name in ["Tair_f_inst","Tmax","Tmin","VegT_tavg","VegTmax","VegTmin",
                        "AvgSurfT_tavg","SurfTmax","SurfTmin","SoilTemp_inst",'TDR','VegTDR','SurfTDR']:
        clevs = [-5.,-4.,-3.,-2.,-1.,-0.5, 0.5,1.,2.,3.,4.,5.]
        cmap  = plt.cm.seismic
    elif var_name in ["FWsoil_tavg","SmLiqFrac_inst","SmFrozFrac_inst"]:
        clevs = [-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0.05,0.1,0.15,0.2,0.25,0.3,0.35]
    elif var_name in ["LAI_inst"]:
        clevs = [-2,-1.8,-1.6,-1.4,-1.2,-1.,-0.8,-0.6,-0.4,-0.2,-0.05,0.05,0.2,0.4,0.6,0.8,1.,1.2,1.4,1.6,1.8,2]
        clevs_percentage =  [-70,-60,-50,-40,-30,-20,-10,-5,5,10,20,30,40,50,60,70]
        cmap  = cmap21
    elif var_name in ["Albedo_inst"]:
        clevs = [-0.08,-0.07,-0.06,-0.05,-0.04,-0.03,-0.02,-0.01,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08]
        clevs_percentage =   [-70,-60,-50,-40,-30,-20,-10,-5,5,10,20,30,40,50,60,70]
        cmap  = cmap17
    else:
        clevs = [-0.5,-0.4,-0.3,-0.2,-0.1,-0.05,0.05,0.1,0.2,0.3,0.4,0.5]

    return ctl_in, sen_in, obs_regrid, ERA_regrid, cmap, clevs

def read_LIS_time_series(file_name, land_ctl_path, land_sen_path, AWAP_T_file, nc_path,
                     var_name, loc_lat, loc_lon, lat_names, lon_names, time_s, time_e, message=None):

    # Read simulations
    land_ctl_files    = [land_ctl_path+'Tair_f_inst/'+file_name]
    land_sen_files    = [land_sen_path+'Tair_f_inst/'+file_name]
    time, Ctl_tmp     = read_var_multi_file(land_ctl_files, 'Tair_f_inst', loc_lat, loc_lon, lat_names, lon_names)
    time, Sen_tmp     = read_var_multi_file(land_sen_files, 'Tair_f_inst', loc_lat, loc_lon, lat_names, lon_names)
    Ctl_tmp           = Ctl_tmp - 273.15
    Sen_tmp           = Sen_tmp - 273.15
    print('time',time)
    print('is any Ctl_tmp not nan ',np.any(~np.isnan(Ctl_tmp)))

    # read lat and lon in simulations
    wrf_path   = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/WRF_output/wrfout_d01_2019-12-01_01:00:00"
    wrf        = Dataset(wrf_path,  mode='r')
    lon_out    = wrf.variables['XLONG'][0,:,:]
    lat_out    = wrf.variables['XLAT'][0,:,:]

    # Read observations
    if 'max' in var_name:
        time_obs, Obs_tmp = read_var_multi_file([AWAP_T_file], 'tmax', loc_lat, loc_lon, 'lat', 'lon')
    elif 'mean' in var_name:
        AWAP_Tmax_file    = '/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/agcd_v1-0-1_tmax_mean_r005_daily_2017-2020.nc'     # air temperature
        AWAP_Tmin_file    = '/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/agcd_v1-0-1_tmin_mean_r005_daily_2017-2020.nc'     # air temperature
        time_obs, Obs_tmax = read_var_multi_file([AWAP_Tmax_file], 'tmax', loc_lat, loc_lon, 'lat', 'lon')
        time_obs, Obs_tmin = read_var_multi_file([AWAP_Tmin_file], 'tmin', loc_lat, loc_lon, 'lat', 'lon')
        Obs_tmp           = (Obs_tmax+Obs_tmin)/2.

    awap              = Dataset(AWAP_T_file,  mode='r')
    lat_in            = awap.variables['lat'][:]
    lon_in            = awap.variables['lon'][:]

    # ctl and sen temporal mean
    if 'max' in var_name:
        # average of daily max
        ctl_in       = time_clip_to_day_max(time, Ctl_tmp, time_s, time_e)
        sen_in       = time_clip_to_day_max(time, Sen_tmp, time_s, time_e)
    else:
        ctl_in       = time_clip_to_day(time, Ctl_tmp, time_s, time_e)
        sen_in       = time_clip_to_day(time, Sen_tmp, time_s, time_e)

    ctl_time_series = np.nanmean(ctl_in, axis=(1,2))
    sen_time_series = np.nanmean(sen_in, axis=(1,2))

    # obs temporal mean

    obs_in           = time_clip_to_day(time_obs, Obs_tmp, time_s, time_e)

    obs_regrid       = np.zeros(np.shape(ctl_in))

    for i in np.arange(len(ctl_in[:,0,0])):
        tmp               = regrid_data(lat_in, lon_in, lat_out, lon_out, obs_in[i,:,:])
        obs_regrid[i,:,:] = np.where(ctl_in[0,:,:]>-9999., tmp, np.nan)

    obs_time_series  = np.nanmean(obs_regrid, axis=(1,2))

    return ctl_time_series, sen_time_series, obs_time_series


def read_LIS_time_series_burnt(file_name, land_ctl_path, land_sen_path, AWAP_T_file, nc_path,
                     var_name, loc_lat, loc_lon, lat_names, lon_names, time_s, time_e, message=None):

    # Read simulations
    land_ctl_files    = [land_ctl_path+'Tair_f_inst/'+file_name]
    land_sen_files    = [land_sen_path+'Tair_f_inst/'+file_name]
    time, Ctl_tmp     = read_var_multi_file(land_ctl_files, 'Tair_f_inst', loc_lat, loc_lon, lat_names, lon_names)
    time, Sen_tmp     = read_var_multi_file(land_sen_files, 'Tair_f_inst', loc_lat, loc_lon, lat_names, lon_names)
    Ctl_tmp           = Ctl_tmp - 273.15
    Sen_tmp           = Sen_tmp - 273.15

    # read lat and lon in simulations
    wrf_path   = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/WRF_output/wrfout_d01_2019-12-01_01:00:00"
    wrf        = Dataset(wrf_path,  mode='r')
    lon_in     = wrf.variables['XLONG'][0,:,:]
    lat_in     = wrf.variables['XLAT'][0,:,:]

    # Read observations
    if 'max' in var_name:
        time_obs, Obs_tmp = read_var_multi_file([AWAP_T_file], 'tmax', loc_lat, loc_lon, 'lat', 'lon')
    elif 'mean' in var_name:
        AWAP_Tmax_file    = '/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/agcd_v1-0-1_tmax_mean_r005_daily_2017-2020.nc'     # air temperature
        AWAP_Tmin_file    = '/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/agcd_v1-0-1_tmin_mean_r005_daily_2017-2020.nc'     # air temperature
        time_obs, Obs_tmax = read_var_multi_file([AWAP_Tmax_file], 'tmax', loc_lat, loc_lon, 'lat', 'lon')
        time_obs, Obs_tmin = read_var_multi_file([AWAP_Tmin_file], 'tmin', loc_lat, loc_lon, 'lat', 'lon')
        Obs_tmp           = (Obs_tmax+Obs_tmin)/2.

    awap              = Dataset(AWAP_T_file,  mode='r')
    lat_obs_in        = awap.variables['lat'][:]
    lon_obs_in        = awap.variables['lon'][:]

    # ctl and sen temporal mean
    if 'max' in var_name:
        # average of daily max
        ctl_in       = time_clip_to_day_max(time, Ctl_tmp, time_s, time_e)
        sen_in       = time_clip_to_day_max(time, Sen_tmp, time_s, time_e)
    else:
        ctl_in       = time_clip_to_day(time, Ctl_tmp, time_s, time_e)
        sen_in       = time_clip_to_day(time, Sen_tmp, time_s, time_e)

    # obs temporal mean
    obs_in           = time_clip_to_day(time_obs, Obs_tmp, time_s, time_e)

    # =========== Read in fire data ============
    fire_path  = '/g/data/w97/mm3972/data/MODIS/MODIS_fire/MCD64A1.061_500m_aid0001.nc'
    fire_file  = Dataset(fire_path, mode='r')
    Burn_Date  = fire_file.variables['Burn_Date'][0:8,:,:]  # 2019-07 - 2020-02
    lat_out    = fire_file.variables['lat'][:]
    lon_out    = fire_file.variables['lon'][:]

    # burnt region from 2019-07 to 2020-02
    burn_area  = np.where( Burn_Date[0,:,:] + Burn_Date[1,:,:] + Burn_Date[2,:,:] + Burn_Date[3,:,:] +
                           Burn_Date[4,:,:] + Burn_Date[5,:,:] + Burn_Date[6,:,:] + Burn_Date[7,:,:] > 0, 1, Burn_Date[0,:,:])

    lons_2D, lats_2D = np.meshgrid(lon_out, lat_out)

    ntime = len(ctl_in[:,0,0])
    nlat  = len(burn_area[:,0])
    nlon  = len(burn_area[0,:])

    burn_area_3D = np.repeat(burn_area[np.newaxis, :, :], ntime, axis=0)
    lons_3D      = np.repeat(lons_2D[np.newaxis, :, :], ntime, axis=0)
    lats_3D      = np.repeat(lats_2D[np.newaxis, :, :], ntime, axis=0)

    obs_regrid  = np.zeros((4,ntime,nlat,nlon))
    ctl_regrid  = np.zeros((4,ntime,nlat,nlon))
    sen_regrid  = np.zeros((4,ntime,nlat,nlon))

    for i in np.arange(ntime):
        ctl_regrid[0,i,:,:] = regrid_data(lat_in, lon_in, lat_out, lon_out, ctl_in[i,:,:])
        sen_regrid[0,i,:,:] = regrid_data(lat_in, lon_in, lat_out, lon_out, sen_in[i,:,:])
        obs_regrid[0,i,:,:] = regrid_data(lat_obs_in, lon_obs_in, lat_out, lon_out, obs_in[i,:,:])

    # set fire burnt focus regions
    #                   North ,        Central,       South
    loc_lats       = [[-32,-28.5],   [-34.5,-32.5], [-38,-34.5]]
    loc_lons       = [[151.5,153.5], [149.5,151.5], [146.5,151]]

    for j in np.arange(3):
        obs_regrid[j+1,:,:,:] = np.where(np.all(( lats_3D>loc_lats[j][0],
                                   lats_3D<loc_lats[j][1],
                                   lons_3D>loc_lons[j][0],
                                   lons_3D<loc_lons[j][1],
                                   burn_area_3D == 1), axis=0),
                                   obs_regrid[0,:,:,:], np.nan)

        ctl_regrid[j+1,:,:,:] = np.where(np.all(( lats_3D>loc_lats[j][0],
                                   lats_3D<loc_lats[j][1],
                                   lons_3D>loc_lons[j][0],
                                   lons_3D<loc_lons[j][1],
                                   burn_area_3D == 1), axis=0),
                                   ctl_regrid[0,:,:,:], np.nan)

        sen_regrid[j+1,:,:,:] = np.where(np.all(( lats_3D>loc_lats[j][0],
                                   lats_3D<loc_lats[j][1],
                                   lons_3D>loc_lons[j][0],
                                   lons_3D<loc_lons[j][1],
                                   burn_area_3D == 1), axis=0),
                                   sen_regrid[0,:,:,:], np.nan)

    ctl_time_series = np.nanmean(ctl_regrid, axis=(2,3))
    sen_time_series = np.nanmean(sen_regrid, axis=(2,3))
    obs_time_series = np.nanmean(obs_regrid, axis=(2,3))

    return ctl_time_series, sen_time_series, obs_time_series

def read_LIS_time_series_burnt_new(file_name, land_ctl_path, land_sen_path, AWAP_T_file, nc_path,
                     var_name, loc_lat, loc_lon, lat_names, lon_names, time_s, time_e, message=None):

    # ctl and sen temporal mean
    path = '/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/'

    if 'max' in var_name:
        keyword = 'max'
    elif 'mean' in var_name:
        keyword = 'mean'

    ctl_files = [ f'{path}LIS.CABLE.201701-202002_day{keyword}_North_clim_regrid_to_fire_map_grids.nc',
                  f'{path}LIS.CABLE.201701-202002_day{keyword}_Central_clim_regrid_to_fire_map_grids.nc',
                  f'{path}LIS.CABLE.201701-202002_day{keyword}_South_clim_regrid_to_fire_map_grids.nc',]

    sen_files = [ f'{path}LIS.CABLE.201701-202002_day{keyword}_North_dyn_regrid_to_fire_map_grids.nc',
                  f'{path}LIS.CABLE.201701-202002_day{keyword}_Central_dyn_regrid_to_fire_map_grids.nc',
                  f'{path}LIS.CABLE.201701-202002_day{keyword}_South_dyn_regrid_to_fire_map_grids.nc',]

    obs_files = [ f'{path}agcd_v1-0-1_tmax_mean_r005_daily_2017-2020_North_regrid_to_fire_map_grids.nc',
                  f'{path}agcd_v1-0-1_tmax_mean_r005_daily_2017-2020_Central_regrid_to_fire_map_grids.nc',
                  f'{path}agcd_v1-0-1_tmax_mean_r005_daily_2017-2020_South_regrid_to_fire_map_grids.nc',]

    if 'mean' in var_name:
        obs_files1 = [ f'{path}agcd_v1-0-1_tmin_mean_r005_daily_2017-2020_North_regrid_to_fire_map_grids.nc',
                      f'{path}agcd_v1-0-1_tmin_mean_r005_daily_2017-2020_Central_regrid_to_fire_map_grids.nc',
                      f'{path}agcd_v1-0-1_tmin_mean_r005_daily_2017-2020_South_regrid_to_fire_map_grids.nc',]

    fire_files= [ f'{path}MCD64A1.061_500m_aid0001_North.nc',
                  f'{path}MCD64A1.061_500m_aid0001_Central.nc',
                  f'{path}MCD64A1.061_500m_aid0001_South.nc',]

    f_ctl      = Dataset(ctl_files[0], mode='r')
    ntime      = len(f_ctl.variables['Tair_f_inst'][:,0,0])

    ctl_time_series = np.zeros((3,ntime))
    sen_time_series = np.zeros((3,ntime))
    obs_time_series = np.zeros((3,ntime))

    for i in np.arange(3):
        f_ctl      = Dataset(ctl_files[i], mode='r')
        ctl_in     = f_ctl.variables['Tair_f_inst'][:] - 273.15

        f_sen      = Dataset(sen_files[i], mode='r')
        sen_in     = f_sen.variables['Tair_f_inst'][:] - 273.15

        if 'max' in var_name:
            f_obs  = Dataset(obs_files[i], mode='r')
            obs_in = f_obs.variables['tmax'][:]
        elif 'mean' in var_name:
            f_obs  = Dataset(obs_files[i], mode='r')
            f_obs1 = Dataset(obs_files1[i], mode='r')
            obs_in = (f_obs.variables['tmax'][:]  + f_obs1.variables['tmin'][:])/2

        # =========== Read in fire data ============
        fire_file  = Dataset(fire_files[i], mode='r')
        Burn_Date  = fire_file.variables['Burn_Date'][0:8,:,:]  # 2019-07 - 2020-02
        lat_out    = fire_file.variables['lat'][:]
        lon_out    = fire_file.variables['lon'][:]

        # burnt region from 2019-07 to 2020-02
        burn_area  = np.where( Burn_Date[0,:,:] + Burn_Date[1,:,:] + Burn_Date[2,:,:] + Burn_Date[3,:,:] +
                               Burn_Date[4,:,:] + Burn_Date[5,:,:] + Burn_Date[6,:,:] + Burn_Date[7,:,:] > 0, 1, Burn_Date[0,:,:])

        lons_2D, lats_2D = np.meshgrid(lon_out, lat_out)
        burn_area_3D     = np.repeat(burn_area[np.newaxis, :, :], ntime, axis=0)

        ctl_time_series[i,:] = np.nanmean(np.where(burn_area_3D == 1,ctl_in[:,:,:], np.nan),axis=(1,2))
        sen_time_series[i,:] = np.nanmean(np.where(burn_area_3D == 1,sen_in[:,:,:], np.nan),axis=(1,2))
        obs_time_series[i,:] = np.nanmean(np.where(burn_area_3D == 1,obs_in[:ntime,:,:], np.nan),axis=(1,2))

    return ctl_time_series, sen_time_series, obs_time_series

def read_LIS_time_series_burnt_version1(file_name, land_ctl_path, land_sen_path, AWAP_T_file, nc_path,
                     var_name, loc_lat, loc_lon, lat_names, lon_names, time_s, time_e, message=None):

    # ctl and sen temporal mean
    if 'max' in var_name:
        file_name         = "LIS.CABLE.201701-202002_daymax_regrid_to_fire_map_grids.nc"
    elif 'mean' in var_name:
        file_name         = "LIS.CABLE.201701-202002_daymean_regrid_to_fire_map_grids.nc"

    f_ctl      = Dataset(f'{land_ctl_path}/Tair_f_inst/{file_name}', mode='r')
    ctl_in     = f_ctl.variables['Tair_f_inst'][:] - 273.15

    f_sen      = Dataset(f'{land_sen_path}/Tair_f_inst/{file_name}', mode='r')
    sen_in     = f_sen.variables['Tair_f_inst'][:] - 273.15

    # Read observations
    if 'max' in var_name:
        f_obs      = Dataset('/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/agcd_v1-0-1_tmax_mean_r005_daily_2017-2020_regrid_to_fire_map_grids.nc', mode='r')
        obs_in     = f_obs.variables['tmax'][:]
    elif 'mean' in var_name:
        f_obs_max  = Dataset('/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/agcd_v1-0-1_tmax_mean_r005_daily_2017-2020_regrid_to_fire_map_grids.nc', mode='r')
        f_obs_min  = Dataset('/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/agcd_v1-0-1_tmin_mean_r005_daily_2017-2020_regrid_to_fire_map_grids.nc', mode='r')
        obs_in     = (f_obs_max.variables['tmax'][:]  + f_obs_min.variables['tmin'][:])/2

    # =========== Read in fire data ============
    fire_path  = '/g/data/w97/mm3972/data/MODIS/MODIS_fire/MCD64A1.061_500m_aid0001.nc'
    fire_file  = Dataset(fire_path, mode='r')
    Burn_Date  = fire_file.variables['Burn_Date'][0:8,:,:]  # 2019-07 - 2020-02
    lat_out    = fire_file.variables['lat'][:]
    lon_out    = fire_file.variables['lon'][:]

    # burnt region from 2019-07 to 2020-02
    burn_area  = np.where( Burn_Date[0,:,:] + Burn_Date[1,:,:] + Burn_Date[2,:,:] + Burn_Date[3,:,:] +
                           Burn_Date[4,:,:] + Burn_Date[5,:,:] + Burn_Date[6,:,:] + Burn_Date[7,:,:] > 0, 1, Burn_Date[0,:,:])

    lons_2D, lats_2D = np.meshgrid(lon_out, lat_out)

    ntime = len(ctl_in[:,0,0])
    nlat  = len(burn_area[:,0])
    nlon  = len(burn_area[0,:])

    burn_area_3D = np.repeat(burn_area[np.newaxis, :, :], ntime, axis=0)
    lons_3D      = np.repeat(lons_2D[np.newaxis, :, :], ntime, axis=0)
    lats_3D      = np.repeat(lats_2D[np.newaxis, :, :], ntime, axis=0)

    # set fire burnt focus regions
    #                   North ,        Central,       South
    loc_lats       = [[-32,-28.5],   [-34.5,-32.5], [-38,-34.5]]
    loc_lons       = [[151.5,153.5], [149.5,151.5], [146.5,151]]

    obs_regrid  = np.zeros((3,ntime,nlat,nlon))
    ctl_regrid  = np.zeros((3,ntime,nlat,nlon))
    sen_regrid  = np.zeros((3,ntime,nlat,nlon))

    for j in np.arange(3):
        obs_regrid[j,:,:,:] = np.where(np.all(( lats_3D>loc_lats[j][0],
                                   lats_3D<loc_lats[j][1],
                                   lons_3D>loc_lons[j][0],
                                   lons_3D<loc_lons[j][1],
                                   burn_area_3D == 1), axis=0),
                                   obs_in[:ntime,:,:], np.nan)

        ctl_regrid[j,:,:,:] = np.where(np.all(( lats_3D>loc_lats[j][0],
                                   lats_3D<loc_lats[j][1],
                                   lons_3D>loc_lons[j][0],
                                   lons_3D<loc_lons[j][1],
                                   burn_area_3D == 1), axis=0),
                                   ctl_in[:,:,:], np.nan)

        sen_regrid[j,:,:,:] = np.where(np.all(( lats_3D>loc_lats[j][0],
                                   lats_3D<loc_lats[j][1],
                                   lons_3D>loc_lons[j][0],
                                   lons_3D<loc_lons[j][1],
                                   burn_area_3D == 1), axis=0),
                                   sen_in[:,:,:], np.nan)

    ctl_time_series = np.nanmean(ctl_regrid, axis=(2,3))
    sen_time_series = np.nanmean(sen_regrid, axis=(2,3))
    obs_time_series = np.nanmean(obs_regrid, axis=(2,3))

    return ctl_time_series, sen_time_series, obs_time_series

def read_three_year_season(file_name, land_ctl_path, land_sen_path, AWAP_T_file, nc_path,
                    loc_lat, loc_lon, lat_names, lon_names, time_ss, time_es, message=None):

    ctl_all = np.zeros((271, 439, 529))
    sen_all = np.zeros((271, 439, 529))
    ERA_all = np.zeros((271, 439, 529))
    obs_all = np.zeros((271, 439, 529))

    t_s     = 0

    for i in np.arange(3):

        time_s = time_ss[i]
        time_e = time_es[i]

        # Read simulations
        land_ctl_files    = [land_ctl_path+'Tair_f_inst/'+file_name]
        land_sen_files    = [land_sen_path+'Tair_f_inst/'+file_name]
        time, Ctl_tmp     = read_var_multi_file(land_ctl_files, 'Tair_f_inst', loc_lat, loc_lon, lat_names, lon_names)
        time, Sen_tmp     = read_var_multi_file(land_sen_files, 'Tair_f_inst', loc_lat, loc_lon, lat_names, lon_names)
        Ctl_tmp           = Ctl_tmp - 273.15
        Sen_tmp           = Sen_tmp - 273.15

        # read lat and lon in simulations
        wrf_path   = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/WRF_output/wrfout_d01_2019-12-01_01:00:00"
        wrf        = Dataset(wrf_path,  mode='r')
        lon_out    = wrf.variables['XLONG'][0,:,:]
        lat_out    = wrf.variables['XLAT'][0,:,:]

        # Read observations
        time_obs, Obs_tmp = read_var_multi_file([AWAP_T_file], 'tmax', loc_lat, loc_lon, 'lat', 'lon')
        awap              = Dataset(AWAP_T_file,  mode='r')
        lat_in            = awap.variables['lat'][:]
        lon_in            = awap.variables['lon'][:]

        # Read ERA5
        current_month = time_s.month
        current_year  = time_s.year
        if message == "last_summer":
            print('current_year',current_year,'current_month',current_month)
            ERA5_T_files      = [ f'{nc_path}2t_era5_oper_sfc_201912_day_max.nc',
                                f'{nc_path}2t_era5_oper_sfc_202001_day_max.nc',
                                f'{nc_path}2t_era5_oper_sfc_202002_day_max.nc',]
        elif current_month == 6:
            print('current_year',current_year,'current_month',current_month)
            ERA5_T_files      = [ f'{nc_path}2t_era5_oper_sfc_{current_year}06_day_max.nc',
                                f'{nc_path}2t_era5_oper_sfc_{current_year}07_day_max.nc',
                                f'{nc_path}2t_era5_oper_sfc_{current_year}08_day_max.nc',]
        elif current_month == 12:
            print('current_year',current_year,'current_month',current_month)
            ERA5_T_files      = [ f'{nc_path}2t_era5_oper_sfc_{current_year}12_day_max.nc',
                                f'{nc_path}2t_era5_oper_sfc_{current_year+1}01_day_max.nc',
                                f'{nc_path}2t_era5_oper_sfc_{current_year+1}02_day_max.nc',]

        time_ERA, ERA_tmp = read_var_multi_file(ERA5_T_files, 't2m', loc_lat, loc_lon, 'latitude', 'longitude')
        era5              = Dataset(ERA5_T_files[0],  mode='r')
        ERA_tmp           = ERA_tmp - 273.15
        lat_ERA_in        = era5.variables['latitude'][:]
        lon_ERA_in        = era5.variables['longitude'][:]

        # sim time series
        ctl_in            = time_clip_to_day_max(time, Ctl_tmp, time_s, time_e)
        sen_in            = time_clip_to_day_max(time, Sen_tmp, time_s, time_e)

        # obs time series
        obs_in            = Obs_tmp[time_mask(time_obs, time_s, time_e)]

        # ERA5 time series
        ERA_in            = ERA_tmp[time_mask(time_ERA, time_s, time_e)]

        # regrid
        obs_length = len(obs_in[:,0,0])
        obs_regrid = np.zeros((obs_length, len(lat_out[:,0]), len(lon_out[0,:])))
        print('obs_length',obs_length)
        for i in np.arange(obs_length):
            obs_regrid[i,:,:] = regrid_data(lat_in, lon_in, lat_out, lon_out, obs_in[i,:,:])

        ERA_length = len(ERA_in[:,0,0])
        ERA_regrid = np.zeros((ERA_length, len(lat_out[:,0]), len(lon_out[0,:])))
        print('ERA_length',ERA_length)
        for i in np.arange(ERA_length):
            ERA_regrid[i,:,:] = regrid_data(lat_ERA_in, lon_ERA_in, lat_out, lon_out, ERA_in[i,:,:])

        ntime = len(ERA_regrid[:,0,0])
        ctl_all[t_s:t_s+ntime,:,:] = ctl_in
        sen_all[t_s:t_s+ntime,:,:] = sen_in
        ERA_all[t_s:t_s+ntime,:,:] = ERA_regrid
        obs_all[t_s:t_s+ntime,:,:] = obs_regrid

        t_s = t_s+ntime

    return ctl_all, sen_all, ERA_all, obs_all, lat_out, lon_out

def calc_temporal_correl(ctl_in, sen_in, ERA_in, obs_in):

    ctl_test = np.nanmean(ctl_in,axis=(1,2))
    sen_test = np.nanmean(sen_in,axis=(1,2))
    ERA_test = np.nanmean(ERA_in,axis=(1,2))
    obs_test = np.nanmean(obs_in,axis=(1,2))
    # if np.any(np.isnan(ctl_test)):
    print('ctl_test',ctl_test)
    print('sen_test',sen_test)
    print('ERA_test',ERA_test)
    print('obs_test',obs_test)

    # calculate temperal correlation
    nlat     = np.shape(ctl_in)[1]
    nlon     = np.shape(ctl_in)[2]
    ctl_r    = np.zeros((nlat, nlon))
    ctl_RMSE = np.zeros((nlat, nlon))

    sen_r    = np.zeros((nlat, nlon))
    sen_RMSE = np.zeros((nlat, nlon))

    ERA_r    = np.zeros((nlat, nlon))
    ERA_RMSE = np.zeros((nlat, nlon))

    for x in np.arange(0,nlat,1):
        for y in np.arange(0, nlon,1):
            # print('x, y',type(x), type(y))

            ctl_tmp = ctl_in[:,x,y]
            sen_tmp = sen_in[:,x,y]
            ERA_tmp = ERA_in[:,x,y]
            obs_tmp = obs_in[:,x,y]

            if np.any(np.isnan(obs_tmp)) or np.any(np.isnan(ctl_tmp)):
                ctl_r[x,y]    = np.nan
                sen_r[x,y]    = np.nan
                ERA_r[x,y]    = np.nan
                ctl_RMSE[x,y] = np.nan
                sen_RMSE[x,y] = np.nan
                ERA_RMSE[x,y] = np.nan
            else:
                ctl_r[x,y]    = stats.spearmanr(obs_tmp, ctl_tmp)[0] #pearsonr(obs_tmp, ctl_tmp)[0]
                sen_r[x,y]    = stats.spearmanr(obs_tmp, sen_tmp)[0] #pearsonr(obs_tmp, sen_tmp)[0]
                ERA_r[x,y]    = stats.spearmanr(obs_tmp, ERA_tmp)[0] #pearsonr(obs_tmp, ERA_tmp)[0]
                ctl_RMSE[x,y] = np.sqrt(mean_squared_error(obs_tmp, ctl_tmp))
                sen_RMSE[x,y] = np.sqrt(mean_squared_error(obs_tmp, sen_tmp))
                ERA_RMSE[x,y] = np.sqrt(mean_squared_error(obs_tmp, ERA_tmp))

    return ctl_r, sen_r, ERA_r, ctl_RMSE, sen_RMSE, ERA_RMSE

def spatial_map_winter_summer_abs(file_name, land_ctl_path, land_sen_path, AWAP_T_file, nc_path, var_names,
                              time_ss=None, time_es=None, lat_names="lat", lon_names="lon",loc_lat=None,
                              loc_lon=None, wrf_path=None,  message=None):

    '''
    plot a single spatial map
    '''

    # read lat and lon outs
    wrf            = Dataset(wrf_path,  mode='r')
    lons           = wrf.variables['XLONG'][0,:,:]
    lats           = wrf.variables['XLAT'][0,:,:]

    # WRF-CABLE
    for var_name in var_names:
        print("plotting "+var_name)

        # ================== Start Plotting =================
        fig, axs = plt.subplots(nrows=3, ncols=3, figsize=[12,10],sharex=False,
                    sharey=False, squeeze=True, subplot_kw={'projection': ccrs.PlateCarree()})

        plt.subplots_adjust(wspace=-0.15, hspace=0.105)

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
        texts = ["(a)","(b)","(c)","(d)",
                 "(e)","(f)","(g)","(h)",
                 "(i)","(j)","(k)","(l)"]

        for i in np.arange(9):

            row = int(i/3)
            col = i % 3

            print('i',i,'row=',row,"col=",col)

            if col == 0:
                ctl_in, sen_in, obs_regrid, ERA_regrid, cmap, clevs = \
                        read_LIS_obs_var(file_name, land_ctl_path, land_sen_path, AWAP_T_file, nc_path, var_name,
                                         loc_lat, loc_lon, lat_names, lon_names, time_ss[row], time_es[row], message=message)

            axs[row,col].coastlines(resolution="50m",linewidth=1)
            axs[row,col].set_extent([135,155,-39,-24])
            axs[row,col].add_feature(states, linewidth=.5, edgecolor="black")

            # Set the ticks on the x-axis and y-axis
            axs[row,col].tick_params(axis='x', direction='out')
            axs[row,col].tick_params(axis='y', direction='out')
            x_ticks = np.arange(135, 156, 5)
            y_ticks = np.arange(-40, -20, 5)
            axs[row,col].set_xticks(x_ticks)
            axs[row,col].set_yticks(y_ticks)
            axs[row, col].set_facecolor('lightgray')

            if row==2:
                axs[row,col].set_xticklabels(['135$\mathregular{^{o}}$E','140$\mathregular{^{o}}$E','145$\mathregular{^{o}}$E',
                                              '150$\mathregular{^{o}}$E','155$\mathregular{^{o}}$E'],rotation=25)
            else:
                axs[row,col].set_xticklabels([])

            if col==0:
                axs[row,col].set_yticklabels(['40$\mathregular{^{o}}$S','35$\mathregular{^{o}}$S',
                                              '30$\mathregular{^{o}}$S','25$\mathregular{^{o}}$S'])
            else:
                axs[row,col].set_yticklabels([])

            if message == 'summer':
                clevs = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]
            elif message == 'winter':
                clevs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
            cmap  = plt.cm.hot_r
            if col == 0:
                plot1 = axs[row,col].contourf(lons, lats, ctl_in, clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
            elif col == 1:
                plot1 = axs[row,col].contourf(lons, lats, sen_in, clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
            elif col == 2:
                plot1 = axs[row,col].contourf(lons, lats, obs_regrid, clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend='both')

            if i == 0:
                # add boxes
                reg_lats      = [  [-36.5,-28],   # east
                                   [-35,-27]    ] # west

                reg_lons      = [  [150,154],     # east
                                   [139,148]    ] # west

                axs[row,col].add_patch(Polygon([[reg_lons[0][0], reg_lats[0][0]], [reg_lons[0][1], reg_lats[0][0]],
                                            [reg_lons[0][1], reg_lats[0][1]], [reg_lons[0][0], reg_lats[0][1]]],
                                            closed=True,color=almost_black, fill=False,linewidth=0.8))

                axs[row,col].add_patch(Polygon([[reg_lons[1][0], reg_lats[1][0]], [reg_lons[1][1], reg_lats[1][0]],
                                            [reg_lons[1][1], reg_lats[1][1]], [reg_lons[1][0], reg_lats[1][1]]],
                                            closed=True,color=almost_black, fill=False,linewidth=0.8))

            # if message == 'summer':
            #     clevs = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
            #     clevs_diff = [-5.,-4.,-3.,-2.,-1.,-0.5, 0.5,1.,2.,3.,4.,5.]
            # elif message == 'winter':
            #     clevs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
            #     clevs_diff = [-5.,-4.,-3.,-2.,-1.,-0.5, 0.5,1.,2.,3.,4.,5.]
            #
            # if col == 0:
            #     cmap  = plt.cm.hot_r
            #     plot1 = axs[row,col].contourf(lons, lats, obs_regrid, clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
            # elif col == 1:
            #     cmap  = plt.cm.bwr
            #     # clevs = [-5.,-4.,-3.,-2.,-1.,-0.5, 0.5,1.,2.,3.,4.,5.]
            #     plot1 = axs[row,col].contourf(lons, lats, ctl_in-obs_regrid, clevs_diff, transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
            # elif col == 2:
            #     cmap  = plt.cm.bwr
            #     # clevs = [-5.,-4.,-3.,-2.,-1.,-0.5, 0.5,1.,2.,3.,4.,5.]
            #     plot1 = axs[row,col].contourf(lons, lats, sen_in-obs_regrid, clevs_diff, transform=ccrs.PlateCarree(), cmap=cmap, extend='both')

            axs[row,col].text(0.02, 0.15, texts[i], transform=axs[row,col].transAxes, fontsize=14, verticalalignment='top', bbox=props)

            if var_name == 'Tmax':
                if row == 2 and col == 0:
                    cbar = plt.colorbar(plot1, ax=axs[:,col], ticklocation="right", pad=0.06, orientation="horizontal", aspect=20, shrink=0.8) # cax=cax,
                    cbar.set_label('T$\mathregular{_{max}}$ ($\mathregular{^{o}}$C)', loc='center',size=12)# rotation=270,
                elif row == 2 and col > 0:
                    cbar = plt.colorbar(plot1, ax=axs[:,col], ticklocation="right", pad=0.06, orientation="horizontal", aspect=20, shrink=0.8) # cax=cax,
                    cbar.set_label('T$\mathregular{_{max}}$ ($\mathregular{^{o}}$C)', loc='center',size=12)# rotation=270,
            elif var_name == 'Tmean':
                if row == 2 and col == 0:
                    cbar = plt.colorbar(plot1, ax=axs[:,col], ticklocation="right", pad=0.06, orientation="horizontal", aspect=20, shrink=0.8) # cax=cax,
                    cbar.set_label('T$\mathregular{_{mean}}$ ($\mathregular{^{o}}$C)', loc='center',size=12)# rotation=270,
                elif row == 2 and col > 0:
                    cbar = plt.colorbar(plot1, ax=axs[:,col], ticklocation="right", pad=0.06, orientation="horizontal", aspect=20, shrink=0.8) # cax=cax,
                    cbar.set_label('T$\mathregular{_{mean}}$ ($\mathregular{^{o}}$C)', loc='center',size=12)# rotation=270,


        # cbar.ax.tick_params(labelsize=10,labelrotation=90)

        if message == 'summer':
            add_season = 'Summer'
        elif message == 'winter':
            add_season = 'Winter'

        axs[0,0].text(-0.25, 0.53, "2017 "+add_season, va='bottom', ha='center',
                rotation='vertical', rotation_mode='anchor',
                transform=axs[0,0].transAxes, fontsize=12)
        axs[1,0].text(-0.25, 0.50, "2018 "+add_season, va='bottom', ha='center',
                rotation='vertical', rotation_mode='anchor',
                transform=axs[1,0].transAxes, fontsize=12)
        axs[2,0].text(-0.25, 0.48, "2019 "+add_season, va='bottom', ha='center',
                rotation='vertical', rotation_mode='anchor',
                transform=axs[2,0].transAxes, fontsize=12)

        axs[0,0].set_title("Clim")
        axs[0,1].set_title("Dyn")
        axs[0,2].set_title("AGCD")
        # axs[0,0].set_title("AGCD")
        # axs[0,1].set_title("Clim-AGCD")
        # axs[0,2].set_title("Dyn-AGCD")
        # axs[0,3].set_title("Dyn bias - Clim bias")

        # Apply tight layout
        # plt.tight_layout()
        plt.savefig('./plots/Fig_spatial_map_evaluation_'+message + "_" + var_name+'_abs.png',dpi=500)

    return

def plot_time_series(file_name, land_ctl_path, land_sen_path, AWAP_T_file, nc_path, var_names,
                              time_s=None, time_e=None, lat_names="lat", lon_names="lon",loc_lat=None,
                              loc_lon=None, wrf_path=None,  message=None):

    # read lat and lon outs
    wrf            = Dataset(wrf_path,  mode='r')
    lons           = wrf.variables['XLONG'][0,:,:]
    lats           = wrf.variables['XLAT'][0,:,:]

    # ================== Start Plotting =================
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=[7,3.5],sharex=False,
                sharey=False, squeeze=False)

    plt.subplots_adjust(wspace=0.12, hspace=0.05)

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

    texts = ["(a)","(b)","(c)","(d)",
             "(e)","(f)","(g)","(h)",
             "(i)","(j)","(k)","(l)"]

    # WRF-CABLE
    for i, var_name in enumerate(var_names):
        print("plotting "+var_name)

        row = int(i/2)
        col = i % 2

        print('i',i,'row=',row,"col=",col)

        ctl_time_series, sen_time_series, obs_time_series = read_LIS_time_series(file_name, land_ctl_path, land_sen_path, AWAP_T_file, nc_path, var_name,
                         loc_lat, loc_lon, lat_names, lon_names, time_s, time_e, message=message)

        print('ctl_time_series',ctl_time_series)
        ctl_series = pd.DataFrame(ctl_time_series,columns=['vals'])
        sen_series = pd.DataFrame(sen_time_series,columns=['vals'])
        obs_series = pd.DataFrame(obs_time_series,columns=['vals'])

        print('ctl_series',ctl_series)
        axs[row,col].plot(ctl_series['vals'].rolling(window=30).mean(), label="Clim", c = "green",  lw=1.5, alpha=1)
        axs[row,col].plot(sen_series['vals'].rolling(window=30).mean(), label="Dyn",  c = "orange", lw=1.5, alpha=1)
        axs[row,col].plot(obs_series['vals'].rolling(window=30).mean(), label="AGCD", c = "red",    lw=1.5, alpha=1)
        axs[row,col].text(0.02, 0.09, texts[i], transform=axs[row,col].transAxes, fontsize=14, verticalalignment='top', bbox=props)

    axs[0,0].legend(fontsize=10, frameon=False, ncol=1)

    axs[0,0].set_title("T$\mathregular{_{mean}}$ ($\mathregular{^{o}}$C)")
    axs[0,1].set_title("T$\mathregular{_{max}}$ ($\mathregular{^{o}}$C)")

    cleaner_dates = ["Jan 2017", "Jul 2017", "Jan 2018", "Jul 2018", "Jan 2019", "Jul 2019", "Jan 2020"]
    xtickslocs    = [         0,        181,        365,        546,       730,         911,       1095]

    axs[0,0].set_xticks(xtickslocs)
    axs[0,0].set_xticklabels(cleaner_dates,rotation=20)
    axs[0,1].set_xticks(xtickslocs)
    axs[0,1].set_xticklabels(cleaner_dates,rotation=20)

    axs[0,0].set_ylim(0, 45)
    axs[0,0].set_yticks([0,5,10,15,20,25,30,35,40,45])
    axs[0,0].set_yticklabels(['0','5','10','15','20','25','30','35','40','45'])

    axs[0,1].set_ylim(0, 45)
    axs[0,1].set_yticks([0,5,10,15,20,25,30,35,40,45])
    axs[0,1].set_yticklabels(['0','5','10','15','20','25','30','35','40','45'])

    plt.tight_layout()
    plt.savefig('./plots/Fig_spatial_map_evaluation_time_series_'+message + "_" + var_name+'.png',dpi=500)

    return

def plot_time_series_burnt(file_name, land_ctl_path, land_sen_path, AWAP_T_file, nc_path, var_names,
                              time_s=None, time_e=None, lat_names="lat", lon_names="lon",loc_lat=None,
                              loc_lon=None, wrf_path=None,  message=None):

    # ================== Start Plotting =================
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=[9,6],sharex=False,
                sharey=False, squeeze=False)

    plt.subplots_adjust(wspace=0.12, hspace=0.05)

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

    texts = ["(a)","(b)","(c)","(d)",
             "(e)","(f)","(g)","(h)",
             "(i)","(j)","(k)","(l)"]

    # WRF-CABLE
    order = 0
    for i, var_name in enumerate(var_names):
        print("plotting "+var_name)
        ctl_time_series, sen_time_series, obs_time_series = read_LIS_time_series_burnt_new(
                         file_name, land_ctl_path, land_sen_path, AWAP_T_file, nc_path, var_name,
                         loc_lat, loc_lon, lat_names, lon_names, time_s, time_e, message=message)
        print('ctl_time_series',ctl_time_series)

        for j in np.arange(0,3):
            row = i
            col = j
            ctl_series = pd.DataFrame(ctl_time_series[j,:], columns=['vals'])
            sen_series = pd.DataFrame(sen_time_series[j,:], columns=['vals'])
            obs_series = pd.DataFrame(obs_time_series[j,:], columns=['vals'])
            print('ctl_series',ctl_series)

            axs[row,col].plot(ctl_series['vals'].rolling(window=30).mean(), label="Clim", c = "green",  lw=1.5, alpha=1) #
            axs[row,col].plot(sen_series['vals'].rolling(window=30).mean(), label="Dyn",  c = "orange", lw=1.5, alpha=1) # .rolling(window=30).mean()
            axs[row,col].plot(obs_series['vals'].rolling(window=30).mean(), label="AGCD", c = "red",    lw=1.5, alpha=1) # .rolling(window=30).mean()
            axs[row,col].text(0.02, 0.09, texts[order], transform=axs[row,col].transAxes, fontsize=14, verticalalignment='top', bbox=props)
            axs[row,col].set_ylim(0, 45)
            axs[row,col].set_yticks([0,5,10,15,20,25,30,35,40,45])
            axs[row,col].set_yticklabels(['0','5','10','15','20','25','30','35','40','45'])
            order = order+1

    axs[0,0].legend(fontsize=10, frameon=False, ncol=1)

    axs[0,0].set_ylabel("T$\mathregular{_{mean}}$ ($\mathregular{^{o}}$C)")
    axs[1,0].set_ylabel("T$\mathregular{_{max}}$ ($\mathregular{^{o}}$C)")

    axs[0,0].set_title("North")
    axs[0,1].set_title("Central")
    axs[0,2].set_title("South")

    cleaner_dates = ["Jan 2017", "Jul 2017", "Jan 2018", "Jul 2018", "Jan 2019", "Jul 2019", "Jan 2020"]
    xtickslocs    = [         0,        181,        365,        546,       730,         911,       1095]
    cleaner_dates_empty = ["", "", "", "", "", "", ""]

    axs[0,0].set_xticks(xtickslocs)
    axs[0,0].set_xticklabels(cleaner_dates_empty,rotation=20)
    axs[0,1].set_xticks(xtickslocs)
    axs[0,1].set_xticklabels(cleaner_dates_empty,rotation=20)
    axs[0,2].set_xticks(xtickslocs)
    axs[0,2].set_xticklabels(cleaner_dates_empty,rotation=20)

    axs[1,0].set_xticks(xtickslocs)
    axs[1,0].set_xticklabels(cleaner_dates,rotation=20)
    axs[1,1].set_xticks(xtickslocs)
    axs[1,1].set_xticklabels(cleaner_dates,rotation=20)
    axs[1,2].set_xticks(xtickslocs)
    axs[1,2].set_xticklabels(cleaner_dates,rotation=20)

    plt.tight_layout()
    plt.savefig('./plots/Fig_spatial_map_evaluation_time_series_'+message + "_" + var_name+'_burnt.png',dpi=500)

    return

def spatial_map_winter_summer(file_name, land_ctl_path, land_sen_path, AWAP_T_file, nc_path, var_names,
                              time_ss=None, time_es=None, lat_names="lat", lon_names="lon",loc_lat=None,
                              loc_lon=None, wrf_path=None,  message=None):

    '''
    plot a single spatial map
    '''

    # read lat and lon outs
    wrf            = Dataset(wrf_path,  mode='r')
    lons           = wrf.variables['XLONG'][0,:,:]
    lats           = wrf.variables['XLAT'][0,:,:]

    # WRF-CABLE
    for var_name in var_names:
        print("plotting "+var_name)

        # ================== Start Plotting =================
        fig, axs = plt.subplots(nrows=3, ncols=4, figsize=[14,10],sharex=False,
                    sharey=False, squeeze=True, subplot_kw={'projection': ccrs.PlateCarree()})

        plt.subplots_adjust(wspace=-0.15, hspace=0.105)

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
        texts = ["(a)","(b)","(c)","(d)",
                 "(e)","(f)","(g)","(h)",
                 "(i)","(j)","(k)","(l)"]

        for i in np.arange(12):

            row = int(i/4)
            col = i % 4

            print('i',i,'row=',row,"col=",col)

            if col == 0:
                ctl_in, sen_in, obs_regrid, ERA_regrid, cmap, clevs = \
                        read_LIS_obs_var(file_name, land_ctl_path, land_sen_path, AWAP_T_file, nc_path, var_name,
                                         loc_lat, loc_lon, lat_names, lon_names, time_ss[row], time_es[row], message=message)

            axs[row,col].coastlines(resolution="50m",linewidth=1)
            axs[row,col].set_extent([135,155,-39,-24])
            axs[row,col].add_feature(states, linewidth=.5, edgecolor="black")

            # Set the ticks on the x-axis and y-axis
            axs[row,col].tick_params(axis='x', direction='out')
            axs[row,col].tick_params(axis='y', direction='out')
            x_ticks = np.arange(135, 156, 5)
            y_ticks = np.arange(-40, -20, 5)
            axs[row,col].set_xticks(x_ticks)
            axs[row,col].set_yticks(y_ticks)
            axs[row, col].set_facecolor('lightgray')

            if row==2:
                axs[row,col].set_xticklabels(['135$\mathregular{^{o}}$E','140$\mathregular{^{o}}$E','145$\mathregular{^{o}}$E',
                                              '150$\mathregular{^{o}}$E','155$\mathregular{^{o}}$E'],rotation=25)
            else:
                axs[row,col].set_xticklabels([])

            if col==0:
                axs[row,col].set_yticklabels(['40$\mathregular{^{o}}$S','35$\mathregular{^{o}}$S',
                                              '30$\mathregular{^{o}}$S','25$\mathregular{^{o}}$S'])
            else:
                axs[row,col].set_yticklabels([])

            if col == 0:
                cmap  = plt.cm.bwr
                clevs = [-5.,-4.,-3.,-2.,-1.,-0.5, 0.5,1.,2.,3.,4.,5.]
                plot1 = axs[row,col].contourf(lons, lats, ctl_in - obs_regrid, clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
            elif col == 1:
                cmap  = plt.cm.bwr
                clevs = [-5.,-4.,-3.,-2.,-1.,-0.5, 0.5,1.,2.,3.,4.,5.]
                plot1 = axs[row,col].contourf(lons, lats, sen_in - obs_regrid, clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
            elif col == 2:
                cmap  = plt.cm.bwr
                clevs = [-5.,-4.,-3.,-2.,-1.,-0.5, 0.5,1.,2.,3.,4.,5.]
                plot1 = axs[row,col].contourf(lons, lats, ERA_regrid - obs_regrid, clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
            elif col == 3:
                cmap  = plt.cm.bwr
                clevs = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.]
                plot1 = axs[row,col].contourf(lons, lats, abs(sen_in - obs_regrid)-abs(ctl_in - obs_regrid), clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
                #

            axs[row,col].text(0.02, 0.15, texts[i], transform=axs[row,col].transAxes, fontsize=14, verticalalignment='top', bbox=props)

            if row == 2 and col < 3:
                cbar = plt.colorbar(plot1, ax=axs[:,col], ticklocation="right", pad=0.06, orientation="horizontal", aspect=20, shrink=0.8) # cax=cax,
                cbar.set_label('ΔT$\mathregular{_{max}}$ ($\mathregular{^{o}}$C)', loc='center',size=12)# rotation=270,
                cbar.set_ticks([-5.,-4.,-3.,-2.,-1.,-0.5, 0.5,1.,2.,3.,4.,5.])
                cbar.set_ticklabels(['-5.0','-4.0','-3.0','-2.0','-1.0','-0.5','0.5','1.0','2.0','3.0','4.0','5.0'],rotation=90) # cax=cax,

            elif row == 2 and col == 3:
                cbar = plt.colorbar(plot1, ax=axs[:,3], ticklocation="right", pad=0.06, orientation="horizontal",
                        aspect=20, shrink=0.8) # cax=cax,
                cbar.set_label('ΔT$\mathregular{_{max}}$ ($\mathregular{^{o}}$C)', loc='center',size=12)# rotation=270,
                cbar.set_ticks([-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.])
                cbar.set_ticklabels(['','-0.9','','-0.7','','-0.5','','-0.3','','-0.1','0.1','','0.3','','0.5','','0.7','','0.9',''],rotation=90) # cax=cax,

        cbar.ax.tick_params(labelsize=10,labelrotation=90)

        axs[0,0].text(-0.25, 0.53, "2017-18 Summer", va='bottom', ha='center',
                rotation='vertical', rotation_mode='anchor',
                transform=axs[0,0].transAxes, fontsize=12)
        axs[1,0].text(-0.25, 0.50, "2018-19 Summer", va='bottom', ha='center',
                rotation='vertical', rotation_mode='anchor',
                transform=axs[1,0].transAxes, fontsize=12)
        axs[2,0].text(-0.25, 0.48, "2019-20 Summer", va='bottom', ha='center',
                rotation='vertical', rotation_mode='anchor',
                transform=axs[2,0].transAxes, fontsize=12)

        axs[0,0].set_title("Clim - AGCD")
        axs[0,1].set_title("Dyn - AGCD")
        axs[0,2].set_title("ERA5 - AGCD")
        axs[0,3].set_title("Dyn bias - Clim bias")

        # Apply tight layout
        # plt.tight_layout()
        plt.savefig('./plots/Fig_spatial_map_evaluation_'+message + "_" + var_name+'_new_colorbar.png',dpi=500)

    return

def spatial_map_correlation(file_name, land_ctl_path, land_sen_path, AWAP_T_file, nc_path, time_ss=None, time_es=None, lat_names="lat", lon_names="lon",loc_lat=None, loc_lon=None,  message=None):

    ctl_all, sen_all, ERA_all, obs_all, lat_out, lon_out = read_three_year_season(file_name, land_ctl_path, land_sen_path, AWAP_T_file, nc_path, loc_lat, loc_lon, lat_names, lon_names, time_ss, time_es, message=message)

    ctl_r, sen_r, ERA_r, ctl_RMSE, sen_RMSE, ERA_RMSE = \
                    calc_temporal_correl(ctl_all, sen_all, ERA_all, obs_all)

    # ================== Start Plotting =================
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=[12,8],sharex=False,
                sharey=False, squeeze=True, subplot_kw={'projection': ccrs.PlateCarree()})

    plt.subplots_adjust(wspace=0.1, hspace=0.2)

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
    texts = ["(a)","(b)","(c)","(d)",
             "(e)","(f)","(g)","(h)",
             "(i)","(j)","(k)","(l)"]

    for i in np.arange(6):

        row = int(i/3)
        col = i % 3

        axs[row,col].coastlines(resolution="50m",linewidth=1)
        axs[row,col].set_extent([135,155,-39,-24])
        axs[row,col].add_feature(states, linewidth=.5, edgecolor="black")

        # Set the ticks on the x-axis and y-axis
        axs[row,col].tick_params(axis='x', direction='out')
        axs[row,col].tick_params(axis='y', direction='out')
        x_ticks = np.arange(135, 156, 5)
        y_ticks = np.arange(-40, -20, 5)
        axs[row,col].set_xticks(x_ticks)
        axs[row,col].set_yticks(y_ticks)
        axs[row, col].set_facecolor('lightgray')

        # if row==1:
        axs[row,col].set_xticklabels(['135$\mathregular{^{o}}$E','140$\mathregular{^{o}}$E','145$\mathregular{^{o}}$E',
                                        '150$\mathregular{^{o}}$E','155$\mathregular{^{o}}$E'],rotation=25)
        # else:
        #     axs[row,col].set_xticklabels([])

        if col==0:
            axs[row,col].set_yticklabels(['40$\mathregular{^{o}}$S','35$\mathregular{^{o}}$S',
                                            '30$\mathregular{^{o}}$S','25$\mathregular{^{o}}$S'])
        else:
            axs[row,col].set_yticklabels([])

        #BrBG #bwr

        if row == 0:
            # clevs = [-1.0,-.9,-.8,-.7,-.6,-.5,-.4,-.3,-.2,-.1,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.]
            # clevs = [.9,.91,0.92,0.93,0.94,.95,.96,0.97,0.98,.99,1.]
            if col == 0:
                clevs = [.5,.55,0.6,0.65,.7,0.75,.8,0.85, 0.9,0.95, 1.]
                cmap  = plt.cm.viridis_r
                plot1 = axs[0,0].contourf(lon_out, lat_out, ctl_r, clevs,transform=ccrs.PlateCarree(), cmap=cmap, extend='both') #
            if col == 1:
                clevs = [.5,.55,0.6,0.65,.7,0.75,.8,0.85, 0.9,0.95, 1.]
                cmap  = plt.cm.viridis_r
                plot1 = axs[0,1].contourf(lon_out, lat_out, sen_r, clevs,transform=ccrs.PlateCarree(), cmap=cmap, extend='both') # clevs,
            if col == 2:
                clevs = [-0.05,-0.04,-0.03,-0.02,-0.01,0.01,0.02,0.03,0.04,0.05]
                cmap  = plt.cm.BrBG
                plot1 = axs[0,2].contourf(lon_out, lat_out, sen_r-ctl_r, clevs,transform=ccrs.PlateCarree(), cmap=cmap, extend='both') #  clevs,
                # plot1 = axs[0,2].contourf(lon_out, lat_out, ERA_r, clevs,transform=ccrs.PlateCarree(), cmap=cmap, extend='both') #  clevs,
            cbar = plt.colorbar(plot1, ax=axs[0,col], ticklocation="right", pad=0.15, orientation="horizontal", aspect=20, shrink=0.8) # cax=cax,
            cbar.set_label('Correlation', loc='center',size=12, rotation=30)
            # cbar.set_ticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.])
            # cbar.set_ticklabels(['0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'],rotation=90)
        if row == 1:
            # clevs = [-3.0,-2.5,-2.,-1.5,-1.,-.5,.5,1.,1.5,2.,2.5,3.]
            # clevs = [0,.5,1.,1.5,2.,2.5,3.,3.5,4.,4.5,5.,5.5,6.,6.5,7.]
            if col == 0:
                cmap  = plt.cm.viridis_r
                clevs = [0,.5,1.,1.5,2.,2.5,3.,3.5,4.,4.5,5.,5.5,6.,6.5,7.]
                plot1 = axs[1,0].contourf(lon_out, lat_out, ctl_RMSE, clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend='both') #clevs,
            if col == 1:
                cmap  = plt.cm.viridis_r
                clevs = [0,.5,1.,1.5,2.,2.5,3.,3.5,4.,4.5,5.,5.5,6.,6.5,7.]
                plot1 = axs[1,1].contourf(lon_out, lat_out, sen_RMSE, clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend='both') # clevs,
            if col == 2:
                cmap  = plt.cm.BrBG
                clevs = [-0.5,-0.4,-0.3,-0.2,-0.1, 0.1,0.2,0.3,0.4,0.5]
                plot1 = axs[1,2].contourf(lon_out, lat_out, sen_RMSE - ctl_RMSE, clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend='both') #clevs,
                # plot1 = axs[1,2].contourf(lon_out, lat_out, ERA_RMSE, clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend='both') #clevs,
            cbar = plt.colorbar(plot1, ax=axs[1,col], ticklocation="right", pad=0.15, orientation="horizontal", aspect=20, shrink=0.8) # cax=cax,
            cbar.set_label('RMSE', loc='center',size=12, rotation=30)# rotation=270,
            # cbar.set_ticks([-3.0,-2.5,-2.,-1.5,-1.,-.5,0.])
            # cbar.set_ticklabels(['-3.0','-2.5','-2.0','-1.5','-1.0','-0.5','0.0'],rotation=90) # cax=cax,

        axs[row,col].text(0.02, 0.15, texts[i], transform=axs[row,col].transAxes, fontsize=14, verticalalignment='top', bbox=props)


    axs[0,0].set_title("Clim")
    axs[0,1].set_title("Dyn")
    axs[0,2].set_title("Dyn-Clim")
    # axs[0,2].set_title("ERA5")

    # Apply tight layout
    # plt.tight_layout()
    plt.savefig("./plots/Fig_spatial_map_evaluation_metrics_summer_Tmax.png",dpi=500)

    return


if __name__ == "__main__":

    # ======================= Option =======================
    region = "SE Aus" #"SE Aus" #"CORDEX" #"SE Aus"

    if region == "Aus":
        loc_lat    = [-44,-10]
        loc_lon    = [112,154]
    elif region == "SE Aus":
        loc_lat    = [-40,-23]
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

        case_name      = "ALB-CTL_new" #"bl_pbl2_mp4_sf_sfclay2" #
        case_ctl       = "drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2"
        case_sen       = "drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2_obs_LAI_ALB"

        wrf_path       = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_ctl+"/WRF_output/wrfout_d01_2019-12-01_01:00:00"
        land_sen_path  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_sen+"/LIS_output/"
        land_ctl_path  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_ctl+"/LIS_output/"
        atmo_sen_path  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_sen+"/WRF_output/"
        atmo_ctl_path  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_ctl+"/WRF_output/"
        AWAP_T_file    = '/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/agcd_v1-0-1_tmax_mean_r005_daily_2017-2020.nc'     # air temperature
        nc_path        = '/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/'
        file_name      = "LIS.CABLE.201701-202002.nc"

        var_names  = [ "Tmean","Tmax" ] #

        # # Winter
        # time_ss    = [  datetime(2017,6,1,0,0,0,0),
        #                 datetime(2018,6,1,0,0,0,0),
        #                 datetime(2019,6,1,0,0,0,0), ]
        # time_es    = [  datetime(2017,9,1,0,0,0,0),
        #                 datetime(2018,9,1,0,0,0,0),
        #                 datetime(2019,9,1,0,0,0,0), ]

        # Summer
        time_ss    = [  datetime(2017,12,1,0,0,0,0),
                        datetime(2018,12,1,0,0,0,0),
                        datetime(2019,12,1,0,0,0,0), ]
        time_es    = [  datetime(2018,3,1,0,0,0,0),
                        datetime(2019,3,1,0,0,0,0),
                        datetime(2020,3,1,0,0,0,0), ]

        # # three months
        # time_ss    = [  datetime(2019,12,1,0,0,0,0),
        #                 datetime(2020,1,1,0,0,0,0),
        #                 datetime(2020,2,1,0,0,0,0), ]
        # time_es    = [  datetime(2020,1,1,0,0,0,0),
        #                 datetime(2020,2,1,0,0,0,0),
        #                 datetime(2020,3,1,0,0,0,0), ]

        message    = "summer"
        # spatial_map_winter_summer(file_name, land_ctl_path, land_sen_path, AWAP_T_file, nc_path, var_names, time_ss=time_ss, time_es=time_es, lat_names="lat", lon_names="lon",loc_lat=loc_lat, loc_lon=loc_lon, wrf_path=wrf_path, message=message)
        # spatial_map_winter_summer_abs(file_name, land_ctl_path, land_sen_path, AWAP_T_file, nc_path, var_names, time_ss=time_ss, time_es=time_es, lat_names="lat", lon_names="lon",loc_lat=loc_lat, loc_lon=loc_lon, wrf_path=wrf_path, message=message)
        # # # spatial_map_correlation(file_name, land_ctl_path, land_sen_path, AWAP_T_file, nc_path, time_ss=time_ss, time_es=time_es, lat_names="lat", lon_names="lon", loc_lat=loc_lat, loc_lon=loc_lon, message=message)

        # =============================== Time series ==============================
        var_names  = [ "Tmean","Tmax" ] #
        message    = 'burnt'
        time_s     = datetime(2017,1,1,0,0,0,0)
        # time_e     = datetime(2017,1,3,0,0,0,0)

        time_e     = datetime(2020,3,1,0,0,0,0)
        # plot_time_series_burnt(file_name, land_ctl_path, land_sen_path, AWAP_T_file, nc_path, var_names, time_s=time_s, time_e=time_e,
        #                  lat_names="lat", lon_names="lon", loc_lat=loc_lat, loc_lon=loc_lon, wrf_path=wrf_path,  message=message)

        # message    = "east"
        # if message == 'east':
        #     # east area
        #     loc_lat    = [-36.5,-28]
        #     loc_lon    = [150,154]
        # elif message == 'west':
        #     # west area
        #     loc_lat    = [-35,-27]
        #     loc_lon    = [139,148]
        # plot_time_series(file_name, land_ctl_path, land_sen_path, AWAP_T_file, nc_path, var_names, time_s=time_s, time_e=time_e,
        #                  lat_names="lat", lon_names="lon", loc_lat=loc_lat, loc_lon=loc_lon, wrf_path=wrf_path,  message=message)

        message    = "west"
        if message == 'east':
            # east area
            loc_lat    = [-36.5,-28]
            loc_lon    = [150,154]
        elif message == 'west':
            # west area
            loc_lat    = [-35,-27]
            loc_lon    = [139,148]
        plot_time_series(file_name, land_ctl_path, land_sen_path, AWAP_T_file, nc_path, var_names, time_s=time_s, time_e=time_e,
                         lat_names="lat", lon_names="lon", loc_lat=loc_lat, loc_lon=loc_lon, wrf_path=wrf_path,  message=message)
