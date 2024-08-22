#!/usr/bin/python

__author__ = "Mengyuan Mu"
__email__  = "mu.mengyuan815@gmail.com" 

import sys
import cartopy
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


def plot_spatial_LIS_wind( file_path_ctl, file_path_sen, time_s, time_e, lat_name, lon_name,
                           loc_lat=None, loc_lon=None, seconds=None, message=None):

    # Extract the pressure, geopotential height, and wind variables
    ua_file  = Dataset(file_path_sen + 'ua/wrfout_201912-202002.nc', mode='r')
    ua_tmp   = ua_file.variables['ua'][:,0,:,:]

    lats     = ua_file.variables['lat'][:,:]
    lons     = ua_file.variables['lon'][:,:]
    time_tmp = nc.num2date(ua_file.variables['time'][:],ua_file.variables['time'].units,
                 only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    ua_file.close()

    va_file  = Dataset(file_path_ctl + 'va/wrfout_201912-202002.nc', mode='r')
    va_tmp   = va_file.variables['va'][:,0,:,:]
    va_file.close()
    
    th_file  = Dataset(file_path_ctl + 'th/wrfout_201912-202002.nc', mode='r')
    th_tmp   = th_file.variables['th'][:,0,:,:]
    th_file.close()

    time     = UTC_to_AEST(time_tmp) - datetime(2000,1,1,0,0,0)
    print('time',time)
    print('time_s',time_s)
    print('time_e',time_e)

    ua       = spatial_var_mean(time, ua_tmp, time_s, time_e, seconds)
    va       = spatial_var_mean(time, va_tmp, time_s, time_e, seconds)
    th       = spatial_var_mean(time, th_tmp, time_s, time_e, seconds)
    th       = th - 273.15

    # ================== Start Plotting =================
    fig = plt.figure(figsize=(6,5))
    ax = plt.axes(projection=ccrs.PlateCarree())

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
    # choose colormap

    # =============== CHANGE HERE ===============
    # clevs = np.linspace( 0.,10., num=11)
    clevs = np.linspace( 10.,40., num=31)
    cmap  = plt.cm.coolwarm # BrBG

    # start plotting
    if loc_lat == None:
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
        gl.xlocator     = mticker.FixedLocator([135,140,145,150,155])
        gl.ylocator     = mticker.FixedLocator([-40,-35,-30,-25])
    else:
        gl.xlocator = mticker.FixedLocator(loc_lon)
        gl.ylocator = mticker.FixedLocator(loc_lat)

    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size':10, 'color':'black'}
    gl.ylabel_style = {'size':10, 'color':'black'}

    plot1 = plt.contourf(lons, lats, th, clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend='both') #
    cb    = plt.colorbar(plot1, ax=ax, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)

    scale =1
    q = plt.quiver(lons[::15,::15], lats[::15,::15], ua[::15, ::15],
             va[::15, ::15],  transform=ccrs.PlateCarree()) #scale=5,scale=scale,

    # ax.quiverkey(q,X=0.95, Y=0.95, labelpos='E', color="black") #U=5, label='5 m/s', 
    cb.ax.tick_params(labelsize=10, labelrotation=45)
    plt.title('Wind & Temperature', size=16)

    plt.savefig('./plots/spatial_map_'+message+'_wind_temperature.png',dpi=300)

if __name__ == "__main__":

    #######################################################
    # Decks to run:
    #    plot_spital_map
    #######################################################


    # ======================= Option =======================
    # 2017-2019 drought polygon shapefile
    shape_path = "/g/data/w97/ad9701/drought_2017to2020/drought_focusArea/smooth_polygon_drought_focusArea.shp"

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

        file_path_sen  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_sen+"/WRF_output/"
        file_path_ctl  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_ctl+"/WRF_output/"

        if 1:
            '''
            Difference plot yearly
            '''
            
            seconds    = [6.*60.*60.,18.*60.*60.] # for daytime
            # seconds    = [18.*60.*60., 6.*60.*60.] # for nighttime
            period     = "201920_summer_day_sen"
            time_s     = datetime(2019,12,1,0,0,0,0)
            time_e     = datetime(2020,3,1,0,0,0,0)
            message    = period

            plot_spatial_LIS_wind(file_path_ctl, file_path_sen, time_s=time_s, time_e=time_e, lat_name="lat", lon_name="lon",
                                  loc_lat=loc_lat, loc_lon=loc_lon, seconds=seconds, message=message)