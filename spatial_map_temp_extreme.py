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

def spatial_map_T_coldest_warmest(land_ctl_path, land_sen_path, var_name, time_s=None,
                                time_e=None, lat_names="lat", lon_names="lon",loc_lat=None,
                                loc_lon=None, wrf_path=None, shape_path=None, message=None,
                                delta=False):

    '''
    plot a single spatial map
    '''

    # WRF-CABLE
    print("plotting "+var_name)

    # read lat and lon
    wrf           = Dataset(wrf_path,  mode='r')
    lons          = wrf.variables['XLONG'][0,:,:]
    lats          = wrf.variables['XLAT'][0,:,:]
    nlat          = len(lons[:,0])
    nlon          = len(lons[0,:])
    wrf.close()

    land_ctl_files = [land_ctl_path+var_name+'/LIS.CABLE.201701-202002.nc']
    land_sen_files = [land_sen_path+var_name+'/LIS.CABLE.201701-202002.nc']
    time, Ctl_tmp  = read_var_multi_file(land_ctl_files, var_name, loc_lat, loc_lon, lat_names, lon_names)
    time, Sen_tmp  = read_var_multi_file(land_sen_files, var_name, loc_lat, loc_lon, lat_names, lon_names)
    Ctl_tmp        = Ctl_tmp - 273.15
    Sen_tmp        = Sen_tmp - 273.15

    # daily max
    ctl_max        = time_clip_to_day_max(time,Ctl_tmp,time_s,time_e)
    sen_max        = time_clip_to_day_max(time,Sen_tmp,time_s,time_e)

    # daily min
    ctl_min        = time_clip_to_day_min(time,Ctl_tmp,time_s,time_e)
    sen_min        = time_clip_to_day_min(time,Sen_tmp,time_s,time_e)

    # average of daily min
    ctl_TDR        = ctl_max - ctl_min
    sen_TDR        = sen_max - sen_min

    # sort out data
    ctl_max_sort       = np.sort(ctl_max,axis=0)
    ctl_min_sort       = np.sort(ctl_min,axis=0)
    sen_max_sort       = np.sort(sen_max,axis=0)
    sen_min_sort       = np.sort(sen_min,axis=0)
    print('np.shape(ctl_max_sort)',np.shape(ctl_max_sort))

    # get the index after sort
    ctl_max_sort_index = np.argsort(ctl_max,axis=0)
    ctl_min_sort_index = np.argsort(ctl_min,axis=0)
    sen_max_sort_index = np.argsort(sen_max,axis=0)
    sen_min_sort_index = np.argsort(sen_min,axis=0)
    print('np.shape(ctl_max_sort_index)',np.shape(ctl_max_sort_index))

    # sorted out TDR by tmax and tmin rankings
    ctl_TDR_max_sort   = np.take_along_axis(ctl_TDR, ctl_max_sort_index, axis=0)
    ctl_TDR_min_sort   = np.take_along_axis(ctl_TDR, ctl_min_sort_index, axis=0)
    sen_TDR_max_sort   = np.take_along_axis(sen_TDR, sen_max_sort_index, axis=0)
    sen_TDR_min_sort   = np.take_along_axis(sen_TDR, sen_min_sort_index, axis=0)
    print('np.shape(ctl_TDR_max_sort)',np.shape(ctl_TDR_max_sort))

    # average 5th, 95th percentile
    # the 18 days (i.e. 5th 365 days) of coldest Tmax
    ctl_max_coldest     = np.nanmean(ctl_max_sort[0:18,:,:],axis=0)
    ctl_TDR_max_coldest = np.nanmean(ctl_TDR_max_sort[0:18,:,:],axis=0)
    sen_max_coldest     = np.nanmean(sen_max_sort[0:18,:,:],axis=0)
    sen_TDR_max_coldest = np.nanmean(sen_TDR_max_sort[0:18,:,:],axis=0)
    print('np.shape(ctl_max_coldest)',np.shape(ctl_max_coldest))

    # the 18 coldest days (i.e. 5th 365 days) of coldest Tmin
    ctl_min_coldest     = np.nanmean(ctl_min_sort[0:18,:,:],axis=0)
    ctl_TDR_min_coldest = np.nanmean(ctl_TDR_min_sort[0:18,:,:],axis=0)
    sen_min_coldest     = np.nanmean(sen_min_sort[0:18,:,:],axis=0)
    sen_TDR_min_coldest = np.nanmean(sen_TDR_min_sort[0:18,:,:],axis=0)

    # the 18 days (i.e. 95th 365 days) of warmest Tmax
    ctl_max_warmest     = np.nanmean(ctl_max_sort[-18:,:,:],axis=0)
    ctl_TDR_max_warmest = np.nanmean(ctl_TDR_max_sort[-18:,:,:],axis=0)
    sen_max_warmest     = np.nanmean(sen_max_sort[-18:,:,:],axis=0)
    sen_TDR_max_warmest = np.nanmean(sen_TDR_max_sort[-18:,:,:],axis=0)

    # the 18 days (i.e. 95th 365 days) of warmest Tmin
    ctl_min_warmest     = np.nanmean(ctl_min_sort[-18:,:,:],axis=0)
    ctl_TDR_min_warmest = np.nanmean(ctl_TDR_min_sort[-18:,:,:],axis=0)
    sen_min_warmest     = np.nanmean(sen_min_sort[-18:,:,:],axis=0)
    sen_TDR_min_warmest = np.nanmean(sen_TDR_min_sort[-18:,:,:],axis=0)

    if delta:
        delta_max_coldest     = sen_max_coldest - ctl_max_coldest
        delta_min_coldest     = sen_min_coldest - ctl_min_coldest
        delta_max_warmest     = sen_max_warmest - ctl_max_warmest
        delta_min_warmest     = sen_min_warmest - ctl_min_warmest

        delta_TDR_max_coldest = sen_TDR_max_coldest - ctl_TDR_max_coldest
        delta_TDR_min_coldest = sen_TDR_min_coldest - ctl_TDR_min_coldest
        delta_TDR_max_warmest = sen_TDR_max_warmest - ctl_TDR_max_warmest
        delta_TDR_min_warmest = sen_TDR_min_warmest - ctl_TDR_min_warmest
    else:
        # set season codes
        seasons              = np.zeros(np.shape(ctl_max))
        seasons[0:92,:,:]    = 1 # MAM
        seasons[92:184,:,:]  = 2 # JJA
        seasons[184:275,:,:] = 3 # SON
        seasons[275:365,:,:] = 4 # DJF

        # sorted out seasons by tmax and tmin rankings
        seasons_ctl_max_sort = np.take_along_axis(seasons, ctl_max_sort_index, axis=0)
        seasons_ctl_min_sort = np.take_along_axis(seasons, ctl_min_sort_index, axis=0)
        seasons_sen_max_sort = np.take_along_axis(seasons, sen_max_sort_index, axis=0)
        seasons_sen_min_sort = np.take_along_axis(seasons, sen_min_sort_index, axis=0)

        # calculate days in different seasons
        ctl_seasons_days_max_warmest = np.zeros((4,nlat,nlon))
        ctl_seasons_days_max_coldest = np.zeros((4,nlat,nlon))
        ctl_seasons_days_min_warmest = np.zeros((4,nlat,nlon))
        ctl_seasons_days_min_coldest = np.zeros((4,nlat,nlon))

        sen_seasons_days_max_warmest = np.zeros((4,nlat,nlon))
        sen_seasons_days_max_coldest = np.zeros((4,nlat,nlon))
        sen_seasons_days_min_warmest = np.zeros((4,nlat,nlon))
        sen_seasons_days_min_coldest = np.zeros((4,nlat,nlon))

        for i in np.arange(4):
            ctl_seasons_days_max_coldest[i,:,:] = np.count_nonzero(
                                                np.where(seasons_ctl_max_sort[0:18,:,:]==(i+1),1,0),
                                                axis=0)
            ctl_seasons_days_min_coldest[i,:,:] = np.count_nonzero(
                                                np.where(seasons_ctl_min_sort[0:18,:,:]==(i+1),1,0),
                                                axis=0)
            ctl_seasons_days_max_warmest[i,:,:] = np.count_nonzero(
                                                np.where(seasons_ctl_max_sort[-18:,:,:]==(i+1),1,0),
                                                axis=0)
            ctl_seasons_days_min_warmest[i,:,:] = np.count_nonzero(
                                                np.where(seasons_ctl_min_sort[-18:,:,:]==(i+1),1,0),
                                                axis=0)
            sen_seasons_days_max_coldest[i,:,:] = np.count_nonzero(
                                                np.where(seasons_sen_max_sort[0:18,:,:]==(i+1),1,0),
                                                axis=0)
            sen_seasons_days_min_coldest[i,:,:] = np.count_nonzero(
                                                np.where(seasons_sen_min_sort[0:18,:,:]==(i+1),1,0),
                                                axis=0)
            sen_seasons_days_max_warmest[i,:,:] = np.count_nonzero(
                                                np.where(seasons_sen_max_sort[-18:,:,:]==(i+1),1,0),
                                                axis=0)
            sen_seasons_days_min_warmest[i,:,:] = np.count_nonzero(
                                                np.where(seasons_sen_min_sort[-18:,:,:]==(i+1),1,0),
                                                axis=0)

    # ================== Start Plotting =================
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=[15,8],sharex=True,
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
    # choose colormap
    # cmap  = plt.cm.BrBG
    cmap  = plt.cm.seismic

    for i in np.arange(2):
        for j in np.arange(4):
            axs[i,j].coastlines(resolution="50m",linewidth=1)
            axs[i,j].set_extent([135,155,-39,-23])
            axs[i,j].add_feature(states, linewidth=.5, edgecolor="black")

            # Add gridlines
            gl = axs[i,j].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color=almost_black, linestyle='--')
            gl.xlabels_top  = False
            gl.ylabels_right= False
            gl.xlines       = False
            gl.ylines       = False
            # gl.xlocator     = mticker.FixedLocator(np.arange(125,160,1))
            # gl.ylocator     = mticker.FixedLocator(np.arange(-40,-20,1))
            gl.xlocator     = mticker.FixedLocator([130,135,140,145,150,155,160])
            gl.ylocator     = mticker.FixedLocator([-40,-35,-30,-25,-20])
            gl.xformatter   = LONGITUDE_FORMATTER
            gl.yformatter   = LATITUDE_FORMATTER
            gl.xlabel_style = {'size':12, 'color':almost_black}#,'rotation': 90}
            gl.ylabel_style = {'size':12, 'color':almost_black}

            gl.xlabels_bottom = True
            gl.ylabels_left   = True

    clevs_T_coldest= [-4,-2,0,2,4,6,8,10,12,14,16,18,20]
    clevs_T_warmest= [24,26,28,30,32,34,36,38,40,42,44,46,48,50]
    clevs_T_diff   = [-1.2,-1.1,-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.,1.1,1.2]
    clevs_TDR      = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    clevs_TDR_diff = [-2,-1.8,-1.6,-1.4,-1.2,-1,-0.8,-0.6,-0.4,-0.2,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2]
    clevs_days     = [-50,-45,-40,-35,-30,-25,-20,-15,-10,-5,-4,-3,-2,-1,1,2,3,4,5,10,15,20,25,30,35,40,45,50]

    if delta:
        # Tmax/Tmin
        plot1 = axs[0,0].contourf(lons, lats, delta_max_warmest, clevs_T_diff, transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
        cbar  = plt.colorbar(plot1, ax=axs[0,0], ticklocation="right", pad=0.1, orientation="horizontal",aspect=40, shrink=1) # cax=cax,
        cbar.ax.tick_params(labelsize=10,labelrotation=45)
        axs[0,0].set_title('delta_max_warmest', size=12)

        plot2 = axs[0,1].contourf(lons, lats, delta_min_warmest, clevs_T_diff, transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
        cbar  = plt.colorbar(plot2, ax=axs[0,1], ticklocation="right", pad=0.1, orientation="horizontal",aspect=40, shrink=1) # cax=cax,
        cbar.ax.tick_params(labelsize=10,labelrotation=45)
        axs[0,1].set_title('delta_min_warmest', size=12)

        plot3 = axs[0,2].contourf(lons, lats, delta_max_coldest, clevs_T_diff, transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
        cbar  = plt.colorbar(plot3, ax=axs[0,2], ticklocation="right", pad=0.1, orientation="horizontal",aspect=40, shrink=1) # cax=cax,
        cbar.ax.tick_params(labelsize=10,labelrotation=45)
        axs[0,2].set_title('delta_max_coldest', size=12)

        plot4 = axs[0,3].contourf(lons, lats, delta_min_coldest, clevs_T_diff, transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
        cbar  = plt.colorbar(plot4, ax=axs[0,3], ticklocation="right", pad=0.1, orientation="horizontal",aspect=40, shrink=1) # cax=cax,
        cbar.ax.tick_params(labelsize=10,labelrotation=45)
        axs[0,3].set_title('delta_min_coldest', size=12)

        # TDR
        plot5 = axs[1,0].contourf(lons, lats, delta_TDR_max_warmest, clevs_TDR_diff, transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
        cbar  = plt.colorbar(plot5, ax=axs[1,0], ticklocation="right", pad=0.1, orientation="horizontal",
                aspect=40, shrink=1) # cax=cax,
        cbar.ax.tick_params(labelsize=10,labelrotation=45)
        axs[1,0].set_title('delta_TDR_max_warmest', size=12)

        plot6 = axs[1,1].contourf(lons, lats, delta_TDR_min_warmest, clevs_TDR_diff, transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
        cbar  = plt.colorbar(plot6, ax=axs[1,1], ticklocation="right", pad=0.1, orientation="horizontal",
                aspect=40, shrink=1) # cax=cax,
        cbar.ax.tick_params(labelsize=10,labelrotation=45)
        axs[1,1].set_title('delta_TDR_min_warmest', size=12)

        plot7 = axs[1,2].contourf(lons, lats, delta_TDR_max_coldest, clevs_TDR_diff, transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
        cbar  = plt.colorbar(plot7, ax=axs[1,2], ticklocation="right", pad=0.1, orientation="horizontal",
                aspect=40, shrink=1) # cax=cax,
        cbar.ax.tick_params(labelsize=10,labelrotation=45)
        axs[1,2].set_title('delta_TDR_max_coldest', size=12)

        plot8 = axs[1,3].contourf(lons, lats, delta_TDR_min_coldest, clevs_TDR_diff, transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
        cbar  = plt.colorbar(plot8, ax=axs[1,3], ticklocation="right", pad=0.1, orientation="horizontal",
                aspect=40, shrink=1) # cax=cax,
        cbar.ax.tick_params(labelsize=10,labelrotation=45)
        axs[1,3].set_title('delta_TDR_min_coldest', size=12)

        plt.savefig('./plots/spatial_map_coldest_warmest_Tmax_Tmin_'+message+'.png',dpi=300)
    else:
        plot_type   = 'ctl_Tmin_coldest'
        var_type    = 'ctl_min_coldest'
        clevs_T     = clevs_T_coldest
        var_extreme = ctl_min_coldest
        var_TDR     = ctl_TDR_min_coldest
        var_season  = ctl_seasons_days_min_coldest

        # Tmax/Tmin
        plot1 = axs[0,0].contourf(lons, lats, var_extreme, clevs_T, transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
        cbar  = plt.colorbar(plot1, ax=axs[0,0], ticklocation="right", pad=0.1, orientation="horizontal",aspect=40, shrink=1) # cax=cax,
        cbar.ax.tick_params(labelsize=10,labelrotation=45)
        axs[0,0].set_title(var_type, size=12)

        # TDR
        plot2 = axs[1,0].contourf(lons, lats, var_TDR, clevs_TDR, transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
        cbar  = plt.colorbar(plot2, ax=axs[1,0], ticklocation="right", pad=0.1, orientation="horizontal",
                aspect=40, shrink=1) # cax=cax,
        cbar.ax.tick_params(labelsize=10,labelrotation=45)
        axs[1,0].set_title('TDR', size=12)

        # MAM
        plot3 = axs[0,1].contourf(lons, lats, var_season[0,:,:], clevs_days, transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
        cbar  = plt.colorbar(plot3, ax=axs[0,1], ticklocation="right", pad=0.1, orientation="horizontal",aspect=40, shrink=1) # cax=cax,
        cbar.ax.tick_params(labelsize=10,labelrotation=45)
        axs[0,1].set_title('MAM', size=12)

        # JJA
        plot4 = axs[0,2].contourf(lons, lats, var_season[1,:,:], clevs_days, transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
        cbar  = plt.colorbar(plot4, ax=axs[0,2], ticklocation="right", pad=0.1, orientation="horizontal",aspect=40, shrink=1) # cax=cax,
        cbar.ax.tick_params(labelsize=10,labelrotation=45)
        axs[0,2].set_title('JJA', size=12)

        # SON
        plot5 = axs[1,1].contourf(lons, lats, var_season[2,:,:], clevs_days, transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
        cbar  = plt.colorbar(plot5, ax=axs[1,1], ticklocation="right", pad=0.1, orientation="horizontal",
                aspect=40, shrink=1) # cax=cax,
        cbar.ax.tick_params(labelsize=10,labelrotation=45)
        axs[1,1].set_title('SON', size=12)

        # DJF
        plot6 = axs[1,2].contourf(lons, lats, var_season[3,:,:], clevs_days, transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
        cbar  = plt.colorbar(plot6, ax=axs[1,2], ticklocation="right", pad=0.1, orientation="horizontal",
                aspect=40, shrink=1) # cax=cax,
        cbar.ax.tick_params(labelsize=10,labelrotation=45)
        axs[1,2].set_title('DJF', size=12)

        plt.savefig('./plots/spatial_map_'+plot_type+'_'+message+'.png',dpi=300)

    return

if __name__ == "__main__":

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

        wrf_path       = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_ctl+"/WRF_output/wrfout_d01_2017-02-01_06:00:00"
        land_sen_path  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_sen+"/LIS_output/"
        land_ctl_path  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_ctl+"/LIS_output/"
        atmo_sen_path  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_sen+"/WRF_output/"
        atmo_ctl_path  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_ctl+"/WRF_output/"

        if 1:
            '''
            Difference plot yearly

            '''
            var_name   = "Tair_f_inst"
            delta      = True

            period     = "201703-201802"
            time_s     = datetime(2017,3,1,0,0,0,0)
            time_e     = datetime(2018,3,1,0,0,0,0)
            message    = period
            spatial_map_T_coldest_warmest(land_ctl_path, land_sen_path, var_name, time_s=time_s, time_e=time_e, lat_names="lat",
                                lon_names="lon",loc_lat=loc_lat, loc_lon=loc_lon, wrf_path=wrf_path, shape_path=shape_path,
                                message=message, delta=delta)

            period     = "201803-201902"
            time_s     = datetime(2018,3,1,0,0,0,0)
            time_e     = datetime(2019,3,1,0,0,0,0)
            message    = period
            spatial_map_T_coldest_warmest(land_ctl_path, land_sen_path, var_name, time_s=time_s, time_e=time_e, lat_names="lat",
                                lon_names="lon",loc_lat=loc_lat, loc_lon=loc_lon, wrf_path=wrf_path, shape_path=shape_path,
                                message=message, delta=delta)

            period     = "201903-202002"
            time_s     = datetime(2019,3,1,0,0,0,0)
            time_e     = datetime(2020,3,1,0,0,0,0)
            message    = period
            spatial_map_T_coldest_warmest(land_ctl_path, land_sen_path, var_name, time_s=time_s, time_e=time_e, lat_names="lat",
                                lon_names="lon",loc_lat=loc_lat, loc_lon=loc_lon, wrf_path=wrf_path, shape_path=shape_path,
                                message=message, delta=delta)
