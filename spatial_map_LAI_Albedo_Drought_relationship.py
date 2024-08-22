#!/usr/bin/python

__author__ = "Mengyuan Mu"
__email__  = "mu.mengyuan815@gmail.com"

'''
Functions:
1. process multi-year dataset and calculate a few metrics
'''

import os
from netCDF4 import Dataset
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import scipy.stats as stats
import cartopy.crs as ccrs
from scipy.interpolate import griddata, interp1d
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.feature import NaturalEarthFeature, OCEAN
from common_utils import *

def regrid_to_lat_lon(var, method='nearest'):

    # =================== Plotting spatial map ===================
    # Read mask_val
    land_file = '/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/Albedo_trend.nc'
    land      = Dataset(land_file, 'r')
    mask_val  = land.variables['lon'][:,:]
    var       = np.where(mask_val <0, np.nan, var)

    # Read input lat and lon
    wrf_file  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/"\
                + "WRF_output/p/wrfout_201701-202002.nc"
    wrf       = Dataset(wrf_file, 'r')
    lat_in    = wrf.variables['lat'][:,:]
    lon_in    = wrf.variables['lon'][:,:]

    # Set output lat and lon
    lat_out   = np.arange(-39,-24,0.04)
    lon_out   = np.arange(135,155,0.04)

    var_regrid= regrid_to_PlateCarree(var, mask_val, lat_in, lon_in, lat_out, lon_out, method='nearest')

    return var_regrid

def read_spatial_data(land_path, var_name, time_ss=None, time_es=None,
                      lat_names="lat", lon_names="lon",loc_lat=None, loc_lon=None):

    '''
    Read WRF-CABLE data
    '''

    print("var_name= "+var_name)

    if var_name in ["Rnet"]:
        land_files      = [land_path+'Lwnet_tavg/LIS.CABLE.201701-202002.nc']
        time, Lwnet_tmp = read_var_multi_file(land_files, 'Lwnet_tavg', loc_lat, loc_lon, lat_names, lon_names)
        land_files      = [land_path+'Swnet_tavg/LIS.CABLE.201701-202002.nc']
        time, Swnet_tmp = read_var_multi_file(land_files, 'Swnet_tavg', loc_lat, loc_lon, lat_names, lon_names)
        tmp             = Lwnet_tmp+Swnet_tmp
    elif var_name in ['VPD']:
        tair_files      = [land_path+'Tair_f_inst/LIS.CABLE.201701-202002.nc']
        qair_files      = [land_path+'Qair_f_inst/LIS.CABLE.201701-202002.nc']
        time, Tair      = read_var_multi_file(tair_files, "Tair_f_inst", loc_lat, loc_lon, lat_names, lon_names)
        time, Qair      = read_var_multi_file(qair_files, "Qair_f_inst", loc_lat, loc_lon, lat_names, lon_names)

        pres_files      = ['/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2_obs_LAI_ALB/'
                            +'WRF_output/slp/wrfout_201701-202002.nc']
        time_wrf, Pres_tmp = read_var_multi_file(pres_files, "slp", loc_lat, loc_lon, lat_names, lon_names)

        time_in = []
        time_out= []
        for t in time_wrf:
            time_in.append(t.total_seconds())
        for t in time:
            time_out.append(t.total_seconds())

        f              = interp1d(np.array(time_in), Pres_tmp[:], kind='linear',fill_value='extrapolate', axis=0)
        Pres           = f(np.array(time_out))
        tmp            = qair_to_vpd(Qair, Tair, Pres)
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
        land_files= [land_path+vname+'/LIS.CABLE.201701-202002.nc']
        time, tmp = read_var_multi_file(land_files, vname, loc_lat, loc_lon, lat_names, lon_names)

    # Read particular periods
    var         = []
    time_series = []
    time_init   = datetime(2000,1,1,0,0,0,0)

    for i in np.arange(len(time_ss)):
        print("np.shape(time)",np.shape(time))
        print("np.shape(tmp)",np.shape(tmp))

        # time-step into daily
        if var_name in ["SurfTmax","Tmax","VegTmax"]:
            # average of daily max
            temp  = time_clip_to_day_max(time,tmp,time_ss[i],time_es[i])
        elif var_name in ["SurfTmin","Tmin","VegTmin"]:
            # average of daily min
            temp  = time_clip_to_day_min(time,tmp,time_ss[i],time_es[i])
        elif var_name in ["SM_top50cm",]:
            # top 1m soil moisture [.022, .058, .154, .409, 1.085, 2.872]
            c_tmp = tmp[:,0,:,:]*0.022 + tmp[:,1,:,:]*0.058 + tmp[:,2,:,:]*0.154 + tmp[:,3,:,:]*0.266
            temp  = time_clip_to_day(time,c_tmp,time_ss[i],time_es[i])
        else:
            temp  = time_clip_to_day(time,tmp,time_ss[i],time_es[i])

        if i == 0:
            var   = temp
        else:
            var   = np.append(var,temp[:],axis=0)

        print('(time_ss[i]-time_init).days',(time_ss[i]-time_init).days,'(time_es[i]-time_init).days',(time_es[i]-time_init).days)

        for j in np.arange((time_ss[i]-time_init).days,(time_es[i]-time_init).days,1):
            time_series.append(j)

    print('np.shape(var)',np.shape(var))

    if var_name in ['WaterTableD_tavg','WatTable']:
        var     = var/1000.
    if var_name in ['ESoil_tavg','Evap_tavg',"ECanop_tavg",'TVeg_tavg',"Rainf_tavg","Snowf_tavg","Qs_tavg","Qsb_tavg"]:
        var     = var*3600*24
    if var_name in ['Qair_f_inst']:
        var     = var*1000
    if var_name in ['GPP_tavg','NPP_tavg']:
        s2d        = 3600*24.          # s-1 to d-1
        GPP_scale  = -0.000001*12*s2d   # umol s-1 to g d-1
        var     = var*GPP_scale

    return time_series, var


def plot_map_correlation(land_path, time_ss=None,time_es=None, lat_names="lat", lon_names="lon",
                    loc_lat=None, loc_lon=None, message=None):

    # Read data
    time_series, LAI     = read_spatial_data(land_path, "LAI_inst",    time_ss, time_es, lat_names, lon_names, loc_lat, loc_lon)
    time_series, Albedo  = read_spatial_data(land_path, "Albedo_inst", time_ss, time_es, lat_names, lon_names, loc_lat, loc_lon)
    time_series, SM50    = read_spatial_data(land_path, "SM_top50cm", time_ss, time_es, lat_names, lon_names, loc_lat, loc_lon)
    time_series, VPD     = read_spatial_data(land_path, "VPD", time_ss, time_es, lat_names, lon_names, loc_lat, loc_lon)

    wrf_file             = '/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2_obs_LAI_ALB/WRF_output/slp/wrfout_201701-202002.nc'
    time, lats           = read_var(wrf_file, lat_names, loc_lat, loc_lon, lat_names, lon_names)
    time, lons           = read_var(wrf_file, lon_names, loc_lat, loc_lon, lat_names, lon_names)

    # set the dimension
    ntime                = np.shape(LAI)[0]
    nlat                 = np.shape(LAI)[1]
    nlon                 = np.shape(LAI)[2]

    r_LAI_Albedo         = np.zeros((nlat,nlon))
    p_LAI_Albedo         = np.zeros((nlat,nlon))

    r_LAI_SM50           = np.zeros((nlat,nlon))
    p_LAI_SM50           = np.zeros((nlat,nlon))

    r_LAI_VPD            = np.zeros((nlat,nlon))
    p_LAI_VPD            = np.zeros((nlat,nlon))

    r_Albedo_SM50        = np.zeros((nlat,nlon))
    p_Albedo_SM50        = np.zeros((nlat,nlon))


    for x in np.arange(nlat):
        for y in np.arange(nlon):
            LAI_tmp = LAI[:,x,y]
            ALB_tmp = Albedo[:,x,y]
            SM50_tmp= SM50[:,x,y]
            VPD_tmp = VPD[:,x,y]
            if np.any(np.isnan(LAI_tmp)) or np.any(np.isnan(ALB_tmp)):
                r_LAI_Albedo[x,y]  = np.nan
                p_LAI_Albedo[x,y]  = np.nan
                r_LAI_SM50[x,y]    = np.nan
                p_LAI_SM50[x,y]    = np.nan
                r_LAI_VPD[x,y]     = np.nan
                p_LAI_VPD[x,y]     = np.nan
                r_Albedo_SM50[x,y] = np.nan
                p_Albedo_SM50[x,y] = np.nan
            else:
                r_LAI_Albedo[x,y],p_LAI_Albedo[x,y]   = stats.spearmanr(LAI_tmp, ALB_tmp)
                r_LAI_SM50[x,y],p_LAI_SM50[x,y]       = stats.spearmanr(LAI_tmp, SM50_tmp)
                r_LAI_VPD[x,y],p_LAI_VPD[x,y]         = stats.spearmanr(LAI_tmp, VPD_tmp)
                r_Albedo_SM50[x,y],p_Albedo_SM50[x,y] = stats.spearmanr(ALB_tmp, SM50_tmp)

    print('np.any(r_LAI_Albedo)',np.any(r_LAI_Albedo),'np.any(p_LAI_Albedo)',np.any(p_LAI_Albedo))

    if 0:
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
        cmap  = plt.cm.BrBG# seismic_r
        clevs = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

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
        gl.xlines        = False
        gl.xlines        = False

        if loc_lat == None:
            gl.xlocator  = mticker.FixedLocator([135,140,145,150,155])
            gl.ylocator  = mticker.FixedLocator([-40,-35,-30,-25])
        else:
            gl.xlocator  = mticker.FixedLocator(loc_lon)
            gl.ylocator  = mticker.FixedLocator(loc_lat)

        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size':10, 'color':'black'}
        gl.ylabel_style = {'size':10, 'color':'black'}

        r               = np.where(p<0.05, r, np.nan)
        plt.contourf(lons, lats, r, clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend='both') #

        cb = plt.colorbar(ax=ax, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
        cb.ax.tick_params(labelsize=10)
        plt.title(message, size=16)

        plt.savefig('./plots/correl_map_LAI_ALB_'+message+'.png',dpi=300)
    if 1:
        fig, ax = plt.subplots(nrows=1, ncols=4, figsize=[12,4],sharex=True, sharey=True, squeeze=True,
                                subplot_kw={'projection': ccrs.PlateCarree()})
        plt.subplots_adjust(wspace=0.05, hspace=0.) # left=0.15,right=0.95,top=0.85,bottom=0.05,

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

        # set the box type of sequence number
        props = dict(boxstyle="round", facecolor='white', alpha=0.0, ec='white')
        # choose colormap

        states= NaturalEarthFeature(category="cultural", scale="50m",
                                            facecolor="none",
                                            name="admin_1_states_provinces_shp")


        # ======================= Set colormap =======================
        label_x = [ "r_LAI_Albedo", "r_LAI_SM50", "r_LAI_VPD", "r_Albedo_SM50",]
        cmap    = plt.cm.BrBG# seismic_r
        # clevs = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

        cnt         = 0
        for i in np.arange(4):
            ax[i].coastlines(resolution="50m",linewidth=1)
            ax[i].set_extent([135,155,-39,-24])
            ax[i].add_feature(states, linewidth=.5, edgecolor="black")

            # Add gridlines
            gl = ax[i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color=almost_black, linestyle='--')
            gl.top_labels   = False
            gl.right_labels = False
            gl.bottom_labels= True
            gl.left_labels  = True
            gl.xlines       = False
            gl.ylines       = False
            gl.xlocator     = mticker.FixedLocator(np.arange(125,160,1))
            gl.ylocator     = mticker.FixedLocator(np.arange(-40,-20,1))
            gl.xformatter   = LONGITUDE_FORMATTER
            gl.yformatter   = LATITUDE_FORMATTER
            gl.xlabel_style = {'size':12, 'color':almost_black}#,'rotation': 90}
            gl.ylabel_style = {'size':12, 'color':almost_black}

            # left: MAM
            print('ccrs.PlateCarree()',ccrs.PlateCarree())
            extent=(135, 155, -39, -24)
            # print(np.min(lon),np.max(lon), np.min(lat), np.max(lat))
            if cnt == 0:
                r_LAI_Albedo_tmp = regrid_to_lat_lon(r_LAI_Albedo, 'nearest')
                # r_LAI_Albedo     = np.where(p_LAI_Albedo<0.05, r_LAI_Albedo, np.nan)
                plot1            = ax[i].imshow(r_LAI_Albedo_tmp, origin="lower", extent=extent, vmin=-1, vmax=1, transform=ccrs.PlateCarree(), cmap=cmap)
                # cbar             = plt.colorbar(plot1, ax=ax[i], ticklocation="right", pad=0.01, orientation="vertical",
                #                    aspect=20, shrink=0.6) # cax=cax,
            elif cnt == 1:
                r_LAI_SM50_tmp = regrid_to_lat_lon(r_LAI_SM50, 'nearest')
                # r_LAI_SM50   = np.where(p_LAI_SM50<0.05, r_LAI_SM50, np.nan)
                plot2          = ax[i].imshow(r_LAI_SM50_tmp, origin="lower", extent=extent, vmin=-1, vmax=1, transform=ccrs.PlateCarree(), cmap=cmap)
                # cbar           = plt.colorbar(plot2, ax=ax[i], ticklocation="right", pad=0.01, orientation="vertical",
                #                aspect=20, shrink=0.6) # cax=cax,
            elif cnt == 2:
                r_LAI_VPD_tmp = regrid_to_lat_lon(r_LAI_VPD, 'nearest')
                # r_LAI_VPD    = np.where(p_LAI_VPD<0.05, r_LAI_VPD, np.nan)
                plot3        = ax[i].imshow(r_LAI_VPD_tmp, origin="lower", extent=extent, vmin=-1, vmax=1, transform=ccrs.PlateCarree(), cmap=cmap)
                # cbar         = plt.colorbar(plot3, ax=ax[i], ticklocation="right", pad=0.01, orientation="vertical",
                #                aspect=20, shrink=0.6) # cax=cax,
            elif cnt == 3:
                r_Albedo_SM50_tmp = regrid_to_lat_lon(r_Albedo_SM50, 'nearest')
                # r_Albedo_SM50 = np.where(p_Albedo_SM50<0.05, r_Albedo_SM50, np.nan)
                plot4         = ax[i].imshow(r_Albedo_SM50_tmp, origin="lower", extent=extent, vmin=-1, vmax=1, transform=ccrs.PlateCarree(), cmap=cmap)
                # cbar          = plt.colorbar(plot4, ax=ax[i], ticklocation="right", pad=0.01, orientation="vertical",
                #                 aspect=20, shrink=0.6) # cax=cax,

            ax[i].text(0.02, 0.15, label_x[cnt], transform=ax[i].transAxes, fontsize=14, verticalalignment='top', bbox=props)
            # ax[0].add_feature(OCEAN,edgecolor='none', facecolor="lightgray")
            ax[i].add_feature(OCEAN,edgecolor='none', facecolor="lightgray")
            cnt = cnt + 1
        cbar = plt.colorbar(plot1, ax=ax, ticklocation="bottom", pad=0.1, orientation="horizontal", aspect=40, shrink=0.6) # cax=cax,
        cbar.ax.tick_params(labelsize=8, labelrotation=45)
        plt.savefig('./plots/correl_map_LAI_ALB_'+message+'.png',dpi=300)

if __name__ == "__main__":

    # ======================= Option =======================
    region = "SE Aus" #"SE Aus" #"CORDEX" #"SE Aus"

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
        '''
        Test WRF-CABLE output
        '''

        case_sen       = "drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2_obs_LAI_ALB"
        land_sen_path  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_sen+"/LIS_output/"

        '''
        make nc file
        '''

        time_ss    = [datetime(2017,12,1,0,0,0,0),datetime(2018,12,1,0,0,0,0),datetime(2019,12,1,0,0,0,0)]
        time_es    = [datetime(2018,3,1,0,0,0,0), datetime(2019,3,1,0,0,0,0), datetime(2020,3,1,0,0,0,0)]

        message    = "Tinderbox_drought"
        plot_map_correlation(land_sen_path, time_ss=time_ss,time_es=time_es, lat_names="lat", lon_names="lon",
                loc_lat=loc_lat, loc_lon=loc_lon, message=message)
