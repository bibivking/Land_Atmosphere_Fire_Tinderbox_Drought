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
import cartopy.crs as wrf_path
import cartopy.feature as cfeature
from matplotlib.cm import get_cmap
from sklearn.metrics import mean_squared_error
from scipy.interpolate import griddata, interp1d
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from wrf import (getvar, interplevel, get_cartopy, cartopy_xlim,
                 cartopy_ylim, to_np, latlon_coords)
from common_utils import *


def read_spatial_data(land_ctl_path, land_sen_path, var_name, time_ss=None,
                      time_es=None, lat_names="lat", lon_names="lon",
                      loc_lat=None, loc_lon=None):

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
    time_series = []
    time_init   = datetime(2000,1,1,0,0,0,0)

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
        
        print('(time_ss[i]-time_init).days',(time_ss[i]-time_init).days,'(time_es[i]-time_init).days',(time_es[i]-time_init).days)

        for j in np.arange((time_ss[i]-time_init).days,(time_es[i]-time_init).days,1):
            time_series.append(j)

    print("ATTENTION!!! check time_series",np.shape(time_series),time_series)

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

    return time_series, ctl_in, sen_in


def make_delta_Tmax_LAI_Albedo_nc(land_ctl_path, land_sen_path, output_file, var_names, time_ss=None,time_es=None, lat_names="lat", lon_names="lon",
                    loc_lat=None, loc_lon=None, message=None):

    # Read data
    time_series, ctl_one, sen_one     = read_spatial_data(land_ctl_path, land_sen_path, var_names[0], time_ss, time_es, lat_names, lon_names,loc_lat, loc_lon)
    time_series, ctl_two, sen_two     = read_spatial_data(land_ctl_path, land_sen_path, var_names[1], time_ss, time_es, lat_names, lon_names,loc_lat, loc_lon)
    time_series, ctl_three, sen_three = read_spatial_data(land_ctl_path, land_sen_path, var_names[2], time_ss, time_es, lat_names, lon_names,loc_lat, loc_lon)
    one_diff             = sen_one - ctl_one
    two_diff             = sen_two - ctl_two
    three_diff           = sen_three - ctl_three

    land_ctl             = land_ctl_path+"LIS.CABLE.201701-201701.d01.nc"
    time, lats           = read_var(land_ctl, lat_names, loc_lat, loc_lon, lat_names, lon_names)
    time, lons           = read_var(land_ctl, lon_names, loc_lat, loc_lon, lat_names, lon_names)

    # set the dimension
    ntime                = np.shape(ctl_one)[0]
    nlat                 = np.shape(ctl_one)[1]
    nlon                 = np.shape(ctl_one)[2]

    print('np.shape(one_diff)', np.shape(one_diff))

    # create file and write global attributes
    f                   = nc.Dataset(output_file, 'w', format='NETCDF4')
    f.history           = "Created by: %s" % (os.path.basename(__file__))
    f.creation_date     = "%s" % (datetime.now())
    f.description       = 'daily ΔTmax (sen-ctl), ΔLAI, Δalbedo for 2017-2020 three DJFs, made by MU Mengyuan'

    # set dimensions
    f.createDimension('time', ntime)
    f.createDimension('north_south', nlat)
    f.createDimension('east_west', nlon)
    f.Conventions        = "CF-1.0"

    time                 = f.createVariable('time', 'f4', ('time'))
    time.units           = "days since 2000-01-01"
    time[:]              = time_series

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

    delta_Tmax               = f.createVariable('delta_Tmax', 'f4', ('time', 'north_south', 'east_west'))
    delta_Tmax.standard_name = 'maximum daiy air tempeature difference'
    delta_Tmax.units         = 'C'
    delta_Tmax[:]            = one_diff

    delta_LAI                = f.createVariable('delta_LAI', 'f4', ('time', 'north_south', 'east_west'))
    delta_LAI.standard_name  = 'Leaf area index (LAI) difference'
    delta_LAI.units          = 'm2/m2'
    delta_LAI[:]             = two_diff

    delta_Albedo               = f.createVariable('delta_Albedo', 'f4', ('time', 'north_south', 'east_west'))
    delta_Albedo.standard_name = 'Albedo difference'
    delta_Albedo[:]            = three_diff

    f.close()

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
        cmap  = plt.cm.BrBG#plt.cm.seismic_r
        cmap.set_bad('lightgrey')
        clevs = [-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5,0.6]

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
        partial_corr_r = np.where(partial_corr_r<0.05,partial_corr_r,np.nan)
        plt.contourf(lons, lats, partial_corr_r, clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend='both') #

        cb = plt.colorbar(ax=ax, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
        cb.ax.tick_params(labelsize=10)
        plt.title("Δ"+var_names[0]+" & Δ"+var_names[1], size=16)

        plt.savefig('./plots/partial_corr_map_'+message+'.png',dpi=300)


def cut_into_pieces(input_file, output_path):

    # Read in input file
    fin       = nc.Dataset(input_file,  mode='r')
    time_in   = fin.variables['time'][:]
    lat_in    = fin.variables['lat'][:]
    lon_in    = fin.variables['lon'][:]
    Tmax_in   = fin.variables['delta_Tmax'][:]
    LAI_in    = fin.variables['delta_LAI'][:]
    Albedo_in = fin.variables['delta_Albedo'][:]
    fin.close()

    # Set lat and lon numbers
    nlat      = 439
    nlon      = 23 # 529 = 23 x 23
    ntime     = len(time_in)
    
    for i in np.arange(23):

        print("Doing part "+str(i))
        
        # set the range of lon
        lon_s = i*23
        lon_e = (i+1)*23

        # define the file name
        output_file          = output_path + "deltaTmax_deltaLAI_deltaAlbedo_2017_2020_DJF_part"+str(i+1)+".nc"
        
        print("output_file="+output_file)

        # create file and write global attributes
        f                    = nc.Dataset(output_file, 'w', format='NETCDF4')
        f.history            = "Created by: %s" % (os.path.basename(__file__))
        f.creation_date      = "%s" % (datetime.now())
        f.description        = 'daily ΔTmax (sen-ctl), ΔLAI, Δalbedo for 2017-2020 three DJFs, made by MU Mengyuan'

        # set dimensions
        f.createDimension('time', ntime)
        f.createDimension('north_south', nlat)
        f.createDimension('east_west', nlon)
        f.Conventions        = "CF-1.0"

        time                 = f.createVariable('time', 'f4', ('time'))
        time.units           = "days since 2000-01-01"
        time[:]              = time_in

        latitude             = f.createVariable('lat', 'f4', ('north_south', 'east_west'))
        latitude.long_name   = "latitude"
        latitude.units       = "degree_north"
        latitude._CoordinateAxisType = "Lat"
        latitude[:]          = lat_in[:,lon_s:lon_e]

        longitude            = f.createVariable('lon', 'f4', ('north_south', 'east_west'))
        longitude.long_name  = "longitude"
        longitude.units      = "degree_east"
        longitude._CoordinateAxisType = "Lon"
        longitude[:]         = lon_in[:,lon_s:lon_e]

        delta_Tmax               = f.createVariable('delta_Tmax', 'f4', ('time', 'north_south', 'east_west'))
        delta_Tmax.standard_name = 'maximum daiy air tempeature difference'
        delta_Tmax.units         = 'C'
        delta_Tmax[:]            = Tmax_in[:,:,lon_s:lon_e]

        delta_LAI                = f.createVariable('delta_LAI', 'f4', ('time', 'north_south', 'east_west'))
        delta_LAI.standard_name  = 'Leaf area index (LAI) difference'
        delta_LAI.units          = 'm2/m2'
        delta_LAI[:]             = LAI_in[:,:,lon_s:lon_e]

        delta_Albedo               = f.createVariable('delta_Albedo', 'f4', ('time', 'north_south', 'east_west'))
        delta_Albedo.standard_name = 'Albedo difference'
        delta_Albedo[:]            = Albedo_in[:,:,lon_s:lon_e]

        f.close()


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
    if 0:
        '''
        Test WRF-CABLE output
        '''

        case_name      = "ALB-CTL_new" #"bl_pbl2_mp4_sf_sfclay2" #
        case_ctl       = "drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2"
        case_sen       = "drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2_obs_LAI_ALB"

        output_file    = "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/deltaTmax_deltaLAI_deltaAlbedo_2017_2020_DJF.nc"
        land_sen_path  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_sen+"/LIS_output/"
        land_ctl_path  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_ctl+"/LIS_output/"

        '''
        make nc file
        '''

        var_names  = ["Tmax","LAI_inst", "Albedo_inst",]
        period     = "Summers_"+var_names[0]+"_vs_"+var_names[1]
        time_ss    = [datetime(2017,12,1,0,0,0,0),datetime(2018,12,1,0,0,0,0),datetime(2019,12,1,0,0,0,0)]
        time_es    = [datetime(2018,3,1,0,0,0,0), datetime(2019,3,1,0,0,0,0), datetime(2020,3,1,0,0,0,0)]
        message    = case_name+"_"+period
        make_delta_Tmax_LAI_Albedo_nc(land_ctl_path, land_sen_path, output_file, var_names, time_ss=time_ss,time_es=time_es, lat_names="lat", lon_names="lon",
                loc_lat=loc_lat, loc_lon=loc_lon, message=message)

    if 1: 
        # cutting the file into 23 pieces   
        input_file  = "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/deltaTmax_deltaLAI_deltaAlbedo_2017_2020_DJF.nc"
        output_path = "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/"

        cut_into_pieces(input_file, output_path)