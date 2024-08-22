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
from netCDF4 import Dataset,num2date
from datetime import datetime, timedelta
from matplotlib.cm import get_cmap
from matplotlib.patches import Polygon
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.feature import NaturalEarthFeature
from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim,
                        cartopy_ylim, latlon_coords, ALL_TIMES)
from common_utils import *

def read_LIS_vars(var_type):

    '''
    List the variables in a LIS file
    '''

    if var_type == "var_3D":
        var_names  =  [ "Albedo_inst","Tair_f_inst","Swnet_tavg","Lwnet_tavg","Qle_tavg","Qh_tavg",
                        "Rainf_tavg","Evap_tavg","Qs_tavg","Qsb_tavg","VegT_tavg","AvgSurfT_tavg",
                        "ECanop_tavg","TVeg_tavg",
                        "FWsoil_tavg","ESoil_tavg","Wind_f_inst",
                        "Qair_f_inst","Psurf_f_inst","SWdown_f_inst","LWdown_f_inst"]
        # "Rainf_f_inst","Albedo_inst",
        # "GPP_tavg","Qg_tavg","Snowf_tavg","SWE_inst","SnowDepth_inst","SoilWet_inst","CanopInt_inst","SnowCover_inst",
    elif var_type == "var_landinfo_3D":
        var_names  =  [ "Landmask_inst","Landcover_inst","Soiltype_inst","SandFrac_inst","ClayFrac_inst","SiltFrac_inst",
                        "SoilFieldCap_inst","SoilSat_inst","SoilWiltPt_inst","Hyds_inst","Bch_inst","Sucs_inst",
                        "Elevation_inst","LAI_inst"]
    elif var_type == "var_4D":
        var_names  =  ["RelSMC_inst","SoilMoist_inst","SoilTemp_inst","SmLiqFrac_inst","SmFrozFrac_inst"]
    elif var_type == "var_3D_basic":
        var_names  = ["Tair_f_inst",'Evap_tavg',"ESoil_tavg","ECanop_tavg",'TVeg_tavg',"FWsoil_tavg","Qle_tavg","Qh_tavg",
                      "Qg_tavg","VegT_tavg","WaterTableD_tavg","Rainf_tavg","Qs_tavg","Qsb_tavg",]
    elif var_type == "var_4D_basic":
        var_names  =  ["SoilMoist_inst","SoilTemp_inst"]
    elif var_type == "var_energy":
        var_names  = ["Tair_f_inst"] #["Swnet_tavg","Lwnet_tavg","Qle_tavg","Qh_tavg","Qg_tavg","Qair_f_inst","Rnet","EF"] #
    elif var_type == "var_albedo":
        var_names  = ["Albedo_inst"]
    elif var_type == "var_wrf_hgt":
        var_names  = [
                    'cape_3d',# 3D CAPE and CIN
                    'p',    # Full Model Pressure
                    'avo',    # Absolute Vorticity
                    'eth',    # Equivalent Potential Temperature
                    'dbz',    # Reflectivity
                    'geopt',  # Geopotential for the Mass Grid
                    'omg',  # Omega
                    'pvo',  # Potential Vorticity
                    'rh',   # Relative Humidity
                    'td',   # Dew Point Temperature
                    'tc',   # Temperature in Celsius
                    'th',   # Potential Temperature
                    'temp', # Temperature (in specified units)
                    'tv',   # Virtual Temperature
                    'twb',  # Wet Bulb Temperature
                    'ua',   # U-component of Wind on Mass Points
                    'va',   # V-component of Wind on Mass Points
                    'wa',   # W-component of Wind on Mass Points
                    'z',    # Model Height for Mass Grid
                    ]
    elif var_type == "var_wrf_hgt":
        var_names  = [
                    'cape_3d',# 3D CAPE and CIN
                    'p',    # Full Model Pressure
                    'avo',    # Absolute Vorticity
                    'eth',    # Equivalent Potential Temperature
                    'dbz',    # Reflectivity
                    'geopt',  # Geopotential for the Mass Grid
                    'omg',  # Omega
                    'pvo',  # Potential Vorticity
                    'rh',   # Relative Humidity
                    'td',   # Dew Point Temperature
                    'tc',   # Temperature in Celsius
                    'th',   # Potential Temperature
                    'temp', # Temperature (in specified units)
                    'tv',   # Virtual Temperature
                    'twb',  # Wet Bulb Temperature
                    'ua',   # U-component of Wind on Mass Points
                    'va',   # V-component of Wind on Mass Points
                    'wa',   # W-component of Wind on Mass Points
                    'z',    # Model Height for Mass Grid
                    ]
    elif var_type == "var_wrf_surf":
        var_names = [
                    'cloudfrac', # Cloud Fraction
                    'td2',  # 2m Dew Point Temperature
                    'rh2',  # 2m Relative Humidity
                    'T2',   # 2m Temperature
                    'slp',  # Sea Level Pressure
                    'ter',  # Model Terrain Height
                    'updraft_helicity', # Updraft Helicity
                    'helicity',        # Storm Relative Helicity
                    'ctt',  # Cloud Top Temperature
                    'mdbz', # Maximum Reflectivity
                    'td2',  # 2m Dew Point Temperature
                    'rh2',  # 2m Relative Humidity
                    'T2',   # 2m Temperature
                    'slp',  # Sea Level Pressure
                    'pw',   # Precipitable Water
                    'cape_2d', # 2D CAPE (MCAPE/MCIN/LCL/LFC)
                    'cloudfrac', # Cloud Fraction
                ]
    elif var_type == "var_wrf_surf_basic":
        var_names = [
                    'cloudfrac', # Cloud Fraction
                    'td2',  # 2m Dew Point Temperature
                    'rh2',  # 2m Relative Humidity
                    'T2',   # 2m Temperature
                    'slp',  # Sea Level Pressure
                    'td2',  # 2m Dew Point Temperature
                    'rh2',  # 2m Relative Humidity
                    'T2',   # 2m Temperature
                    'slp',  # Sea Level Pressure
                    'pw',   # Precipitable Water
                    'cape_2d', # 2D CAPE (MCAPE/MCIN/LCL/LFC)
                ]
    elif var_type == "var_wrf_surf_other":
        var_names  = [  "SWDNB", # INSTANTANEOUS DOWNWELLING SHORTWAVE FLUX AT BOTTOM
                        "LWDNB", # INSTANTANEOUS DOWNWELLING LONGWAVE FLUX AT BOTTOM
                        "SWUPB", # INSTANTANEOUS UPWELLING SHORTWAVE FLUX AT BOTTOM
                        "LWUPB", # INSTANTANEOUS UPWELLING LONGWAVE FLUX AT BOTTOM
                        ]
        # ['RAINC','RAINNC','PSFC','U10','V10','TSK','PBLH']
    return var_names

def spatial_map_single_plot(file_path, var_name, time_s, time_e, lat_name, lon_name,
                            loc_lat=None, loc_lon=None, wrf_path=None,message=None):

    '''
    plot a single spatial map
    '''

    time, Var  = read_var(file_path, var_name, loc_lat, loc_lon, lat_name, lon_name)
    print(time)
    var        = spatial_var(time,Var,time_s,time_e)

    if 'LIS' in file_path:
        wrf        = Dataset(wrf_path,  mode='r')
        lons       = wrf.variables['XLONG'][0,:,:]
        lats       = wrf.variables['XLAT'][0,:,:]
        var        =  var/1000.
    else:
        time, lats = read_var(file_path, lat_name, loc_lat, loc_lon, lat_name, lon_name)
        time, lons = read_var(file_path, lon_name, loc_lat, loc_lon, lat_name, lon_name)

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
    clevs = np.linspace( 0.,600., num=31)
    cmap  = plt.cm.GnBu_r # BrBG

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

    plt.contourf(lons, lats, var,clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend='both') #

    cb = plt.colorbar(ax=ax, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
    cb.ax.tick_params(labelsize=10)
    plt.title(var_name, size=16)

    if message == None:
        message = var_name
    else:
        message = message + "_" + var_name

    plt.savefig('./plots/weather/spatial_map_'+message+'.png',dpi=300)

def spatial_map_single_plot_diff(file_paths, var_names, time_s=None, time_e=None, lat_names="lat",
                                 lon_names="lon",loc_lat=None, loc_lon=None, wrf_path=None,
                                 shape_path=None, message=None):

    '''
    plot a single spatial map
    '''

    # WRF-CABLE
    if 'LIS_HIST_' in file_paths[0]:
        var_file     = Dataset(file_paths[0], mode='r')
        var1         = var_file.variables[var_names[0]][:,:]
        lats         = var_file.variables[lat_names[0]][:,:]
        lons         = var_file.variables[lon_names[0]][:,:]
    else:
        time1, Var1  = read_var(file_paths[0], var_names[0], loc_lat, loc_lon, lat_names[0], lon_names[0])
        time1, lats  = read_var(file_paths[0], lat_names[0], loc_lat, loc_lon, lat_names[0], lon_names[0])
        time1, lons  = read_var(file_paths[0], lon_names[0], loc_lat, loc_lon, lat_names[0], lon_names[0])
        var1         = spatial_var(time1,Var1,time_s,time_e)

    if 'LIS' in file_paths[0] and var_names[0] in ['WaterTableD_tavg','WatTable']:
        var1     = var1/1000.
    if var_names[0] in ['ESoil_tavg','Evap_tavg','TVeg_tavg']:
        var1     = var1*3600

    # read lat and lon outs
    wrf          = Dataset(wrf_path,  mode='r')
    lons_out     = wrf.variables['XLONG'][0,:,:]
    lats_out     = wrf.variables['XLAT'][0,:,:]

    # offline-CABLE

    if 'cable_out' in file_paths[1]:
        # offline sim
        time2, Var2 = read_var(file_paths[1], var_names[1], loc_lat, loc_lon, lat_names[1], lon_names[1])
        var2        = spatial_var(time2,Var2,time_s,time_e)
    elif '-' in file_paths[1]:
        # lis-cable hist
        time2, Var2 = read_var(file_paths[1], var_names[1], loc_lat, loc_lon, lat_names[1], lon_names[1])
        var2        = Var2[0,:,:]
    elif 'LIS_HIST_' in file_paths[1]:
        var_file2   = Dataset(file_paths[1], mode='r')
        var2        = var_file2.variables[var_names[1]][:,:]
    elif 'LIS' in file_paths[0]:
        # lis restart
        time2, Var2 = read_var(file_paths[1], var_names[1], loc_lat, loc_lon, lat_names[1], lon_names[1])
        var2        = Var2[-1]

    if 'LIS' in file_paths[1] and var_names[1] in ['WaterTableD_tavg','WatTable']:
        var2     = var2/1000.
    if var_names[1] in ['ESoil_tavg','Evap_tavg','TVeg_tavg']:
        var2     = var2*3600

    if 'cable_out' in file_paths[1] :
        time, lats_in= read_var(file_paths[1], lat_names[1], loc_lat, loc_lon, lat_names[1], lon_names[1])
        time, lons_in= read_var(file_paths[1], lon_names[1], loc_lat, loc_lon, lat_names[1], lon_names[1])

        if var_names[1] in ['SoilMoist','SoilMoist_inst','ssat','sucs']:
            nlat   = len(lats_out[:,0])
            nlon   = len(lats_out[0,:])
            var2_regrid  = np.zeros((6,nlat,nlon))
            for j in np.arange(6):
                var2_regrid[j,:,:]  = regrid_data(lats_in, lons_in, lats_out, lons_out, var2[j,:,:])
        else:
            var2_regrid  = regrid_data(lats_in, lons_in, lats_out, lons_out, var2)

        if var_names[1] in ['ssat','sucs']:
            nlat   = len(lats_out[:,0])
            nlon   = len(lats_out[0,:])
            var    = np.zeros((6,nlat,nlon))
            for j in np.arange(6):
                var[j,:,:] = var1-var2_regrid[j,:,:]
        else:
            var    = var1-var2_regrid

    elif 'LIS' in file_paths[1]:
        var          = var1-var2

    if var_names[0] in ['WaterTableD_tavg','WatTable']:
        clevs = [-4,-3,-2,-1.5,-1,-0.5,0.5,1,1.5,2,3,4]
    elif var_names[0] in ['GWwb_tavg','GWMoist']:
        clevs = [-0.05,-0.04,-0.03,-0.02,-0.01,-0.005,0.005,0.01,0.02,0.03,0.04,0.05]
    elif var_names[0] in ['SoilMoist_inst','SoilMoist']:
        clevs = [-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0.05,0.1,0.15,0.2,0.25,0.3]
    elif var_names[0] in ['Sucs_inst','sucs_inst']:
        clevs = [-100,-80,-60,-40,-20,-10,10,20,40,60,80,100]
    elif var_names[0] in ['ESoil_tavg','Evap_tavg','TVeg_tavg']:
        clevs = [-1,-0.8,-0.6,-0.4,-0.2,-0.1,0.1,0.2,0.4,0.6,0.8,1]
    elif var_names[0] in ['Tair_f_inst','VegT_tavg']:
        clevs = [-5,-4.5,-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0.5,1,1.5,2,2.5,3,3.5,4.,4.5,5]
    elif var_names[0] in ['LAI_inst']:
        clevs = [-2,-1.5,-1,-0.5,-0.1,0.1,0.5,1,1.5,2]
    else:
        # clevs = [-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0.05,0.1,0.15,0.2,0.25,0.3]
        clevs = [-0.5,-0.4,-0.3,-0.2,-0.1,-0.05,0.05,0.1,0.2,0.3,0.4,0.5]

    print("len(np.shape(var))",len(np.shape(var)))

    if len(np.shape(var)) >=3:

        for j in np.arange(6):

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
            cmap  = plt.cm.seismic

            #hot_r # BrBG

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
                gl.xlocator  = mticker.FixedLocator([135,140,145,150,155])
                gl.ylocator  = mticker.FixedLocator([-40,-35,-30,-25])
            else:
                gl.xlocator  = mticker.FixedLocator(loc_lon)
                gl.ylocator  = mticker.FixedLocator(loc_lat)

            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlabel_style = {'size':10, 'color':'black'}
            gl.ylabel_style = {'size':10, 'color':'black'}
            plt.contourf(lons, lats, var[j,:,:], clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend='both') #

            cb = plt.colorbar(ax=ax, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
            cb.ax.tick_params(labelsize=10)
            plt.title(var_names[0], size=16)

            if shape_path != None:
            # Requires the pyshp package
                sf = shp.Reader(shape_path)

                for shape in sf.shapeRecords():
                    x = [i[0] for i in shape.shape.points[:]]
                    y = [i[1] for i in shape.shape.points[:]]
                    plt.plot(x,y,c="black")

            if j == 0:
                if message == None:
                    message = var_names[0]
                else:
                    message = message + "_" + var_names[0]

            plt.savefig('./plots/WTD_sudden_change/spatial_map_'+message+'_layer='+str(j)+'.png',dpi=300)
            cb = None
            gl = None
            ax = None
            fig= None

    elif len(np.shape(var)) ==2:
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
        cmap  = plt.cm.seismic

        #hot_r # BrBG

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
            gl.xlocator  = mticker.FixedLocator([135,140,145,150,155])
            gl.ylocator  = mticker.FixedLocator([-40,-35,-30,-25])
        else:
            gl.xlocator  = mticker.FixedLocator(loc_lon)
            gl.ylocator  = mticker.FixedLocator(loc_lat)

        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size':10, 'color':'black'}
        gl.ylabel_style = {'size':10, 'color':'black'}

        plt.contourf(lons, lats, var, clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend='both') #
        print(var)
        cb = plt.colorbar(ax=ax, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
        cb.ax.tick_params(labelsize=10)
        plt.title(var_names[0], size=16)

        if shape_path != None:
            # Requires the pyshp package
            sf = shp.Reader(shape_path)

            for shape in sf.shapeRecords():
                x = [i[0] for i in shape.shape.points[:]]
                y = [i[1] for i in shape.shape.points[:]]
                plt.plot(x,y,c="black")

        if message == None:
            message = var_names[0]
        else:
            message = message + "_" + var_names[0]

        plt.savefig('./plots/WTD_sudden_change/spatial_map_'+message+'.png',dpi=300)

def spatial_map_single_plot_LIS_diff(land_ctl_path, land_sen_path, var_names, time_s=None,
                                     time_e=None, lat_names="lat", lon_names="lon",loc_lat=None,
                                     loc_lon=None, wrf_path=None, shape_path=None, message=None):

    '''
    plot a single spatial map
    '''

    # WRF-CABLE
    for var_name in var_names:
        print("plotting "+var_name)

        if var_name in ["Tmax","Tmin",]:
            land_ctl_files= [land_ctl_path+'Tair_f_inst/LIS.CABLE.201701-201912.nc']
            land_sen_files= [land_sen_path+'Tair_f_inst/LIS.CABLE.201701-201912.nc']
            time, Ctl_tmp = read_var_multi_file(land_ctl_files, 'Tair_f_inst', loc_lat, loc_lon, lat_names, lon_names)
            time, Sen_tmp = read_var_multi_file(land_sen_files, 'Tair_f_inst', loc_lat, loc_lon, lat_names, lon_names)
        elif var_name in ["VegTmax","VegTmin"]:
            land_ctl_files= [land_ctl_path+'VegT_tavg/LIS.CABLE.201701-201912.nc']
            land_sen_files= [land_sen_path+'VegT_tavg/LIS.CABLE.201701-201912.nc']
            time, Ctl_tmp = read_var_multi_file(land_ctl_files, 'VegT_tavg', loc_lat, loc_lon, lat_names, lon_names)
            time, Sen_tmp = read_var_multi_file(land_sen_files, 'VegT_tavg', loc_lat, loc_lon, lat_names, lon_names)
        elif var_name in ["SurfTmax","SurfTmin"]:
            land_ctl_files= [land_ctl_path+'AvgSurfT_tavg/LIS.CABLE.201701-201912.nc']
            land_sen_files= [land_sen_path+'AvgSurfT_tavg/LIS.CABLE.201701-201912.nc']
            time, Ctl_tmp = read_var_multi_file(land_ctl_files, 'AvgSurfT_tavg', loc_lat, loc_lon, lat_names, lon_names)
            time, Sen_tmp = read_var_multi_file(land_sen_files, 'AvgSurfT_tavg', loc_lat, loc_lon, lat_names, lon_names)
        elif var_name in ["Rnet",]:
            land_ctl_files= [land_ctl_path+'Lwnet_tavg/LIS.CABLE.201701-201912.nc']
            land_sen_files= [land_sen_path+'Lwnet_tavg/LIS.CABLE.201701-201912.nc']
            time, Ctl_Lwnet_tmp = read_var_multi_file(land_ctl_files, 'Lwnet_tavg', loc_lat, loc_lon, lat_names, lon_names)
            time, Sen_Lwnet_tmp = read_var_multi_file(land_sen_files, 'Lwnet_tavg', loc_lat, loc_lon, lat_names, lon_names)
            land_ctl_files= [land_ctl_path+'Swnet_tavg/LIS.CABLE.201701-201912.nc']
            land_sen_files= [land_sen_path+'Swnet_tavg/LIS.CABLE.201701-201912.nc']
            time, Ctl_Swnet_tmp = read_var_multi_file(land_ctl_files, 'Swnet_tavg', loc_lat, loc_lon, lat_names, lon_names)
            time, Sen_Swnet_tmp = read_var_multi_file(land_sen_files, 'Swnet_tavg', loc_lat, loc_lon, lat_names, lon_names)
            Ctl_tmp = Ctl_Lwnet_tmp+Ctl_Swnet_tmp
            Sen_tmp = Sen_Lwnet_tmp+Sen_Swnet_tmp
        elif var_name in ["SM_top50cm",]:
            land_ctl_files= [land_ctl_path+'SoilMoist_inst/LIS.CABLE.201701-201912.nc']
            land_sen_files= [land_sen_path+'SoilMoist_inst/LIS.CABLE.201701-201912.nc']
            time, Ctl_tmp = read_var_multi_file(land_ctl_files, 'SoilMoist_inst', loc_lat, loc_lon, lat_names, lon_names)
            time, Sen_tmp = read_var_multi_file(land_sen_files, 'SoilMoist_inst', loc_lat, loc_lon, lat_names, lon_names)
        else:
            land_ctl_files= [land_ctl_path+var_name+'/LIS.CABLE.201701-201912.nc']
            land_sen_files= [land_sen_path+var_name+'/LIS.CABLE.201701-201912.nc']
            time, Ctl_tmp = read_var_multi_file(land_ctl_files, var_name, loc_lat, loc_lon, lat_names, lon_names)
            time, Sen_tmp = read_var_multi_file(land_sen_files, var_name, loc_lat, loc_lon, lat_names, lon_names)

        if var_name in ["SurfTmax","Tmax","VegTmax"]:
            # average of daily max
            ctl_in       = spatial_var_max(time,Ctl_tmp,time_s,time_e)
            sen_in       = spatial_var_max(time,Sen_tmp,time_s,time_e)
        elif var_name in ["SurfTmin","Tmin","VegTmin"]:
            # average of daily min
            ctl_in       = spatial_var_min(time,Ctl_tmp,time_s,time_e)
            sen_in       = spatial_var_min(time,Sen_tmp,time_s,time_e)
        elif var_name in ["SM_top50cm",]:
            # top 1m soil moisture [.022, .058, .154, .409, 1.085, 2.872]
            c_tmp        = (Ctl_tmp[:,0,:,:]*0.022 + Ctl_tmp[:,1,:,:]*0.058 + Ctl_tmp[:,2,:,:]*0.154 + Ctl_tmp[:,3,:,:]*0.266)/0.5
            s_tmp        = (Sen_tmp[:,0,:,:]*0.022 + Sen_tmp[:,1,:,:]*0.058 + Sen_tmp[:,2,:,:]*0.154 + Sen_tmp[:,3,:,:]*0.266)/0.5
            ctl_in       = spatial_var_min(time,c_tmp,time_s,time_e)
            sen_in       = spatial_var_min(time,s_tmp,time_s,time_e)
        else:
            ctl_in       = spatial_var(time,Ctl_tmp,time_s,time_e)
            sen_in       = spatial_var(time,Sen_tmp,time_s,time_e)

        wrf            = Dataset(wrf_path,  mode='r')
        lons           = wrf.variables['XLONG'][0,:,:]
        lats           = wrf.variables['XLAT'][0,:,:]

        if var_name in ['WaterTableD_tavg','WatTable']:
            ctl_in     = ctl_in/1000.
            sen_in     = sen_in/1000.
        if var_name in ['ESoil_tavg','Evap_tavg',"ECanop_tavg",'TVeg_tavg',"Rainf_tavg","Snowf_tavg","Qs_tavg","Qsb_tavg"]:
            t_s        = time_s - datetime(2000,1,1,0,0,0,0)
            t_e        = time_e - datetime(2000,1,1,0,0,0,0)
            ctl_in     = ctl_in*3600*24 #*(t_e.days - t_s.days)
            sen_in     = sen_in*3600*24 #*(t_e.days - t_s.days)
        if var_name in ['Qair_f_inst']:
            ctl_in     = ctl_in*1000
            sen_in     = sen_in*1000
        if var_name in ['GPP_tavg','NPP_tavg']:
            t_s        = time_s - datetime(2000,1,1,0,0,0,0)
            t_e        = time_e - datetime(2000,1,1,0,0,0,0)
            s2d        = 3600*24.          # s-1 to d-1
            GPP_scale  = -0.000001*12*s2d   # umol s-1 to g d-1
            ctl_in     = ctl_in*GPP_scale #*(t_e.days - t_s.days)
            sen_in     = sen_in*GPP_scale #*(t_e.days - t_s.days)
        var_diff     = sen_in - ctl_in

        # read lat and lon outs

        if var_name in ['WaterTableD_tavg','WatTable']:
            clevs = [-4,-3,-2,-1.5,-1,-0.5,0.5,1,1.5,2,3,4]
        elif var_name in ['GWwb_tavg','GWMoist']:
            clevs = [-0.05,-0.04,-0.03,-0.02,-0.01,-0.005,0.005,0.01,0.02,0.03,0.04,0.05]
        elif  var_name in ["Qair_f_inst"]:
            clevs = [-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]
            #clevs = [-2.,-1.8,-1.6,-1.4,-1.2,-1.,-0.6,-0.4,-0.2,0.2,0.4,0.6,0.8,1.,1.2,1.4,1.6,1.8,2]
        elif var_name in ['SoilMoist_inst','SoilMoist',"SM_top50cm"]:
            clevs = [-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0.05,0.1,0.15,0.2,0.25,0.3]
        elif var_name in ['ESoil_tavg','Evap_tavg',"ECanop_tavg",'TVeg_tavg',"Rainf_tavg","Snowf_tavg","Qs_tavg","Qsb_tavg"]:
            # clevs = [-30,-28,-26,-24,-22,-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]
            clevs = [-5.,-4.5,-4.,-3.5,-3.,-2.5,-2,-1.5,-1,-0.5,0.5,1.,1.5,2.,2.5,3.,3.5,4.,4.5,5.]
            # clevs = [-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,-5,5,10,20.,30,40,50,60,70,80,90,100]
            # clevs = [-140,-120,-100,-80,-60,-40,-20,20,40,60,80,100,120,140]
        elif var_name in ["GPP_tavg","NPP_tavg",]:
            # clevs = [-200,-190,-180,-170,-160,-150,-140,-130,-120,-110,-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,
            #          -5,5,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
            clevs = [-2.0,-1.8,-1.6,-1.4,-1.2,-1.0,-0.8,-0.6,-0.4,-0.2,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0]
        elif var_name in ["CanopInt_inst","SnowCover_inst"]:
            clevs = [-2.,-1.8,-1.6,-1.4,-1.2,-1.,-0.8,-0.6,-0.4,-0.2,0.2,0.4,0.6,0.8,1.,1.2,1.4,1.6,1.8,2.]
            # clevs = [-4,-3,-2,-1.5,-1,-0.5,0.5,1,1.5,2,3,4]
        elif var_name in ["Qle_tavg","Qh_tavg","Qg_tavg","Rnet",]:
            clevs = [-140,-120,-100,-80,-60,-40,-20,-10,-5,5,10,20,40,60,80,100,120,140]
        elif var_name in ["Swnet_tavg","Lwnet_tavg","SWdown_f_inst","LWdown_f_inst"]:
            clevs = [-22,-18,-14,-10,-6,-2,2,6,10,14,18,22]
        elif var_name in ["Wind_f_inst",]:
            clevs = [-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4]
        elif var_name in ["Psurf_f_inst"]:
            clevs = [-22,-18,-14,-10,-6,-2,2,6,10,14,18,22]
        elif var_name in ["Tair_f_inst","Tmax","Tmin","VegT_tavg","VegTmax","VegTmin",
                          "AvgSurfT_tavg","SurfTmax","SurfTmin","SoilTemp_inst",]:
            # clevs = [-2.,-1.8,-1.6,-1.4,-1.2,-1.,-0.8,-0.6,-0.4,-0.2,0.2,0.4,0.6,0.8,1.,1.2,1.4,1.6,1.8,2.]
            clevs = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.]
            # clevs = [-3,-2.5,-2,-1.5,-1,-0.5,-0.1,0.1,0.5,1.,1.5,2,2.5,3.]
        elif var_name in ["Wind_f_inst",]:
            clevs = [-2.,-1.5,-1,-0.5,-0.1,0.1,0.5,1.,1.5,2.]
        elif var_name in ["FWsoil_tavg","SmLiqFrac_inst","SmFrozFrac_inst"]:
            clevs = [-0.5,-0.45,-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
        elif var_name in ["LAI_inst"]:
            clevs = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        elif var_name in ["Albedo_inst"]:
            clevs = [-0.05,-0.045,-0.04,-0.035,-0.03,-0.025,-0.02,-0.015,-0.01,-0.005,0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05]
        else:
            clevs = [-0.5,-0.4,-0.3,-0.2,-0.1,-0.05,0.05,0.1,0.2,0.3,0.4,0.5]

        clevs_percentage =  [-50,-45,-40,-35,-30,-25,-20,-15,-10,-5,5,10,15,20,25,30,35,40,45,50]

        print("len(np.shape(var_diff))",len(np.shape(var_diff)))

        if len(np.shape(var_diff)) >=3:

            for j in np.arange(6):

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
                cmap  = plt.cm.seismic

                #hot_r # BrBG

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
                    gl.xlocator  = mticker.FixedLocator([135,140,145,150,155])
                    gl.ylocator  = mticker.FixedLocator([-40,-35,-30,-25])
                else:
                    gl.xlocator  = mticker.FixedLocator(loc_lon)
                    gl.ylocator  = mticker.FixedLocator(loc_lat)

                gl.xformatter = LONGITUDE_FORMATTER
                gl.yformatter = LATITUDE_FORMATTER
                gl.xlabel_style = {'size':10, 'color':'black'}
                gl.ylabel_style = {'size':10, 'color':'black'}
                plt.contourf(lons, lats, var_diff[j,:,:], clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend='both') #

                cb = plt.colorbar(ax=ax, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
                cb.ax.tick_params(labelsize=10)
                plt.title(var_name, size=16)

                if shape_path != None:
                    # Requires the pyshp package
                    sf = shp.Reader(shape_path)

                    for shape in sf.shapeRecords():
                        x = [i[0] for i in shape.shape.points[:]]
                        y = [i[1] for i in shape.shape.points[:]]
                        plt.plot(x,y,c="black")

                if j == 0:
                    if message == None:
                        message = var_name
                    else:
                        message = message + "_" + var_name

                plt.savefig('./plots/spatial_map_'+message+'_layer='+str(j)+'.png',dpi=300)
                cb = None
                gl = None
                ax = None
                fig= None

        elif len(np.shape(var_diff)) ==2:
            # ================== Start Plotting =================
            # fig = plt.figure(figsize=(6,5))
            # ax = plt.axes(projection=ccrs.PlateCarree())

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

            # =============== CHANGE HERE ===============
            # choose colormap
            cmap  = plt.cm.seismic

            #hot_r # BrBG
            # for i in np.arange(2):
            #     # start plotting
            #     if loc_lat == None:
            #         axs[i].set_extent([135,155,-40,-25])
            #     else:
            #         axs[i].set_extent([loc_lon[0],loc_lon[1],loc_lat[0],loc_lat[1]])

            #     axs[i].coastlines(resolution="50m",linewidth=1)

            #     # Add gridlines
            #     gl = axs[i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=1, color='black', linestyle='--')     
            #     gl.xlabels_bottom= True
            #     gl.ylabels_left  = True       
            #     gl.xlabels_top   = False
            #     gl.ylabels_right = False
            #     gl.xlines        = True

            #     if loc_lat == None:
            #         gl.xlocator  = mticker.FixedLocator([135,140,145,150,155])
            #         gl.ylocator  = mticker.FixedLocator([-40,-35,-30,-25])
            #     else:
            #         gl.xlocator  = mticker.FixedLocator(loc_lon)
            #         gl.ylocator  = mticker.FixedLocator(loc_lat)

            #     gl.xformatter = LONGITUDE_FORMATTER
            #     gl.yformatter = LATITUDE_FORMATTER
            #     gl.xlabel_style = {'size':10, 'color':'black'}
            #     gl.ylabel_style = {'size':10, 'color':'black'}


            for i in np.arange(2):

                axs[i].coastlines(resolution="50m",linewidth=1)
                axs[i].set_extent([135,155,-39,-23])
                axs[i].add_feature(states, linewidth=.5, edgecolor="black")

                # Add gridlines
                gl = axs[i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color=almost_black, linestyle='--')
                gl.xlabels_top  = False
                gl.ylabels_right= False
                gl.xlines       = True
                gl.ylines       = True
                gl.xlocator     = mticker.FixedLocator(np.arange(125,160,1))
                gl.ylocator     = mticker.FixedLocator(np.arange(-40,-20,1))
                # gl.xlocator     = mticker.FixedLocator([130,135,140,145,150,155,160])
                # gl.ylocator     = mticker.FixedLocator([-40,-35,-30,-25,-20])
                gl.xformatter   = LONGITUDE_FORMATTER
                gl.yformatter   = LATITUDE_FORMATTER
                gl.xlabel_style = {'size':12, 'color':almost_black}#,'rotation': 90}
                gl.ylabel_style = {'size':12, 'color':almost_black}

                gl.xlabels_bottom = True
                gl.ylabels_left   = True

            # print("any(not np.isnan(var_diff))",any(not np.isnan(var_diff)))
            plot1 = axs[0].contourf(lons, lats, var_diff, clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
            cbar = plt.colorbar(plot1, ax=axs[0], ticklocation="right", pad=0.08, orientation="horizontal",
                    aspect=40, shrink=1) # cax=cax,
            cbar.ax.tick_params(labelsize=8)
            plt.title(var_name, size=16)
            if shape_path != None:
                # Requires the pyshp package
                sf = shp.Reader(shape_path)

                for shape in sf.shapeRecords():
                    x = [i[0] for i in shape.shape.points[:]]
                    y = [i[1] for i in shape.shape.points[:]]
                    plt.plot(x,y,c="black")

            rate = np.where( ctl_in != 0, var_diff/ctl_in, np.nan)
            plot2 = axs[1].contourf(lons, lats, rate*100., clevs_percentage, transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
            cbar  = plt.colorbar(plot2, ax=axs[1], ticklocation="right", pad=0.08, orientation="horizontal",
                    aspect=40, shrink=1) # cax=cax,
            cbar.ax.tick_params(labelsize=8)
            plt.title(var_name+"_diff_percentage", size=16)

            if shape_path != None:
                # Requires the pyshp package
                sf = shp.Reader(shape_path)

                for shape in sf.shapeRecords():
                    x = [i[0] for i in shape.shape.points[:]]
                    y = [i[1] for i in shape.shape.points[:]]
                    plt.plot(x,y,c="black")

            plt.savefig('./plots/spatial_map_'+message + "_" + var_name+'.png',dpi=300)

def spatial_map_total_soil_water_diff(file_paths, lis_path, loc_lat=None, loc_lon=None, wrf_path=None,message=None):

    '''
    calculate the total soil water change to estimate water table depth changes
    '''

    # read WRF-CABLE
    var_file   = Dataset(file_paths[0], mode='r')
    wb1        = var_file.variables['SoilMoist_inst'][0,:,:,:]
    gwwb1      = var_file.variables['GWwb_tavg'][0,:,:]

    # read LIS_RST
    rst_file   = Dataset(file_paths[1], mode='r')
    wb2        = rst_file.variables['SoilMoist_inst'][-1,:,:,:]
    gwwb2      = rst_file.variables['GWwb_tavg'][-1,:,:]

    # calculate soil water storage
    lis_file   = Dataset(lis_path, mode='r')
    dtb        = lis_file.variables['DTB']

    sws1       = wb1[0,:,:]*0.005 + wb1[1,:,:]*0.075 + wb1[2,:,:]*0.154 + wb1[3,:,:]*0.409 + \
                 wb1[4,:,:]*1.085 + wb1[5,:,:]*2.872 + gwwb1*dtb

    sws2       = wb2[0,:,:]*0.005 + wb2[1,:,:]*0.075 + wb2[2,:,:]*0.154 + wb2[3,:,:]*0.409 + \
                 wb2[4,:,:]*1.085 + wb2[5,:,:]*2.872 + gwwb2*dtb

    sws_diff   = sws1-sws2

    # read lat and lon outs
    wrf        = Dataset(wrf_path,  mode='r')
    lons       = wrf.variables['XLONG'][0,:,:]
    lats       = wrf.variables['XLAT'][0,:,:]

    clevs      = [-4,-3,-2,-1.5,-1,-0.5,0.5,1,1.5,2,3,4]

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
    cmap  = plt.cm.seismic_r

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
        gl.xlocator  = mticker.FixedLocator([135,140,145,150,155])
        gl.ylocator  = mticker.FixedLocator([-40,-35,-30,-25])
    else:
        gl.xlocator  = mticker.FixedLocator(loc_lon)
        gl.ylocator  = mticker.FixedLocator(loc_lat)

    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size':10, 'color':'black'}
    gl.ylabel_style = {'size':10, 'color':'black'}

    plt.contourf(lons, lats, sws_diff, clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend='both') #

    cb = plt.colorbar(ax=ax, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
    cb.ax.tick_params(labelsize=10)
    plt.title(message, size=16)

    if message == None:
        message = 'Soil_watar_storage_diff'
    else:
        message = message + "_" + 'Soil_watar_storage_diff'

    plt.savefig('./plots/weather/spatial_map_'+message+'.png',dpi=300)

def spatial_map_land_info_diff(file_paths, loc_lat=None, loc_lon=None, wrf_path=None,message=None):

    '''
    compare land information
    '''

    var_names = ['SiltFrac_inst','silt'] #['ClayFrac_inst','clay']
    #['SandFrac_inst','sand'] #['sand_vec','SAND'] #['Sucs_inst','sucs'] # ['SoilSat_inst', 'ssat']

    if var_names[0] in ['Sucs_inst','sucs','sand_vec','SAND']:
        loop = 6
    else:
        loop = 1

    # read WRF-CABLE
    var_file   = Dataset(file_paths[0], mode='r')
    var1       = var_file.variables[var_names[0]][0,:,:] #

    # read off-CABLE
    off_file   = Dataset(file_paths[1], mode='r')
    var2       = off_file.variables[var_names[1]]#[:,:,:]
    print(file_paths[1])
    print(var2)
    print(off_file)
    lats_in    = off_file.variables['latitude'][:]
    lons_in    = off_file.variables['longitude'][:]

    # read lat and lon outs
    if wrf_path == None:
        lats       = var_file.variables['lat']
        lons       = var_file.variables['lon']
        lons_out, lats_out = np.meshgrid(lons, lats)
    else:
        wrf        = Dataset(wrf_path,  mode='r')
        lons_out   = wrf.variables['XLONG'][0,:,:]
        lats_out   = wrf.variables['XLAT'][0,:,:]

    nlat       = len(lons_out[:,0])
    nlon       = len(lons_out[0,:])


    var2_regrid = regrid_data(lats_in, lons_in, lats_out, lons_out, var2) #[j,:,:]
    if loop > 1:
        var_diff   = np.zeros((loop,nlat,nlon))
        for j in np.arange(loop):
            var_diff[j,:,:] = var1[j,:,:] - var2_regrid
    else:
        var_diff = var1 - var2_regrid

    if var_names[0] in ['SoilSat_inst', 'ssat']:
        clevs      = [-0.03,-0.02,-0.01,-0.005,0.005,0.01,0.02,0.03]
    elif var_names[0] in ['SAND','sand_vec','SandFrac_inst','ClayFrac_inst','SiltFrac_inst']:
        clevs      = [-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0.05,0.1,0.15,0.2,0.25,0.3]
    elif var_names[0] in ['Sucs_inst','sucs']:
        clevs      = [-100.,-80,-60,-40,-20,-10,10,20,40,60,80,100]

    # ================== Start Plotting =================
    for j in np.arange(loop):
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
        cmap  = plt.cm.seismic_r

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
            gl.xlocator  = mticker.FixedLocator([135,140,145,150,155])
            gl.ylocator  = mticker.FixedLocator([-40,-35,-30,-25])
        else:
            gl.xlocator  = mticker.FixedLocator(loc_lon)
            gl.ylocator  = mticker.FixedLocator(loc_lat)

        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size':10, 'color':'black'}
        gl.ylabel_style = {'size':10, 'color':'black'}

        if loop >1:
            plt.contourf(lons_out, lats_out, var_diff[j,:,:],  clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
        else:
            plt.contourf(lons_out, lats_out, var_diff,  clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend='both')

        cb = plt.colorbar(ax=ax, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
        cb.ax.tick_params(labelsize=10)
        plt.title(message, size=16)
        if j == 0:
            if message == None:
                message = var_names[0]+'_diff'
            else:
                message = message +"_"+ var_names[0]+'_diff'
        if loop > 1:
            plt.savefig('./plots/weather/spatial_map_'+message+'_layer='+str(j)+'.png',dpi=300)
        else:
            plt.savefig('./plots/weather/spatial_map_'+message+'.png',dpi=300)

def spatial_map_wrf_hgt(file_paths, var_name, height, time_s, time_e, var_unit=None, loc_lat=None, loc_lon=None, message=None):

    time, var_tmp = read_wrf_hgt_var_multi_files(file_paths, var_name, var_unit, height, loc_lat, loc_lon)

    scale      = get_scale(var_name)
    var        = spatial_var(time,var_tmp,time_s,time_e)*scale

    # Get the lat/lon coordinates
    ncfile     = Dataset(file_paths[0])
    pressure   = getvar(ncfile, "pressure", timeidx=ALL_TIMES)
    lats, lons = latlon_coords(pressure)

    # Get the cartopy mapping object
    cart_proj  = get_cartopy(pressure)

    # Create the figure
    fig = plt.figure(figsize=(12,9))
    fig.subplots_adjust(hspace=0.3)
    fig.subplots_adjust(wspace=0.2)

    plt.rcParams['text.usetex']     = False
    plt.rcParams['font.family']     = "sans-serif"
    plt.rcParams['font.serif']      = "Helvetica"
    plt.rcParams['axes.linewidth']  = 1.5
    plt.rcParams['axes.labelsize']  = 14
    plt.rcParams['font.size']       = 14
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
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

    # Set the GeoAxes to the projection used by WRF
    ax = plt.axes(projection=cart_proj)

    # Download and add the states and coastlines
    states = NaturalEarthFeature(category="cultural", scale="50m",
                                         facecolor="none",
                                         name="admin_1_states_provinces_shp")
    ax.add_feature(states, linewidth=.5, edgecolor="black")
    ax.coastlines('50m', linewidth=0.8)

    # start plotting
    if loc_lat == None:
        ax.set_extent([135,155,-40,-25])
    else:
        ax.set_extent([loc_lon[0],loc_lon[1],loc_lat[0],loc_lat[1]])

    # Set the map bounds
    ax.set_xlim(cartopy_xlim(pressure))
    ax.set_ylim(cartopy_ylim(pressure))

    # Add the var contours

    var_contours = plt.contourf(to_np(lons), to_np(lats), to_np(var),
                   transform=ccrs.PlateCarree(), cmap=get_cmap("bwr"),extend='both') #,"jet" #“rainbow”#"coolwarm" levels = levels,
    plt.colorbar(var_contours, ax=ax, orientation="horizontal", pad=.05)

    if var_unit == None:
        plt.title(str(height)+"hPa, "+var_name)
    else:
        plt.title(str(height)+"hPa, Geopotential Height (gpm),"+var_name+" ("+var_unit+") and Barbs (m s-1)")

    if message == None:
        message = var_name+'_'+str(height)+"hPa"
    else:
        message = message+"_"+var_name+'_'+str(height)+"hPa"

    fig.savefig('./plots/wrf_output/spatial_map_wrf_hgt_'+message , bbox_inches='tight', pad_inches=0.1)

def spatial_map_wrf_surf(file_paths, var_name, time_s, time_e, loc_lat=None, loc_lon=None, message=None):

    # Open the NetCDF file
    time, Var = read_wrf_surf_var_multi_files(file_paths, var_name, loc_lat, loc_lon)

    if var_name in ['T2','td2']:
        var   = spatial_var(time,Var,time_s,time_e)-273.15
    else:
        scale = get_scale(var_name)
        var   = spatial_var(time,Var,time_s,time_e)*scale

    # Get the lat/lon coordinates
    ncfile    = Dataset(file_paths[0])
    pressure  = getvar(ncfile, "pressure", timeidx=ALL_TIMES)
    lats, lons= latlon_coords(pressure)

    # Get the cartopy mapping object
    cart_proj = get_cartopy(pressure)

    # Create the figure
    fig = plt.figure(figsize=(12,9))
    fig.subplots_adjust(hspace=0.3)
    fig.subplots_adjust(wspace=0.2)

    plt.rcParams['text.usetex']     = False
    plt.rcParams['font.family']     = "sans-serif"
    plt.rcParams['font.serif']      = "Helvetica"
    plt.rcParams['axes.linewidth']  = 1.5
    plt.rcParams['axes.labelsize']  = 14
    plt.rcParams['font.size']       = 14
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
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

    # Set the GeoAxes to the projection used by WRF
    ax = plt.axes(projection=cart_proj)

    # Download and add the states and coastlines
    states = NaturalEarthFeature(category="cultural", scale="50m",
                                         facecolor="none",
                                         name="admin_1_states_provinces_shp")
    ax.add_feature(states, linewidth=.5, edgecolor="black")
    ax.coastlines('50m', linewidth=0.8)

    # start plotting
    if loc_lat == None:
        ax.set_extent([135,155,-40,-25])
    else:
        ax.set_extent([loc_lon[0],loc_lon[1],loc_lat[0],loc_lat[1]])

    # levels = np.linspace(-10.,10.,num=21)
    cmap = plt.cm.seismic

    var_contours = plt.contourf(to_np(lons), to_np(lats), to_np(var),
                    transform=ccrs.PlateCarree(), cmap=cmap,extend='both') #,"jet" #“rainbow”#"coolwarm" levels = levels[levels!=0]
    plt.colorbar(var_contours, ax=ax, orientation="horizontal", pad=.05)

    # Set the map bounds
    ax.set_xlim(cartopy_xlim(pressure))
    ax.set_ylim(cartopy_ylim(pressure))

    plt.title(var_name)

    if message == None:
        message = var_name
    else:
        message = message+"_"+var_name

    fig.savefig('./plots/wrf_output/spatial_wrf_surf_'+message , bbox_inches='tight', pad_inches=0.1)

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
        Test WRF-CABLE output
        '''

        case_name      = "ALB-CTL_new" #"bl_pbl2_mp4_sf_sfclay2" #
        case_ctl       = "drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2"
        case_sen       = "drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2_obs_LAI_ALB"

        wrf_path       = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_ctl+"/WRF_output/wrfout_d01_2017-02-01_06:00:00"
        land_sen_path  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_sen+"/LIS_output/"
        land_ctl_path  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_ctl+"/LIS_output/"
        atmo_sen_path  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_sen+"/WRF_output/"
        atmo_ctl_path  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_ctl+"/WRF_output/"

        land_sen_files = [land_sen_path+"LIS.CABLE.201701-201701.d01.nc",]

        land_ctl_files = [land_ctl_path+"LIS.CABLE.201701-201701.d01.nc",]

        # atmo_sen_files = [ atmo_sen_path+"wrfout_d01_2018-01-01_11:00:00",]
        # atmo_ctl_files = [ atmo_ctl_path+"wrfout_d01_2018-01-01_11:00:00",]

        if 1:
            '''
            Difference plot yearly
            '''

            var_names  = [
                        #   "GPP_tavg","NPP_tavg",
                        #   "Tmax","Tmin","VegTmax","VegTmin",
                        #   "TVeg_tavg","VegT_tavg","Tair_f_inst",
                        #   "Evap_tavg","ESoil_tavg",
                        #   "Qle_tavg","Qh_tavg","Qg_tavg",
                        #   "LAI_inst",
                          "Albedo_inst","FWsoil_tavg",
                          "AvgSurfT_tavg","SurfTmax","SurfTmin",
                          "Rainf_tavg",
                          "Rnet",
                          # "SM_top50cm",
                          ]

            # period     = "2019_preHW_8_12Jan"
            # time_s     = datetime(2019,1,8,0,0,0,0)
            # time_e     = datetime(2019,1,13,0,0,0,0)
            # message    = case_name+"_"+period
            # spatial_map_single_plot_LIS_diff(land_ctl_path, land_sen_path, var_names, time_s=time_s, time_e=time_e, lat_names="lat",
            #                     lon_names="lon",loc_lat=loc_lat, loc_lon=loc_lon, wrf_path=wrf_path, shape_path=shape_path,
            #                     message=message)

            period     = "2019_HW_14_18Jan"
            time_s     = datetime(2019,1,14,0,0,0,0)
            time_e     = datetime(2019,1,19,0,0,0,0)
            message    = case_name+"_"+period
            spatial_map_single_plot_LIS_diff(land_ctl_path, land_sen_path, var_names, time_s=time_s, time_e=time_e, lat_names="lat",
                                lon_names="lon",loc_lat=loc_lat, loc_lon=loc_lon, wrf_path=wrf_path, shape_path=shape_path,
                                message=message)

            period     = "2019_HW_22_26Jan"
            time_s     = datetime(2019,1,22,0,0,0,0)
            time_e     = datetime(2019,1,27,0,0,0,0)
            message    = case_name+"_"+period
            spatial_map_single_plot_LIS_diff(land_ctl_path, land_sen_path, var_names, time_s=time_s, time_e=time_e, lat_names="lat",
                                lon_names="lon",loc_lat=loc_lat, loc_lon=loc_lon, wrf_path=wrf_path, shape_path=shape_path,
                                message=message)
        
            period     = "2019_postHW_28_1Feb"
            time_s     = datetime(2019,1,28,0,0,0,0)
            time_e     = datetime(2019,2,2,0,0,0,0)
            message    = case_name+"_"+period
            spatial_map_single_plot_LIS_diff(land_ctl_path, land_sen_path, var_names, time_s=time_s, time_e=time_e, lat_names="lat",
                                lon_names="lon",loc_lat=loc_lat, loc_lon=loc_lon, wrf_path=wrf_path, shape_path=shape_path,
                                message=message)
                                