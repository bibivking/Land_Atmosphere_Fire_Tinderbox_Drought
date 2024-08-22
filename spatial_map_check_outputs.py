#!/usr/bin/python

__author__ = "Mengyuan Mu"
__email__  = "mu.mengyuan815@gmail.com" 

'''
Functions:
1. Analyze weather situation during heatwaves
2. Process ERAI, AWAP, offline CABLE and LIS-CABLE data
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
# from spatial_wrf_hgt_var import plot_spatial_map_hgt

def select_case(time, Var, var_name, time_s, time_e, month=None, diff=False):

    if var_name in ['t2m','tas','Tair','Tair_f_inst']:
        if diff == False:
            clevs    = np.linspace( 15.,45., num=31)
            cmap     = plt.cm.RdYlBu_r
        else:
            clevs    = [-5,-4,-3,-2,-1,-0.5,0.5,1,2,3,4,5] # np.linspace( -5., 5., num=11)
            cmap     = plt.cm.RdYlBu_r
        var      = spatial_var(time,Var,time_s,time_e)-273.15
        
    elif var_name in ['Rainf','Rainf_tavg','tp']:
        if diff == False:
            clevs    = np.linspace( 0., 200., num=11)
            cmap     = plt.cm.Blues
        else:
            clevs    = [-180,-160,-140,-120,-100,-80,-60,-40,-20,20,40,60,80,100,120,140,160,180]
            cmap     = plt.cm.BrBG #RdYlBu_r
        if month == 1:
            var      = spatial_var(time,Var,time_s,time_e)*24*60*60.*30
        elif month == 2:
            var      = spatial_var(time,Var,time_s,time_e)*24*60*60.*28
        elif month == 3:
            var      = spatial_var(time,Var,time_s,time_e)*24*60*60.*31
        else:
            var      = spatial_var(time,Var,time_s,time_e)*24*60*60.
            
    elif var_name in ['LWdown','LWdown_f_inst','SWdown','SWdown_f_inst']:
        if diff == False:
            clevs = np.arange( 80.,520.,20.) #np.linspace( 15.,45., num=31)
        else:
            clevs = np.arange( -90.,100.,10.)
        cmap  = plt.cm.BrBG_r
        scale = get_scale(var_name)
        var   = spatial_var(time,Var,time_s,time_e)*scale
        
    elif var_name in ['Wind','Wind_f_inst']:
        if diff == False:
            clevs = np.arange( 0,10.5,0.5) #np.linspace( 15.,45., num=31)
            var   = spatial_var(time,Var,time_s,time_e)*2
        else:
            clevs = np.arange( -5,5.5,0.5)
            var   = spatial_var(time,Var,time_s,time_e)
        cmap  = plt.cm.BrBG
        
    elif var_name in ['Qair','Qair_f_inst']:
        # kg kg-1
        if diff == False:
            clevs = np.arange( 0.,0.02, 0.001) #np.linspace( 15.,45., num=31)
        else:
            clevs = np.arange( -0.006,0.007, 0.001)
        cmap  = plt.cm.BrBG
        scale = get_scale(var_name)
        var   = spatial_var(time,Var,time_s,time_e)*scale
        
    elif var_name in ['Qle_tavg','Qh_tavg']:
        # W m-2
        if diff == False:
            clevs = np.arange( 0.,220, 20.) #np.linspace( 15.,45., num=31)
        else:
            clevs = np.arange( -100.,110., 10.)
        cmap  = plt.cm.RdYlBu_r``
        scale = get_scale(var_name)
        var   = spatial_var(time,Var,time_s,time_e)*scale
        
    else:
        if diff == False:
            clevs = np.linspace( 0.,5., num=11)
        else:
            clevs = np.linspace( -5.,5., num=11)
        cmap  = plt.cm.GnBu # BrBG
        scale = get_scale(var_name)
        var   = spatial_var(time,Var,time_s,time_e)*scale
        
    return var, clevs, cmap 

def plot_check_spital_map(wrf_path, file_paths, var_names, time_s, time_e, 
                          loc_lat=None, loc_lon=None, lat_names=None, lon_names=None, 
                          message=None, metric=False, month=None):

    print("======== In plot_check_spital_map =========")

    # ================== Plot setting ==================
    fig, ax = plt.subplots( nrows=3, ncols=2, figsize=[12,10],sharex=True, sharey=True, squeeze=True,
                            subplot_kw={'projection': ccrs.PlateCarree()})
    plt.subplots_adjust(wspace=0, hspace=0.2) 

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

        
    # ================== Reading data =================
    
    for i, file_path in enumerate(file_paths):
        row = i//2
        col = i%2
        print("row=",row, "col=",col)
        time, Var  = read_var(file_path, var_names[i], loc_lat, loc_lon, lat_names[i], lon_names[i])
        time, lats = read_var(file_path, lat_names[i], loc_lat, loc_lon, lat_names[i], lon_names[i])
        time, lons = read_var(file_path, lon_names[i], loc_lat, loc_lon, lat_names[i], lon_names[i])
        var, clevs, cmap = select_case(time, Var, var_names[i], time_s, time_e, month=month)

        # =============== setting plots ===============
        if loc_lat == None:
            ax[row,col].set_extent([135,155,-40,-25])
        else:
            ax[row,col].set_extent([loc_lon[0],loc_lon[1],loc_lat[0],loc_lat[1]])

        ax[row,col].coastlines(resolution="50m",linewidth=1)

        # Add gridlines
        gl              = ax[row,col].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=1, color='black', linestyle='--')

        if loc_lat == None:
            gl.xlocator = mticker.FixedLocator([135,140,145,150,155])
            gl.ylocator = mticker.FixedLocator([-40,-35,-30,-25])
        else:
            gl.xlocator = mticker.FixedLocator([135,140,144.2,150,155])
            gl.ylocator = mticker.FixedLocator([-40,-35,-31.8,-25])

        gl.xformatter    = LONGITUDE_FORMATTER
        gl.yformatter    = LATITUDE_FORMATTER
        gl.xlabel_style  = {'size':10, 'color':'black'}
        gl.ylabel_style  = {'size':10, 'color':'black'}
        gl.xlabels_bottom= True
        gl.xlabels_top   = False
        gl.ylabels_left  = True
        gl.ylabels_right = False
        gl.xlines        = True
        gl.ylines        = True
        plot1            = ax[row, col].contourf(lons, lats, var, clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend='both') 
        cb               = plt.colorbar(plot1, ax=ax[row, col], orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
        cb.ax.tick_params(labelsize=7)
        ax[row, col].set_title(var_names[i], size=12)

    plt.savefig('./plots/weather/spatial_map_check_output_'+message+'.png',dpi=300)

if __name__ == "__main__":

    # ======================= Plot region =======================
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
        
    # ====================== Pre-load =======================
    ERA5_path    = "/g/data/rt52/era5/single-levels/reanalysis"
    ERA5_T_file  = ERA5_path + '/2t/2017/2t_era5_oper_sfc_20170301-20170331.nc' # air temperatur


    # #################################
    # Plot WRF-CABLE - AWAP
    # #################################
    metric       = False
    year         = 2017
    AWAP_path    = '/g/data/w97/W35_GDATA_MOVED/Shared_data/AWAP_3h_v1'
    AWAP_T_file  = AWAP_path + '/Tair/AWAP.Tair.3hr.'+str(year)+'.nc'     # air temperature
    AWAP_R_file  = AWAP_path + '/Rainf/AWAP.Rainf.3hr.'+str(year)+'.nc'   # Daily rainfall
    
    wrf_path     = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/drght_2017_2019/run_Jan2017/WRF_output/wrfout_d01_2017-01-01_11:00:00"
    cable_path   = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/drght_2017_2019/LIS_output/"   
    
    
    for month in np.arange(1,2):
        
        message    = "WRF-AWAP_"+str(year)+"-"+ "%02d" % month
        print(message)
        
        cable_file = cable_path+"LIS.CABLE."+str(year)+"%02d" % month+"-"+str(year)+"%02d" % month+".d01.nc"
        time_s     = datetime(year,month,1,0,0,0,0) 
        time_e     = datetime(year,month+1,1,0,0,0,0)         

        # ============= Tair =============
        file_paths  = [ AWAP_T_file, cable_file, AWAP_R_file, cable_file, cable_file, cable_file]
        var_names   = ['Tair',     'Tair_f_inst', 'Rainf',   'Rainf_tavg', 'Qle_tavg', 'Qh_tavg']
        lat_names   = ['lat', 'lat', 'lat', 'lat', 'lat', 'lat']
        lon_names   = ['lon', 'lon', 'lon', 'lon', 'lon', 'lon']

        plot_check_spital_map(wrf_path, file_paths, var_names, time_s, time_e, 
                          loc_lat=loc_lat, loc_lon=loc_lon, lat_names=lat_names, lon_names=lon_names, 
                          message=message, metric=False, month=month)

  
 