#!/usr/bin/python

__author__ = "Mengyuan Mu"
__email__  = "mu.mengyuan815@gmail.com" 

'''
Functions:
1. Compare LIS-CABLE with GRACE, GLEAM, & DOLCE
2. GW vs FD
3. plot time-series and spitial (difference) map
'''

from netCDF4 import Dataset
import netCDF4 as nc
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy.interpolate import griddata
import matplotlib.ticker as mticker
from convert_units import get_land_var_scale, get_land_var_scale_offline
from common_utils import *
from spatial_map_weather_analysis import spital_map_temperal_metrics, plot_map_temperal_metrics


def calc_cable_tws(file_path, loc_lat, loc_lon, lat_name, lon_name):

    # calculate TWS for CABLE simulations

    print("calc_cable_tws")

    Soil_thickness = [0.022, 0.058, 0.154, 0.409, 1.085, 2.872]
    off_file       = Dataset(file_path, mode='r')
    GWdz           = off_file.variables['GWdz'][:]
    SoilMoist      = off_file.variables['SoilMoist'][:]

    time1, CanopInt= read_var(file_path, "CanopInt", loc_lat, loc_lon, lat_name, lon_name)

    time2, GWMoist = read_var(file_path, "GWMoist", loc_lat, loc_lon, lat_name, lon_name)
    GWMoist        = GWMoist*GWdz*1000.

    time3, SWE     = read_var(file_path, "SWE", loc_lat, loc_lon, lat_name, lon_name)
    print('np.shape(SWE)')
    print(np.shape(SWE))

    TWS            = GWMoist + CanopInt + SWE
    print('np.shape(TWS)')
    print(np.shape(TWS))

    for i in np.arange(6):
        TWS = TWS + SoilMoist[:,i,:,:]*Soil_thickness[i]*1000
    
    year_2004       = datetime(2004,1,1)
    year_2009       = datetime(2009,12,31)

    TWS_0409 = spatial_var(time1,TWS,year_2004,year_2009)
    print('np.shape(TWS_0409)')
    print(np.shape(TWS_0409))
    TWS      = TWS - TWS_0409
    print('np.shape(TWS)')
    print(np.shape(TWS))
    return time1,TWS

def plot_spital_map(file_paths, var_names, time_s, time_e, file_paths2=None, loc_lat=None, loc_lon=None, 
                    lat_names=None, lon_names=None, message=None):

    '''
    Plot either value or difference
    '''
    print("======== In plot_spital_map =========")

    # Open the NetCDF4 file (add a directory path if necessary) for reading:
    
    time1, Var1     = read_var_multi_file(file_paths, var_names[0], loc_lat, loc_lon, lat_names[0], lon_names[0])
    time_tmp, lats1 = read_var(file_paths[0], lat_names[0], loc_lat, loc_lon, lat_names[0], lon_names[0])
    time_tmp, lons1 = read_var(file_paths[0], lon_names[0], loc_lat, loc_lon, lat_names[0], lon_names[0])
    
    if var_names[0] in ['Evap_tavg','TVeg_tavg','ESoil_tavg','ECanop_tavg']:
        var1        = spatial_var_sum(time1,Var1,time_s,time_e)*3600.
    else:
        scale           = get_scale(var_names[0])
        var1        = spatial_var_mean(time1,Var1,time_s,time_e)*scale
        
    if file_paths2 is not None:
        
        time2, Var2     = read_var_multi_file(file_paths2, var_names[1], loc_lat, loc_lon, lat_names[1], lon_names[1])
        time_tmp, lats2 = read_var(file_paths2[0], lat_names[1], loc_lat, loc_lon, lat_names[1], lon_names[1])
        time_tmp, lons2 = read_var(file_paths2[0], lon_names[1], loc_lat, loc_lon, lat_names[1], lon_names[1])
        scale           = get_scale(var_names[1])
        if var_names[1] in ['E','Et','Ei','Es']:
            var2            = spatial_var_sum(time2,Var2,time_s,time_e)
        else:
            var2            = spatial_var_mean(time2,Var2,time_s,time_e)*scale

        lat_in_1D       = lats2.flatten()
        lon_in_1D       = lons2.flatten()
        var2_in_1D      = var2.flatten()
        
        lat_in_1D       = lat_in_1D[~np.isnan(lat_in_1D)]  # here I make nan in values as the standard
        lon_in_1D       = lon_in_1D[~np.isnan(lon_in_1D)]
        var2_in_1D      = var2_in_1D[~np.isnan(var2_in_1D)]
        
        var2_regrid     = griddata(lat_in_1D,lon_in_1D, var2_in_1D, (lats1, lons1), method='nearest') # 'linear' 'cubic'
        var             = var1 - var2_regrid
    else:
        var             = var1
    # np.savetxt("test_var.txt",var,delimiter=",")

    fig = plt.figure(figsize=(6,5))
    ax = plt.axes(projection=ccrs.PlateCarree())

    if loc_lat == None:
        ax.set_extent([140,154,-40,-28])
    else:
        ax.set_extent([loc_lon[0],loc_lon[1],loc_lat[0],loc_lat[1]])

    ax.coastlines(resolution="50m",linewidth=1)
    # Add gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=1, color='black', linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlines = True

    if loc_lat == None:
        gl.xlocator = mticker.FixedLocator([140,145,150])
        gl.ylocator = mticker.FixedLocator([-40,-35,-30])
    else:
        gl.xlocator = mticker.FixedLocator(loc_lon)
        gl.ylocator = mticker.FixedLocator(loc_lat)

    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size':10, 'color':'black'}
    gl.ylabel_style = {'size':10, 'color':'black'}
    # Plot windspeed

    # clevs = np.linspace( 0.,1500., num=31)
    plt.contourf(lons1, lats1, var,  transform=ccrs.PlateCarree(), extend='both',cmap=plt.cm.BrBG) # clevs,
    plt.title(var_names[0], size=16)
    cb = plt.colorbar(ax=ax, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
    # cb.set_label(units,size=14,rotation=270,labelpad=15)
    cb.ax.tick_params(labelsize=10)
    if message == None:
        message = var_names[0]
    else:
        message = message + "_" + var_names[0]
    plt.savefig('./plots/spatial_map_obs_'+message+'.png',dpi=300)
    
def plot_time_series( file_paths, var_names, year_s, year_e, loc_lat=None, loc_lon=None,
                      lat_names=None, lon_names=None, message=None ):

    print("======== In plot_time_series =========")

    fig, ax = plt.subplots()

    # plot line 1
    if var_names[0] == "TWS":
        Time1, Var1 = calc_cable_tws(file_paths[0], loc_lat, loc_lon, lat_names[0], lon_names[0])
        time1, var1 = time_series_var(Time1, Var1, year_s, year_e)
    elif var_names[0] == "lwe_thickness":
        Time1, Var1 = read_var(file_paths[0], var_names[0], loc_lat, loc_lon, lat_names[0], lon_names[0])
        time1, var1 = time_series_var(Time1, Var1*10., year_s, year_e)
    else:
        Time1, Var1 = read_var(file_paths[0], var_names[0], loc_lat, loc_lon, lat_names[0], lon_names[0])
        time1, var1 = time_series_var(Time1, Var1, year_s, year_e) # np.cumsum()

    t1 = []
    for i in np.arange(len(time1)):
        t1.append(time1[i].days)
    if var_names[0] in ['Qs','Qsb','Rainf','Evap','ESoil','ECanop','TVeg']:
        scale = 24.*3600.
    else:
        scale = 1.
    print("var1*scale")
    print(var1*scale)
    ax.plot(t1, var1*scale, c = "blue", label="GW", alpha=0.5)


    # plot line 2
    if len(file_paths) > 1:
        if var_names[1] == "TWS":
            Time2, Var2 = calc_cable_tws(file_paths[1], loc_lat, loc_lon, lat_names[1], lon_names[1])
            time2, var2 = time_series_var(Time2, Var2, year_s, year_e)
        elif var_names[1] == "lwe_thickness":
            Time2, Var2 = read_var(file_paths[1], var_names[1], loc_lat, loc_lon, lat_names[1], lon_names[1])
            time2, var2 = time_series_var(Time2, Var2*10., year_s, year_e)
        else:
            Time2, Var2 = read_var(file_paths[1], var_names[1], loc_lat, loc_lon, lat_names[1], lon_names[1])
            time2, var2 = time_series_var(Time2, Var2, year_s, year_e) #np.cumsum()
        t2 = []
        for i in np.arange(len(time2)):
            t2.append(time2[i].days)
        if var_names[1] in ['Qs','Qsb','Rainf','Evap','ESoil','ECanop','TVeg']:
            scale = 24.*3600.
        else:
            scale = 1.

        print("var2*scale")
        print(var2*scale)
        ax.plot(t2, var2*scale, c = "green", label="FD", alpha=0.5)

    # plot line 3
    if len(file_paths) > 2:
        if var_names[2] == "TWS":
            Time3, Var3 = calc_cable_tws(file_paths[2], loc_lat, loc_lon, lat_names[2], lon_names[2])
            time3, var3 = time_series_var(Time3, Var3, year_s, year_e)
        elif var_names[2] == "lwe_thickness":
            Time3, Var3 = read_var(file_paths[2], var_names[2], loc_lat, loc_lon, lat_names[2], lon_names[2])
            time3, var3 = time_series_var(Time3, Var3*10., year_s, year_e)
        else:
            Time3, Var3 = read_var(file_paths[2], var_names[2], loc_lat, loc_lon, lat_names[2], lon_names[2])
            time3, var3 = time_series_var(Time3, Var3, year_s, year_e) # np.cumsum()
        t3 = []
        for i in np.arange(len(time3)):
            t3.append(time3[i].days)
        if var_names[2] in ['Qs','Qsb','Rainf','Evap','ESoil','ECanop','TVeg']:
            scale = 24.*3600.
        else:
            scale = 1.

        print("var3*scale")
        print(var3*scale)
        ax.plot(t3, var3*scale, c = "red", label="FD", alpha=0.5)

    # ax.set_xlim([np.min(var1*scale,var2*scale), np.max(var1*scale,var2*scale)])
    # Time2, Var2 = read_var(file_paths[1], var_names[1], loc_lat, loc_lon, lat_name[1], lon_name[1])
    # time2, var2 = time_series_var(Time2,Var2,year_s,year_e)
    # var = np.zeros((2,len(var1)))
    # var[0,:] = var1
    # var[1,:] = var2
    # ax.plot(t1, var*scale, alpha=0.5)
    # ax.set_ylabel('mm')
    ax.set_title(var_names[0])
    # ax.set_xticks(x1[::1440])
    # ax.set_xticklabels(np.arange(year_s,year_e,1))
    ax.legend()

    fig.tight_layout()
    if message == None:
        message = var_names[0]
    else:
        message = message + "_" + var_names[0]
    if loc_lat != None:
        message = message + "_lat="+str(loc_lat[0]) +"-"+str(loc_lat[1]) + "_lon="+str(loc_lon[0])+"-"+str(loc_lon[1])

    plt.savefig('./plots/19Oct/time_series_lis_vs_off_'+message+'.png',dpi=300)

if __name__ == "__main__":

    # #######################
    #     path setting      #
    # #######################

    DOLCE_path = "/g/data/w97/mm3972/data/DOLCE/v3/"
    DOLCE_file = DOLCE_path+"DOLCE_v3_2000-2018.nc"
    GLEAM_path = "/g/data/w97/W35_GDATA_MOVED/Shared_data/Observations/Global_ET_products/GLEAM_v3_3/3_3a/monthly/"
    GLEAM_file = GLEAM_path + "E_1980_2018_GLEAM_v3.3a_MO.nc"
    GRACE_path = "/g/data/w97/mm3972/data/GRACE/GRACE_JPL_RL06/GRACE_JPLRL06M_MASCON/"
    GRACE_file = GRACE_path + "GRCTellus.JPL.200204_202004.GLO.RL06M.MSCNv02CRI.nc"

    GW_off_path = "/g/data/w97/mm3972/model/cable/runs/runs_4_coupled/gw_after_sp30yrx3/outputs/"
    GW_off_file = GW_off_path + "cable_out_2000-2019.nc"

    FD_off_path = "/g/data/w97/mm3972/model/cable/runs/runs_4_coupled/fd_after_sp30yrx3/outputs/"
    FD_off_file = FD_off_path + "cable_out_2000-2019.nc"

    SP1_off_path = "/g/data/w97/mm3972/model/cable/runs/runs_4_coupled/spinup_30yrx3/outputs_1st/"
    SP2_off_path = "/g/data/w97/mm3972/model/cable/runs/runs_4_coupled/spinup_30yrx3/outputs_2nd/"
    SP3_off_path = "/g/data/w97/mm3972/model/cable/runs/runs_4_coupled/spinup_30yrx3/outputs_3rd/"

    SP1_off_file = SP1_off_path + "cable_out_1970-1999.nc"
    SP2_off_file = SP2_off_path + "cable_out_1970-1999.nc"
    SP3_off_file = SP3_off_path + "cable_out_1970-1999.nc"
    
    LIS_path     = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/drght_2017_2019_bl_pbl2_mp4_sf_sfclay2/LIS_output/"

    # #######################
    #        variable       #
    # #######################


    # #######################
    #   plot_spital_map     #
    # #######################
    if 1:
        year_s      = datetime(2017,1,1)
        year_e      = datetime(2018,7,1)
        loc_lat     = [-40,-28]
        loc_lon     = [140,154]

        file_paths1 = [ LIS_path+"LIS.CABLE.201701-201701.d01.nc",
                        LIS_path+"LIS.CABLE.201702-201702.d01.nc",
                        LIS_path+"LIS.CABLE.201703-201703.d01.nc",
                        LIS_path+"LIS.CABLE.201704-201704.d01.nc",
                        LIS_path+"LIS.CABLE.201705-201705.d01.nc",
                        LIS_path+"LIS.CABLE.201706-201706.d01.nc",
                        LIS_path+"LIS.CABLE.201707-201707.d01.nc",
                        LIS_path+"LIS.CABLE.201708-201708.d01.nc",
                        LIS_path+"LIS.CABLE.201709-201709.d01.nc",
                        LIS_path+"LIS.CABLE.201710-201710.d01.nc",
                        LIS_path+"LIS.CABLE.201711-201711.d01.nc",
                        LIS_path+"LIS.CABLE.201712-201712.d01.nc",
                        LIS_path+"LIS.CABLE.201801-201801.d01.nc",
                        LIS_path+"LIS.CABLE.201802-201802.d01.nc",
                        LIS_path+"LIS.CABLE.201803-201803.d01.nc",
                        LIS_path+"LIS.CABLE.201804-201804.d01.nc",
                        LIS_path+"LIS.CABLE.201805-201805.d01.nc",
                        LIS_path+"LIS.CABLE.201806-201806.d01.nc"  ]

        ## plot Evap vs GLEAM ###
        print("plot Evap vs GLEAM")
        file_paths2 = [GLEAM_file]
        var_name    = ["Evap_tavg","E"]
        lat_names   = ["lat","lat"]#"lat"
        lon_names   = ["lon","lon"]#"lon"
        message     = "LIS-GLEAM_Evap_2017"

        plot_spital_map(file_paths1, var_name, year_s, year_e, file_paths2=file_paths2, loc_lat=loc_lat, loc_lon=loc_lon, lat_names=lat_names,
                        lon_names=lon_names,message=message)
 
        ### plot Qle vs DOLCE ###
        print("plot Qle vs DOLCE")
        file_paths2 = [DOLCE_file]
        var_name    = ["Qle_tavg","hfls"]
        lat_names   = ["lat","lat"]#"lat"
        lon_names   = ["lon","lon"]#"lon"
        message     = "LIS-DOLCE_Qle_2017"

        plot_spital_map(file_paths1, var_name, year_s, year_e, file_paths2=file_paths2, loc_lat=loc_lat, loc_lon=loc_lon, lat_names=lat_names,
                        lon_names=lon_names,message=message)        




    # #################################
    #   plot_map_temperal_metrics     #
    # #################################
    if 1:
        file_paths1 = [ LIS_path+"LIS.CABLE.201701-201701.d01.nc",
                        LIS_path+"LIS.CABLE.201702-201702.d01.nc",
                        LIS_path+"LIS.CABLE.201703-201703.d01.nc",
                        LIS_path+"LIS.CABLE.201704-201704.d01.nc",
                        LIS_path+"LIS.CABLE.201705-201705.d01.nc",
                        LIS_path+"LIS.CABLE.201706-201706.d01.nc",
                        LIS_path+"LIS.CABLE.201707-201707.d01.nc",
                        LIS_path+"LIS.CABLE.201708-201708.d01.nc",
                        LIS_path+"LIS.CABLE.201709-201709.d01.nc",
                        LIS_path+"LIS.CABLE.201710-201710.d01.nc",
                        LIS_path+"LIS.CABLE.201711-201711.d01.nc",
                        LIS_path+"LIS.CABLE.201712-201712.d01.nc",
                        LIS_path+"LIS.CABLE.201801-201801.d01.nc",
                        LIS_path+"LIS.CABLE.201802-201802.d01.nc",
                        LIS_path+"LIS.CABLE.201803-201803.d01.nc",
                        LIS_path+"LIS.CABLE.201804-201804.d01.nc",
                        LIS_path+"LIS.CABLE.201805-201805.d01.nc",
                        LIS_path+"LIS.CABLE.201806-201806.d01.nc"  ]

    ####################################
    #         plot_time_series         #
    ####################################
    if 0:
        year_s       = datetime(2000,1,1)
        year_e       = datetime(2019,12,31)
        loc_lat      = [-40,-28]
        loc_lon      = [140,154]

        # var_names       = [ ["GWMoist","GWMoist"],
        #                     ["Evap","Evap"],
        #                     ["TVeg","TVeg"],
        #                     ["ESoil","ESoil"],
        #                     ["ECanop","ECanop"],
        #                     ["Qs","Qs"],
        #                     ["Qsb","Qsb"],
        #                     ["WatTable","WatTable"],
        #                     ["Qle","Qle"],
        #                     ["Qh","Qh"],
        #                     ["Qg","Qg"],
        #                     ["RadT","RadT"],
        #                     ["VegT","VegT"],
        #                     ["Fwsoil","Fwsoil"]]

        # file_paths  = [GW_off_file, FD_off_file]
        
        # for var_name in var_names:
        #     lat_names = ["latitude","latitude"]
        #     lon_names = ["longitude","longitude"]
        #     message   = "GW_vs_FD"
        #     plot_time_series(file_paths, var_name, year_s, year_e, loc_lat=loc_lat, loc_lon=loc_lon,
        #                      lat_names=lat_names, lon_names=lon_names, message=message)

        # file_paths= [GRACE_file, GW_off_file, FD_off_file]
        # var_names = ["lwe_thickness","TWS","TWS"]

        # GLEAM_vs_CABLE
        file_paths= [GLEAM_file, GW_off_file, FD_off_file]
        var_names = ["E","Evap","Evap"]    
        lat_names = ["lat","latitude","latitude"]
        lon_names = ["lon","longitude","longitude"]
        message   = "GLEAM_vs_CABLE"
        plot_time_series(file_paths, var_names, year_s, year_e, loc_lat=loc_lat, loc_lon=loc_lon,
                        lat_names=lat_names, lon_names=lon_names, message=message)

        # DOLCE_vs_CABLE
        # file_paths= [DOLCE_file, GW_off_file, FD_off_file]
        # var_names = ["hfls","Qle","Qle"]    
        # lat_names = ["lat","latitude","latitude"]
        # lon_names = ["lon","longitude","longitude"]
        # message   = "DOLCE_vs_CABLE"
        # plot_time_series(file_paths, var_names, year_s, year_e, loc_lat=loc_lat, loc_lon=loc_lon,
        #                  lat_names=lat_names, lon_names=lon_names, message=message)
                    
