import sys
import cartopy
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.patches import Polygon
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from scipy.interpolate import griddata, interp1d
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature, OCEAN
from common_utils import *

def plot_LAI_MODIS(LAI_MODIS_path):

    # =========== Read in data ===========
    LAI_file   = Dataset(LAI_MODIS_path, mode='r')
    LAI_input  = LAI_file.variables['LAI'][:]
    lat        = LAI_file.variables['latitude'][:]
    lon        = LAI_file.variables['longitude'][:]
    time       = nc.num2date( LAI_file.variables['time'][:], LAI_file.variables['time'].units,
                              only_use_cftime_datetimes=False, only_use_python_datetimes=True )

    # =========== Plotting ============
    # for i in np.arange( 1, len(time) ):

    #     print('time[i]', time[i])

    fig, ax    = plt.subplots(nrows=1, ncols=1, figsize=[5,4],sharex=True, sharey=True, squeeze=True,
                                subplot_kw={'projection': ccrs.PlateCarree()})

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
    cmap    = plt.cm.BrBG

    ax.coastlines(resolution="50m",linewidth=1)
    ax.set_extent([135,155,-39,-23])
    ax.add_feature(states, linewidth=.5, edgecolor="black")

    # Add gridlines
    gl              = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color=almost_black, linestyle='--')
    gl.xlabels_top  = False
    gl.ylabels_right= False
    gl.xlines       = False
    gl.ylines       = False
    gl.xlocator     = mticker.FixedLocator(np.arange(125,160,1))
    gl.ylocator     = mticker.FixedLocator(np.arange(-40,-20,1))
    gl.xformatter   = LONGITUDE_FORMATTER
    gl.yformatter   = LATITUDE_FORMATTER
    gl.xlabel_style = {'size':12, 'color':almost_black}#,'rotation': 90}
    gl.ylabel_style = {'size':12, 'color':almost_black}

    gl.xlabels_bottom = True
    gl.ylabels_left   = True

    clevs_diff     = [-0.5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5]

    # left: LAI_obs_mean
    plot           = ax.contourf( lon, lat, LAI_input[9] - LAI_input[1], levels=clevs_diff,
                                    transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
    # ax.text(0.02, 0.15, time[i], transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    ax.add_feature(OCEAN,edgecolor='none', facecolor="lightgray")
    cbar = plt.colorbar(plot, ax=ax, ticklocation="right", pad=0.08, orientation="horizontal",
                        aspect=40, shrink=0.6)

    cbar.ax.tick_params(labelsize=8, labelrotation=45)

    plt.savefig('./plots/spatial_map_LAI_'+ str(time[9]) +'.png',dpi=300)

    return

def plot_fire_map(fire_path):

    day_in_month = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]

    # =========== Read in data ===========
    fire_file  = Dataset(fire_path, mode='r')
    Burn_Date  = fire_file.variables['Burn_Date'][:]
    First_Day  = fire_file.variables['First_Day'][:]
    Last_Day   = fire_file.variables['Last_Day'][:]
    lat        = fire_file.variables['lat'][:]
    lon        = fire_file.variables['lon'][:]
    time       = nc.num2date( fire_file.variables['time'][:], fire_file.variables['time'].units,
                              only_use_cftime_datetimes=False, only_use_python_datetimes=True )

    Burn_Date  = np.where(Burn_Date<=0, np.nan, Burn_Date)
    First_Day  = np.where(First_Day<=0, np.nan, First_Day)
    Last_Day   = np.where(Last_Day<=0, np.nan, Last_Day)

    # ============= Plotting ==============
    for i in np.arange( len(time) ):

        print('time[i].month', time[i].month)

        fig, ax    = plt.subplots(nrows=1, ncols=1, figsize=[5,4],sharex=True, sharey=True, squeeze=True,
                                    subplot_kw={'projection': ccrs.PlateCarree()})

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
        cmap    = plt.cm.BrBG
        cmap.set_bad(color='lightgrey')

        ax.coastlines(resolution="50m",linewidth=1)
        ax.set_extent([135,155,-39,-23])
        ax.add_feature(states, linewidth=.5, edgecolor="black")

        # Add gridlines
        gl              = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color=almost_black, linestyle='--')
        gl.xlabels_top  = False
        gl.ylabels_right= False
        gl.xlines       = False
        gl.ylines       = False
        gl.xlocator     = mticker.FixedLocator(np.arange(125,160,1))
        gl.ylocator     = mticker.FixedLocator(np.arange(-40,-20,1))
        gl.xformatter   = LONGITUDE_FORMATTER
        gl.yformatter   = LATITUDE_FORMATTER
        gl.xlabel_style = {'size':12, 'color':almost_black}#,'rotation': 90}
        gl.ylabel_style = {'size':12, 'color':almost_black}

        gl.xlabels_bottom = True
        gl.ylabels_left   = True

        month_int = int(time[i].month)

        # clevs = np.arange(day_in_month[month_int-1]+1, day_in_month[month_int]+1)
        # clevs_diff = [-5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5]

        plot  = ax.contourf( lon, lat, Burn_Date[i,:,:]-day_in_month[month_int-1],transform=ccrs.PlateCarree(), cmap=cmap, extend='both') #levels=clevs, # levels=clevs_diff,
        # ax.text(0.02, 0.15, time[i], transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
        ax.add_feature(OCEAN,edgecolor='none', facecolor="lightgray")
        cbar  = plt.colorbar(plot, ax=ax, ticklocation="right", pad=0.08, orientation="horizontal",
                            aspect=40, shrink=0.6)

        cbar.ax.tick_params(labelsize=8, labelrotation=45)

        plt.savefig('./plots/spatial_map_fire_Burn_Date_'+ str(time[i]) +'.png',dpi=300)

    return

def plot_LAI_fire_map(fire_path,LAI_MODIS_path):

    # =========== Read in fire data ============
    fire_file  = Dataset(fire_path, mode='r')
    Burn_Date  = fire_file.variables['Burn_Date'][4:8,:,:]  # 2019-11 - 2020-02
    lat_out    = fire_file.variables['lat'][:]
    lon_out    = fire_file.variables['lon'][:]

    time_fire  = nc.num2date( fire_file.variables['time'][:], fire_file.variables['time'].units,
                              only_use_cftime_datetimes=False, only_use_python_datetimes=True )
    nlat       = len(lat_out)
    nlon       = len(lon_out)

    # =========== Read in MODIS data ===========
    LAI_file   = Dataset(LAI_MODIS_path, mode='r')
    LAI_in     = LAI_file.variables['LAI'][:]
    lat_in     = LAI_file.variables['latitude'][:]
    lon_in     = LAI_file.variables['longitude'][:]
    time       = nc.num2date( LAI_file.variables['time'][:], LAI_file.variables['time'].units,
                              only_use_cftime_datetimes=False, only_use_python_datetimes=True )
    ntime      = len(time)
    LAI_regrid = np.zeros((ntime,nlat,nlon))

    for i in np.arange(ntime):
        LAI_regrid[i,:,:] = regrid_data(lat_in, lon_in, lat_out, lon_out, LAI_in[i,:,:], method='nearest',threshold=0)

    # ============= Plotting ==============
    fig, ax    = plt.subplots(nrows=1, ncols=1, figsize=[5,4],sharex=True, sharey=True, squeeze=True,
                                subplot_kw={'projection': ccrs.PlateCarree()})

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
    cmap    = plt.cm.BrBG
    cmap.set_bad(color='lightgrey')

    ax.coastlines(resolution="50m",linewidth=1)
    ax.set_extent([135,155,-39,-23])
    ax.add_feature(states, linewidth=.5, edgecolor="black")

    # Add gridlines
    gl              = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color=almost_black, linestyle='--')
    gl.xlabels_top  = False
    gl.ylabels_right= False
    gl.xlines       = False
    gl.ylines       = False
    gl.xlocator     = mticker.FixedLocator(np.arange(125,160,1))
    gl.ylocator     = mticker.FixedLocator(np.arange(-40,-20,1))
    gl.xformatter   = LONGITUDE_FORMATTER
    gl.yformatter   = LATITUDE_FORMATTER
    gl.xlabel_style = {'size':12, 'color':almost_black}#,'rotation': 90}
    gl.ylabel_style = {'size':12, 'color':almost_black}

    gl.xlabels_bottom = True
    gl.ylabels_left   = True

    clevs_diff     = [-0.5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5]

    # Nov changes
    # LAI_plot       = np.where(Burn_Date[0,:,:] + Burn_Date[1,:,:]> 0, LAI_regrid[9,:,:] - LAI_regrid[0,:,:], np.nan)
    LAI_plot       = np.where( Burn_Date[1,:,:]> 0, LAI_regrid[9,:,:] - LAI_regrid[4,:,:], np.nan)
    plot           = ax.contourf( lon_out, lat_out, LAI_plot, levels=clevs_diff, transform=ccrs.PlateCarree(), cmap=cmap, extend='both') #

    # ax.text(0.02, 0.15, time[i], transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    ax.add_feature(OCEAN,edgecolor='none', facecolor="lightgray")
    cbar = plt.colorbar(plot, ax=ax, ticklocation="right", pad=0.08, orientation="horizontal",
                        aspect=40, shrink=0.6)

    cbar.ax.tick_params(labelsize=8, labelrotation=45)

    plt.savefig('./plots/spatial_map_fire_Burnt_LAI_Dec.png',dpi=300)

def read_LIS_diff(var_name,file_name,land_ctl_path,land_sen_path, lat_names, lon_names,time_s=None,time_e=None):

    print("plotting "+var_name)

    if var_name in ["Tmax","Tmin","TDR"]:
        land_ctl_files= [land_ctl_path+'Tair_f_inst/'+file_name]
        land_sen_files= [land_sen_path+'Tair_f_inst/'+file_name]
        time, Ctl_tmp = read_var_multi_file(land_ctl_files, 'Tair_f_inst', loc_lat, loc_lon, lat_names, lon_names)
        time, Sen_tmp = read_var_multi_file(land_sen_files, 'Tair_f_inst', loc_lat, loc_lon, lat_names, lon_names)
        Ctl_tmp       = Ctl_tmp -273.15
        Sen_tmp       = Sen_tmp -273.15
    elif var_name in ["VegTmax","VegTmin","VegTDR"]:
        land_ctl_files= [land_ctl_path+'VegT_tavg/'+file_name]
        land_sen_files= [land_sen_path+'VegT_tavg/'+file_name]
        time, Ctl_tmp = read_var_multi_file(land_ctl_files, 'VegT_tavg', loc_lat, loc_lon, lat_names, lon_names)
        time, Sen_tmp = read_var_multi_file(land_sen_files, 'VegT_tavg', loc_lat, loc_lon, lat_names, lon_names)
        Ctl_tmp       = Ctl_tmp -273.15
        Sen_tmp       = Sen_tmp -273.15
    elif var_name in ["SurfTmax","SurfTmin","SurfTDR"]:
        land_ctl_files= [land_ctl_path+'AvgSurfT_tavg/'+file_name]
        land_sen_files= [land_sen_path+'AvgSurfT_tavg/'+file_name]
        time, Ctl_tmp = read_var_multi_file(land_ctl_files, 'AvgSurfT_tavg', loc_lat, loc_lon, lat_names, lon_names)
        time, Sen_tmp = read_var_multi_file(land_sen_files, 'AvgSurfT_tavg', loc_lat, loc_lon, lat_names, lon_names)
        Ctl_tmp       = Ctl_tmp -273.15
        Sen_tmp       = Sen_tmp -273.15
    elif var_name in ["Rnet",]:
        land_ctl_files= [land_ctl_path+'Lwnet_tavg/'+file_name]
        land_sen_files= [land_sen_path+'Lwnet_tavg/'+file_name]
        time, Ctl_Lwnet_tmp = read_var_multi_file(land_ctl_files, 'Lwnet_tavg', loc_lat, loc_lon, lat_names, lon_names)
        time, Sen_Lwnet_tmp = read_var_multi_file(land_sen_files, 'Lwnet_tavg', loc_lat, loc_lon, lat_names, lon_names)
        land_ctl_files= [land_ctl_path+'Swnet_tavg/'+file_name]
        land_sen_files= [land_sen_path+'Swnet_tavg/'+file_name]
        time, Ctl_Swnet_tmp = read_var_multi_file(land_ctl_files, 'Swnet_tavg', loc_lat, loc_lon, lat_names, lon_names)
        time, Sen_Swnet_tmp = read_var_multi_file(land_sen_files, 'Swnet_tavg', loc_lat, loc_lon, lat_names, lon_names)
        Ctl_tmp = Ctl_Lwnet_tmp+Ctl_Swnet_tmp
        Sen_tmp = Sen_Lwnet_tmp+Sen_Swnet_tmp
    elif var_name in ["SM_top50cm",]:
        land_ctl_files = [land_ctl_path+'SoilMoist_inst/'+file_name]
        land_sen_files = [land_sen_path+'SoilMoist_inst/'+file_name]
        time, Ctl_temp = read_var_multi_file(land_ctl_files, 'SoilMoist_inst', loc_lat, loc_lon, lat_names, lon_names)
        time, Sen_temp = read_var_multi_file(land_sen_files, 'SoilMoist_inst', loc_lat, loc_lon, lat_names, lon_names)
        # [.022, .058, .154, .409, 1.085, 2.872]
        Ctl_tmp    = (Ctl_temp[:,0,:,:]*0.022 + Ctl_temp[:,1,:,:]*0.058 + Ctl_temp[:,2,:,:]*0.154 + Ctl_temp[:,3,:,:]*0.266)/0.5
        Sen_tmp    = (Sen_temp[:,0,:,:]*0.022 + Sen_temp[:,1,:,:]*0.058 + Sen_temp[:,2,:,:]*0.154 + Sen_temp[:,3,:,:]*0.266)/0.5
    elif var_name in ['VPD','VPDmax','VPDmin']:
        tair_ctl_files= [land_ctl_path+'Tair_f_inst/'+file_name]
        tair_sen_files= [land_sen_path+'Tair_f_inst/'+file_name]
        qair_ctl_files= [land_ctl_path+'Qair_f_inst/'+file_name]
        qair_sen_files= [land_sen_path+'Qair_f_inst/'+file_name]
        pres_ctl_files= ['/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/'
                            +'WRF_output/slp/wrfout_201701-202002.nc']
        pres_sen_files= ['/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2_obs_LAI_ALB/'
                            +'WRF_output/slp/wrfout_201701-202002.nc']

        time, Tair_ctl    = read_var_multi_file(tair_ctl_files, "Tair_f_inst", loc_lat, loc_lon, lat_names, lon_names)
        time, Tair_sen    = read_var_multi_file(tair_sen_files, "Tair_f_inst", loc_lat, loc_lon, lat_names, lon_names)
        time, Qair_ctl    = read_var_multi_file(qair_ctl_files, "Qair_f_inst", loc_lat, loc_lon, lat_names, lon_names)
        time, Qair_sen    = read_var_multi_file(qair_sen_files, "Qair_f_inst", loc_lat, loc_lon, lat_names, lon_names)
        time_wrf, Pres_ctl_tmp= read_var_multi_file(pres_ctl_files, "slp", loc_lat, loc_lon, lat_names, lon_names)
        time_wrf, Pres_sen_tmp= read_var_multi_file(pres_sen_files, "slp", loc_lat, loc_lon, lat_names, lon_names)

        time_in = []
        time_out= []
        for t in time_wrf:
            time_in.append(t.total_seconds())
        for t in time:
            time_out.append(t.total_seconds())

        f_ctl             = interp1d(np.array(time_in), Pres_ctl_tmp[:], kind='linear',fill_value='extrapolate', axis=0)
        f_sen             = interp1d(np.array(time_in), Pres_sen_tmp[:],kind='linear', fill_value='extrapolate', axis=0)
        Pres_ctl          = f_ctl(np.array(time_out))
        Pres_sen          = f_sen(np.array(time_out))
        Ctl_tmp           = qair_to_vpd(Qair_ctl, Tair_ctl, Pres_ctl)
        Sen_tmp           = qair_to_vpd(Qair_sen, Tair_sen, Pres_sen)
    else:
        land_ctl_files= [land_ctl_path+var_name+'/'+file_name]
        land_sen_files= [land_sen_path+var_name+'/'+file_name]
        time, Ctl_tmp = read_var_multi_file(land_ctl_files, var_name, loc_lat, loc_lon, lat_names, lon_names)
        time, Sen_tmp = read_var_multi_file(land_sen_files, var_name, loc_lat, loc_lon, lat_names, lon_names)

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

    if var_name in ['WaterTableD_tavg','WatTable']:
        ctl_in     = ctl_in/1000.
        sen_in     = sen_in/1000.
    if var_name in ['ESoil_tavg','Evap_tavg',"ECanop_tavg",'TVeg_tavg',"Rainf_tavg","Snowf_tavg","Qs_tavg","Qsb_tavg"]:
        t_s        = time_s - datetime(2000,1,1,0,0,0,0)
        t_e        = time_e - datetime(2000,1,1,0,0,0,0)
        ctl_in     = ctl_in*3600*24*(t_e.days - t_s.days)
        sen_in     = sen_in*3600*24*(t_e.days - t_s.days)
    if var_name in ['Qair_f_inst']:
        ctl_in     = ctl_in*1000
        sen_in     = sen_in*1000
    if var_name in ['GPP_tavg','NPP_tavg']:
        t_s        = time_s - datetime(2000,1,1,0,0,0,0)
        t_e        = time_e - datetime(2000,1,1,0,0,0,0)
        s2d        = 3600*24.          # s-1 to d-1
        GPP_scale  = -0.000001*12*s2d   # umol s-1 to g d-1
        ctl_in     = ctl_in*GPP_scale*(t_e.days - t_s.days)
        sen_in     = sen_in*GPP_scale*(t_e.days - t_s.days)

    var_diff     = sen_in - ctl_in


    # Select clevs
    cmap  = plt.cm.BrBG
    if var_name in ['WaterTableD_tavg','WatTable']:
        clevs = [-4,-3,-2,-1.5,-1,-0.5,0.5,1,1.5,2,3,4]
    elif var_name in ['GWwb_tavg','GWMoist']:
        clevs = [-0.05,-0.04,-0.03,-0.02,-0.01,-0.005,0.005,0.01,0.02,0.03,0.04,0.05]
    elif  var_name in ["Qair_f_inst"]:
        clevs = [-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]
    elif var_name in ['SoilMoist_inst','SoilMoist',"SM_top50cm"]:
        clevs = [-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0.05,0.1,0.15,0.2,0.25,0.3]
    elif var_name in ['ESoil_tavg','Evap_tavg',"ECanop_tavg",'TVeg_tavg',"Rainf_tavg","Snowf_tavg","Qs_tavg","Qsb_tavg"]:
        clevs = [-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,-5,5,10,20.,30,40,50,60,70,80,90,100]
    elif var_name in ["GPP_tavg","NPP_tavg",]:
        clevs = [-200,-190,-180,-170,-160,-150,-140,-130,-120,-110,-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,
                    -5,5,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
        clevs_percentage =  [-70,-60,-50,-40,-30,-20,-10,10,20,30,40,50,60,70]
        cmap  = plt.cm.BrBG
    elif var_name in ["CanopInt_inst","SnowCover_inst"]:
        clevs = [-2.,-1.8,-1.6,-1.4,-1.2,-1.,-0.8,-0.6,-0.4,-0.2,0.2,0.4,0.6,0.8,1.,1.2,1.4,1.6,1.8,2.]
    elif var_name in ["Qle_tavg","Qh_tavg","Qg_tavg","Rnet",]:
        clevs = [-140,-120,-100,-80,-60,-40,-20,-10,-5,5,10,20,40,60,80,100,120,140]
    elif var_name in ["Swnet_tavg","Lwnet_tavg","SWdown_f_inst","LWdown_f_inst","Rnet"]:
        clevs = [-22,-18,-14,-10,-6,-2,2,6,10,14,18,22]
    elif var_name in ["Wind_f_inst",]:
        clevs = [-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4]
    elif var_name in ["Psurf_f_inst"]:
        clevs = [-22,-18,-14,-10,-6,-2,2,6,10,14,18,22]
    elif var_name in ["Tair_f_inst","Tmax","Tmin","VegT_tavg","VegTmax","VegTmin",
                        "AvgSurfT_tavg","SurfTmax","SurfTmin","SoilTemp_inst",'TDR','VegTDR','SurfTDR']:
        clevs = [-1.2,-1.1,-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.,1.1,1.2]
        cmap  = plt.cm.seismic
    elif var_name in ["Wind_f_inst",]:
        clevs = [-2.,-1.5,-1,-0.5,-0.1,0.1,0.5,1.,1.5,2.]
    elif var_name in ["FWsoil_tavg","SmLiqFrac_inst","SmFrozFrac_inst"]:
        clevs = [-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0.05,0.1,0.15,0.2,0.25,0.3,0.35]
    elif var_name in ["LAI_inst"]:
        clevs = [-2,-1.8,-1.6,-1.4,-1.2,-1.,-0.8,-0.6,-0.4,-0.2,0.2,0.4,0.6,0.8,1.,1.2,1.4,1.6,1.8,2]
        clevs_percentage =  [-70,-60,-50,-40,-30,-20,-10,-5,5,10,20,30,40,50,60,70]
        cmap  = plt.cm.BrBG
    elif var_name in ["VPD","VPDmax","VPDmin",]:
        clevs = [-0.1,-0.09,-0.08,-0.07,-0.06,-0.05,-0.04,-0.03,-0.02,-0.01,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
        cmap  = plt.cm.BrBG
    elif var_name in ["Albedo_inst"]:
        clevs = [-0.08,-0.07,-0.06,-0.05,-0.04,-0.03,-0.02,-0.01,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08]
        clevs_percentage =   [-70,-60,-50,-40,-30,-20,-10,-5,5,10,20,30,40,50,60,70]
        cmap  = plt.cm.BrBG_r
    else:
        clevs = [-0.5,-0.4,-0.3,-0.2,-0.1,-0.05,0.05,0.1,0.2,0.3,0.4,0.5]

    return sen_in, var_diff, clevs, cmap

def regrid_to_fire_map_resolution(fire_path, var_in, lat_in, lon_in, loc_lat=None, loc_lon=None, burn=0):

    # =========== Read in fire data ============
    fire_file  = Dataset(fire_path, mode='r')
    Burn_Date  = fire_file.variables['Burn_Date'][0:8,:,:]  # 2019-07 - 2020-02
    lat_out    = fire_file.variables['lat'][:]
    lon_out    = fire_file.variables['lon'][:]

    var_regrid = regrid_data(lat_in, lon_in, lat_out, lon_out, var_in, method='nearest')

    # burnt region from 2019-07 to 2020-02
    burn_area  = np.where( Burn_Date[0,:,:] + Burn_Date[1,:,:] + Burn_Date[2,:,:] + Burn_Date[3,:,:] +
                           Burn_Date[4,:,:] + Burn_Date[5,:,:] + Burn_Date[6,:,:] + Burn_Date[7,:,:] > 0, 1, Burn_Date[0,:,:])
    if burn == 1:
        # burnt region
        var_regrid = np.where(burn_area==1, var_regrid, np.nan )
    elif burn == 0:
        # all region
        var_regrid = var_regrid
    elif burn == -1:
        # unburnt region
        var_regrid = np.where(burn_area==0, var_regrid, np.nan )

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

def plot_LIS_diff(fire_path, file_name, land_ctl_path, land_sen_path, var_names, time_s=None, time_e=None,
                  lat_names="lat", lon_names="lon",loc_lat=None, loc_lon=None, wrf_path=None, message=None, burn=0):

    '''
    plot LIS variables in burnt / unburnt / all regions
    '''

    # Read in WRF lat and lon
    wrf            = Dataset(wrf_path,  mode='r')
    lon_in         = wrf.variables['XLONG'][0,:,:]
    lat_in         = wrf.variables['XLAT'][0,:,:]

    for var_name in var_names:

        # read in var
        sen_in, var_diff, clevs, cmap = read_LIS_diff(var_name, file_name, land_ctl_path, land_sen_path,
                                                       lat_names, lon_names, time_s, time_e)
        # regrid to burn map resolution ~ 400m
        var_regrid, lats, lons = regrid_to_fire_map_resolution(fire_path, var_diff, lat_in, lon_in, burn=burn)
        sen_regrid, lats, lons = regrid_to_fire_map_resolution(fire_path, sen_in, lat_in, lon_in, burn=burn)

        # ================== Start Plotting =================
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=[8,6],sharex=True,
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

        for i in np.arange(2):

            axs[i].set_facecolor('lightgray')
            axs[i].coastlines(resolution="50m",linewidth=1)
            axs[i].set_extent([146,154,-39,-27])
            axs[i].add_feature(states, linewidth=.5, edgecolor="black")

            # Add gridlines
            gl = axs[i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color=almost_black, linestyle='--')
            gl.xlabels_top  = False
            gl.ylabels_right= False
            gl.xlines       = False
            gl.ylines       = False
            # gl.xlines       = True
            # gl.ylines       = True
            gl.xlocator     = mticker.FixedLocator(np.arange(126,160,2))
            gl.ylocator     = mticker.FixedLocator(np.arange(-40,-20,2))
            # gl.xlocator     = mticker.FixedLocator([130,135,140,145,150,155,160])
            # gl.ylocator     = mticker.FixedLocator([-40,-35,-30,-25,-20])
            gl.xformatter   = LONGITUDE_FORMATTER
            gl.yformatter   = LATITUDE_FORMATTER
            gl.xlabel_style = {'size':12, 'color':almost_black}#,'rotation': 90}
            gl.ylabel_style = {'size':12, 'color':almost_black}

            gl.xlabels_bottom = True
            gl.ylabels_left   = True

        plot1 = axs[0].contourf(lons, lats, var_regrid, clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
        cbar = plt.colorbar(plot1, ax=axs[0], ticklocation="right", pad=0.05, orientation="vertical",
                aspect=35, shrink=0.6) # cax=cax,
        cbar.ax.tick_params(labelsize=10,labelrotation=45)

        plot2 = axs[1].contourf(lons, lats, sen_regrid, transform=ccrs.PlateCarree(), cmap=cmap, extend='both') # clevs_percentage,
        cbar  = plt.colorbar(plot2, ax=axs[1], ticklocation="right", pad=0.05, orientation="vertical",
                aspect=35, shrink=0.6) # cax=cax,
        cbar.ax.tick_params(labelsize=10,labelrotation=45)

        plt.savefig('./plots/spatial_map_' +message + '_' + var_name + '.png',dpi=300)

def plot_LIS_burn_vs_unburn(fire_path, file_name, land_ctl_path, land_sen_path, var_names, time_s=None, time_e=None,
                  lat_names="lat", lon_names="lon",loc_lat=None, loc_lon=None, wrf_path=None, message=None):

    '''
    plot LIS variables in burnt / unburnt / all regions
    '''

    # Read in WRF lat and lon
    wrf            = Dataset(wrf_path,  mode='r')
    lon_in         = wrf.variables['XLONG'][0,:,:]
    lat_in         = wrf.variables['XLAT'][0,:,:]

    for var_name in var_names:

        # read in var
        sen_in, var_diff, clevs, cmap = read_LIS_diff(var_name, file_name, land_ctl_path, land_sen_path,
                                                      lat_names, lon_names, time_s, time_e)

        # regrid to burn map resolution ~ 400m
        var_regrid_burn, lats, lons   = regrid_to_fire_map_resolution(fire_path, var_diff, lat_in, lon_in, burn=1)
        sen_regrid_burn, lats, lons   = regrid_to_fire_map_resolution(fire_path, sen_in, lat_in, lon_in, burn=1)
        var_regrid_unburn, lats, lons = regrid_to_fire_map_resolution(fire_path, var_diff, lat_in, lon_in, burn=-1)
        sen_regrid_unburn, lats, lons = regrid_to_fire_map_resolution(fire_path, sen_in, lat_in, lon_in, burn=-1)

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

        plot1 = axs[0].contourf(lons, lats, var_regrid_burn, clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
        cbar = plt.colorbar(plot1, ax=axs[0], ticklocation="right", pad=0.08, orientation="horizontal",
                aspect=40, shrink=1) # cax=cax,
        cbar.ax.tick_params(labelsize=10,labelrotation=45)

        plot2 = axs[0].contourf(lons, lats, var_regrid_unburn, clevs, transform=ccrs.PlateCarree(), cmap=cmap, alpha=0.5, extend='both')
        cbar = plt.colorbar(plot2, ax=axs[0], ticklocation="right", pad=0.08, orientation="horizontal",
                aspect=40, shrink=1) # cax=cax,
        cbar.ax.tick_params(labelsize=10,labelrotation=45)

        plot3 = axs[1].contourf(lons, lats, sen_regrid_burn, transform=ccrs.PlateCarree(), cmap=cmap, extend='both') # clevs_percentage,
        cbar  = plt.colorbar(plot3, ax=axs[1], ticklocation="right", pad=0.08, orientation="horizontal",
                aspect=40, shrink=1) # cax=cax,
        cbar.ax.tick_params(labelsize=10,labelrotation=45)

        plot4 = axs[1].contourf(lons, lats, sen_regrid_unburn, transform=ccrs.PlateCarree(), cmap=cmap, alpha=0.5, extend='both') # clevs_percentage,
        cbar  = plt.colorbar(plot4, ax=axs[1], ticklocation="right", pad=0.08, orientation="horizontal",
                aspect=40, shrink=1) # cax=cax,
        cbar.ax.tick_params(labelsize=10,labelrotation=45)

        # Check:
        # all_nan = np.where(np.all([np.isnan(var_regrid_burn),np.isnan(var_regrid_unburn)],axis=0), lons, np.nan)
        # plot1 = axs[0].contourf(lons, lats, all_nan, clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
        # cbar = plt.colorbar(plot1, ax=axs[0], ticklocation="right", pad=0.08, orientation="horizontal",
        #         aspect=40, shrink=1) # cax=cax,
        # cbar.ax.tick_params(labelsize=10,labelrotation=45)
        #
        # all_values = np.where(np.all([~ np.isnan(var_regrid_burn), ~ np.isnan(var_regrid_unburn)],axis=0), lons, np.nan)
        # plot3 = axs[1].contourf(lons, lats, all_values, transform=ccrs.PlateCarree(), cmap=cmap, extend='both') # clevs_percentage,
        # cbar  = plt.colorbar(plot3, ax=axs[1], ticklocation="right", pad=0.08, orientation="horizontal",
        #         aspect=40, shrink=1) # cax=cax,
        # cbar.ax.tick_params(labelsize=10,labelrotation=45)

        plt.savefig('./plots/spatial_map_' +message + '_' + var_name + '.png',dpi=300)

def plot_time_series_burn_region(fire_path, wrf_path, file_name, land_ctl_path, land_sen_path, var_name=None,
                          time_s=None, time_e=None, loc_lat=None, loc_lon=None,
                          lat_name=None, lon_name=None, message=None, burn=0):

    # plot multiple lines on one plot

    if var_name == "EF":
        time_ctl, Var_ctl_Qle = read_var_multi_file([land_ctl_path +"Qle_tavg/"+ file_name], "Qle_tavg", loc_lat, loc_lon, lat_name, lon_name)
        time_ctl, Var_ctl_Qh  = read_var_multi_file([land_ctl_path +"Qh_tavg/" + file_name], "Qh_tavg", loc_lat, loc_lon, lat_name, lon_name)
        time_sen, Var_sen_Qle = read_var_multi_file([land_sen_path +"Qle_tavg/"+ file_name], "Qle_tavg", loc_lat, loc_lon, lat_name, lon_name)
        time_sen, Var_sen_Qh  = read_var_multi_file([land_sen_path +"Qh_tavg/ "+ file_name], "Qh_tavg", loc_lat, loc_lon, lat_name, lon_name)
        ctl_QleQh = Var_ctl_Qle+Var_ctl_Qh
        sen_QleQh = Var_sen_Qle+Var_sen_Qh
        Var_ctl = np.where(abs(ctl_QleQh)>0.01, Var_ctl_Qle/ctl_QleQh,np.nan)
        Var_sen = np.where(abs(sen_QleQh)>0.01, Var_sen_Qle/sen_QleQh,np.nan)
    elif var_name in ["Tmax","Tmin"]:
        time_ctl, Var_ctl = read_var_multi_file([land_ctl_path +"Tair_f_inst/"+ file_name], "Tair_f_inst", loc_lat, loc_lon, lat_name, lon_name)
        time_sen, Var_sen = read_var_multi_file([land_sen_path +"Tair_f_inst/"+ file_name], "Tair_f_inst", loc_lat, loc_lon, lat_name, lon_name)
    else:
        time_ctl, Var_ctl = read_var_multi_file([land_ctl_path +var_name+"/"+ file_name], var_name, loc_lat, loc_lon, lat_name, lon_name)
        time_sen, Var_sen = read_var_multi_file([land_sen_path +var_name+"/"+ file_name], var_name, loc_lat, loc_lon, lat_name, lon_name)

    if var_name in ["Tmax"]:
        Var_daily_ctl = time_clip_to_day_max(time_ctl, Var_ctl, time_s, time_e, seconds=None)
        Var_daily_ctl = Var_daily_ctl-273.15
        Var_daily_sen = time_clip_to_day_max(time_sen, Var_sen, time_s, time_e, seconds=None)
        Var_daily_sen = Var_daily_sen-273.15
    elif var_name in ["Tmin"]:
        Var_daily_ctl = time_clip_to_day_min(time_ctl, Var_ctl, time_s, time_e, seconds=None)
        Var_daily_ctl = Var_daily_ctl-273.15
        Var_daily_sen = time_clip_to_day_min(time_sen, Var_sen, time_s, time_e, seconds=None)
        Var_daily_sen = Var_daily_sen-273.15
    else:
        Var_daily_ctl = time_clip_to_day(time_ctl, Var_ctl, time_s, time_e, seconds=None)
        Var_daily_sen = time_clip_to_day(time_sen, Var_sen, time_s, time_e, seconds=None)


    wrf     = Dataset(wrf_path,  mode='r')
    lon_in  = wrf.variables['XLONG'][0,:,:]
    lat_in  = wrf.variables['XLAT'][0,:,:]

    ntime   = np.shape(Var_daily_ctl)[0]
    print("ntime =",ntime)


    for i in np.arange(ntime):
        print("i=",i)
        # regrid to burn map resolution ~ 400m
        if i == 0:
            ctl_regrid_tmp, lats, lons = regrid_to_fire_map_resolution(fire_path, Var_daily_ctl[i,:,:], lat_in, lon_in, loc_lat, loc_lon, burn=burn)
            sen_regrid_tmp, lats, lons = regrid_to_fire_map_resolution(fire_path, Var_daily_sen[i,:,:], lat_in, lon_in, loc_lat, loc_lon, burn=burn)
            nlat = np.shape(ctl_regrid_tmp)[0]
            nlon = np.shape(ctl_regrid_tmp)[1]
            ctl_regrid = np.zeros((ntime, nlat, nlon))
            sen_regrid = np.zeros((ntime, nlat, nlon))
            ctl_regrid[i,:,:] = ctl_regrid_tmp
            sen_regrid[i,:,:] = sen_regrid_tmp
        else:
            ctl_regrid[i,:,:], lats, lons = regrid_to_fire_map_resolution(fire_path, Var_daily_ctl[i,:,:], lat_in, lon_in, loc_lat, loc_lon, burn=burn)
            sen_regrid[i,:,:], lats, lons = regrid_to_fire_map_resolution(fire_path, Var_daily_sen[i,:,:], lat_in, lon_in, loc_lat, loc_lon, burn=burn)

    # Check whether mask right
    if 0:
        fig1, ax1    = plt.subplots(nrows=1, ncols=1, figsize=[5,4],sharex=True, sharey=True, squeeze=True,
                                    subplot_kw={'projection': ccrs.PlateCarree()})

        states= NaturalEarthFeature(category="cultural", scale="50m",
                                            facecolor="none",
                                            name="admin_1_states_provinces_shp")

        # ======================= Set colormap =======================
        cmap    = plt.cm.BrBG
        cmap.set_bad(color='lightgrey')
        ax1.coastlines(resolution="50m",linewidth=1)
        ax1.set_extent([135,155,-39,-23])
        ax1.add_feature(states, linewidth=.5, edgecolor="black")

        ctl_avg = np.nanmean(ctl_regrid, axis=0)

        plot1  = ax1.contourf( lons, lats, ctl_avg, transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
        cbar1  = plt.colorbar(plot1, ax=ax1, ticklocation="right", pad=0.08, orientation="horizontal",
                            aspect=40, shrink=0.6)
        cbar1.ax.tick_params(labelsize=8, labelrotation=45)

        plt.savefig('./plots/spatial_map_check_burn_region_mask.png',dpi=300)


    ctl_time_series = np.nanmean(ctl_regrid,axis=(1,2))
    sen_time_series = np.nanmean(sen_regrid,axis=(1,2))
    ctl_std         = np.nanstd(ctl_regrid, axis=(1,2))
    sen_std         = np.nanstd(sen_regrid, axis=(1,2))

    ctl_low         = ctl_time_series - ctl_std
    ctl_high        = ctl_time_series + ctl_std
    sen_low         = sen_time_series - sen_std
    sen_high        = sen_time_series + sen_std

    cleaner_dates = ["Oct 2019", "Nov 2019", "Dec 2019", "Jan 2020", "Feb 2020"]
    xtickslocs    = [         0,         31,         61,         92,       123 ]

    # ===================== Plotting =====================
    fig, ax = plt.subplots(figsize=[5,3.5])

    # df_ctl  = pd.DataFrame({'ctl': ctl_time_series})
    # df_sen  = pd.DataFrame({'sen': sen_time_series})
    # ax.plot(df_ctl['ctl'].rolling(window=30).mean(), label=var_name+"_ctl", c = "red", alpha=1)
    # ax.plot(df_sen['sen'].rolling(window=30).mean(), label=var_name+"_sen", c = "blue", alpha=1)

    if var_name == "Tmax":
        # bot_val = 15
        # up_val  = 36
        bot_val = -1.
        up_val  = 1.
    elif var_name == "LAI_inst":
        bot_val = 0
        up_val  = 6
    elif var_name == "Albedo_inst":
        bot_val = 0.05
        up_val  = 0.15
    elif var_name == "FWsoil_tavg":
        bot_val = 0.
        up_val  = 1.

    time_steps = np.arange(len(ctl_time_series))


    df_ctl         = pd.DataFrame({'ctl': ctl_time_series})
    df_ctl['low']  = ctl_low
    df_ctl['high'] = ctl_high

    df_sen         = pd.DataFrame({'sen': sen_time_series})
    df_sen['low']  = sen_low
    df_sen['high'] = sen_high

    if var_name == "Tmax":
        ax.axhline(y=0, color='grey', linestyle='--')
        ax.plot(df_sen['sen'].rolling(window=5).mean()-df_ctl['ctl'].rolling(window=5).mean(), label="sen-ctl", c = "red", lw=1., alpha=1)
    else:
        ax.fill_between(time_steps, df_ctl['low'].rolling(window=5).mean(), df_ctl['high'].rolling(window=5).mean(),
                        color="green", edgecolor="none", alpha=0.15)
        ax.fill_between(time_steps, df_sen['low'].rolling(window=5).mean(), df_sen['high'].rolling(window=5).mean(),
                        color="orange", edgecolor="none", alpha=0.15)

        ax.plot(df_ctl['ctl'].rolling(window=5).mean(), label="ctl", c = "green", lw=0.5, alpha=1)
        ax.plot(df_sen['sen'].rolling(window=5).mean(), label="sen", c = "orange", lw=0.5, alpha=1)

    ax.set_ylim(bot_val,up_val)
    # ax.set_xlim(0,time_steps[-1])

    ax.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax.legend()
    fig.tight_layout()

    plt.savefig('./plots/time_series_'+message+'_'+var_name+'.png',dpi=300)

if __name__ == "__main__":
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

    if 0:
        LAI_MODIS_path = "/g/data/w97/mm3972/data/MODIS/MODIS_LAI/MCD15A3H_c61_bigWRFroi_LAI_for_WRF_5000m_201911_202002.nc"
        plot_LAI_MODIS(LAI_MODIS_path)

    if 0:
        fire_path = '/g/data/w97/mm3972/data/MODIS/MODIS_fire/MCD64A1.061_500m_aid0001.nc'
        plot_fire_map(fire_path)

    if 0:
        fire_path = '/g/data/w97/mm3972/data/MODIS/MODIS_fire/MCD64A1.061_500m_aid0001.nc'
        LAI_MODIS_path = "/g/data/w97/mm3972/data/MODIS/MODIS_LAI/MCD15A3H_c61_bigWRFroi_LAI_for_WRF_5000m_201911_202002.nc"
        plot_LAI_fire_map(fire_path,LAI_MODIS_path)

    if 0:
        '''
        Difference plot yearly

        '''
        case_ctl       = "drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2"
        case_sen       = "drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2_obs_LAI_ALB"

        wrf_path       = "/scratch/w97/mm3972/model/NUWRF/Tinderbox_drght_LAI_ALB/output/drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/WRF_output/wrfout_d01_2017-02-01_06:00:00"
        land_sen_path  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_sen+"/LIS_output/"
        land_ctl_path  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_ctl+"/LIS_output/"

        fire_path = '/g/data/w97/mm3972/data/MODIS/MODIS_fire/MCD64A1.061_500m_aid0001.nc'
        var_names  = [  "Tmax","LAI_inst","Albedo_inst","FWsoil_tavg"]

        if 1:

            year         = 2019
            burn_message = "_burnt"
            burn         = 1
            file_name    = 'LIS.CABLE.'+str(year)+'12-'+str(year+1)+'02.nc'

            message      = "HW_Dec"+str(year)+burn_message
            time_s       = datetime(year,12,1,0,0,0,0)
            time_e       = datetime(year+1,1,1,0,0,0,0)
            plot_LIS_diff(fire_path, file_name, land_ctl_path, land_sen_path, var_names, time_s=time_s, time_e=time_e, lat_names="lat",
                          lon_names="lon",loc_lat=loc_lat, loc_lon=loc_lon, wrf_path=wrf_path,
                          message=message, burn=burn)

            message      = "HW_Jan"+str(year+1)+burn_message
            time_s       = datetime(year+1,1,1,0,0,0,0)
            time_e       = datetime(year+1,2,1,0,0,0,0)
            plot_LIS_diff(fire_path, file_name, land_ctl_path, land_sen_path, var_names, time_s=time_s, time_e=time_e, lat_names="lat",
                          lon_names="lon",loc_lat=loc_lat, loc_lon=loc_lon, wrf_path=wrf_path,
                          message=message, burn=burn)

            message      = "HW_Feb"+str(year+1)+burn_message
            time_s       = datetime(year+1,2,1,0,0,0,0)
            time_e       = datetime(year+1,3,1,0,0,0,0)
            plot_LIS_diff(fire_path, file_name, land_ctl_path, land_sen_path, var_names, time_s=time_s, time_e=time_e, lat_names="lat",
                          lon_names="lon",loc_lat=loc_lat, loc_lon=loc_lon, wrf_path=wrf_path,
                          message=message, burn=burn)

        if 0:
            year       = 2019
            file_name  = 'LIS.CABLE.'+str(year)+'12-'+str(year+1)+'02.nc'

            message    = "HW_Dec"+str(year)+"_burn_vs_unburn"
            time_s     = datetime(year,12,1,0,0,0,0)
            time_e     = datetime(year+1,1,1,0,0,0,0)
            plot_LIS_burn_vs_unburn(fire_path, file_name, land_ctl_path, land_sen_path, var_names, time_s=time_s, time_e=time_e, lat_names="lat",
                         lon_names="lon",loc_lat=loc_lat, loc_lon=loc_lon, wrf_path=wrf_path,
                         message=message)

            message    = "HW_Jan"+str(year+1)+"_burn_vs_unburn"
            time_s     = datetime(year+1,1,1,0,0,0,0)
            time_e     = datetime(year+1,2,1,0,0,0,0)
            plot_LIS_burn_vs_unburn(fire_path, file_name, land_ctl_path, land_sen_path, var_names, time_s=time_s, time_e=time_e, lat_names="lat",
                          lon_names="lon",loc_lat=loc_lat, loc_lon=loc_lon, wrf_path=wrf_path,
                          message=message)

            message    = "HW_Feb"+str(year+1)+"_burn_vs_unburn"
            time_s     = datetime(year+1,2,1,0,0,0,0)
            time_e     = datetime(year+1,3,1,0,0,0,0)
            plot_LIS_burn_vs_unburn(fire_path, file_name, land_ctl_path, land_sen_path, var_names, time_s=time_s, time_e=time_e, lat_names="lat",
                          lon_names="lon",loc_lat=loc_lat, loc_lon=loc_lon, wrf_path=wrf_path,
                          message=message)

        if 0:
            year         = 2017
            file_name    = 'LIS.CABLE.'+str(year)+'12-'+str(year+1)+'02.nc'

            burn_message = "_burnt"
            burn         = 1

            message    = "HW_Dec"+str(year)+burn_message
            time_s     = datetime(year,12,1,0,0,0,0)
            time_e     = datetime(year+1,1,1,0,0,0,0)

            plot_LIS_diff(fire_path, file_name, land_ctl_path, land_sen_path, var_names, time_s=time_s, time_e=time_e, lat_names="lat",
                          lon_names="lon",loc_lat=loc_lat, loc_lon=loc_lon, wrf_path=wrf_path,
                          message=message, burn=burn)

            message    = "HW_Jan"+str(year+1)+burn_message
            time_s     = datetime(year+1,1,1,0,0,0,0)
            time_e     = datetime(year+1,2,1,0,0,0,0)

            plot_LIS_diff(fire_path, file_name, land_ctl_path, land_sen_path, var_names, time_s=time_s, time_e=time_e, lat_names="lat",
                          lon_names="lon",loc_lat=loc_lat, loc_lon=loc_lon, wrf_path=wrf_path,
                          message=message, burn=burn)

            message    = "HW_Feb"+str(year+1)+burn_message
            time_s     = datetime(year+1,2,1,0,0,0,0)
            time_e     = datetime(year+1,3,1,0,0,0,0)

            plot_LIS_diff(fire_path, file_name, land_ctl_path, land_sen_path, var_names, time_s=time_s, time_e=time_e, lat_names="lat",
                          lon_names="lon",loc_lat=loc_lat, loc_lon=loc_lon, wrf_path=wrf_path,
                          message=message, burn=burn)

            burn_message = "_unburnt"
            burn         = -1

            message    = "HW_Dec"+str(year)+burn_message
            time_s     = datetime(year,12,1,0,0,0,0)
            time_e     = datetime(year+1,1,1,0,0,0,0)

            plot_LIS_diff(fire_path, file_name, land_ctl_path, land_sen_path, var_names, time_s=time_s, time_e=time_e, lat_names="lat",
                          lon_names="lon",loc_lat=loc_lat, loc_lon=loc_lon, wrf_path=wrf_path,
                          message=message, burn=burn)

            message    = "HW_Jan"+str(year+1)+burn_message
            time_s     = datetime(year+1,1,1,0,0,0,0)
            time_e     = datetime(year+1,2,1,0,0,0,0)

            plot_LIS_diff(fire_path, file_name, land_ctl_path, land_sen_path, var_names, time_s=time_s, time_e=time_e, lat_names="lat",
                          lon_names="lon",loc_lat=loc_lat, loc_lon=loc_lon, wrf_path=wrf_path,
                          message=message, burn=burn)

            message    = "HW_Feb"+str(year+1)+burn_message
            time_s     = datetime(year+1,2,1,0,0,0,0)
            time_e     = datetime(year+1,3,1,0,0,0,0)

            plot_LIS_diff(fire_path, file_name, land_ctl_path, land_sen_path, var_names, time_s=time_s, time_e=time_e, lat_names="lat",
                          lon_names="lon",loc_lat=loc_lat, loc_lon=loc_lon, wrf_path=wrf_path,
                          message=message, burn=burn)

        if 0:
            '''
            Three heatwaves in Dec 2019 - Feb 2020
            '''

            file_name    = 'LIS.CABLE.201912-202002.nc'
            burn_message = "_burnt"
            burn         = 1

            message    = "HW_20191220-20191223"+burn_message
            time_s     = datetime(2019,12,20,0,0,0,0)
            time_e     = datetime(2019,12,24,0,0,0,0)

            plot_LIS_diff(fire_path, file_name, land_ctl_path, land_sen_path, var_names, time_s=time_s, time_e=time_e, lat_names="lat",
                          lon_names="lon",loc_lat=loc_lat, loc_lon=loc_lon, wrf_path=wrf_path,
                          message=message, burn=burn)

            message    = "HW_20191230-20200101"+burn_message
            time_s     = datetime(2019,12,30,0,0,0,0)
            time_e     = datetime(2020,1,2,0,0,0,0)

            plot_LIS_diff(fire_path, file_name, land_ctl_path, land_sen_path, var_names, time_s=time_s, time_e=time_e, lat_names="lat",
                          lon_names="lon",loc_lat=loc_lat, loc_lon=loc_lon, wrf_path=wrf_path,
                          message=message, burn=burn)

            message    = "HW_20200131-20200203"+burn_message
            time_s     = datetime(2020,1,31,0,0,0,0)
            time_e     = datetime(2020,2,4,0,0,0,0)

            plot_LIS_diff(fire_path, file_name, land_ctl_path, land_sen_path, var_names, time_s=time_s, time_e=time_e, lat_names="lat",
                          lon_names="lon",loc_lat=loc_lat, loc_lon=loc_lon, wrf_path=wrf_path,
                          message=message, burn=burn)

            burn_message = "_unburnt"
            burn         = -1

            message    = "HW_20191220-20191223"+burn_message
            time_s     = datetime(2019,12,20,0,0,0,0)
            time_e     = datetime(2019,12,24,0,0,0,0)

            plot_LIS_diff(fire_path, file_name, land_ctl_path, land_sen_path, var_names, time_s=time_s, time_e=time_e, lat_names="lat",
                          lon_names="lon",loc_lat=loc_lat, loc_lon=loc_lon, wrf_path=wrf_path,
                          message=message, burn=burn)

            message    = "HW_20191230-20200101"+burn_message
            time_s     = datetime(2019,12,30,0,0,0,0)
            time_e     = datetime(2020,1,2,0,0,0,0)

            plot_LIS_diff(fire_path, file_name, land_ctl_path, land_sen_path, var_names, time_s=time_s, time_e=time_e, lat_names="lat",
                          lon_names="lon",loc_lat=loc_lat, loc_lon=loc_lon, wrf_path=wrf_path,
                          message=message, burn=burn)

            message    = "HW_20200131-20200203"+burn_message
            time_s     = datetime(2020,1,31,0,0,0,0)
            time_e     = datetime(2020,2,4,0,0,0,0)

            plot_LIS_diff(fire_path, file_name, land_ctl_path, land_sen_path, var_names, time_s=time_s, time_e=time_e, lat_names="lat",
                          lon_names="lon",loc_lat=loc_lat, loc_lon=loc_lon, wrf_path=wrf_path,
                          message=message, burn=burn)

    if 1:
        # plot burnt region time series
        case_ctl       = "drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2"
        case_sen       = "drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2_obs_LAI_ALB"

        file_name      = 'LIS.CABLE.201701-202002.nc'
        fire_path      = '/g/data/w97/mm3972/data/MODIS/MODIS_fire/MCD64A1.061_500m_aid0001.nc'
        wrf_path       = "/scratch/w97/mm3972/model/NUWRF/Tinderbox_drght_LAI_ALB/output/drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/WRF_output/wrfout_d01_2017-02-01_06:00:00"

        land_sen_path  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_sen+"/LIS_output/"
        land_ctl_path  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_ctl+"/LIS_output/"

        var_names      = ["LAI_inst"]#"Albedo_inst","Tmax","FWsoil_tavg"]

        time_s         = datetime(2019,10,1,0,0,0,0)
        time_e         = datetime(2020,3,1,0,0,0,0)

        burn_message   = "_burn_reg1"
        loc_lat        = [-32,-28.5]
        loc_lon        = [151.5,153.5]

        message        = "time_series_"+burn_message
        for var_name in var_names:
            plot_time_series_burn_region(fire_path, wrf_path, file_name, land_ctl_path, land_sen_path, var_name=var_name,
                                time_s=time_s, time_e=time_e, loc_lat=loc_lat, loc_lon=loc_lon,
                                lat_name="lat", lon_name="lon", message=message, burn=1)
