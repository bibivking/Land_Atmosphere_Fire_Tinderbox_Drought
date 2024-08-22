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

def read_LIS_var(file_name, land_ctl_path, land_sen_path, var_name, loc_lat, loc_lon, lat_names, lon_names, time_ss, time_es):

    if var_name in ["Tmax","Tmin","TDR"]:
        land_ctl_files= [land_ctl_path+'Tair_f_inst/'+file_name]
        land_sen_files= [land_sen_path+'Tair_f_inst/'+file_name]
        time, Ctl_tmp = read_var_multi_file(land_ctl_files, 'Tair_f_inst', loc_lat, loc_lon, lat_names, lon_names)
        time, Sen_tmp = read_var_multi_file(land_sen_files, 'Tair_f_inst', loc_lat, loc_lon, lat_names, lon_names)
        Ctl_tmp       = Ctl_tmp - 273.15
        Sen_tmp       = Sen_tmp - 273.15
    elif var_name in ["VegTmax","VegTmin","VegTDR"]:
        land_ctl_files= [land_ctl_path+'VegT_tavg/'+file_name]
        land_sen_files= [land_sen_path+'VegT_tavg/'+file_name]
        time, Ctl_tmp = read_var_multi_file(land_ctl_files, 'VegT_tavg', loc_lat, loc_lon, lat_names, lon_names)
        time, Sen_tmp = read_var_multi_file(land_sen_files, 'VegT_tavg', loc_lat, loc_lon, lat_names, lon_names)
        Ctl_tmp       = Ctl_tmp - 273.15
        Sen_tmp       = Sen_tmp - 273.15
    elif var_name in ["SurfTmax","SurfTmin","SurfTDR"]:
        land_ctl_files= [land_ctl_path+'AvgSurfT_tavg/'+file_name]
        land_sen_files= [land_sen_path+'AvgSurfT_tavg/'+file_name]
        time, Ctl_tmp = read_var_multi_file(land_ctl_files, 'AvgSurfT_tavg', loc_lat, loc_lon, lat_names, lon_names)
        time, Sen_tmp = read_var_multi_file(land_sen_files, 'AvgSurfT_tavg', loc_lat, loc_lon, lat_names, lon_names)
        Ctl_tmp       = Ctl_tmp - 273.15
        Sen_tmp       = Sen_tmp - 273.15
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
    elif var_name in ["EF",]:
        land_ctl_files      = [land_ctl_path+'Lwnet_tavg/'+file_name]
        land_sen_files      = [land_sen_path+'Lwnet_tavg/'+file_name]
        time, Ctl_Lwnet_tmp = read_var_multi_file(land_ctl_files, 'Lwnet_tavg', loc_lat, loc_lon, lat_names, lon_names)
        time, Sen_Lwnet_tmp = read_var_multi_file(land_sen_files, 'Lwnet_tavg', loc_lat, loc_lon, lat_names, lon_names)
        land_ctl_files      = [land_ctl_path+'Swnet_tavg/'+file_name]
        land_sen_files      = [land_sen_path+'Swnet_tavg/'+file_name]
        time, Ctl_Swnet_tmp = read_var_multi_file(land_ctl_files, 'Swnet_tavg', loc_lat, loc_lon, lat_names, lon_names)
        time, Sen_Swnet_tmp = read_var_multi_file(land_sen_files, 'Swnet_tavg', loc_lat, loc_lon, lat_names, lon_names)
        land_ctl_files      = [land_ctl_path+'Qle_tavg/'+file_name]
        land_sen_files      = [land_sen_path+'Qle_tavg/'+file_name]
        time, Ctl_Qle_tmp   = read_var_multi_file(land_ctl_files, 'Qle_tavg', loc_lat, loc_lon, lat_names, lon_names)
        time, Sen_Qle_tmp   = read_var_multi_file(land_sen_files, 'Qle_tavg', loc_lat, loc_lon, lat_names, lon_names)
        ctl_Rnet            = Ctl_Lwnet_tmp + Ctl_Swnet_tmp
        sen_Rnet            = Sen_Lwnet_tmp + Sen_Swnet_tmp
        Ctl_tmp             = np.where(abs(ctl_Rnet)>0.01, Ctl_Qle_tmp/ctl_Rnet,np.nan)
        Sen_tmp             = np.where(abs(sen_Rnet)>0.01, Sen_Qle_tmp/sen_Rnet,np.nan)
    elif var_name in ["SM_top50cm",]:
        land_ctl_files = [land_ctl_path+'SoilMoist_inst/'+file_name]
        land_sen_files = [land_sen_path+'SoilMoist_inst/'+file_name]
        time, Ctl_temp = read_var_multi_file(land_ctl_files, 'SoilMoist_inst', loc_lat, loc_lon, lat_names, lon_names)
        time, Sen_temp = read_var_multi_file(land_sen_files, 'SoilMoist_inst', loc_lat, loc_lon, lat_names, lon_names)
        # [.022, .058, .154, .409, 1.085, 2.872]
        Ctl_tmp    = Ctl_temp[:,0,:,:]*0.022 + Ctl_temp[:,1,:,:]*0.058 + Ctl_temp[:,2,:,:]*0.154 + Ctl_temp[:,3,:,:]*0.266
        Sen_tmp    = Sen_temp[:,0,:,:]*0.022 + Sen_temp[:,1,:,:]*0.058 + Sen_temp[:,2,:,:]*0.154 + Sen_temp[:,3,:,:]*0.266
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
        land_ctl_files= [land_ctl_path+var_name+'/LIS.CABLE.201701-202002.nc']
        land_sen_files= [land_sen_path+var_name+'/LIS.CABLE.201701-202002.nc']
        time, Ctl_tmp = read_var_multi_file(land_ctl_files, var_name, loc_lat, loc_lon, lat_names, lon_names)
        time, Sen_tmp = read_var_multi_file(land_sen_files, var_name, loc_lat, loc_lon, lat_names, lon_names)

    nlat           = np.shape(Ctl_tmp)[1]
    nlon           = np.shape(Ctl_tmp)[2]
    var_diff       = np.zeros((6,nlat,nlon))
    var_ctl        = np.zeros((6,nlat,nlon))

    for j in np.arange(6):

        print("j=",j)

        if 'max' in var_name:
            # average of daily max
            ctl_in       = spatial_var_max(time,Ctl_tmp,time_ss[j],time_es[j])
            sen_in       = spatial_var_max(time,Sen_tmp,time_ss[j],time_es[j])
        elif 'min' in var_name:
            # average of daily min
            ctl_in       = spatial_var_min(time,Ctl_tmp,time_ss[j],time_es[j])
            sen_in       = spatial_var_min(time,Sen_tmp,time_ss[j],time_es[j])
        elif 'TDR' in var_name:
            # average of daily min
            ctl_in_max   = spatial_var_max(time,Ctl_tmp,time_ss[j],time_es[j])
            sen_in_max   = spatial_var_max(time,Sen_tmp,time_ss[j],time_es[j])
            ctl_in_min   = spatial_var_min(time,Ctl_tmp,time_ss[j],time_es[j])
            sen_in_min   = spatial_var_min(time,Sen_tmp,time_ss[j],time_es[j])
            ctl_in       = ctl_in_max - ctl_in_min
            sen_in       = sen_in_max - sen_in_min
        else:
            ctl_in       = spatial_var(time,Ctl_tmp,time_ss[j],time_es[j])
            sen_in       = spatial_var(time,Sen_tmp,time_ss[j],time_es[j])
        var_ctl[j,:,:]   = ctl_in
        var_diff[j,:,:]  = sen_in - ctl_in

    if var_name in ['WaterTableD_tavg','WatTable']:
        var_diff     = var_diff/1000.
    if var_name in ['ESoil_tavg','Evap_tavg',"ECanop_tavg",'TVeg_tavg',"Rainf_tavg","Snowf_tavg","Qs_tavg","Qsb_tavg"]:
        var_diff   = var_diff*3600*24
    if var_name in ['Qair_f_inst']:
        var_diff     = var_diff*1000
    if var_name in ['GPP_tavg','NPP_tavg']:
        s2d        = 3600*24.          # s-1 to d-1
        GPP_scale  = -0.000001*12*s2d   # umol s-1 to g d-1
        var_diff     = var_diff*GPP_scale

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
        # clevs = [-1.2,-1.1,-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.,1.1,1.2]
        clevs = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.]
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

    return var_ctl, var_diff, cmap, clevs

def spatial_map_winter_summer(file_name, land_ctl_path, land_sen_path, var_names, time_ss=None,
                              time_es=None, lat_names="lat", lon_names="lon",loc_lat=None,
                              loc_lon=None, wrf_path=None,  message=None, landcover_impact=False):

    '''
    plot a single spatial map
    '''

    # read lat and lon outs
    wrf            = Dataset(wrf_path,  mode='r')
    lons           = wrf.variables['XLONG'][0,:,:]
    lats           = wrf.variables['XLAT'][0,:,:]

    if landcover_impact:
        f_lc           = Dataset(land_ctl_path+'LIS.CABLE.201912-201912.d01.nc',  mode='r')
        landcover      = f_lc.variables['Landcover_inst'][0,:,:]

        # Forest
        mask_forest    = (landcover==2)

        # Non Forest
        mask_nonforest = (landcover>2) & (landcover<14)

    # WRF-CABLE
    for var_name in var_names:
        print("plotting "+var_name)
        var_ctl, var_diff, cmap, clevs = read_LIS_var(file_name, land_ctl_path, land_sen_path, var_name, loc_lat, loc_lon, lat_names, lon_names, time_ss, time_es)

        if landcover_impact:
            var_diff = np.where(var_diff==-9999.,np.nan,var_diff)
            var_ctl  = np.where(var_ctl==-9999.,np.nan,var_ctl)
            vals_pft = np.zeros((6,3,2)) # 6 seasons x all, forest, non-forest x absolute vals, percentage

            for i in np.arange(6):

                forest_ctl         = np.where(mask_forest, var_ctl[i,:,:], np.nan)
                forest_diff        = np.where(mask_forest, var_diff[i,:,:], np.nan)

                nonforest_ctl      = np.where(mask_nonforest, var_ctl[i,:,:], np.nan)
                nonforest_diff     = np.where(mask_nonforest, var_diff[i,:,:], np.nan)

                var_diff_avg       = np.nanmean(var_diff[i,:,:])
                var_per_avg        = np.nanmean(np.where( var_ctl[i,:,:]!=0, var_diff[i,:,:]/var_ctl[i,:,:]*100., np.nan))

                forest_diff_avg    = np.nanmean(forest_diff)
                forest_per_avg     = np.nanmean(np.where( forest_ctl!=0, forest_diff/forest_ctl*100., np.nan))

                nonforest_diff_avg = np.nanmean(nonforest_diff)
                nonforest_per_avg  = np.nanmean(np.where( nonforest_ctl!=0, nonforest_diff/nonforest_ctl*100., np.nan))

                # all pixels
                vals_pft[i,0,0] = var_diff_avg
                vals_pft[i,0,1] = var_per_avg

                # forests
                vals_pft[i,1,0] = forest_diff_avg
                vals_pft[i,1,1] = forest_per_avg

                # non-forests
                vals_pft[i,2,0] = nonforest_diff_avg
                vals_pft[i,2,1] = nonforest_per_avg


        # ================== Start Plotting =================
        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=[8,10],sharex=False,
                    sharey=False, squeeze=True, subplot_kw={'projection': ccrs.PlateCarree()})

        plt.subplots_adjust(wspace=-0.23, hspace=0.105)

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
        texts = ["(a)","(b)",
                 "(c)","(d)",
                 "(e)","(f)"]
        for i in np.arange(6):

            row = int(i/2)
            col = i % 2

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

            plot1 = axs[row,col].contourf(lons, lats, var_diff[i,:,:], clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
            axs[row,col].text(0.02, 0.15, texts[i], transform=axs[row,col].transAxes, fontsize=14, verticalalignment='top', bbox=props)


            if landcover_impact:
                if var_name == "Tmax":
                    axs[row,col].text(0.72, 0.15, f" all   {vals_pft[i,0,0]:.2g}"   +
                                                  f"\n fst   {vals_pft[i,1,0]:.2g}" +
                                                  f"\n nfst {vals_pft[i,2,0]:.2g}",
                                      transform=axs[row,col].transAxes, fontsize=6, verticalalignment='top', bbox=props)
                else:
                    axs[row,col].text(0.72, 0.15, f" all   {vals_pft[i,0,0]:.2g},{vals_pft[i,0,1]:.2g}%"   +
                                                  f"\n fst   {vals_pft[i,1,0]:.2g},{vals_pft[i,1,1]:.2g}%" +
                                                  f"\n nfst {vals_pft[i,2,0]:.2g},{vals_pft[i,2,1]:.2g}%",
                                      transform=axs[row,col].transAxes, fontsize=6, verticalalignment='top', bbox=props)

        cbar = plt.colorbar(plot1, ax=axs, ticklocation="right", pad=0.06, orientation="horizontal",
                aspect=50, shrink=0.8) # cax=cax,

        if var_name == "Albedo_inst":
            cbar.set_label('Δ$α$ (-)', loc='center',size=12)# rotation=270,
            cbar.set_ticks([-0.08,-0.07,-0.06,-0.05,-0.04,-0.03,-0.02,-0.01,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08])
            cbar.set_ticklabels(['-0.08','-0.07','-0.06','-0.05','-0.04','-0.03','-0.02','-0.01',
                                 '0.01','0.02','0.03','0.04','0.05','0.06','0.07','0.08']) # cax=cax,
        elif var_name == "LAI_inst":
            cbar.set_label('ΔLAI (m$\mathregular{^{2}}$ m$\mathregular{^{-2}}$)' , loc='center',size=12)# rotation=270,
            cbar.set_ticks([-2,-1.8,-1.6,-1.4,-1.2,-1.,-0.8,-0.6,-0.4,-0.2,-0.05,0.05,0.2,0.4,0.6,0.8,1.,1.2,1.4,1.6,1.8,2])
            cbar.set_ticklabels(['-2.0','-1.8','-1.6','-1.4','-1.2','-1.0','-0.8','-0.6','-0.4','-0.2','-0.05',
                                 '0.05','0.2','0.4','0.6','0.8','1.0','1.2','1.4','1.6','1.8','2.0']) # cax=cax,
        elif var_name == "Tmax":
            cbar.set_label('ΔT$\mathregular{_{max}}$ ($\mathregular{^{o}}$C)', loc='center',size=12)# rotation=270,
            # cbar.set_ticks([-1.2,-1.1,-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.,1.1,1.2])
            # cbar.set_ticklabels(['','-1.1','','-0.9','','-0.7','','-0.5','','-0.3','','-0.1',
            #                      '0.1','','0.3','','0.5','','0.7','','0.9','','1.1','']) # cax=cax,
            cbar.set_ticks([-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.])
            cbar.set_ticklabels(['-1.0','-0.9','-0.8','-0.7','-0.6','-0.5','-0.4','-0.3','-0.2','-0.1',
                                 '0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0']) # cax=cax,
        elif var_name == "Tmin":
            cbar.set_label('ΔT$\mathregular{_{min}}$ ($\mathregular{^{o}}$C)', loc='center',size=12)# rotation=270,
            cbar.set_ticks([1009,1010,1011,1012,1013,1014])
            cbar.set_ticklabels(['Sep','Oct','Nov','Dec','Jan','Feb']) # cax=cax,

        cbar.ax.tick_params(labelsize=10,labelrotation=90)

        axs[0,0].text(-0.25, 0.53, "2017-18", va='bottom', ha='center',
                rotation='vertical', rotation_mode='anchor',
                transform=axs[0,0].transAxes, fontsize=12)
        axs[1,0].text(-0.25, 0.50, "2018-19", va='bottom', ha='center',
                rotation='vertical', rotation_mode='anchor',
                transform=axs[1,0].transAxes, fontsize=12)
        axs[2,0].text(-0.25, 0.48, "2019-20", va='bottom', ha='center',
                rotation='vertical', rotation_mode='anchor',
                transform=axs[2,0].transAxes, fontsize=12)

        axs[0,0].set_title("Winter")
        axs[0,1].set_title("Summer")

        # Apply tight layout
        # plt.tight_layout()
        plt.savefig('./plots/Fig_spatial_map_'+message + "_" + var_name+'.png',dpi=500)


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

        var_names  = [ "Albedo_inst","LAI_inst","Tmax",]#]#,"Tmin" ]

        time_ss    = [  datetime(2017,6,1,0,0,0,0), datetime(2017,12,1,0,0,0,0),
                        datetime(2018,6,1,0,0,0,0), datetime(2018,12,1,0,0,0,0),
                        datetime(2019,6,1,0,0,0,0), datetime(2019,12,1,0,0,0,0)]

        time_es    = [datetime(2017,9,1,0,0,0,0), datetime(2018,3,1,0,0,0,0),
                      datetime(2018,9,1,0,0,0,0), datetime(2019,3,1,0,0,0,0),
                      datetime(2019,9,1,0,0,0,0), datetime(2020,3,1,0,0,0,0)]

        message    = "Winter_Summer"
        file_name  = "LIS.CABLE.201701-202002.nc"
        landcover_impact = False
        spatial_map_winter_summer(file_name, land_ctl_path, land_sen_path, var_names, time_ss=time_ss, time_es=time_es, lat_names="lat",
                            lon_names="lon",loc_lat=loc_lat, loc_lon=loc_lon, wrf_path=wrf_path, message=message, landcover_impact=landcover_impact)
