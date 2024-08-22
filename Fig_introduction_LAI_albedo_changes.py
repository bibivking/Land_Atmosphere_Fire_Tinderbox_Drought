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

def spatial_map_single_plot_LIS_diff(wrf_path, land_sen_path, var_name):

    '''
    plot a single spatial map
    '''
    time_index_s = 0
    time_index_e = 26256-1

    f         = Dataset(land_sen_path+var_name+'/LIS.CABLE.201701-201912.nc', mode='r')
    var_first = f.variables[var_name][time_index_s,:,:]
    var_last  = f.variables[var_name][time_index_e,:,:]
    var_diff  = var_last- var_first


    # read lat and lon outs
    wrf            = Dataset(wrf_path,  mode='r')
    lons           = wrf.variables['XLONG'][0,:,:]
    lats           = wrf.variables['XLAT'][0,:,:]

    # =============== CHANGE HERE ===============
    cmap  = plt.cm.BrBG
    if var_name == 'LAI_inst':
        clevs = [-4.0,-3.8,-3.6,-3.4,-3.2,-3.0,-2.8,-2.6,-2.4,-2.2,-2,-1.8,-1.6,-1.4,-1.2,-1.,-0.8,-0.6,-0.4,-0.2,-0.05,
                 0.05,0.2,0.4,0.6,0.8,1.,1.2,1.4,1.6,1.8,2,2.2,2.4,2.6,2.8,3.0,3.2,3.4,3.6,3.8,4.0]
    elif var_name == 'Albedo_inst':
        clevs = [-0.15,-0.14,-0.13,-0.12,-0.11,-0.10,-0.09,-0.08,-0.07,-0.06,-0.05,-0.04,-0.03,-0.02,-0.01,
                  0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.11,0.12,0.13,0.14,0.15]


    # ================== Start Plotting =================
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=[8,6],sharex=True,
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
    # cmap  = plt.cm.seismic

    axs.coastlines(resolution="50m",linewidth=1)
    axs.set_extent([135,155,-39,-23])
    axs.add_feature(states, linewidth=.5, edgecolor="black")

    # Add gridlines
    gl              = axs.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color=almost_black, linestyle='--')
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

    # print("any(not np.isnan(var_diff))",any(not np.isnan(var_diff)))
    plot1 = axs.contourf(lons, lats, var_diff, clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
    cbar = plt.colorbar(plot1, ax=axs, ticklocation="right", pad=0.08, orientation="horizontal",
            aspect=40, shrink=1) # cax=cax,
    cbar.ax.tick_params(labelsize=10,labelrotation=45)

    plt.savefig('./plots/Introduction_' + var_name+'_changes_in_Tinderbox_drought.png',dpi=300)

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

    '''
    Test WRF-CABLE LIS output
    '''
    case_ctl       = "drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2"
    case_sen       = "drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2_obs_LAI_ALB"

    wrf_path       = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_ctl+"/WRF_output/wrfout_d01_2019-12-07_01:00:00"
    land_sen_path  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_sen+"/LIS_output/"
    land_ctl_path  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_ctl+"/LIS_output/"

    var_name       = 'LAI_inst' #'Albedo_inst'
    spatial_map_single_plot_LIS_diff(wrf_path, land_sen_path, var_name)


    var_name       = 'Albedo_inst'
    spatial_map_single_plot_LIS_diff(wrf_path, land_sen_path, var_name)