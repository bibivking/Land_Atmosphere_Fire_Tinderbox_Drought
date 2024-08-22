#!/usr/bin/env python
"""
Produce the Australia map with the label of EucFACE site - Figure 1
"""

__author__ = "Mengyuan Mu"
__email__  = "mu.mengyuan815@gmail.com"

import sys
import cartopy
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.patches import Polygon
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap
from scipy.interpolate import griddata, interp1d
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature, OCEAN
from common_utils import *


def plot_pft_state_names(iveg_path, wrf_path):

    '''
    plot LIS variables in burnt / unburnt / all regions
    '''

    # Read in WRF lat and lon
    wrf       = Dataset(wrf_path,  mode='r')
    lon_in    = wrf.variables['XLONG'][0,:,:]
    lat_in    = wrf.variables['XLAT'][0,:,:]

    lis       = Dataset(iveg_path,  mode='r')
    landcover = lis.variables['Landcover_inst'][0,:,:]
    LC_Mask   = lis.variables['lon'][:,:]
    landcover = np.where(landcover == 2, 1, landcover)
    landcover = np.where(landcover == 5, 2, landcover)
    landcover = np.where(landcover == 6, 3, landcover)
    landcover = np.where(landcover == 9, 4, landcover)
    landcover = np.where(landcover == 14,5, landcover)
    landcover = np.where(landcover == -9999,0, landcover)


    if 1:
        # Set the output lat and lon
        lat_out      = np.arange(-40,-24,0.04)
        lon_out      = np.arange(135,155,0.04)
        lon_out_2D,lat_out_2D = np.meshgrid(lon_out,lat_out)

        # Regrid to lat-lon projection
        LC_Mask      = np.where(LC_Mask < 0, 0, 1)
        LC_Mask_in_1D= LC_Mask.flatten()
        lat_in_1D    = lat_in.flatten()
        lon_in_1D    = lon_in.flatten()
        Mask_regrid  = np.zeros((len(lat_out),len(lon_out)))
        Mask_regrid  = griddata((lat_in_1D, lon_in_1D), LC_Mask_in_1D, (lat_out_2D, lon_out_2D), method='nearest')

        LC_regrid    = np.zeros((len(lat_out),len(lon_out)))
        LC_in_1D_tmp = landcover.flatten()
        LC_in_1D     = LC_in_1D_tmp[~np.isnan(LC_in_1D_tmp)]
        lat_in_1D    = lat_in_1D[~np.isnan(LC_in_1D_tmp)]    # here I make nan in values as the standard
        lon_in_1D    = lon_in_1D[~np.isnan(LC_in_1D_tmp)]
        LC_regrid     = griddata((lat_in_1D, lon_in_1D), LC_in_1D, (lat_out_2D, lon_out_2D), method='nearest')
        # LC_regrid     = np.where(Mask_regrid==1,LC_regrid,np.nan)

    # ================== Start Plotting =================
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=[4,4],sharex=True,
                sharey=False, squeeze=True, subplot_kw={'projection': ccrs.PlateCarree()})

    plt.subplots_adjust(wspace=0.18, hspace=0.)

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

    # =============== CHANGE HERE ===============
    axs.set_extent([135,155,-39.5,-24])
    axs.add_feature(states, linewidth=.5, edgecolor="black")
    axs.set_facecolor('lightgray')
    axs.coastlines(resolution="50m",linewidth=1)

    axs.tick_params(axis='x', direction='out')
    axs.tick_params(axis='y', direction='out')
    x_ticks = np.arange(135, 156, 5)
    y_ticks = np.arange(-35, -20, 5)
    axs.set_xticks(x_ticks)
    axs.set_yticks(y_ticks)

    axs.set_xticklabels(['135$\mathregular{^{o}}$E','140$\mathregular{^{o}}$E','145$\mathregular{^{o}}$E',
                                        '150$\mathregular{^{o}}$E','155$\mathregular{^{o}}$E'],rotation=25)
    axs.set_yticklabels(['35$\mathregular{^{o}}$S','30$\mathregular{^{o}}$S','25$\mathregular{^{o}}$S'])

    cint = [-1,0.1,1.1,2.1,3.1,4.1,5.1]

    # cmap = plt.cm.Paired(np.arange(12))
    #         # ["BEF","shrub","grass","crop","barren"]
    # cmap = [cmap[0],cmap[3],cmap[2],cmap[6],cmap[7],cmap[11]] #

    cmap    = ListedColormap(["skyblue", # 0: no trend
                              "olivedrab",             # 1: increase
                              "yellowgreen",#"mediumspringgreen",               # 2: decrease
                              "wheat",
                              "orange",            # 3: increase then decrease
                              "peru"
                              ]) #plt.cm.tab10 #BrBG
    # contourf
    # cf   = axs.contourf(lon_in, lat_in, landcover, levels=cint, colors=cmap,
    #                    transform=ccrs.PlateCarree()) # rasterized=True doesn't work

    # imshow
    extent=(135, 155, -40, -24)
    cf = axs.imshow(LC_regrid, origin="lower", extent=extent, interpolation="none", vmin=-0.5, vmax=5.5, transform=ccrs.PlateCarree(), cmap=cmap) # resample=False,
    cbar = plt.colorbar(cf, ax=axs, orientation="horizontal", pad=.13,  aspect=20, extend='neither', shrink=1.)
    cbar.set_ticks([])
    axs.text(0.02, 0.15,'(a)', transform=axs.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    axs.text(0.53, 0.5, "NSW", transform=axs.transAxes, verticalalignment='top', color=almost_black, bbox=props, fontsize=14)
    axs.text(0.38, 0.22, "VIC", transform=axs.transAxes, verticalalignment='top', color=almost_black,bbox=props, fontsize=14)
    axs.text(0.53, 0.9, "QLD", transform=axs.transAxes, verticalalignment='top', color=almost_black,bbox=props, fontsize=14)
    axs.text(0.1,  0.65, "SA", transform=axs.transAxes, verticalalignment='top', color=almost_black,bbox=props, fontsize=14)

    axs.text(-0.02, -0.28, "Water", transform=axs.transAxes, verticalalignment='top', color=almost_black,bbox=props, fontsize=10)
    axs.text(0.15, -0.28, "Forest", transform=axs.transAxes, verticalalignment='top', color=almost_black,bbox=props, fontsize=10)
    axs.text(0.33, -0.28, "Shrub", transform=axs.transAxes, verticalalignment='top', color=almost_black,bbox=props, fontsize=10)
    axs.text(0.53, -0.28, "Grass", transform=axs.transAxes, verticalalignment='top', color=almost_black,bbox=props, fontsize=10)
    axs.text(0.72, -0.28, "Crop", transform=axs.transAxes, verticalalignment='top', color=almost_black,bbox=props, fontsize=10)
    axs.text(0.88, -0.28, "Barren\n  land", transform=axs.transAxes, verticalalignment='top', color=almost_black,bbox=props, fontsize=10)
    # fig.tight_layout()
    plt.savefig('./plots/Fig1_spatial_map_pft_state_names.png',dpi=300)

if __name__ == "__main__":

    #######################################################
    # Decks to run:
    #    heat_advection
    #######################################################

    iveg_path  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/LIS_output/LIS.CABLE.201701-201701.d01.nc"
    wrf_path   = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/WRF_output/wrfout_d01_2019-12-01_01:00:00"

    plot_pft_state_names(iveg_path, wrf_path)
