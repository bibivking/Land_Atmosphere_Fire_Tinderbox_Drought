#!/usr/bin/python

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from netCDF4 import Dataset
import cartopy.crs as crs
from cartopy.feature import NaturalEarthFeature
from scipy.interpolate import griddata
from wrf import (getvar, to_np, vertcross, CoordPair,
                 get_cartopy, latlon_coords, ALL_TIMES)
from common_utils import *

def plot_profile_wrf_wind(file_outs,message=None):

    # ****************** plotting ******************

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=[7,10],sharex=False, sharey=True, squeeze=True)
    plt.subplots_adjust(wspace=0.05, hspace=0.13)

    plt.rcParams['text.usetex']     = False
    plt.rcParams['font.family']     = "sans-serif"
    plt.rcParams['font.serif']      = "Helvetica"
    plt.rcParams['axes.linewidth']  = 1.5
    plt.rcParams['axes.labelsize']  = 12
    plt.rcParams['font.size']       = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

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

    # ===== set colormap =====
    # color map
    color_map_1   = get_cmap("seismic")

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

    # quiver scale
    scale = 1.

    # contour levels
    levels1    = [-1.2,-1.1,-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.,1.1,1.2]
    levels_lai = [-2,-1.8,-1.6,-1.4,-1.2,-1.,-0.8,-0.6,-0.4,-0.2,0.2,0.4,0.6,0.8,1.,1.2,1.4,1.6,1.8,2]
    levels_alb = [-0.08,-0.07,-0.06,-0.05,-0.04,-0.03,-0.02,-0.01,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08]

    # ****************** Read data *****************
    for i, file_out in enumerate(file_outs):

        row = int(i / 3)
        col = i % 3

        ncfile        = Dataset(file_out, mode='r')
        vertical      = ncfile.variables['level'][:]
        xy_loc        = ncfile.variables['lon'][:]

        th_day_diff   = ncfile.variables['th_day_diff'][:]
        ua_day_diff   = ncfile.variables['ua_day_diff'][:]
        wa_day_diff   = ncfile.variables['wa_day_diff'][:]

        lai_diff      = ncfile.variables['lai_diff'][:]
        alb_diff      = ncfile.variables['alb_diff'][:]
        pbl_day_ctl   = ncfile.variables['pbl_day_ctl'][:]
        pbl_day_sen   = ncfile.variables['pbl_day_sen'][:]

        # Water table depth height
        lai_hgt          = [0,100]
        lai_diff_2D      = np.zeros((2,len(lai_diff)))
        lai_diff_2D[0,:] = lai_diff
        lai_diff_2D[1,:] = lai_diff

        alb_hgt          = [-100,0]
        alb_diff_2D      = np.zeros((2,len(alb_diff)))
        alb_diff_2D[0,:] = alb_diff
        alb_diff_2D[1,:] = alb_diff

        # Set color for topography
        ax[row, col].set_facecolor('lightgray')

        # Set the lons boundaries of the transect
        if row ==0:
            ax[0, col].set_xlim(149, 154)
            ax[0, col].set_xticks([150, 152, 154])
            ax[0, col].set_xticklabels(['150','152','154'])
        elif row ==1:
            ax[1, col].set_xlim(148, 153)
            ax[1, col].set_xticks([148, 150, 152])
            ax[1, col].set_xticklabels(['148','150','152'])
        elif row ==2:
            ax[2, col].set_xlim(146, 151)
            ax[2, col].set_xticks([146, 148, 150])
            ax[2, col].set_xticklabels(['146','148','150'])


        # Plot variables
        contour   = ax[row, col].contourf(xy_loc, vertical, th_day_diff, levels=levels1, cmap=color_map_1, extend='both')
        cntr_lai  = ax[row, col].contourf(xy_loc, lai_hgt,  lai_diff_2D, levels=levels_lai, cmap=cmap21, extend='both')
        cntr_alb  = ax[row, col].contourf(xy_loc, alb_hgt,  alb_diff_2D, levels=levels_alb, cmap=cmap17, extend='both')
        line1     = ax[row, col].plot(xy_loc,pbl_day_ctl,ls="-", lw=0.5, color=almost_black)
        line2     = ax[row, col].plot(xy_loc,pbl_day_sen,ls="--", lw=0.5, color=almost_black)
        line3     = ax[row, col].axhline(y=0,   color=almost_black, lw=0.5, linestyle='-')
        line4     = ax[row, col].axhline(y=100, color=almost_black, lw=0.5, linestyle='-')

        # q         = ax[row, col].quiver(xy_loc[::30], vertical[::3], ua_day_diff[::3,::30],
        #                         wa_day_diff[::3,::30], angles='xy', scale_units='xy',
        #                         scale=scale, pivot='middle', color="lightgrey")
        ax[row, col].set_ylim(-100,3000)


    # Set titles
    ax[0,0].set_title("Dec 2019")
    ax[0,1].set_title("Jan 2020")
    ax[0,2].set_title("Feb 2020")

    ax[0,0].set_ylabel("North", fontsize=12)
    ax[1,0].set_ylabel("Central", fontsize=12)
    ax[2,0].set_ylabel("South", fontsize=12)

    fig.text(0.01, 0.5, 'Geopotential Height (m)', va='center', rotation='vertical')

    # Add Tmax colorbar
    cbar  = plt.colorbar(contour, ax=ax, ticklocation="right", pad=0.03, orientation="vertical",
            aspect=40, shrink=0.9) # cax=cax,
    cbar.set_label('ΔT$\mathregular{_{max}}$ ($\mathregular{^{o}}$C)', loc='center',size=12)# rotation=270,
    cbar.ax.tick_params(labelsize=12) # ,labelrotation=45

    # Add LAI and albedo colorbar
    position_lai  = fig.add_axes([0.1, 0.06, 0.34, 0.014]) # [left, bottom, width, height]
    cb_lai        = fig.colorbar(cntr_lai, ax=ax, pad=0.08, cax=position_lai, orientation="horizontal", aspect=60, shrink=0.8)
    cb_lai.set_label('LAI (m$\mathregular{^{2}}$ m$\mathregular{^{-2}}$)', loc='center',size=12)# rotation=270,
    cb_lai.ax.tick_params(labelsize=12,rotation=45)

    position_alb  = fig.add_axes([0.48, 0.06, 0.34, 0.014]) # [left, bottom, width, height]
    cb_alb        = fig.colorbar(cntr_alb, ax=ax, pad=0.08, cax=position_alb, orientation="horizontal", aspect=60, shrink=0.8)
    cb_alb.set_label('Δ$α$ (-)', loc='center',size=12)# rotation=270,
    cb_alb.ax.tick_params(labelsize=12,rotation=45)

    fig.savefig("./plots/profile_wrf_"+message, bbox_inches='tight', pad_inches=0.3,dpi=300)

if __name__ == "__main__":


    lat_slt        = -37.5
    lon_min        = 145.0
    lon_max        = 154.0

    file_name_wrf  = "wrfout_201912-202002.nc"
    file_name_lis  = "LIS.CABLE.201912-202002.nc"
    path           = '/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/'


    wrf_path      = path+ 'drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/WRF_output/wrfout_d01_2017-02-01_06:00:00'
    land_path     = path+ 'drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/geo_em.d01.nc'

    atmo_path_ctl = path + 'drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/WRF_output/'
    atmo_path_sen = path + 'drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2_obs_LAI_ALB/WRF_output/'

    land_path_ctl = path + 'drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/LIS_output/'
    land_path_sen = path + 'drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2_obs_LAI_ALB/LIS_output/'

    message        = "profile_transect"

    file_outs = [ "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/transect_201920_Dec_lat-30_houly.nc",
                  "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/transect_201920_Jan_lat-30_houly.nc",
                  "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/transect_201920_Feb_lat-30_houly.nc",
                  "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/transect_201920_Dec_lat-33_houly.nc",
                  "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/transect_201920_Jan_lat-33_houly.nc",
                  "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/transect_201920_Feb_lat-33_houly.nc",
                  "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/transect_201920_Dec_lat-375_houly.nc",
                  "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/transect_201920_Jan_lat-375_houly.nc",
                  "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/transect_201920_Feb_lat-375_houly.nc"]

    plot_profile_wrf_wind(file_outs,message=message)
