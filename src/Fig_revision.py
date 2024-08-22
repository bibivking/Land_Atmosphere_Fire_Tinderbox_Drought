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


# ===================================================
wrf_path       = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/WRF_output/wrfout_d01_2019-12-01_01:00:00"
land_sen_path  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2_obs_LAI_ALB/LIS_output/LAI_inst/LIS.CABLE.201701-202002.nc"

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

clevs = [-2,-1.8,-1.6,-1.4,-1.2,-1.,-0.8,-0.6,-0.4,-0.2,-0.05,0.05,0.2,0.4,0.6,0.8,1.,1.2,1.4,1.6,1.8,2]
clevs_percentage =  [-70,-60,-50,-40,-30,-20,-10,-5,5,10,20,30,40,50,60,70]
cmap  =  plt.cm.colors.ListedColormap(rgb_21colors)


# ================== Reading data ===================
# read lat and lon outs
wrf            = Dataset(wrf_path,  mode='r')
lons           = wrf.variables['XLONG'][0,:,:]
lats           = wrf.variables['XLAT'][0,:,:]

f              = Dataset(land_sen_path,  mode='r')
LAI            = f.variables['LAI_inst']
# LAI_diff       = np.nanmean(LAI[0:26280,:,:],axis=0)#-LAI[0,:,:]

LAI_diff       = np.nanstd(LAI[0:26280,:,:],axis=0)#-LAI[0,:,:]

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

print(np.shape(LAI_diff))
plot1 = axs.contourf(lons, lats, LAI_diff, clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend='both')

cbar = plt.colorbar(plot1, ax=axs, ticklocation="right", pad=0.1, orientation="horizontal",
        aspect=50, shrink=1.) # cax=cax,
cbar.set_label('ΔLAI (m$\mathregular{^{2}}$ m$\mathregular{^{-2}}$)' , loc='center',size=12)# rotation=270,
# cbar.set_ticks([-2,-1.8,-1.6,-1.4,-1.2,-1.,-0.8,-0.6,-0.4,-0.2,-0.05,0.05,0.2,0.4,0.6,0.8,1.,1.2,1.4,1.6,1.8,2])
# cbar.set_ticklabels(['-2.0','-1.8','-1.6','-1.4','-1.2','-1.0','-0.8','-0.6','-0.4','-0.2','-0.05',
#                      '0.05','0.2','0.4','0.6','0.8','1.0','1.2','1.4','1.6','1.8','2.0']) # cax=cax,


cbar.set_ticks([0.5,1.,1.5,2.,2.5,3.,3.5,4,4.5,5])
cbar.set_ticklabels(['0.5','1.0','1.5','2.0','2.5','3.0','3.5','4.0','4.5','5.0']) # cax=cax,

plt.savefig('./plots/Fig_spatial_map_LAI_diff_during_drought.png',dpi=500)
