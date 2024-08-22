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

def read_LIS_var(file_name, land_ctl_path, land_sen_path, var_name, loc_lat, loc_lon, lat_names, lon_names, time_s, time_e):

    if var_name in ["Tmax","Tmin","TDR"]:
        land_ctl_files= [land_ctl_path+'Tair_f_inst/'+file_name]
        land_sen_files= [land_sen_path+'Tair_f_inst/'+file_name]
        time, Ctl_tmp = read_var_multi_file(land_ctl_files, 'Tair_f_inst', loc_lat, loc_lon, lat_names, lon_names)
        time, Sen_tmp = read_var_multi_file(land_sen_files, 'Tair_f_inst', loc_lat, loc_lon, lat_names, lon_names)
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
    elif var_name in ["SM_top50cm",]:
        land_ctl_files = [land_ctl_path+'SoilMoist_inst/'+file_name]
        land_sen_files = [land_sen_path+'SoilMoist_inst/'+file_name]
        time, Ctl_temp = read_var_multi_file(land_ctl_files, 'SoilMoist_inst', loc_lat, loc_lon, lat_names, lon_names)
        time, Sen_temp = read_var_multi_file(land_sen_files, 'SoilMoist_inst', loc_lat, loc_lon, lat_names, lon_names)
        # [.022, .058, .154, .409, 1.085, 2.872]
        Ctl_tmp    = (Ctl_temp[:,0,:,:]*0.022 + Ctl_temp[:,1,:,:]*0.058 + Ctl_temp[:,2,:,:]*0.154 + Ctl_temp[:,3,:,:]*0.266)/0.5
        Sen_tmp    = (Sen_temp[:,0,:,:]*0.022 + Sen_temp[:,1,:,:]*0.058 + Sen_temp[:,2,:,:]*0.154 + Sen_temp[:,3,:,:]*0.266)/0.5
    else:
        land_ctl_files= [land_ctl_path+var_name+'/LIS.CABLE.201701-202002.nc']
        land_sen_files= [land_sen_path+var_name+'/LIS.CABLE.201701-202002.nc']
        time, Ctl_tmp = read_var_multi_file(land_ctl_files, var_name, loc_lat, loc_lon, lat_names, lon_names)
        time, Sen_tmp = read_var_multi_file(land_sen_files, var_name, loc_lat, loc_lon, lat_names, lon_names)

    if 'max' in var_name:
        # average of daily max
        ctl_in       = spatial_var_max(time,Ctl_tmp,time_s,time_e)
        sen_in       = spatial_var_max(time,Sen_tmp,time_s,time_e)
    else:
        ctl_in       = spatial_var(time,Ctl_tmp,time_s,time_e)
        sen_in       = spatial_var(time,Sen_tmp,time_s,time_e)
    var_diff  = sen_in - ctl_in

    # =============== CHANGE HERE ===============
    cmap  = plt.cm.BrBG
    if var_name in ['SoilMoist_inst','SoilMoist',"SM_top50cm"]:
        # clevs = [0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.18,0.20,0.22,0.24,0.26,0.28,0.30,0.32,0.34,0.36,0.38]
        # clevs = [0.05,0.1,0.15,0.2,0.25,0.30,0.35,0.40,0.45,0.5,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1]
        clevs = [0.1,0.2,0.30,0.40,0.5,0.60,0.70,0.80,0.90]
        # clevs = [-0.08,-0.07,-0.06,-0.05,-0.04,-0.03,-0.02,-0.01,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08]
    elif var_name in ["Swnet_tavg","Lwnet_tavg","SWdown_f_inst","LWdown_f_inst","Qle_tavg","Qh_tavg","Qg_tavg"]:
        # clevs = [-22,-18,-14,-10,-6,-2,2,6,10,14,18,22]
        clevs = [-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,2,4,6,8,10,12,14,16,18,20]
    elif var_name in ["Rnet"]:
        clevs = [-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,2,4,6,8,10,12,14,16,18,20]
        # clevs = [-22,-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,2,4,6,8,10,12,14,16,18,20,22]
        # cmap  = plt.cm.BrBG_r
    elif var_name in ["Tair_f_inst","Tmax","Tmin","VegT_tavg","VegTmax","VegTmin",
                        "AvgSurfT_tavg","SurfTmax","SurfTmin","SoilTemp_inst",'TDR','VegTDR','SurfTDR']:
        # clevs = [-1.2,-1.1,-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.,1.1,1.2]
        clevs = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.]
        cmap  = plt.cm.seismic
    elif var_name in ["FWsoil_tavg","SmLiqFrac_inst","SmFrozFrac_inst"]:
        clevs = [-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0.05,0.1,0.15,0.2,0.25,0.3,0.35]
    else:
        clevs = [-0.5,-0.4,-0.3,-0.2,-0.1,-0.05,0.05,0.1,0.2,0.3,0.4,0.5]

    return ctl_in, sen_in, var_diff, cmap, clevs

def interpolate_ssat_to_wrf_domain(wrf_path):

    # # read in ssat data
    # f        = Dataset('/g/data/w97/mm3972/model/cable/src/CABLE-AUX/offline/mmy_gridinfo_Global/gridinfo_mmy_MD_elev_orig_std_avg-sand_mask.nc', mode='r')

    # lat_in   = f.variables['y'][:]
    # lon_in   = f.variables['x'][:]
    # ssat_vec = f.variables['ssat_vec'][:]

    # # calculate ssat for the top 0.5 m
    # ssat_top50cm = (ssat_vec[0,:,:]*0.022 + ssat_vec[1,:,:]*0.058 + ssat_vec[2,:,:]*0.154 + ssat_vec[3,:,:]*0.266)/0.5
    # ssat_top50cm = np.where(ssat_top50cm<0,np.nan,ssat_top50cm)

    # wrf      = Dataset(wrf_path,  mode='r')
    # lon_out  = wrf.variables['XLONG'][0,:,:]
    # lat_out  = wrf.variables['XLAT'][0,:,:]

    # # interpolation to wrf domain
    # ssat_top50cm = regrid_data(lat_in, lon_in, lat_out, lon_out, ssat_top50cm, method='nearest')


    f        = Dataset('/g/data/w97/mm3972/scripts/wrf_scripts/make_LIS_landinfo/nc_file/Openlandmap_soilcomposition_CORDEX_180E_depth_varying.nc', mode='r')

    lat_in   = f.variables['lat'][:]
    lon_in   = f.variables['lon'][:]
    sand_vec = f.variables['SAND_VEC'][:]
    clay_vec = f.variables['CLAY_VEC'][:]
    silt_vec = f.variables['SILT_VEC'][:]
    org_vec  = f.variables['OC_VEC'][:]

    # multivariate equation in Cosby et al 1984
    ssat_vec = (1.0-org_vec) * ( 0.505 - 0.142*sand_vec - 0.037*clay_vec ) + org_vec*0.6

    # calculate ssat for the top 0.5 m
    ssat_top50cm = (ssat_vec[0,:,:]*0.022 + ssat_vec[1,:,:]*0.058 + ssat_vec[2,:,:]*0.154 + ssat_vec[3,:,:]*0.266)/0.5
    ssat_top50cm = np.where(ssat_top50cm<0,np.nan,ssat_top50cm)

    wrf      = Dataset(wrf_path,  mode='r')
    lon_out  = wrf.variables['XLONG'][0,:,:]
    lat_out  = wrf.variables['XLAT'][0,:,:]

    # interpolation to wrf domain
    ssat_top50cm = regrid_data(lat_in, lon_in, lat_out, lon_out, ssat_top50cm, method='nearest')

    return ssat_top50cm


def spatial_map_summer_Tmax_Rnet_Qle_SM(file_names, land_ctl_path, land_sen_path, time_ss=None,
                              time_es=None, lat_names="lat", lon_names="lon",loc_lat=None,
                              loc_lon=None, wrf_path=None,  message=None):

    '''
    plot a single spatial map
    '''

    # Read lat and lon outs
    wrf            = Dataset(wrf_path,  mode='r')
    lons           = wrf.variables['XLONG'][0,:,:]
    lats           = wrf.variables['XLAT'][0,:,:]

    ssat_top50cm   = interpolate_ssat_to_wrf_domain(wrf_path)
    print('ssat_top50cm.shape',ssat_top50cm.shape)
    print('lons.shape',lons.shape)

    # ================== Start Plotting =================
    fig, axs = plt.subplots(nrows=5, ncols=3, figsize=[10,15],sharex=False,
                sharey=False, squeeze=True, subplot_kw={'projection': ccrs.PlateCarree()})

    plt.subplots_adjust(wspace=0.14, hspace=-0.6) #-0.2

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

    # WRF-CABLE
    for col in np.arange(3):

        Tmax_ctl, Tmax_sen, Tmax_diff, Tmax_cmap, Tmax_clevs = read_LIS_var(file_names[col], land_ctl_path, land_sen_path, 'Tmax', None, None, lat_names, lon_names, time_ss[col], time_es[col])
        Rnet_ctl, Rnet_sen, Rnet_diff, Rnet_cmap, Rnet_clevs = read_LIS_var(file_names[col], land_ctl_path, land_sen_path, 'Rnet', None, None, lat_names, lon_names,  time_ss[col], time_es[col])
        Qle_ctl, Qle_sen, Qle_diff, Qle_cmap, Qle_clevs      = read_LIS_var(file_names[col], land_ctl_path, land_sen_path, 'Qle_tavg', None, None, lat_names, lon_names,  time_ss[col], time_es[col])
        SM_ctl, SM_sen, SM_diff, SM_cmap, SM_clevs           = read_LIS_var(file_names[col], land_ctl_path, land_sen_path, 'SM_top50cm', None, None, lat_names, lon_names,  time_ss[col], time_es[col])

        for row in np.arange(5):

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

            if row==4:
                axs[row,col].set_xticklabels(['135$\mathregular{^{o}}$E','140$\mathregular{^{o}}$E','145$\mathregular{^{o}}$E',
                                              '150$\mathregular{^{o}}$E','155$\mathregular{^{o}}$E'],rotation=25)
            else:
                axs[row,col].set_xticklabels([])

            if col==0:
                axs[row,col].set_yticklabels(['40$\mathregular{^{o}}$S','35$\mathregular{^{o}}$S',
                                              '30$\mathregular{^{o}}$S','25$\mathregular{^{o}}$S'],color=almost_black,alpha=1.)
            else:
                axs[row,col].set_yticklabels(['40$\mathregular{^{o}}$S','35$\mathregular{^{o}}$S',
                                              '30$\mathregular{^{o}}$S','25$\mathregular{^{o}}$S'],color='white',alpha=0)

        Tmax_diff = np.where((SM_ctl<0) & (ssat_top50cm!=0), np.nan, Tmax_diff)
        plot1 = axs[0,col].contourf(lons, lats, Tmax_diff, Tmax_clevs, transform=ccrs.PlateCarree(), cmap=Tmax_cmap, extend='both')
        Rnet_diff= np.where((SM_ctl<0) & (ssat_top50cm!=0), np.nan, Rnet_diff)
        Qle_diff = np.where((SM_ctl<0) & (ssat_top50cm!=0), np.nan, Qle_diff)
        plot2 = axs[1,col].contourf(lons, lats, Rnet_diff, Rnet_clevs, transform=ccrs.PlateCarree(), cmap=Rnet_cmap, extend='both')
        plot3 = axs[2,col].contourf(lons, lats, Qle_diff,  Qle_clevs, transform=ccrs.PlateCarree(), cmap=Qle_cmap, extend='both')
        SM_ctl= np.where((SM_ctl<0) & (ssat_top50cm!=0), np.nan, SM_ctl/ssat_top50cm)
        SM_sen= np.where((SM_sen<0) & (ssat_top50cm!=0), np.nan, SM_sen/ssat_top50cm)
        # plot4 = axs[3,col].contourf(lons, lats, SM_ctl, SM_clevs, transform=ccrs.PlateCarree(), cmap=SM_cmap, extend='both')
        # plot5 = axs[4,col].contourf(lons, lats, SM_sen, SM_clevs, transform=ccrs.PlateCarree(), cmap=SM_cmap, extend='both')
        plot4 = axs[3,col].contourf(lons, lats, SM_ctl, SM_clevs, transform=ccrs.PlateCarree(), cmap=plt.cm.YlGnBu,alpha=0.7, extend='both')
        plot5 = axs[4,col].contourf(lons, lats, SM_sen, SM_clevs, transform=ccrs.PlateCarree(), cmap=plt.cm.YlGnBu,alpha=0.7, extend='both')

    # Adding colormap
    cbar = plt.colorbar(plot1, ax=axs[0], ticklocation="left", pad=0.01, orientation="vertical",
        aspect=20, shrink=0.35, )# drawedges=True) # cax=cax,
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('$\mathregular{^{o}}$C', loc='center',size=12)# rotation=270,
    cbar.set_ticks([-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.],visible=True)
    cbar.set_ticklabels(['','-0.9','','-0.7','','-0.5','','-0.3','','-0.1', '0.1','','0.3','','0.5','','0.7','','0.9',''])

    cbar = plt.colorbar(plot2, ax=axs[1], ticklocation="left", pad=0.01, orientation="vertical",
        aspect=20, shrink=0.35, )# drawedges=True) # cax=cax,
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('W m$\mathregular{^{-2}}$', loc='center',size=12)# rotation=270,
    cbar.set_ticks([-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,2,4,6,8,10,12,14,16,18,20],visible=True)
    cbar.set_ticklabels(['','-18','','-14','','-10','','-6','','-2','2','','6','','10','','14','','18','']) # cax=cax,

    cbar = plt.colorbar(plot3, ax=axs[2], ticklocation="left", pad=0.01, orientation="vertical",
        aspect=20, shrink=0.35, )# drawedges=True) # cax=cax,
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('W m$\mathregular{^{-2}}$', loc='center',size=12)# rotation=270,
    cbar.set_ticks([-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,2,4,6,8,10,12,14,16,18,20],visible=True)
    cbar.set_ticklabels(['','-18','','-14','','-10','','-6','','-2','2','','6','','10','','14','','18','']) # cax=cax,

    cbar = plt.colorbar(plot4, ax=axs[3], ticklocation="left", pad=0.01, orientation="vertical",
        aspect=20, shrink=0.35, )# drawedges=True) # cax=cax,
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('', loc='center',size=12)# rotation=270, # 'm$\mathregular{^{3}}$ m$\mathregular{^{-3}}$'
    cbar.set_ticks([0.1,0.2,0.30,0.40,0.5,0.60,0.70,0.80,0.90],visible=True)
    cbar.set_ticklabels(['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9']) # cax=cax,

    cbar = plt.colorbar(plot5, ax=axs[4], ticklocation="left", pad=0.01, orientation="vertical",
        aspect=20, shrink=0.35, )# drawedges=True) # cax=cax,
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('', loc='center',size=12)# rotation=270, # 'm$\mathregular{^{3}}$ m$\mathregular{^{-3}}$'
    cbar.set_ticks([0.1,0.2,0.30,0.40,0.5,0.60,0.70,0.80,0.90],visible=True)
    cbar.set_ticklabels(['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9']) # cax=cax,

    cbar.ax.tick_params(labelsize=10)#,labelrotation=45)

    # Add titles
    axs[0,0].set_title("2017-18", fontsize=12)
    axs[0,1].set_title("2018-19", fontsize=12)
    axs[0,2].set_title("2019-20", fontsize=12)

    axs[0,0].set_ylabel("ΔT$\mathregular{_{max}}$")
    axs[1,0].set_ylabel("ΔR$\mathregular{_{net}}$")
    axs[2,0].set_ylabel("ΔLH")
    axs[3,0].set_ylabel("θ$\mathregular{_{0.5m}}$/θ$\mathregular{_{sat,0.5m}}$ in Clim")
    axs[4,0].set_ylabel("θ$\mathregular{_{0.5m}}$/θ$\mathregular{_{sat,0.5m}}$ in Dyn")

    texts = ["(a)","(b)","(c)","(d)",
             "(e)","(f)","(g)","(h)",
             "(i)","(j)","(k)","(l)",
             "(m)","(n)","(o)","(p)",
             "(q)","(r)","(s)","(t)"]

    cnt = 0
    for i in np.arange(5):
        for j in np.arange(3):
            axs[i,j].text(0.02, 0.15, texts[cnt], transform=axs[i,j].transAxes, fontsize=14, verticalalignment='top', bbox=props)
            cnt = cnt + 1

    # Apply tight layout
    # plt.tight_layout()

    plt.savefig('./plots/Fig5_spatial_map_summer_Tmax_Rnet_Qle_SM_new.png',dpi=300)


if __name__ == "__main__":

    # ======================= Option =======================
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

        time_ss    = [datetime(2017,12,1,0,0,0,0),
                      datetime(2018,12,1,0,0,0,0),
                      datetime(2019,12,1,0,0,0,0)]

        time_es    = [datetime(2018,3,1,0,0,0,0),
                      datetime(2019,3,1,0,0,0,0),
                      datetime(2020,3,1,0,0,0,0)]

        message    = "Summer_Tmax_Rnet_Qle_SM_all"

        file_names = ["LIS.CABLE.201712-201802.nc",  "LIS.CABLE.201812-201902.nc",  "LIS.CABLE.201912-202002.nc"]

        spatial_map_summer_Tmax_Rnet_Qle_SM(file_names, land_ctl_path, land_sen_path, time_ss=time_ss, time_es=time_es, lat_names="lat",
                            lon_names="lon",loc_lat=loc_lat, loc_lon=loc_lon, wrf_path=wrf_path, message=message)
