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
import pandas as pd
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature, OCEAN
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy.interpolate import griddata
import matplotlib.ticker as mticker
from common_utils import *


def spital_map(file_paths_ctl,file_paths_sen,obs_path, time_s, time_e, loc_lat=None, loc_lon=None,
                lat_var_name=None, lon_var_name=None):

    '''
    plot annual GPP
    '''

    # convert from umol/m2/s to g C/h
    s2h               = 3600.*24              # s-1 to d-1
    GPP_scale         = -0.000001*12*s2h   # umol s-1 to g C d-1

    # ================== Reading data =================
    time_var, ctl_tmp = read_var_multi_file(file_paths_ctl, 'GPP_tavg', loc_lat, loc_lon, lat_var_name, lon_var_name)
    time_var, sen_tmp = read_var_multi_file(file_paths_sen, 'GPP_tavg', loc_lat, loc_lon, lat_var_name, lon_var_name)
    time_obs, obs_tmp = read_var_multi_file(obs_path, 'GPP_median', loc_lat, loc_lon, 'latitude', 'longitude')

    # read latitude and longitude from observation file
    time, lats_obs    = read_var(obs_path[0], 'latitude', loc_lat, loc_lon, 'latitude', 'longitude')
    time, lons_obs    = read_var(obs_path[0], 'longitude', loc_lat, loc_lon, 'latitude', 'longitude')

    # read latitude and longitude from lis-cable file
    lis_cable         = Dataset(file_paths_ctl[0],  mode='r')
    lats_out          = lis_cable.variables['lat'][:,:]
    lons_out          = lis_cable.variables['lon'][:,:]

    # clip and resample sims to daily data
    ctl_sum           = spatial_var(time_var, ctl_tmp, time_s, time_e, seconds=None)
    sen_sum           = spatial_var(time_var, sen_tmp, time_s, time_e, seconds=None)
    obs_sum           = spatial_var_sum(time_obs, obs_tmp, time_s, time_e, seconds=None)

    ctl_sum           = ctl_sum * GPP_scale * (time_e-time_s).days
    sen_sum           = sen_sum * GPP_scale * (time_e-time_s).days
    obs_sum           = obs_sum

    # interpolate observation to WRF domain
    obs_regrid = regrid_data(lats_obs, lons_obs, lats_out, lons_out, obs_sum, threshold=0)


    # ================= plotting =================
    if 1:
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=[10,4],sharex=True, sharey=True, squeeze=True,
                                subplot_kw={'projection': ccrs.PlateCarree()})
        # plt.subplots_adjust(wspace=-0.44, hspace=0) # left=0.15,right=0.95,top=0.85,bottom=0.05,

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

        states= NaturalEarthFeature(category="cultural", scale="50m",facecolor="none",
                                            name="admin_1_states_provinces_shp")

        # ======================= Set colormap =======================
        cmap1    = plt.cm.gist_earth_r 
        cmap2    = plt.cm.BrBG

        texts   = [ "(a)","(b)","(c)","(d)","(e)",
                    "(f)","(g)","(h)","(i)","(j)",
                    "(k)","(l)","(m)","(n)","(o)",
                    "(p)","(q)","(r)","(s)","(t)"]

        label_x = ["GPP$\mathregular{_{sen}}$",
                   "GPP$\mathregular{_{obs}}$",
                   "ΔGPP$\mathregular{_{sen-obs}}$",
                #    "ΔGPP$\mathregular{_{alb-ctl}}$",
                   ]

        label_y = ["Annual LAI","Spring LAI","Summer LAI","Autumn LAI","Winter LAI"]
        loc_y   = [0.63,0.55,0.47,0.38]

        for i in np.arange(3):

            ax[i].coastlines(resolution="50m",linewidth=1)
            ax[i].set_extent([135,155,-39,-23])
            ax[i].add_feature(states, linewidth=.5, edgecolor="black")

            # Add gridlines
            gl = ax[i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color=almost_black, linestyle='--')
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

        clevs          = np.arange(500,4500.,100)
        clevs_diff     = [-1500,-1400,-1300,-1200,-1100,-1000,-900,-800,-700,-600,-500,-400,-300,-200,-100,-50,
                          50,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500]

        # left: ctl_sum[:,:]
        plot1    = ax[0].contourf(lons_out, lats_out, sen_sum[:,:], levels=clevs, transform=ccrs.PlateCarree(),cmap=cmap1,extend='both') #
        ax[0].text(0.02, 0.15, texts[0], transform=ax[0].transAxes, fontsize=14, verticalalignment='top', bbox=props)
        ax[0].add_feature(OCEAN,edgecolor='none', facecolor="lightgray")

        # middle: ctl_sum[:,:] - obs_regrid[:,:]
        plot2    = ax[1].contourf(lons_out, lats_out, obs_regrid[:,:], levels=clevs, transform=ccrs.PlateCarree(),cmap=cmap1,extend='both') # levels=clevs,
        ax[1].text(0.02, 0.15, texts[1], transform=ax[1].transAxes, fontsize=14, verticalalignment='top', bbox=props)
        ax[1].add_feature(OCEAN,edgecolor='none', facecolor="lightgray")

        # right:  sen_sum[:,:]-ctl_sum[:,:]
        plot3   = ax[2].contourf(lons_out, lats_out, sen_sum[:,:]-obs_regrid[:,:], levels=clevs_diff, transform=ccrs.PlateCarree(),cmap=cmap2,extend='both') #  levels=clevs_diff,
        ax[2].text(0.02, 0.15, texts[2], transform=ax[2].transAxes, fontsize=14, verticalalignment='top', bbox=props)
        ax[2].add_feature(OCEAN,edgecolor='none', facecolor="lightgray")

        cbar = plt.colorbar(plot1, ax=ax[0], ticklocation="right", pad=0.08, orientation="horizontal",
                            aspect=40, shrink=1) # cax=cax,
        cbar.ax.tick_params(labelsize=8,rotation=45)

        cbar = plt.colorbar(plot2, ax=ax[1], ticklocation="right", pad=0.08, orientation="horizontal",
                            aspect=40, shrink=1) # cax=cax,
        cbar.ax.tick_params(labelsize=8,rotation=45)

        cbar = plt.colorbar(plot3, ax=ax[2], ticklocation="right", pad=0.08, orientation="horizontal",
                            aspect=40, shrink=1)
        cbar.ax.tick_params(labelsize=8,rotation=45)

        # set top x label
        ax[0].set_title(label_x[0])#,labelpad=-0.1)#, fontsize=12)
        ax[1].set_title(label_x[1])#,labelpad=-0.1)#, fontsize=12)
        ax[2].set_title(label_x[2])#,labelpad=-0.1)#, fontsize=12)
        if message != None:
            plt.savefig('./plots/spatial_map_GPP_'+message+'_compare.png',dpi=300)
        else:
            plt.savefig('./plots/spatial_map_GPP_compare_2017_2019.png',dpi=300)

def plot_time_series(file_paths_ctl, file_paths_sen, file_paths_sen_2=None,var_name=None,
                          time_s=None, time_e=None, loc_lat=None, loc_lon=None,
                          lat_name=None, lon_name=None, message=None, multi=None,iveg=None):

    # GPP scaling
    s2d               = 3600*24.          # s-1 to d-1
    GPP_scale         = -0.000001*12*s2d   # umol s-1 to g d-1

    # read data
    time_ctl, Var_ctl = read_var_multi_file(file_paths_ctl, var_name, loc_lat, loc_lon, lat_name, lon_name)
    time_sen, Var_sen = read_var_multi_file(file_paths_sen, var_name, loc_lat, loc_lon, lat_name, lon_name)

    Var_daily_ctl = time_clip_to_day(time_ctl, Var_ctl, time_s, time_e, seconds=None)
    Var_daily_sen = time_clip_to_day(time_sen, Var_sen, time_s, time_e, seconds=None)
    Var_daily_ctl = Var_daily_ctl*GPP_scale
    Var_daily_sen = Var_daily_sen*GPP_scale

    if file_paths_sen_2 != None:
        time_sen_2, Var_sen_2 = read_var_multi_file(file_paths_sen_2, 'GPP_median', loc_lat, loc_lon, 'latitude', 'longitude')
        Var_daily_sen_2       = time_clip_to_day(time_sen_2, Var_sen_2, time_s, time_e, seconds=None)

        # interpolate observation to WRF domain
        # read latitude and longitude from observation file
        time_sen_2, lats_sen_2    = read_var(file_paths_sen_2[0], 'latitude', loc_lat, loc_lon, 'latitude', 'longitude')
        time_sen_2, lons_sen_2    = read_var(file_paths_sen_2[0], 'longitude', loc_lat, loc_lon, 'latitude', 'longitude')
        print("np.shape(lats_sen_2)",np.shape(lats_sen_2))
        print("np.shape(lons_sen_2)",np.shape(lons_sen_2))
        print("np.shape(Var_daily_sen_2)",np.shape(Var_daily_sen_2))

        # read latitude and longitude from lis-cable file
        lis_cable         = Dataset(file_paths_ctl[0],  mode='r')
        lats_out          = lis_cable.variables['lat'][:,:]
        lons_out          = lis_cable.variables['lon'][:,:]
        sen_2_regrid      = np.zeros((len(Var_daily_sen_2[:,0,0]),len(lats_out[:,0]),len(lats_out[0,:])))
       
        for i in np.arange(len(Var_daily_sen_2[:,0,0])):
            sen_2_regrid[i,:,:]    = regrid_data(lats_sen_2, lons_sen_2, lats_out, lons_out, Var_daily_sen_2[i,:,:], threshold=0)
        print('np.shape(sen_2_regrid)',np.shape(sen_2_regrid))


    if iveg != None:
        LC_file        = ["/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/LIS_output/Landcover_inst/LIS.CABLE.201701-202002.nc"]
        time_lc, LC    = read_var_multi_file(LC_file, "Landcover_inst", loc_lat, loc_lon, lat_name, lon_name)
        landcover      = time_clip_to_day(time_lc, LC,  time_s, time_e, seconds=None)
        ntime          = np.shape(landcover)[0]
        nlat           = len(landcover[0,:,0])
        nlon           = len(landcover[0,0,:])

        print("ntime = ",ntime,", nlat = ",nlat,", nlon = ",nlon)
        # landcover      = np.zeros((ntime,nlat,nlon))
        # for i in np.arange(ntime):
        #     landcover[i,:,:]= landcover_tmp[0,:,:]

    colors        = [ "forestgreen", "yellowgreen","orange","red","black",]
                    # [ "black", "grey","lightcoral","red","orange","gold",
                    #  "yellow","yellowgreen","forestgreen","aquamarine","skyblue",
                    #  "blue","blueviolet","violet","purple","crimson","pink"]
    cleaner_dates = ["2017","2018", "2019", "2020" ]
    xtickslocs    = [0,        365,    730,  1095  ]

    print(var_name, "dimension=",np.shape(Var_daily_ctl))
    fig, ax = plt.subplots(figsize=[9,9])

    if multi == None:
        if iveg == None:
            var = np.nanmean(Var_daily_diff[:,:,:],axis=(1,2)) # 1
            ax.plot(np.arange(len(var)), var, c = "blue", label="Δ"+var_name, alpha=0.5)
        elif np.shape(iveg)[0]>1:
            for i,pft in enumerate(iveg):
                var = np.nanmean(np.where(landcover == pft, Var_daily_diff, np.nan),axis=(1,2))
                ax.plot(np.arange(len(var)), var, c = colors[i], label="Δ"+var_name+" PFT="+str(pft), alpha=0.5)
        else:
            var = np.nanmean(np.where(landcover == iveg, Var_daily_diff, np.nan),axis=(1,2))
            ax.plot(np.arange(len(var)), var, c = "blue", label="Δ"+var_name, alpha=0.5)

    if multi == True:
        if iveg == None:
            var_ctl = np.nanmean(Var_daily_ctl[:,:,:],axis=(1,2))
            var_sen = np.nanmean(Var_daily_sen[:,:,:],axis=(1,2))
            df_ctl  = pd.DataFrame({'ctl': var_ctl})
            df_sen  = pd.DataFrame({'sen': var_sen})
            ax.plot(df_ctl['ctl'].rolling(window=30).mean(), c = "red",  label=var_name+"_ctl", alpha=0.5) # np.arange(len(var_ctl)),
            ax.plot(df_sen['sen'].rolling(window=30).mean(), c = "blue", label=var_name+"_sen", alpha=0.5) # np.arange(len(var_sen)),
            if file_paths_sen_2 != None:
                var_sen_2 = np.nanmean(Var_daily_sen_2[:,:,:],axis=(1,2))
                df_sen_2  = pd.DataFrame({'sen_2': var_sen_2})
                ax.plot(df_sen_2['sen_2'].rolling(window=30).mean(), c = "green", label=var_name+"_sen_2", alpha=0.5)
        elif np.shape(iveg)[0]>1:
            for i,pft in enumerate(iveg):
                var_ctl = np.nanmean(np.where(landcover == pft, Var_daily_ctl, np.nan),axis=(1,2))
                var_sen = np.nanmean(np.where(landcover == pft, Var_daily_sen, np.nan),axis=(1,2))
                df_ctl  = pd.DataFrame({'ctl': var_ctl})
                df_sen  = pd.DataFrame({'sen': var_sen})
                ax.plot( df_ctl['ctl'].rolling(window=30).mean(), c = colors[i], label=var_name+"_ctl PFT="+str(pft), alpha=0.8) # .rolling(window=30).mean()
                ax.plot( df_sen['sen'].rolling(window=30).mean(), c = colors[i], label=var_name+"_sen PFT="+str(pft), alpha=0.5) # .rolling(window=30).mean()
                print("iveg = ",pft)
                if file_paths_sen_2 != None:
                    sen_2_tmp = sen_2_regrid*1.0
                    for j in np.arange(len(sen_2_regrid[:,0,0])):
                        sen_2_tmp[j,:,:] = np.where(landcover[0,:,:] == pft, sen_2_regrid[j,:,:], np.nan)
                    var_sen_2 = np.nanmean(sen_2_tmp,axis=(1,2))
                    df_sen_2  = pd.DataFrame({'sen_2': var_sen_2})
                    time_sen_2 = [6224,6255,6283,6314,6344,6375,6405,6436,6467,6497,6528,6558,6589,6620,6648,
                                  6679,6709,6740,6770,6801,6832,6862,6893,6923,6954,6985,7013,7044,7074,7105,
                                  7135,7166,7197,7227,7258,7288,7319,7350]
                    ax.plot(time_sen_2, df_sen_2['sen_2'], c = colors[i], label=var_name+"_sen_2 PFT="+str(pft), alpha=0.3) # .rolling(window=30).mean()
                for j in np.arange(len(df_ctl['ctl'])):
                    print("day = ", j, ", values = ", df_ctl['ctl'][j], " & ", df_sen['sen'][j])
        else:
            var_ctl = np.nanmean(np.where(landcover == iveg, Var_daily_ctl, np.nan),axis=(1,2))
            var_sen = np.nanmean(np.where(landcover == iveg, Var_daily_sen, np.nan),axis=(1,2))
            ax.plot(np.arange(len(var_ctl)), var_ctl, c = "red",  label=var_name+"_ctl", alpha=0.5)
            ax.plot(np.arange(len(var_sen)), var_sen, c = "blue", label=var_name+"_sen", alpha=0.5)
            if file_paths_sen_2 != None:
                var_sen_2 = np.nanmean(np.where(landcover == iveg, Var_daily_sen_2, np.nan),axis=(1,2))
                ax.plot(np.arange(len(var_sen_2)), var_sen_2, c = "green", label=var_name+"_sen_2", alpha=0.5)  



    ax.set_title(var_name)
    ax.set(xticks=xtickslocs, xticklabels=cleaner_dates)
    ax.legend()
    fig.tight_layout()
    if message == None:
        message = var_name
    else:
        message = message + "_" + var_name
    if multi == True:
        message = message + "_multi"
    if iveg != None and np.shape(iveg)[0] >1:
        message = message + "_iveg="+str(iveg)
    elif iveg != None:
        message = message + "_iveg="+str(iveg[0])+"-"+str(iveg[-1])

    plt.savefig('./plots/time_series_'+message+'.png',dpi=300)

if __name__ == "__main__":

    # ======================= Option =======================
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


    # small box
    # loc_lat    = [-33,-29]
    # loc_lon    = [147,149]

    # # east coast
    # loc_lat    = [-33,-27]
    # loc_lon    = [152,154]
    # PFT        = False

    ####################################
    #         plot_spatial_map         #
    ####################################
    if 1:
        lat_name       = "lat"
        lon_name       = "lon"
        iveg           = None #[2,5,6,9,14] #2
        var_name       = "GPP_tavg"

        case_name_ctl  = "drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2"
        case_name_sen  = "drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2_obs_LAI_ALB"

        LIS_path_ctl   = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_name_ctl+"/LIS_output/"
        LIS_path_sen   = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_name_sen+"/LIS_output/"

        file_paths_ctl = [ LIS_path_ctl+var_name+'/LIS.CABLE.201701-201912.nc' ]
        file_paths_sen = [ LIS_path_sen+var_name+'/LIS.CABLE.201701-201912.nc' ]
        obs_path       = ["/g/data/w97/mm3972/data/GPP/AusEFlux_GPP_2003_2022_5km_quantiles_v1.1.nc"]

        message        = "ALB_new_2017"
        time_s         = datetime(2017,1,1,0,0,0,0)
        time_e         = datetime(2018,1,1,0,0,0,0)
        spital_map(file_paths_ctl,file_paths_sen,obs_path, time_s, time_e, loc_lat, loc_lon, lat_name, lon_name)


        message        = "ALB_new_2018"
        time_s         = datetime(2018,1,1,0,0,0,0)
        time_e         = datetime(2019,1,1,0,0,0,0)
        spital_map(file_paths_ctl,file_paths_sen,obs_path, time_s, time_e, loc_lat, loc_lon, lat_name, lon_name)


        message        = "ALB_new_2019"
        time_s         = datetime(2019,1,1,0,0,0,0)
        time_e         = datetime(2020,1,1,0,0,0,0)
        spital_map(file_paths_ctl,file_paths_sen,obs_path, time_s, time_e, loc_lat, loc_lon, lat_name, lon_name)

    ####################################
    #         plot_time_series         #
    ####################################
    if 0:
        message        = "ALB_CTL_new"
        lat_name       = "lat"
        lon_name       = "lon"
        iveg           = [2,5,6,9,14] #2

        case_name_ctl  = "drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2"
        case_name_sen  = "drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2_obs_LAI_ALB"

        time_s         = datetime(2017,1,1,0,0,0,0)
        time_e         = datetime(2020,3,1,0,0,0,0)

        wrf_path       = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_name_ctl+"/WRF_output/wrfout_d01_2017-02-01_06:00:00"

        LIS_path_ctl   = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_name_ctl+"/LIS_output/"
        LIS_path_sen   = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_name_sen+"/LIS_output/"


        var_name       = "GPP_tavg"
        file_paths_ctl = [ LIS_path_ctl+var_name+'/LIS.CABLE.201701-202002.nc' ]
        file_paths_sen = [ LIS_path_sen+var_name+'/LIS.CABLE.201701-202002.nc' ]
        file_paths_sen_2 = None # ["/g/data/w97/mm3972/data/GPP/AusEFlux_GPP_2003_2022_5km_quantiles_v1.1.nc"]
        plot_time_series(file_paths_ctl,file_paths_sen,file_paths_sen_2, var_name,
                            time_s=time_s,time_e=time_e, loc_lat=loc_lat, loc_lon=loc_lon,
                            lat_name=lat_name, lon_name=lon_name, message=message,
                            multi=True, iveg=iveg)
