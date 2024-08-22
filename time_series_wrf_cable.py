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
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy.interpolate import griddata
import matplotlib.ticker as mticker
from common_utils import *

def calc_diurnal_cycle(Var, time, time_s, time_e, seconds=None):

    # select time periods
    time_cood = time_mask(time, time_s, time_e, seconds)
    time_slt  = time[time_cood]
    var_slt   = Var[time_cood]

    seconds   = [t.seconds for t in time_slt]
    print("seconds",seconds)

    data        = pd.DataFrame([t.seconds for t in time_slt], columns=['seconds'])
    data['var'] = var_slt[:]

    data_group  = data.groupby(by=["seconds"]).mean()
    print(data_group)

    var = data_group['var'].values
    print(var)

    return var

def plot_time_series(file_paths, var_name, time_s=None, time_e=None, loc_lat=None, loc_lon=None, lat_name=None, lon_name=None, message=None):

    print("======== In plot_time_series =========")
    time, Var = read_var_multi_file(file_paths, var_name, loc_lat, loc_lon, lat_name, lon_name)

    if var_name in ["Evap_tavg","TVeg_tavg","ESoil_tavg","ECanop_tavg","Qs_tavg","Qsb_tavg"]:
        Var_daily = time_clip_to_day_sum(time, Var, time_s, time_e, seconds=None)
        Var_daily = Var_daily*3600.
    elif var_name in ["WaterTableD_tavg"]:
        Var_daily = time_clip_to_day(time, Var, time_s, time_e, seconds=None)
        Var_daily = Var_daily/1000.
    else:
        Var_daily = time_clip_to_day(time, Var, time_s, time_e, seconds=None)

    colors        = [ "forestgreen", "yellowgreen","orange","red","black",]
                    # [ "black", "grey","lightcoral","red","orange","gold",
                    #  "yellow","yellowgreen","forestgreen","aquamarine","skyblue",
                    #  "blue","blueviolet","violet","purple","crimson","pink"]
    cleaner_dates = ["2017","2018", "2019", "2020" ]
    xtickslocs    = [0,     365,      730,   1095  ]

    if len(np.shape(Var_daily)) == 3:

        print(var_name, "dimension=",np.shape(Var_daily))

        var = np.nanmean(Var_daily[1:,:,:],axis=(1,2))
        fig, ax = plt.subplots(figsize=[9,9])
        ax.plot(np.arange(len(var)), var, c = "blue", label=var_name, alpha=0.5)
        # ax.set_title(var_name)
        # ax.set(xticks=xtickslocs, xticklabels=cleaner_dates)
        ax.legend()
        fig.tight_layout()
        if message == None:
            message = var_name
        else:
            message = message + "_" + var_name

        plt.savefig('./plots/time_series_'+message+'.png',dpi=300)

    elif len(np.shape(Var_daily)) == 4:

        print(var_name, "dimension=",np.shape(Var_daily))

        labels = ["lyr1","lyr2","lyr3","lyr4","lyr5","lyr6"]
        var = np.nanmean(Var_daily[1:,:,:,:],axis=(2,3))

        fig, ax = plt.subplots(figsize=[9,9])
        ax.plot(np.arange(len(var[:,0])), var, label=labels, alpha=0.5) #c = "blue",

        # ax.set(xticks=xtickslocs, xticklabels=cleaner_dates)
        ax.legend()
        fig.tight_layout()
        if message == None:
            message = var_name
        else:
            message = message + "_" + var_name

        plt.savefig('./plots/time_series_'+message+'.png',dpi=300)

def plot_time_series_diff(file_paths_ctl, file_paths_sen, file_paths_sen_2=None,var_name=None,
                          time_s=None, time_e=None, loc_lat=None, loc_lon=None,
                          lat_name=None, lon_name=None, message=None, multi=None,iveg=None):

    print("======== In plot_time_series =========")
    if var_name == "EF":
        time_ctl, Var_ctl_Qle = read_var_multi_file(file_paths_ctl, "Qle_tavg", loc_lat, loc_lon, lat_name, lon_name)
        time_ctl, Var_ctl_Qh = read_var_multi_file(file_paths_ctl, "Qh_tavg", loc_lat, loc_lon, lat_name, lon_name)

        time_sen, Var_sen_Qle = read_var_multi_file(file_paths_sen, "Qle_tavg", loc_lat, loc_lon, lat_name, lon_name)
        time_sen, Var_sen_Qh = read_var_multi_file(file_paths_sen, "Qh_tavg", loc_lat, loc_lon, lat_name, lon_name)

        ctl_QleQh = Var_ctl_Qle+Var_ctl_Qh
        sen_QleQh = Var_sen_Qle+Var_sen_Qh
        Var_ctl = np.where(abs(ctl_QleQh)>0.01, Var_ctl_Qle/ctl_QleQh,np.nan)
        Var_sen = np.where(abs(sen_QleQh)>0.01, Var_sen_Qle/sen_QleQh,np.nan)
    elif var_name in ["Tmax","Tmin"]:
        time_ctl, Var_ctl = read_var_multi_file(file_paths_ctl, "Tair_f_inst", loc_lat, loc_lon, lat_name, lon_name)
        time_sen, Var_sen = read_var_multi_file(file_paths_sen, "Tair_f_inst", loc_lat, loc_lon, lat_name, lon_name)
    else:
        time_ctl, Var_ctl = read_var_multi_file(file_paths_ctl, var_name, loc_lat, loc_lon, lat_name, lon_name)
        time_sen, Var_sen = read_var_multi_file(file_paths_sen, var_name, loc_lat, loc_lon, lat_name, lon_name)

    if file_paths_sen_2 != None:
        if var_name == "EF":
            time_sen_2, Var_sen_2_Qle = read_var_multi_file(file_paths_sen_2, "Qle_tavg", loc_lat, loc_lon, lat_name, lon_name)
            time_sen_2, Var_sen_2_Qh  = read_var_multi_file(file_paths_sen_2, "Qh_tavg", loc_lat, loc_lon, lat_name, lon_name)
            sen_2_QleQh = Var_sen_2_Qle+Var_sen_2_Qh
            Var_sen_2 = np.where(abs(sen_2_QleQh)>0.01, Var_sen_2_Qle/sen_2_QleQh,np.nan)
        elif var_name in ["Tmax","Tmin"]:
            time_sen_2, Var_sen_2 = read_var_multi_file(file_paths_sen_2, "Tair_f_inst", loc_lat, loc_lon, lat_name, lon_name)
        else:
            time_sen_2, Var_sen_2 = read_var_multi_file(file_paths_sen_2, var_name, loc_lat, loc_lon, lat_name, lon_name)

    if var_name in ["Rainf_tavg","Evap_tavg","TVeg_tavg","ESoil_tavg","ECanop_tavg","Qs_tavg","Qsb_tavg"]:
        Var_daily_ctl = time_clip_to_day_sum(time_ctl, Var_ctl, time_s, time_e, seconds=None)
        Var_daily_ctl = Var_daily_ctl*3600.
    elif var_name in ["WaterTableD_tavg"]:
        Var_daily_ctl = time_clip_to_day(time_ctl, Var_ctl, time_s, time_e, seconds=None)
        Var_daily_ctl = Var_daily_ctl/1000.
    elif var_name in ["Tmax"]:
        Var_daily_ctl = time_clip_to_day_max(time_ctl, Var_ctl, time_s, time_e, seconds=None)
        Var_daily_ctl = Var_daily_ctl-273.15
    elif var_name in ["Tmin"]:
        Var_daily_ctl = time_clip_to_day_min(time_ctl, Var_ctl, time_s, time_e, seconds=None)
        Var_daily_ctl = Var_daily_ctl-273.15
    else:
        Var_daily_ctl = time_clip_to_day(time_ctl, Var_ctl, time_s, time_e, seconds=None)

    if var_name in ["Rainf_tavg","Evap_tavg","TVeg_tavg","ESoil_tavg","ECanop_tavg","Qs_tavg","Qsb_tavg"]:
        Var_daily_sen = time_clip_to_day_sum(time_sen, Var_sen, time_s, time_e, seconds=None)
        Var_daily_sen = Var_daily_sen*3600.
    elif var_name in ["WaterTableD_tavg"]:
        Var_daily_sen = time_clip_to_day(time_sen, Var_sen, time_s, time_e, seconds=None)
        Var_daily_sen = Var_daily_sen/1000.
    elif var_name in ["Tmax"]:
        Var_daily_sen = time_clip_to_day_max(time_sen, Var_sen, time_s, time_e, seconds=None)
        Var_daily_sen = Var_daily_sen-273.15
    elif var_name in ["Tmin"]:
        Var_daily_sen = time_clip_to_day_min(time_sen, Var_sen, time_s, time_e, seconds=None)
        Var_daily_sen = Var_daily_sen-273.15
    else:
        Var_daily_sen = time_clip_to_day(time_sen, Var_sen, time_s, time_e, seconds=None)

    if file_paths_sen_2 != None:
        if var_name in ["Rainf_tavg","Evap_tavg","TVeg_tavg","ESoil_tavg","ECanop_tavg","Qs_tavg","Qsb_tavg"]:
            Var_daily_sen_2 = time_clip_to_day_sum(time_sen_2, Var_sen_2, time_s, time_e, seconds=None)
            Var_daily_sen_2 = Var_daily_sen_2*3600.
        elif var_name in ["WaterTableD_tavg"]:
            Var_daily_sen_2 = time_clip_to_day(time_sen_2, Var_sen_2, time_s, time_e, seconds=None)
            Var_daily_sen_2 = Var_daily_sen_2/1000.
        elif var_name in ["Tmax"]:
            Var_daily_sen_2 = time_clip_to_day_max(time_sen_2, Var_sen_2, time_s, time_e, seconds=None)
            Var_daily_sen_2 = Var_daily_sen_2-273.15
        elif var_name in ["Tmin"]:
            Var_daily_sen_2 = time_clip_to_day_min(time_sen_2, Var_sen_2, time_s, time_e, seconds=None)
            Var_daily_sen_2 = Var_daily_sen_2-273.15
        else:
            Var_daily_sen_2 = time_clip_to_day(time_sen_2, Var_sen_2, time_s, time_e, seconds=None)

    if multi == None:
        Var_daily_diff = Var_daily_sen - Var_daily_ctl

    if iveg != None:
        LC_file        = ["/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/LIS_output/LIS.CABLE.201702-201702.d01.nc"]
        time_lc, LC    = read_var_multi_file(LC_file, "Landcover_inst", loc_lat, loc_lon, lat_name, lon_name)
        landcover_tmp  = time_clip_to_day(time_lc, LC,  datetime(2017,2,1,0,0,0,0),  datetime(2017,3,1,0,0,0,0), seconds=None)
        ntime          = np.shape(Var_daily_ctl)[0]
        nlat           = len(landcover_tmp[0,:,0])
        nlon           = len(landcover_tmp[0,0,:])
        print("ntime = ",ntime,", nlat = ",nlat,", nlon = ",nlon)
        landcover      = np.zeros((ntime,nlat,nlon))
        for i in np.arange(ntime):
            landcover[i,:,:]= landcover_tmp[0,:,:]

    colors        = [ "forestgreen", "yellowgreen","orange","red","black",]
                    # [ "black", "grey","lightcoral","red","orange","gold",
                    #  "yellow","yellowgreen","forestgreen","aquamarine","skyblue",
                    #  "blue","blueviolet","violet","purple","crimson","pink"]
    cleaner_dates = ["2017","2018", "2019", "2020" ]
    xtickslocs    = [0,     365,      730,   1095  ]

    if len(np.shape(Var_daily_ctl)) == 3:

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
            elif np.shape(iveg)[0]>1:
                for i,pft in enumerate(iveg):
                    var_ctl = np.nanmean(np.where(landcover == pft, Var_daily_ctl, np.nan),axis=(1,2))
                    var_sen = np.nanmean(np.where(landcover == pft, Var_daily_sen, np.nan),axis=(1,2))
                    df_ctl  = pd.DataFrame({'ctl': var_ctl})
                    df_sen  = pd.DataFrame({'sen': var_sen})
                    ax.plot( df_ctl['ctl'].rolling(window=30).mean(), c = colors[i], label=var_name+"_ctl PFT="+str(pft), alpha=0.8) # .rolling(window=30).mean()
                    ax.plot( df_sen['sen'].rolling(window=30).mean(), c = colors[i], label=var_name+"_sen PFT="+str(pft), alpha=0.5) # .rolling(window=30).mean()
                    print("iveg = ",pft)
                    for j in np.arange(len(df_ctl['ctl'])):
                        print("day = ", j, ", values = ", df_ctl['ctl'][j], " & ", df_sen['sen'][j])
            else:
                var_ctl = np.nanmean(np.where(landcover == iveg, Var_daily_ctl, np.nan),axis=(1,2))
                var_sen = np.nanmean(np.where(landcover == iveg, Var_daily_sen, np.nan),axis=(1,2))
                ax.plot(np.arange(len(var_ctl)), var_ctl, c = "red",  label=var_name+"_ctl", alpha=0.5)
                ax.plot(np.arange(len(var_sen)), var_sen, c = "blue", label=var_name+"_sen", alpha=0.5)

            if file_paths_sen_2 != None:
                if iveg == None:
                    var_sen_2 = np.nanmean(Var_daily_sen_2[:,:,:],axis=(1,2))
                    ax.plot(np.arange(len(var_sen_2)), var_sen_2, c = "green", label=var_name+"_sen_2", alpha=0.5)
                elif np.shape(iveg)[0]>1:
                    for i,pft in enumerate(iveg):
                        var_sen_2 = np.nanmean(np.where(landcover == pft, Var_daily_sen_2, np.nan),axis=(1,2))
                        ax.plot(np.arange(len(var_sen_2)), var_sen_2, ls="-.", c = colors[i], label=var_name+"_sen_2 PFT="+str(pft), alpha=0.5)
                else:
                    var_sen_2 = np.nanmean(np.where(landcover == iveg, Var_daily_sen_2, np.nan),axis=(1,2))
                    ax.plot(np.arange(len(var_sen_2)), var_sen_2, c = "green", label=var_name+"_sen_2", alpha=0.5)

        # ax.set_title(var_name)
        # ax.set(xticks=xtickslocs, xticklabels=cleaner_dates)
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

    elif len(np.shape(Var_daily_ctl)) == 4:
        # note that 4-D var doesn't support PFT lines
        print(var_name, "dimension=",np.shape(Var_daily_ctl))

        fig, ax = plt.subplots(figsize=[9,9])
        labels = ["lyr1","lyr2","lyr3","lyr4","lyr5","lyr6"]

        if multi == None:
            if iveg == None:
                var = np.nanmean(Var_daily_diff[:,:,:,:],axis=(2,3))
                ax.plot(np.arange(len(var[:,0])), var, label=labels, alpha=0.5) #c = "blue",
            else:
                var = np.nanmean(Var_daily_diff[:,:,:,:],axis=(2,3))
                for i in np.arange(6):
                    var[:,i] = np.nanmean(np.where(landcover == iveg, Var_daily_diff[:,i,:,:], np.nan),axis=(1,2))
        if multi == True:
            if iveg == None:
                var_ctl = np.nanmean(Var_daily_ctl[:,:,:,:],axis=(2,3))
                var_sen = np.nanmean(Var_daily_sen[:,:,:,:],axis=(2,3))
                ax.plot(np.arange(len(var_ctl[:,0])), var_ctl, ls = "-",  label=labels+"_ctl", alpha=0.5) #c = "blue",
                ax.plot(np.arange(len(var_sen[:,0])), var_sen, ls = "-.", label=labels+"_sen", alpha=0.5) #c = "blue",
            else:
                var_ctl = np.nanmean(Var_daily_ctl[:,:,:,:],axis=(2,3))
                var_sen = np.nanmean(Var_daily_sen[:,:,:,:],axis=(2,3))
                for i in np.arange(6):
                    var_ctl[:,i] = np.nanmean(np.where(landcover == iveg, Var_daily_ctl[:,i,:,:], np.nan),axis=(1,2))
                    var_sen[:,i] = np.nanmean(np.where(landcover == iveg, Var_daily_sen[:,i,:,:], np.nan),axis=(1,2))
                ax.plot(np.arange(len(var_ctl[:,0])), var_ctl, ls = "-",  label=labels+"_ctl", alpha=0.5) #c = "blue",
                ax.plot(np.arange(len(var_sen[:,0])), var_sen, ls = "-.", label=labels+"_sen", alpha=0.5) #c = "blue",

            if file_paths_sen_2 != None:
                if iveg == None:
                    var_sen_2 = np.nanmean(Var_daily_sen_2[:,:,:,:],axis=(2,3))
                    ax.plot(np.arange(len(var_sen_2[:,0])), var_sen_2, ls = "--", label=var_name+"_sen_2", alpha=0.5)
                else:
                    var_sen_2 = np.nanmean(Var_daily_sen_2[:,:,:,:],axis=(2,3))
                    for i in np.arange(6):
                        var_sen_2[:,i] = np.nanmean(np.where(landcover == iveg, Var_daily_sen_2[:,i,:,:], np.nan),axis=(1,2))
                    ax.plot(np.arange(len(var_sen_2[:,0])), var_sen_2, ls = "-.", label=labels+"_sen", alpha=0.5) #c = "blue",

        # ax.set(xticks=xtickslocs, xticklabels=cleaner_dates)
        ax.legend()
        fig.tight_layout()

        if message == None:
            message = var_name
        else:
            message = message + "_" + var_name
        if multi == True:
            message = message + "_multi"
        if iveg != None:
            message = message + "_iveg="+str(iveg)
        plt.savefig('./plots/time_series_diff_'+message+'.png',dpi=300)

def plot_time_series_twolines(file_paths_ctl, file_paths_sen, file_paths_sen_2=None,var_name=None,
                          time_s=None, time_e=None, loc_lat=None, loc_lon=None,
                          lat_name=None, lon_name=None, message=None, multi=None,iveg=None):

    # plot multiple lines on one plot
    # for GPP
    s2d        = 3600*24.          # s-1 to d-1
    GPP_scale  = -0.000001*12*s2d   # umol s-1 to g d-1

    print("======== In plot_time_series =========")
    if var_name == "EF":
        time_ctl, Var_ctl_Qle = read_var_multi_file(file_paths_ctl, "Qle_tavg", loc_lat, loc_lon, lat_name, lon_name)
        time_ctl, Var_ctl_Qh  = read_var_multi_file(file_paths_ctl, "Qh_tavg", loc_lat, loc_lon, lat_name, lon_name)
        time_sen, Var_sen_Qle = read_var_multi_file(file_paths_sen, "Qle_tavg", loc_lat, loc_lon, lat_name, lon_name)
        time_sen, Var_sen_Qh  = read_var_multi_file(file_paths_sen, "Qh_tavg", loc_lat, loc_lon, lat_name, lon_name)
        ctl_QleQh = Var_ctl_Qle+Var_ctl_Qh
        sen_QleQh = Var_sen_Qle+Var_sen_Qh
        Var_ctl = np.where(abs(ctl_QleQh)>0.01, Var_ctl_Qle/ctl_QleQh,np.nan)
        Var_sen = np.where(abs(sen_QleQh)>0.01, Var_sen_Qle/sen_QleQh,np.nan)
    elif var_name in ["Tmax","Tmin"]:
        time_ctl, Var_ctl = read_var_multi_file(file_paths_ctl, "Tair_f_inst", loc_lat, loc_lon, lat_name, lon_name)
        time_sen, Var_sen = read_var_multi_file(file_paths_sen, "Tair_f_inst", loc_lat, loc_lon, lat_name, lon_name)
    else:
        time_ctl, Var_ctl = read_var_multi_file(file_paths_ctl, var_name, loc_lat, loc_lon, lat_name, lon_name)
        time_sen, Var_sen = read_var_multi_file(file_paths_sen, var_name, loc_lat, loc_lon, lat_name, lon_name)

    if file_paths_sen_2 != None:
        if var_name == "EF":
            time_sen_2, Var_sen_2_Qle = read_var_multi_file(file_paths_sen_2, "Qle_tavg", loc_lat, loc_lon, lat_name, lon_name)
            time_sen_2, Var_sen_2_Qh  = read_var_multi_file(file_paths_sen_2, "Qh_tavg", loc_lat, loc_lon, lat_name, lon_name)
            sen_2_QleQh = Var_sen_2_Qle+Var_sen_2_Qh
            Var_sen_2 = np.where(abs(sen_2_QleQh)>0.01, Var_sen_2_Qle/sen_2_QleQh,np.nan)
        elif var_name in ["Tmax","Tmin"]:
            time_sen_2, Var_sen_2 = read_var_multi_file(file_paths_sen_2, "Tair_f_inst", loc_lat, loc_lon, lat_name, lon_name)
        else:
            time_sen_2, Var_sen_2 = read_var_multi_file(file_paths_sen_2, var_name, loc_lat, loc_lon, lat_name, lon_name)

    if var_name in ["Rainf_tavg","Evap_tavg","TVeg_tavg","ESoil_tavg","ECanop_tavg","Qs_tavg","Qsb_tavg"]:
        Var_daily_ctl = time_clip_to_day_sum(time_ctl, Var_ctl, time_s, time_e, seconds=None)
        Var_daily_ctl = Var_daily_ctl*3600.
    elif var_name in ["WaterTableD_tavg"]:
        Var_daily_ctl = time_clip_to_day(time_ctl, Var_ctl, time_s, time_e, seconds=None)
        Var_daily_ctl = Var_daily_ctl/1000.
    elif var_name in ["Tmax"]:
        Var_daily_ctl = time_clip_to_day_max(time_ctl, Var_ctl, time_s, time_e, seconds=None)
        Var_daily_ctl = Var_daily_ctl-273.15
    elif var_name in ["Tmin"]:
        Var_daily_ctl = time_clip_to_day_min(time_ctl, Var_ctl, time_s, time_e, seconds=None)
        Var_daily_ctl = Var_daily_ctl-273.15
    elif var_name in ['GPP_tavg','NPP_tavg']:
        Var_daily_ctl = time_clip_to_day(time_ctl, Var_ctl, time_s, time_e, seconds=None)
        Var_daily_ctl = Var_daily_ctl*GPP_scale
    else:
        Var_daily_ctl = time_clip_to_day(time_ctl, Var_ctl, time_s, time_e, seconds=None)

    if var_name in ["Rainf_tavg","Evap_tavg","TVeg_tavg","ESoil_tavg","ECanop_tavg","Qs_tavg","Qsb_tavg"]:
        Var_daily_sen = time_clip_to_day_sum(time_sen, Var_sen, time_s, time_e, seconds=None)
        Var_daily_sen = Var_daily_sen*3600.
    elif var_name in ["WaterTableD_tavg"]:
        Var_daily_sen = time_clip_to_day(time_sen, Var_sen, time_s, time_e, seconds=None)
        Var_daily_sen = Var_daily_sen/1000.
    elif var_name in ["Tmax"]:
        Var_daily_sen = time_clip_to_day_max(time_sen, Var_sen, time_s, time_e, seconds=None)
        Var_daily_sen = Var_daily_sen-273.15
    elif var_name in ["Tmin"]:
        Var_daily_sen = time_clip_to_day_min(time_sen, Var_sen, time_s, time_e, seconds=None)
        Var_daily_sen = Var_daily_sen-273.15
    elif var_name in ['GPP_tavg','NPP_tavg']:
        Var_daily_sen = time_clip_to_day(time_sen, Var_sen, time_s, time_e, seconds=None)
        Var_daily_sen = Var_daily_sen*GPP_scale
    else:
        Var_daily_sen = time_clip_to_day(time_sen, Var_sen, time_s, time_e, seconds=None)

    if file_paths_sen_2 != None:
        if var_name in ["Rainf_tavg","Evap_tavg","TVeg_tavg","ESoil_tavg","ECanop_tavg","Qs_tavg","Qsb_tavg"]:
            Var_daily_sen_2 = time_clip_to_day_sum(time_sen_2, Var_sen_2, time_s, time_e, seconds=None)
            Var_daily_sen_2 = Var_daily_sen_2*3600.
        elif var_name in ["WaterTableD_tavg"]:
            Var_daily_sen_2 = time_clip_to_day(time_sen_2, Var_sen_2, time_s, time_e, seconds=None)
            Var_daily_sen_2 = Var_daily_sen_2/1000.
        elif var_name in ["Tmax"]:
            Var_daily_sen_2 = time_clip_to_day_max(time_sen_2, Var_sen_2, time_s, time_e, seconds=None)
            Var_daily_sen_2 = Var_daily_sen_2-273.15
        elif var_name in ["Tmin"]:
            Var_daily_sen_2 = time_clip_to_day_min(time_sen_2, Var_sen_2, time_s, time_e, seconds=None)
            Var_daily_sen_2 = Var_daily_sen_2-273.15
        elif var_name in ['GPP_tavg','NPP_tavg']:
            Var_daily_sen_2 = time_clip_to_day(time_sen_2, Var_sen_2, time_s, time_e, seconds=None)
            Var_daily_sen_2 = Var_daily_sen_2*GPP_scale
        else:
            Var_daily_sen_2 = time_clip_to_day(time_sen_2, Var_sen_2, time_s, time_e, seconds=None)

    if multi == None:
        Var_daily_diff = Var_daily_sen - Var_daily_ctl

    if iveg != None:
        LC_file        = ["/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/LIS_output/LIS.CABLE.201702-201702.d01.nc"]
        time_lc, LC    = read_var_multi_file(LC_file, "Landcover_inst", loc_lat, loc_lon, lat_name, lon_name)
        landcover_tmp  = time_clip_to_day(time_lc, LC,  datetime(2017,2,1,0,0,0,0),  datetime(2017,3,1,0,0,0,0), seconds=None)
        ntime          = np.shape(Var_daily_ctl)[0]
        nlat           = len(landcover_tmp[0,:,0])
        nlon           = len(landcover_tmp[0,0,:])
        print("ntime = ",ntime,", nlat = ",nlat,", nlon = ",nlon)
        landcover      = np.zeros((ntime,nlat,nlon))
        for i in np.arange(ntime):
            landcover[i,:,:]= landcover_tmp[0,:,:]

    colors        = [ "forestgreen", "yellowgreen","orange","red","black",]
                    # [ "black", "grey","lightcoral","red","orange","gold",
                    #  "yellow","yellowgreen","forestgreen","aquamarine","skyblue",
                    #  "blue","blueviolet","violet","purple","crimson","pink"]
    cleaner_dates = ["2017","2018", "2019", "2020" ]
    xtickslocs    = [0,     365,      730,   1095  ]

    if len(np.shape(Var_daily_ctl)) == 3:

        print(var_name, "dimension=",np.shape(Var_daily_ctl))
        fig, ax = plt.subplots(figsize=[9,9])

        if multi == None:
            if iveg == None:
                var = np.nanmean(Var_daily_diff[:,:,:],axis=(1,2)) # 1
                ax.plot(np.arange(len((var)), var, c = "blue", label="Δ"+var_name, alpha=1.))
            elif np.shape(iveg)[0]>1:
                for i,pft in enumerate(iveg):
                    var = np.nanmean(np.where(landcover == pft, Var_daily_diff, np.nan),axis=(1,2))
                    ax.plot(np.arange(len(var)), var, c = colors[i], label="Δ"+var_name+" PFT="+str(pft), alpha=0.5)
            else:
                var = np.nanmean(np.where(landcover == iveg, Var_daily_diff, np.nan),axis=(1,2))
                ax.plot(np.arange(len(var)), var, c = "blue", label="Δ"+var_name, alpha=0.5)

        if multi == True:
            if iveg == None:
                nt = np.shape(Var_daily_ctl)[0]
                nx = np.shape(Var_daily_ctl)[1] # lats number
                ny = np.shape(Var_daily_ctl)[2] # lons number
                for x in np.arange(nx):
                    for y in np.arange(ny):
                        if np.any(~np.isnan(Var_daily_ctl[:,x,y])):
                            var_ctl = Var_daily_ctl[:,x,y]
                            var_sen = Var_daily_sen[:,x,y]
                            df_ctl  = pd.DataFrame({'ctl': var_ctl})
                            df_sen  = pd.DataFrame({'sen': var_sen})
                            ax.plot(df_ctl['ctl'].rolling(window=30).mean(), c = "red", alpha=0.1) # np.arange(len(var_ctl)),
                            ax.plot(df_sen['sen'].rolling(window=30).mean(), c = "blue", alpha=0.1) # np.arange(len(var_sen)),

                var_ctl = np.nanmean(Var_daily_ctl[:,:,:],axis=(1,2))
                var_sen = np.nanmean(Var_daily_sen[:,:,:],axis=(1,2))
                df_ctl  = pd.DataFrame({'ctl': var_ctl})
                df_sen  = pd.DataFrame({'sen': var_sen})
                ax.plot(df_ctl['ctl'].rolling(window=30).mean(), c = "red",  label=var_name+"_ctl", alpha=1.) # np.arange(len(var_ctl)),
                ax.plot(df_sen['sen'].rolling(window=30).mean(), c = "blue", label=var_name+"_sen", alpha=1.) # np.arange(len(var_sen)),

            elif np.shape(iveg)[0]>1:
                for i,pft in enumerate(iveg):
                    var_ctl = np.nanmean(np.where(landcover == pft, Var_daily_ctl, np.nan),axis=(1,2))
                    var_sen = np.nanmean(np.where(landcover == pft, Var_daily_sen, np.nan),axis=(1,2))
                    df_ctl  = pd.DataFrame({'ctl': var_ctl})
                    df_sen  = pd.DataFrame({'sen': var_sen})
                    ax.plot( df_ctl['ctl'].rolling(window=30).mean(), c = colors[i], label=var_name+"_ctl PFT="+str(pft), alpha=0.8) # .rolling(window=30).mean()
                    ax.plot( df_sen['sen'].rolling(window=30).mean(), c = colors[i], label=var_name+"_sen PFT="+str(pft), alpha=0.5) # .rolling(window=30).mean()
                    print("iveg = ",pft)
                    for j in np.arange(len(df_ctl['ctl'])):
                        print("day = ", j, ", values = ", df_ctl['ctl'][j], " & ", df_sen['sen'][j])
            else:
                var_ctl = np.nanmean(np.where(landcover == iveg, Var_daily_ctl, np.nan),axis=(1,2))
                var_sen = np.nanmean(np.where(landcover == iveg, Var_daily_sen, np.nan),axis=(1,2))
                ax.plot(np.arange(len(var_ctl)), var_ctl, c = "red",  label=var_name+"_ctl", alpha=0.5)
                ax.plot(np.arange(len(var_sen)), var_sen, c = "blue", label=var_name+"_sen", alpha=0.5)

            if file_paths_sen_2 != None:
                if iveg == None:
                    var_sen_2 = np.nanmean(Var_daily_sen_2[:,:,:],axis=(1,2))
                    ax.plot(np.arange(len(var_sen_2)), var_sen_2, c = "green", label=var_name+"_sen_2", alpha=0.5)
                elif np.shape(iveg)[0]>1:
                    for i,pft in enumerate(iveg):
                        var_sen_2 = np.nanmean(np.where(landcover == pft, Var_daily_sen_2, np.nan),axis=(1,2))
                        ax.plot(np.arange(len(var_sen_2)), var_sen_2, ls="-.", c = colors[i], label=var_name+"_sen_2 PFT="+str(pft), alpha=0.5)
                else:
                    var_sen_2 = np.nanmean(np.where(landcover == iveg, Var_daily_sen_2, np.nan),axis=(1,2))
                    ax.plot(np.arange(len(var_sen_2)), var_sen_2, c = "green", label=var_name+"_sen_2", alpha=0.5)

        # ax.set_title(var_name)
        # ax.set(xticks=xtickslocs, xticklabels=cleaner_dates)
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

    elif len(np.shape(Var_daily_ctl)) == 4:
        # note that 4-D var doesn't support PFT lines
        print(var_name, "dimension=",np.shape(Var_daily_ctl))

        fig, ax = plt.subplots(figsize=[9,9])
        labels = ["lyr1","lyr2","lyr3","lyr4","lyr5","lyr6"]

        if multi == None:
            if iveg == None:
                var = np.nanmean(Var_daily_diff[:,:,:,:],axis=(2,3))
                ax.plot(np.arange(len(var[:,0])), var, label=labels, alpha=0.5) #c = "blue",
            else:
                var = np.nanmean(Var_daily_diff[:,:,:,:],axis=(2,3))
                for i in np.arange(6):
                    var[:,i] = np.nanmean(np.where(landcover == iveg, Var_daily_diff[:,i,:,:], np.nan),axis=(1,2))
        if multi == True:
            if iveg == None:
                var_ctl = np.nanmean(Var_daily_ctl[:,:,:,:],axis=(2,3))
                var_sen = np.nanmean(Var_daily_sen[:,:,:,:],axis=(2,3))
                ax.plot(np.arange(len(var_ctl[:,0])), var_ctl, ls = "-",  label=labels+"_ctl", alpha=0.5) #c = "blue",
                ax.plot(np.arange(len(var_sen[:,0])), var_sen, ls = "-.", label=labels+"_sen", alpha=0.5) #c = "blue",
            else:
                var_ctl = np.nanmean(Var_daily_ctl[:,:,:,:],axis=(2,3))
                var_sen = np.nanmean(Var_daily_sen[:,:,:,:],axis=(2,3))
                for i in np.arange(6):
                    var_ctl[:,i] = np.nanmean(np.where(landcover == iveg, Var_daily_ctl[:,i,:,:], np.nan),axis=(1,2))
                    var_sen[:,i] = np.nanmean(np.where(landcover == iveg, Var_daily_sen[:,i,:,:], np.nan),axis=(1,2))
                ax.plot(np.arange(len(var_ctl[:,0])), var_ctl, ls = "-",  label=labels+"_ctl", alpha=0.5) #c = "blue",
                ax.plot(np.arange(len(var_sen[:,0])), var_sen, ls = "-.", label=labels+"_sen", alpha=0.5) #c = "blue",

            if file_paths_sen_2 != None:
                if iveg == None:
                    var_sen_2 = np.nanmean(Var_daily_sen_2[:,:,:,:],axis=(2,3))
                    ax.plot(np.arange(len(var_sen_2[:,0])), var_sen_2, ls = "--", label=var_name+"_sen_2", alpha=0.5)
                else:
                    var_sen_2 = np.nanmean(Var_daily_sen_2[:,:,:,:],axis=(2,3))
                    for i in np.arange(6):
                        var_sen_2[:,i] = np.nanmean(np.where(landcover == iveg, Var_daily_sen_2[:,i,:,:], np.nan),axis=(1,2))
                    ax.plot(np.arange(len(var_sen_2[:,0])), var_sen_2, ls = "-.", label=labels+"_sen", alpha=0.5) #c = "blue",

        # ax.set(xticks=xtickslocs, xticklabels=cleaner_dates)
        ax.legend()
        fig.tight_layout()

        if message == None:
            message = var_name
        else:
            message = message + "_" + var_name
        if multi == True:
            message = message + "_multi"
        if iveg != None:
            message = message + "_iveg="+str(iveg)
        plt.savefig('./plots/time_series_diff_'+message+'.png',dpi=300)

def plot_diurnal_cycle_multilines(file_paths, var_name=None, time_ss=None, time_es=None, loc_lat=None, loc_lon=None,
                                lat_name=None, lon_name=None, message=None):

    # for GPP
    s2d        = 3600*24.          # s-1 to d-1
    GPP_scale  = -0.000001*12*s2d   # umol s-1 to g d-1

    # Set line numbers dimension
    nlines     = len(file_paths)

    # Loop the lines I want to draw
    for i in np.arange(nlines):
        # Read in data
        if var_name == "EF":
            time, Var_Qle = read_var(file_paths[i], "Qle_tavg", loc_lat, loc_lon, lat_name, lon_name)
            time, Var_Qh  = read_var(file_paths[i], "Qh_tavg",  loc_lat, loc_lon, lat_name, lon_name)
            QleQh         = Var_Qle+Var_Qh
            Var           = np.where(abs(QleQh)>0.01, Var_Qle/QleQh,np.nan)
        elif var_name in ["Tmax","Tmin"]:
            time, Var     = read_var(file_paths[i], "Tair_f_inst", loc_lat, loc_lon, lat_name, lon_name)
        else:
            time, Var     = read_var(file_paths[i], var_name, loc_lat, loc_lon, lat_name, lon_name)

        # Calculate regional average
        var = np.nanmean(Var[:], axis=(1,2))

        # Ajust the units
        if var_name in ["Rainf_tavg","Evap_tavg","TVeg_tavg","ESoil_tavg","ECanop_tavg","Qs_tavg","Qsb_tavg"]:
            var = var*3600.
        elif var_name in ["WaterTableD_tavg"]:
            var = var/1000.
        elif var_name in ["Tmax","Tmin","Tair_f_inst"]:
            var = var-273.15
        elif var_name in ['GPP_tavg','NPP_tavg']:
            var = var*GPP_scale
        
        # Calculate diurnal cycle
        var_diurnal = calc_diurnal_cycle(var, time, time_ss[i], time_es[i])

        # Set up new variable
        if i == 0:
            ntime   = len(var_diurnal)
            var_all = np.zeros((nlines, ntime))

        # Give values to the new variable 
        var_all[i,:] = var_diurnal[:]

    # ====================== Plotting ======================
    colors        = [ "forestgreen", "forestgreen", "forestgreen",
                      "lightcoral",  "lightcoral",  "lightcoral"]
    alpha         = [0.3,0.4,0.7, 0.3,0.4,0.7]
    labels        = [ 'ctl_2017', 'ctl_2018', 'ctl_2019', 
                      'sen_2017', 'sen_2018', 'sen_2019']

    cleaner_dates = ["0", "6", "12", "18" ]
    xtickslocs    = [  0,   6,   12,  18  ]

    fig, ax = plt.subplots(figsize=[9,9])

    for i in np.arange(nlines):
        ax.plot(var_all[i], c = colors[i],  label=labels[i], alpha=alpha[i]) 

    ax.legend()
    fig.tight_layout()

    if message == None:
        message = var_name
    else:
        message = message + "_multi_" + var_name

    plt.savefig('./plots/time_series_'+message+'.png',dpi=300)


if __name__ == "__main__":

    # ======================= Option =======================
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


    # small box
    # loc_lat    = [-33,-29]
    # loc_lon    = [147,149]

    # east coast
    loc_lat    = [-33,-27]
    loc_lon    = [152,154]
    PFT        = False

    ####################################
    #         plot_time_series         #
    ####################################

    if 1:
        lat_name       = "lat"
        lon_name       = "lon"
        
        case_name_ctl  = "drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2"
        case_name_sen  = "drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2_obs_LAI_ALB"

        wrf_path       = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_name_ctl+"/WRF_output/wrfout_d01_2017-02-01_06:00:00"
        LIS_path_ctl   = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_name_ctl+"/LIS_output/"
        LIS_path_sen   = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_name_sen+"/LIS_output/"
        
        var_names      = [ "Tair_f_inst",]

        if 1: 

            # ========== Plotting summer ==========
            if 1:
                lat_name       = "lat"
                lon_name       = "lon"
                iveg           = None #[2,5,6,9,14] #2

                case_name_ctl  = "drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2"
                case_name_sen  = "drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2_obs_LAI_ALB"
                case_name_sen_2= None

                time_s         = datetime(2017,1,1,0,0,0,0)
                time_e         = datetime(2020,3,1,0,0,0,0)

                wrf_path       = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_name_ctl+"/WRF_output/wrfout_d01_2017-02-01_06:00:00"
                LIS_path_ctl   = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_name_ctl+"/LIS_output/"
                LIS_path_sen   = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_name_sen+"/LIS_output/"
                LIS_path_sen_2 = None

                var_names      = [ "Tmax","Albedo_inst","LAI_inst"]

                message        = "burn_region_1"
                loc_lat        = [-33,-27]
                loc_lon        = [152,154]

                for var_name in var_names:
                    if var_name in ["Tmax","Tmin"]:
                        file_paths_ctl = [ LIS_path_ctl+'Tair_f_inst/LIS.CABLE.201701-202002.nc' ]
                        file_paths_sen = [ LIS_path_sen+'Tair_f_inst/LIS.CABLE.201701-202002.nc' ]
                        file_paths_sen_2 = None
                    else:
                        file_paths_ctl = [ LIS_path_ctl+var_name+'/LIS.CABLE.201701-202002.nc' ]
                        file_paths_sen = [ LIS_path_sen+var_name+'/LIS.CABLE.201701-202002.nc' ]
                        file_paths_sen_2 = None
                    plot_time_series_twolines(file_paths_ctl,file_paths_sen,file_paths_sen_2, var_name,
                                        time_s=time_s,time_e=time_e, loc_lat=loc_lat, loc_lon=loc_lon,
                                        lat_name=lat_name, lon_name=lon_name, message=message,
                                        multi=True, iveg=iveg)

                # message        = "burn_region_2"
                # loc_lat        = [-33,-27]
                # loc_lon        = [152,154]

                # for var_name in var_names:
                #     if var_name in ["Tmax","Tmin"]:
                #         file_paths_ctl = [ LIS_path_ctl+'Tair_f_inst/LIS.CABLE.201701-202002.nc' ]
                #         file_paths_sen = [ LIS_path_sen+'Tair_f_inst/LIS.CABLE.201701-202002.nc' ]
                #         file_paths_sen_2 = None
                #     else:
                #         file_paths_ctl = [ LIS_path_ctl+var_name+'/LIS.CABLE.201701-202002.nc' ]
                #         file_paths_sen = [ LIS_path_sen+var_name+'/LIS.CABLE.201701-202002.nc' ]
                #         file_paths_sen_2 = None
                #     plot_time_series_twolines(file_paths_ctl,file_paths_sen,file_paths_sen_2, var_name,
                #                         time_s=time_s,time_e=time_e, loc_lat=loc_lat, loc_lon=loc_lon,
                #                         lat_name=lat_name, lon_name=lon_name, message=message,
                #                         multi=True, iveg=iveg)

                # message        = "burn_region_3"
                # loc_lat        = [-33,-27]
                # loc_lon        = [152,154]

                # for var_name in var_names:
                #     if var_name in ["Tmax","Tmin"]:
                #         file_paths_ctl = [ LIS_path_ctl+'Tair_f_inst/LIS.CABLE.201701-202002.nc' ]
                #         file_paths_sen = [ LIS_path_sen+'Tair_f_inst/LIS.CABLE.201701-202002.nc' ]
                #         file_paths_sen_2 = None
                #     else:
                #         file_paths_ctl = [ LIS_path_ctl+var_name+'/LIS.CABLE.201701-202002.nc' ]
                #         file_paths_sen = [ LIS_path_sen+var_name+'/LIS.CABLE.201701-202002.nc' ]
                #         file_paths_sen_2 = None
                #     plot_time_series_twolines(file_paths_ctl,file_paths_sen,file_paths_sen_2, var_name,
                #                         time_s=time_s,time_e=time_e, loc_lat=loc_lat, loc_lon=loc_lon,
                #                         lat_name=lat_name, lon_name=lon_name, message=message,
                #                         multi=True, iveg=iveg)

        if 0:
            '''
            Plot diurnal cycle
            '''

            # ========== Plotting summer ==========
            time_ss        = [ datetime(2017,12,1,0,0,0,0),
                            datetime(2018,12,1,0,0,0,0),
                            datetime(2019,12,1,0,0,0,0),

                            datetime(2017,12,1,0,0,0,0),
                            datetime(2018,12,1,0,0,0,0),
                            datetime(2019,12,1,0,0,0,0),]

            time_es        = [ datetime(2018,3,1,0,0,0,0),
                            datetime(2019,3,1,0,0,0,0),
                            datetime(2020,3,1,0,0,0,0),
                                
                            datetime(2018,3,1,0,0,0,0),
                            datetime(2019,3,1,0,0,0,0),
                            datetime(2020,3,1,0,0,0,0),]



            for var_name in var_names:
                if var_name in ["Tmax","Tmin"]:
                    vname  = 'Tair_f_inst'
                else:
                    vname  = var_name

                file_paths = [  LIS_path_ctl + vname + '/LIS.CABLE.201701-202002.nc',
                                LIS_path_ctl + vname + '/LIS.CABLE.201701-202002.nc',
                                LIS_path_ctl + vname + '/LIS.CABLE.201701-202002.nc',
                                LIS_path_sen + vname + '/LIS.CABLE.201701-202002.nc',
                                LIS_path_sen + vname + '/LIS.CABLE.201701-202002.nc',
                                LIS_path_sen + vname + '/LIS.CABLE.201701-202002.nc' ]
                    
            
                message        = "summer_east_coast"
                
                loc_lat        = [-33,-27]
                loc_lon        = [151,154]

                plot_diurnal_cycle_multilines(file_paths, var_name,
                                        time_ss=time_ss,time_es=time_es, 
                                        loc_lat=loc_lat, loc_lon=loc_lon,
                                        lat_name=lat_name, lon_name=lon_name,
                                        message=message)

                message    = "summer_crop_failure"
                
                loc_lat    = [-36,-33]
                loc_lon    = [144,148]

                plot_diurnal_cycle_multilines(file_paths, var_name,
                                        time_ss=time_ss,time_es=time_es, 
                                        loc_lat=loc_lat, loc_lon=loc_lon,
                                        lat_name=lat_name, lon_name=lon_name,
                                        message=message)

            # ========== Plotting winter ==========
            time_ss        = [ datetime(2017,6,1,0,0,0,0),
                            datetime(2018,6,1,0,0,0,0),
                            datetime(2019,6,1,0,0,0,0),

                            datetime(2017,6,1,0,0,0,0),
                            datetime(2018,6,1,0,0,0,0),
                            datetime(2019,6,1,0,0,0,0),]

            time_es        = [ datetime(2018,8,1,0,0,0,0),
                            datetime(2019,8,1,0,0,0,0),
                            datetime(2020,8,1,0,0,0,0),
                                
                            datetime(2018,8,1,0,0,0,0),
                            datetime(2019,8,1,0,0,0,0),
                            datetime(2020,8,1,0,0,0,0),]



            for var_name in var_names:
                if var_name in ["Tmax","Tmin"]:
                    vname  = 'Tair_f_inst'
                else:
                    vname  = var_name

                file_paths = [  LIS_path_ctl + vname + '/LIS.CABLE.201701-202002.nc',
                                LIS_path_ctl + vname + '/LIS.CABLE.201701-202002.nc',
                                LIS_path_ctl + vname + '/LIS.CABLE.201701-202002.nc',
                                LIS_path_sen + vname + '/LIS.CABLE.201701-202002.nc',
                                LIS_path_sen + vname + '/LIS.CABLE.201701-202002.nc',
                                LIS_path_sen + vname + '/LIS.CABLE.201701-202002.nc' ]
                    
            
                message        = "winter_east_coast"
                
                loc_lat        = [-33,-27]
                loc_lon        = [151,154]

                plot_diurnal_cycle_multilines(file_paths, var_name,
                                        time_ss=time_ss,time_es=time_es, 
                                        loc_lat=loc_lat, loc_lon=loc_lon,
                                        lat_name=lat_name, lon_name=lon_name,
                                        message=message)

                message    = "winter_crop_failure"
                
                loc_lat    = [-36,-33]
                loc_lon    = [144,148]

                plot_diurnal_cycle_multilines(file_paths, var_name,
                                        time_ss=time_ss,time_es=time_es, 
                                        loc_lat=loc_lat, loc_lon=loc_lon,
                                        lat_name=lat_name, lon_name=lon_name,
                                        message=message)