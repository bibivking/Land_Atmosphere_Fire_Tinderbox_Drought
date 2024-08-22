#!/usr/bin/python

__author__ = "Mengyuan Mu"
__email__  = "mu.mengyuan815@gmail.com"


'''
Functions:
1. 
'''

import sys
import numpy as np
import seaborn as sns
import netCDF4 as nc
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from netCDF4 import Dataset
from datetime import datetime, timedelta
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy.interpolate import griddata
from common_utils import *

def calc_stat(pixel):

    # calculate statistics 
    Median  = pixel.groupby(by=['year','season']).median()
    P25     = pixel.groupby(by=['year','season']).quantile(0.25)
    P75     = pixel.groupby(by=['year','season']).quantile(0.75)
    IQR     = P75-P25
    Minimum = P25 - 1.5*IQR # the lowest data point excluding any outliers.
    Maximum = P75 + 1.5*IQR # the largest data point excluding any outliers. Ref: https://www.simplypsychology.org/boxplots.html#:~:text=When%20reviewing%20a%20box%20plot,whiskers%20of%20the%20box%20plot.&text=For%20example%2C%20outside%201.5%20times,Q3%20%2B%201.5%20*%20IQR).
    print("Median ", Median)
    print("P25 ", P25)    
    print("P75 ", P75)    
    print("IQR ", IQR)
    print("Minimum ", Minimum)
    print("Maximum ", Maximum)
    
    return Median,P25,P75,Minimum,Maximum


def read_simulations(file_paths,case_name,time_s=None, time_e=None, loc_lat=None, loc_lon=None,lat_name=None, lon_name=None):
    
    # Read simulations 
    if var_name == "EF":
        time, Var_Qle = read_var_multi_file(file_paths, "Qle_tavg", loc_lat, loc_lon, lat_name, lon_name)
        time, Var_Qh  = read_var_multi_file(file_paths, "Qh_tavg", loc_lat, loc_lon, lat_name, lon_name)
        QleQh         = Var_Qle+Var_Qh
        Var           = np.where(abs(QleQh)>0.01, Var_Qle/QleQh,np.nan)
    else:
        time, Var = read_var_multi_file(file_paths, var_name, loc_lat, loc_lon, lat_name, lon_name)
        
    # calculate daily values
    if var_name in ["Rainf_tavg","Evap_tavg","TVeg_tavg","ESoil_tavg","ECanop_tavg","Qs_tavg","Qsb_tavg"]:
        Var_daily = time_clip_to_day_sum(time, Var, time_s, time_e, seconds=None)
        Var_daily = Var_daily*3600.
    elif var_name in ["WaterTableD_tavg"]:
        Var_daily = time_clip_to_day(time, Var, time_s, time_e, seconds=None)
        Var_daily = Var_daily/1000.
    elif var_name in ["AvgSurfT_tavg","VegT_tavg",'Tair_f_inst']:
        Var_daily = time_clip_to_day(time, Var, time_s, time_e, seconds=None)
        Var_daily = Var_daily-273.15
    else:
        Var_daily = time_clip_to_day(time, Var, time_s, time_e, seconds=None)
    pixel         =  pd.DataFrame({'values': Var_daily.flatten()})
    pixel['case'] = case_name
    print("pixel in case_name=",case_name,pixel)
    
    return pixel

    
def calc_boxplot(file_paths_ctl, file_paths_sen=None, file_paths_sen_2=None,var_name=None,
                    time_s=None, time_e=None, loc_lat=None, loc_lon=None,
                    lat_name=None, lon_name=None, message=None,iveg=False):

    # read simulations
    pixel_ctl = read_simulations(file_paths_ctl,"ctl",time_s, time_e, loc_lat, loc_lon,lat_name,lon_name)
    print("pixel_ctl",pixel_ctl)
    
    if file_paths_sen != None:
        pixel_sen = read_simulations(file_paths_sen,"sen",time_s, time_e, loc_lat, loc_lon,lat_name,lon_name)    
        print("pixel_sen",pixel_sen)
    if file_paths_sen_2 != None:
        pixel_sen_2 = read_simulations(file_paths_sen_2,"sen2",time_s, time_e, loc_lat, loc_lon,lat_name,lon_name)
        print("pixel_sen_2",pixel_sen_2)

    # add the column of year and season
    time_1D             = pd.DataFrame()
    time_1D['datetime'] = pd.date_range(time_s, time_e-timedelta(days=1), freq ='D') # date_range includes the end date so need to minus 1 day
    time_1D['year']     = time_1D['datetime'].dt.year
    time_1D['month']    = time_1D['datetime'].dt.month
    time_1D['season']   = (time_1D['month'].values+10)%12//3+1
    time_1D['year_adjust'] = np.where(time_1D['month'].values==12,time_1D['year'].values+1,time_1D['year'])
    print("time_1D",time_1D)
    
    time_3D_year        = np.zeros((len(time_1D),439,529))
    time_3D_season      = np.zeros((len(time_1D),439,529))
    time_3D_year_adjust = np.zeros((len(time_1D),439,529))
    
    for i in np.arange(len(time_1D)):
        time_3D_year[i,:,:]        = time_1D['year'].values[i]
        time_3D_season[i,:,:]      = time_1D['season'].values[i]
        time_3D_year_adjust[i,:,:] = time_1D['year_adjust'].values[i]
        
    print("time_3D_year",time_3D_year)
    print("time_3D_season",time_3D_season)
    print("time_3D_year_adjust",time_3D_year_adjust)
    print("np.shape(time_3D_year)",np.shape(time_3D_year))
    print("np.shape(time_3D_season)",np.shape(time_3D_season))
    print("np.shape(time_3D_year_adjust)",np.shape(time_3D_year_adjust))
    
    pixel_ctl['year']        = time_3D_year.flatten()
    pixel_ctl['season']      = time_3D_season.flatten()
    pixel_ctl['year_adjust'] = time_3D_year_adjust.flatten()
    print("pixel_ctl",pixel_ctl)
    
    if file_paths_sen != None:
        pixel_sen['year']     = time_3D_year.flatten()
        pixel_sen['season']   = time_3D_season.flatten()
        pixel_sen['year_adjust'] = time_3D_year_adjust.flatten()
        print("pixel_sen",pixel_sen)
    if file_paths_sen_2 != None:
        pixel_sen_2['year']   = time_3D_year.flatten()
        pixel_sen_2['season'] = time_3D_season.flatten()
        pixel_sen_2['year_adjust'] = time_3D_year_adjust.flatten()
        print("pixel_sen_2",pixel_sen_2)
    
    if iveg != None:
        LC_file              = ["/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/drght_2017_2019_bl_pbl2_mp4_sf_sfclay2/LIS_output/LIS.CABLE.201701-202006_ALB_LAI.nc"]
        time_lc, LC          = read_var_multi_file(LC_file, "Landcover_inst", loc_lat, loc_lon, lat_name, lon_name)
        landcover            = time_clip_to_day(time_lc, LC, time_s, time_e, seconds=None)
        pixel_ctl['pft']     = landcover.flatten()
        if file_paths_sen    != None:
            pixel_sen['pft'] = landcover.flatten()
        if file_paths_sen_2  != None:
            pixel_sen_2['pft']= landcover.flatten()
    
    pixel = pixel_ctl
    if file_paths_sen != None:
        pixel = pd.concat([pixel,pixel_sen])
    if file_paths_sen_2 != None:
        pixel = pd.concat([pixel,pixel_sen_2])
    print("pixel",pixel)
    
    pixel_out = pixel[~np.isnan(pixel['values'].values)]
    print("pixel_out",pixel_out)

    # output the dataframe
    fileout_name = var_name+"_ctl"
    if file_paths_sen!=None:
        fileout_name = fileout_name+"_sen"
    if file_paths_sen_2!=None:
        fileout_name = fileout_name+"_sen2"
    pixel_out.to_csv(fileout_name, index=False)
    
    return fileout_name 

def plot_boxplot(fileout_name,var_name,xyhue=['year','values','pft'],year=None,pft=None,season=None,case=None):

    if var_name in ["EF","FWsoil_tavg"]:
        ylabel  = "(-)"
        ranges  = [-0.1,0.4]
    elif var_name in [ "Evap_tavg","Rainf_tavg","TVeg_tavg","ESoil_tavg","ECanop_tavg","Qs_tavg","Qsb_tavg"]:
        ylabel  = "(mm d$\mathregular{^{-1}}$)"
        ranges  = [0,2.5]
    elif var_name in [ "Qle_tavg","Qh_tavg", "Qg_tavg"]:
        ylabel  = "(W m$\mathregular{^{-2}}$)"
        ranges  = [0,200]
    elif var_name in ["Qair_f_inst"]:
        ylabel  = "(g kg$\mathregular{^{-1}}$)"
        ranges  = [0,0.1]
    elif var_name in ["AvgSurfT_tavg","VegT_tavg",'Tair_f_inst']:
        ylabel  = "($\mathregular{^{o}}$C)"
        ranges  = [5,30]
    elif var_name in ["LAI_inst"]:
        ranges = [0,3]
    elif var_name in ["Albedo_inst"]:
        ranges = [0,0.4]
        
    # read the data
    df = pd.read_csv(fileout_name)
    print("df ",df)
    
    message = var_name
    if year != None:
        if "year_adjust" in xyhue:
            df = df[df['year_adjust'].values == year]
            message = message+"_year_adjust="+str(year)
        else:
            df = df[df['year'].values == year]
            message = message+"_year="+str(year)
            
    if pft != None:
        df = df[df['pft'].values == pft]
        message = message+"_pft="+str(pft)
        
    if season != None:
        df = df[df['season'].values == season]
        message = message+"_season="+str(season)
        
    if case != None:
        df = df[df['case'].values == case]
        message = message+"_case="+case
        
    print("df ",df)
    
    # plotting
    sns.set_style("ticks")
    sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
    sns.set_theme(style="ticks", palette="pastel")

    fig, axs = plt.subplots() #1, 1, figsize=(12,7), constrained_layout = True
    
    plt.rcParams['text.usetex']     = False
    plt.rcParams['font.family']     = "sans-serif"
    plt.rcParams['font.serif']      = "Times New Roman"
    plt.rcParams['axes.linewidth']  = 1.5
    plt.rcParams['axes.labelsize']  = 14
    plt.rcParams['font.size']       = 14
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams["legend.markerscale"] = 3.0

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

    # Plotting boxplot
    sns.boxplot(x=xyhue[0], y=xyhue[1], data=df, ax=axs, showfliers=False, hue=xyhue[2],whis=0) # ["m", "g"], palette=["b"], whis=0
    axs.set_ylim(ranges)
    fig.savefig("./plots/boxplot/boxplot_Tinderbox_"+message+".png", bbox_inches='tight', dpi=300, pad_inches=0.1) #

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

    ####################################
    #         plot_time_series         #
    ####################################
    if 1:
        message        = "bl_pbl2_mp4_sf_sfclay2_CTL_ALB"
        lat_name       = "lat"
        lon_name       = "lon"
        iveg           = True
        
        case_name_ctl  = "drght_2017_2019_bl_pbl2_mp4_sf_sfclay2"
        case_name_sen  = "drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2_obs_LAI_ALB"
        case_name_sen_2= "drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2_obs_LAI_ALB_HR"

        time_s         = datetime(2017,1,1,0,0,0,0)
        time_e         = datetime(2020,7,1,0,0,0,0)

        wrf_path       = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/"+case_name_ctl+"/WRF_output/wrfout_d01_2017-02-01_06:00:00"
        LIS_path_ctl   = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/"+case_name_ctl+"/LIS_output/"
        LIS_path_sen   = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/"+case_name_sen+"/LIS_output/"
        LIS_path_sen_2 = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/"+case_name_sen_2+"/LIS_output/"

        file_paths_ctl = [ LIS_path_ctl+"LIS.CABLE.201701-202006_met.nc" ]

        file_paths_sen = [ LIS_path_sen+"LIS.CABLE.201701-202006_met.nc" ]

        file_paths_sen_2=None

        # =============== save pixels  ================
        if 0:
            var_names      = ["Tair_f_inst","Wind_f_inst","Qair_f_inst","Psurf_f_inst","SWdown_f_inst","LWdown_f_inst"] 
                            # [ "Evap_tavg","Rainf_tavg","TVeg_tavg","ESoil_tavg"] # water
                            # [ "Qh_tavg","Qle_tavg","EF",] # energy
                            # ["AvgSurfT_tavg","VegT_tavg"] # temperature
                            # "WaterTableD_tavg",'Albedo_inst',"SoilMoist_inst", "FWsoil_tavg", "GWwb_tavg",,"ECanop_tavg","Qs_tavg","Qsb_tavg",]

            for var_name in var_names:
                fileout_name = calc_boxplot(file_paths_ctl,file_paths_sen,file_paths_sen_2, var_name,
                                time_s=time_s,time_e=time_e, loc_lat=loc_lat, loc_lon=loc_lon,
                                lat_name=lat_name, lon_name=lon_name, message=message,iveg=iveg)


        # =============== plot boxplot ================
        if 1:
            year   = None
            season = None # 1: MAM 2:JJA 3:SON 4:DJF
            case   = None
            xyhue  = ["year","values","case"]
            
            pft    = 2 # [2,5,6,9,14]
            fileout_name = "Albedo_inst_ctl_sen"
            var_name     = "Albedo_inst"
            plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)       
            fileout_name = "LAI_inst_ctl_sen"
            var_name     = "LAI_inst"
            plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)    
            fileout_name = "EF_ctl_sen"
            var_name     = "EF"
            plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)
            fileout_name = "Qle_tavg_ctl_sen"
            var_name     = "Qle_tavg"
            plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)
            fileout_name = "Qh_tavg_ctl_sen"
            var_name     = "Qh_tavg"
            plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)
            fileout_name = "Tair_f_inst_ctl_sen"
            var_name     = "Tair_f_inst"
            plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)
            fileout_name = "TVeg_tavg_ctl_sen"
            var_name     = "TVeg_tavg"
            plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)
            fileout_name = "Evap_tavg_ctl_sen"
            var_name     = "Evap_tavg"
            plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)
                 
            
            pft    = 5 # [2,5,6,9,14]
            fileout_name = "Albedo_inst_ctl_sen"
            var_name     = "Albedo_inst"
            plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)       
            fileout_name = "LAI_inst_ctl_sen"
            var_name     = "LAI_inst"
            plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)    
            fileout_name = "EF_ctl_sen"
            var_name     = "EF"
            plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)
            fileout_name = "Qle_tavg_ctl_sen"
            var_name     = "Qle_tavg"
            plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)
            fileout_name = "Qh_tavg_ctl_sen"
            var_name     = "Qh_tavg"
            plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)
            fileout_name = "Tair_f_inst_ctl_sen"
            var_name     = "Tair_f_inst"
            plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)
            fileout_name = "TVeg_tavg_ctl_sen"
            var_name     = "TVeg_tavg"
            plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)
            fileout_name = "Evap_tavg_ctl_sen"
            var_name     = "Evap_tavg"
            plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)
            
            pft    = 6 # [2,5,6,9,14]
            fileout_name = "Albedo_inst_ctl_sen"
            var_name     = "Albedo_inst"
            plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)       
            fileout_name = "LAI_inst_ctl_sen"
            var_name     = "LAI_inst"
            plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)    
            fileout_name = "EF_ctl_sen"
            var_name     = "EF"
            plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)
            fileout_name = "Qle_tavg_ctl_sen"
            var_name     = "Qle_tavg"
            plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)
            fileout_name = "Qh_tavg_ctl_sen"
            var_name     = "Qh_tavg"
            plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)
            fileout_name = "Tair_f_inst_ctl_sen"
            var_name     = "Tair_f_inst"
            plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)
            fileout_name = "TVeg_tavg_ctl_sen"
            var_name     = "TVeg_tavg"
            plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)
            fileout_name = "Evap_tavg_ctl_sen"
            var_name     = "Evap_tavg"
            plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)  
            
            pft    = 9 # [2,5,6,9,14]
            fileout_name = "Albedo_inst_ctl_sen"
            var_name     = "Albedo_inst"
            plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)       
            fileout_name = "LAI_inst_ctl_sen"
            var_name     = "LAI_inst"
            plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)    
            fileout_name = "EF_ctl_sen"
            var_name     = "EF"
            plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)
            fileout_name = "Qle_tavg_ctl_sen"
            var_name     = "Qle_tavg"
            plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)
            fileout_name = "Qh_tavg_ctl_sen"
            var_name     = "Qh_tavg"
            plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)
            fileout_name = "Tair_f_inst_ctl_sen"
            var_name     = "Tair_f_inst"
            plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)
            fileout_name = "TVeg_tavg_ctl_sen"
            var_name     = "TVeg_tavg"
            plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)
            fileout_name = "Evap_tavg_ctl_sen"
            var_name     = "Evap_tavg"
            plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)    
            
            pft    = 14 # [2,5,6,9,14]
            fileout_name = "Albedo_inst_ctl_sen"
            var_name     = "Albedo_inst"
            plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)       
            fileout_name = "LAI_inst_ctl_sen"
            var_name     = "LAI_inst"
            plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)    
            fileout_name = "EF_ctl_sen"
            var_name     = "EF"
            plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)
            fileout_name = "Qle_tavg_ctl_sen"
            var_name     = "Qle_tavg"
            plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)
            fileout_name = "Qh_tavg_ctl_sen"
            var_name     = "Qh_tavg"
            plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)
            fileout_name = "Tair_f_inst_ctl_sen"
            var_name     = "Tair_f_inst"
            plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)
            fileout_name = "TVeg_tavg_ctl_sen"
            var_name     = "TVeg_tavg"
            plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)
            fileout_name = "Evap_tavg_ctl_sen"
            var_name     = "Evap_tavg"
            plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case) 
            
            
            

            
            
            # fileout_name = "VegT_tavg_ctl_sen"
            # var_name     = "VegT_tavg"
            # plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)
            
            # fileout_name = "ESoil_tavg_ctl_sen"
            # var_name     = "ESoil_tavg"
            # plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case) 
            
            # fileout_name = "SWdown_f_inst_ctl_sen"
            # var_name     = "SWdown_f_inst"
            # plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)
            
            # fileout_name = "Wind_f_inst_ctl_sen"
            # var_name     = "Wind_f_inst"
            # plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)
            
            # fileout_name = "Psurf_f_inst_ctl_sen"
            # var_name     = "Psurf_f_inst"
            # plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)
            
            # fileout_name = "Qair_f_inst_ctl_sen"
            # var_name     = "Qair_f_inst"
            # plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)
            
            # fileout_name = "LWdown_f_inst_ctl_sen"
            # var_name     = "LWdown_f_inst"
            # plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)
            
            # fileout_name = "AvgSurfT_tavg_ctl_sen"
            # var_name     = "AvgSurfT_tavg"
            # plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)

            # fileout_name = "Rainf_tavg_ctl_sen"
            # var_name     = "Rainf_tavg"
            # plot_boxplot(fileout_name,var_name,xyhue,year,pft,season,case)