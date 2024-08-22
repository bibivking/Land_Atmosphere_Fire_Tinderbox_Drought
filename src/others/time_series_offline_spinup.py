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
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy.interpolate import griddata
import matplotlib.ticker as mticker
from common_utils import *

def plot_time_series(file_paths, var_name, loc_lat=None, loc_lon=None, lat_name=None, lon_name=None, message=None):

    print("======== In plot_time_series =========")
    time,Var = read_var_multi_file(file_paths, var_name, loc_lat, loc_lon, lat_name, lon_name)
    var = np.nanmean(Var,axis=(1,2))

    if var_name in ['Qs','Qsb','Rainf','Evap','ESoil','ECanop','TVeg']:
        var = var*24.*3600.

    fig, ax = plt.subplots()
    ax.plot(np.arange(len(var)), var, c = "blue", label=var_name, alpha=0.5)
    # ax.set_title(var_name)
    ax.legend()
    fig.tight_layout()
    if message == None:
        message = var_name
    else:
        message = message + "_" + var_name

    plt.savefig('./plots/time_series_offline_spinup_'+message+'.png',dpi=300)

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

    case_name    = "spinup_30yrx3" #"spinup_30yrx3_rerun" #
    SP1_off_path = "/g/data/w97/mm3972/model/cable/runs/runs_4_coupled/"+case_name+"/outputs_1st/"
    SP2_off_path = "/g/data/w97/mm3972/model/cable/runs/runs_4_coupled/"+case_name+"/outputs_2nd/"
    SP3_off_path = "/g/data/w97/mm3972/model/cable/runs/runs_4_coupled/"+case_name+"/outputs_3rd/"
    # SP4_off_path = "/g/data/w97/mm3972/model/cable/runs/runs_4_coupled/"+case_name+"/outputs_4th/"

    file_paths   = []
    for year in np.arange(1970,2000):
        file_paths.append(SP1_off_path+"cable_out_"+str(year)+".nc")
    for year in np.arange(1970,2000):
        file_paths.append(SP2_off_path+"cable_out_"+str(year)+".nc")
    for year in np.arange(1970,2000):
        file_paths.append(SP3_off_path+"cable_out_"+str(year)+".nc")
    # for year in np.arange(1970,1996):
    #     file_paths.append(SP4_off_path+"cable_out_"+str(year)+".nc")
    print(file_paths)

    message   = case_name
    var_names = ["WatTable"]
    # var_names = ["GWMoist","Evap","TVeg","ESoil","ECanop","Qs","Qsb",
    #              "WatTable","Qle","Qh","Qg","RadT","VegT","Fwsoil"]
    lat_name  = "latitude"
    lon_name  = "longitude"
    for var_name in var_names:
        plot_time_series(file_paths, var_name, loc_lat=loc_lat, loc_lon=loc_lon, lat_name=lat_name, lon_name=lon_name, message=message)
