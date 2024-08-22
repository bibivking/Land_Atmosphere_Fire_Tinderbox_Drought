#!/usr/bin/python

from netCDF4 import Dataset
import xarray as xr
import netCDF4 as nc
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.interpolate import griddata
from wrf import (getvar, interplevel, get_cartopy, cartopy_xlim,
                 cartopy_ylim, to_np, latlon_coords, ALL_TIMES)

# =============================== Basci Functions ==============================
def leap_year(year):
   if (year % 4) == 0:
       return 366
   else:
       return 365

def get_scale(var_name):
    pa2hpa     = ['ps']
    m2mm       = ['tp']
    mm_s2mm_yr = ['Qs','Qsb','Rainf','Evap','ESoil','ECanop','TVeg']
    mm_s2mm_day= ['Rainf_tavg']
    if var_name in mm_s2mm_yr:
        scale = 24.*3600.*365
    elif var_name in mm_s2mm_day:
        scale = 24.*3600.
    elif var_name in pa2hpa:
        scale = 0.01
    elif var_name in m2mm:
        scale = 1000.
    else:
        scale = 1.
    return scale

def UTC_to_AEST(time):

    Time = time + timedelta(hours=10)

    return Time

# ===================================== Mask ===================================
def tree_mask(file_path,pft_var_name):

    var = Dataset(file_path, mode='r')
    if pft_var_name == "Landcover_inst":
        pft = var.variables[pft_var_name][0,:,:]
    elif pft_var_name == "iveg":
        pft = var.variables[pft_var_name][:,:]

    return pft

def mask_by_lat_lon(file_path, loc_lat, loc_lon, lat_name, lon_name):

    '''
    make mask for the selected region
    '''

    file = nc.Dataset(file_path, mode='r')
    if len(np.shape(file.variables[lat_name][:])) == 3:
        # #print("len(np.shape(file.variables[lat_name][:])) == 3")
        # #print(lat_name)
        lat  = file.variables[lat_name][0,:,:]
        lon  = file.variables[lon_name][0,:,:]
    else:
        lat  = file.variables[lat_name][:]
        lon  = file.variables[lon_name][:]

    if 'GLEAM' in file_path:
        lat = lat[::-1]
    # #print(lat)
    # #print(lon)

    if len(np.shape(lat)) == 1:
        # #print("len(np.shape(lat)) == 1")
        lat_spc = lat[1] - lat[0]
        lon_spc = lon[1] - lon[0]
        lons, lats = np.meshgrid(lon, lat)
        mask  = (lats > (loc_lat[0] - lat_spc/2)) & (lats < (loc_lat[1] + lat_spc/2)) & (lons > (loc_lon[0] - lon_spc/2)) & (lons < (loc_lon[1] + lon_spc/2))
    elif len(np.shape(lat)) == 2:
        # #print("len(np.shape(lat)) == 2")
        ### caution: lat=100, lon=100 is a random pixel, lis run over a small domain may not have such a point
        lat_spc = lat[150,150] - lat[149,150]
        lon_spc = lon[150,150] - lon[150,149]
        #print("#print lat_spc  & lon_spc in def mask_by_lat_lon ", lat_spc, lon_spc)
        # #print(lat_spc)
        # #print(lon_spc)
        ### caution: due to irregular space in lis, using lat/lon +lat/lon_spc/2 may includes more than 1 pixel.
        ### I therefore let the space divied by 2.1 rather than 2
        mask  = (lat > (loc_lat[0] - lat_spc/2.1)) & (lat < (loc_lat[1] + lat_spc/2.1)) & (lon > (loc_lon[0] - lon_spc/2.1)) & (lon < (loc_lon[1] + lon_spc/2.1))
    # #print(np.shape(mask))
    return mask

def time_mask(time, time_s, time_e, seconds=None):

    '''
    Checked on 14 Dec 2021, no problem was identified
    '''

    # #print("In time_mask")

    Time_s = time_s - datetime(2000,1,1,0,0,0)
    Time_e = time_e - datetime(2000,1,1,0,0,0)

    if seconds == None:
        time_cood = (time>=Time_s) & (time<Time_e)
    else:
        time_cood = []
        for j in np.arange(len(time)):
            if seconds[0] >= seconds[1]:
                if_seconds = (time[j].seconds >= seconds[0]) | (time[j].seconds < seconds[1])
            else:
                if_seconds = (time[j].seconds >= seconds[0]) & (time[j].seconds < seconds[1])
            time_cood.append( (time[j]>=Time_s) & (time[j]<Time_e) & if_seconds)

    return time_cood

# ================================ Read variables ==============================

def read_var(file_path, var_name, loc_lat=None, loc_lon=None, lat_name=None, lon_name=None):

    '''
    Read observation data, output time coordinate and variable array
    Output: AEST time
    '''

    #print(var_name)

    obs_file   = Dataset(file_path, mode='r')
    time_tmp   = nc.num2date(obs_file.variables['time'][:],obs_file.variables['time'].units,
                 only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    if 'AWAP' in file_path or 'cable_out' in file_path or 'EHF' in file_path:
        time       = time_tmp - datetime(2000,1,1,0,0,0)
    else:
        time       = UTC_to_AEST(time_tmp) - datetime(2000,1,1,0,0,0)
    ntime      = len(time)

    if loc_lat == None:
        Var_tmp = obs_file.variables[var_name][:]
        # #print('Var_tmp',Var_tmp)
        if hasattr(obs_file.variables[var_name], '_FillValue'):
            # hasattr(a,"b"): check whether object a has attribute 'b'
            def_val = obs_file.variables[var_name]._FillValue
            Var = np.where(Var_tmp == def_val, np.nan, Var_tmp)
        elif hasattr(obs_file.variables[var_name], '_fillvalue'):
            def_val = obs_file.variables[var_name]._fillvalue
            Var = np.where(Var_tmp == def_val, np.nan, Var_tmp)
        else:
            Var = Var_tmp
    else:
        # selected region
        if var_name == lat_name or var_name == lon_name:
            # read lat or lon
            mask = mask_by_lat_lon(file_path, loc_lat, loc_lon, lat_name, lon_name)
            if 'GLEAM' in file_path:
                #print("1")
                lat  = obs_file.variables[lat_name][:]
                lat  = lat[::-1]
                lon  = obs_file.variables[lon_name]
            else:
                #print("2")
                lat  = obs_file.variables[lat_name]
                lon  = obs_file.variables[lon_name]

            if len(np.shape(lat)) == 1:
                lons, lats = np.meshgrid(lon, lat)
                if var_name == lat_name:
                    Var = np.where(mask,lats,np.nan)
                if var_name == lon_name:
                    Var = np.where(mask,lons,np.nan)
                # #print(np.shape(Var))
            elif len(np.shape(lat)) == 2:
                Var = np.where(mask, obs_file.variables[var_name][:], np.nan)
                # #print(np.shape(Var))
            elif len(np.shape(lat)) == 3:
                Var = np.where(mask, obs_file.variables[var_name][0,:,:], np.nan)
                # #print(np.shape(Var))
        else:
            # read var except lat or lat
            mask = mask_by_lat_lon(file_path, loc_lat, loc_lon, lat_name, lon_name)
            mask_multi = [ mask ] * ntime

            if 'v3-6a' in file_path:
                #print("1")
                tmp = obs_file.variables[var_name][:,::-1,:]
            elif 'GLEAM' in file_path:
                # change GLEAM's coordinates from (time, lon, lat) to (time, lat, lon)
                #print("2")
                tmp = np.moveaxis(obs_file.variables[var_name], -1, 1)
                tmp = tmp[:,::-1,:]
            else:
                #print("3")
                tmp = obs_file.variables[var_name][:]

            if var_name in ["SoilMoist_inst","SoilTemp_inst", "SoilMoist", "SoilTemp"]:
                nlat    = len(mask[:,0])
                nlon    = len(mask[0,:])
                Var_tmp = np.zeros((ntime,6,nlat,nlon))
                for j in np.arange(6):
                    Var_tmp[:,j,:,:] = np.where(mask_multi,tmp[:,j,:,:],np.nan)
            else:
                Var_tmp = np.where(mask_multi,tmp,np.nan)

            ##print("#print Var_tmp in def read_var: ", Var_tmp)
            # #print(np.shape(Var_tmp))
            if hasattr(obs_file.variables[var_name], '_FillValue'):
                def_val = obs_file.variables[var_name]._FillValue
                Var = np.where(Var_tmp == def_val, np.nan, Var_tmp)
            elif hasattr(obs_file.variables[var_name], '_fillvalue'):
                def_val = obs_file.variables[var_name]._fillvalue
                Var = np.where(Var_tmp == def_val, np.nan, Var_tmp)
            else:
                Var = Var_tmp
    return time,Var

def read_var_multi_file(file_paths, var_name, loc_lat=None, loc_lon=None, lat_name=None, lon_name=None):

    '''
    Read observation data, output time coordinate and variable array
    Output: AEST time

    Please don't use this function to read lat and lon
    '''

    #print(var_name)

    # Initilizing
    time = []

    for i, file_path in enumerate(file_paths):

        #print("file_path = ", file_path)

        var_file   = Dataset(file_path, mode='r')
        time_tmp   = nc.num2date(var_file.variables['time'][:],var_file.variables['time'].units,
                     only_use_cftime_datetimes=False, only_use_python_datetimes=True)
        if 'AWAP' in file_path or 'cable_out' in file_path:
            time_tmp   = time_tmp - datetime(2000,1,1,0,0,0)
        else:
            time_tmp   = UTC_to_AEST(time_tmp) - datetime(2000,1,1,0,0,0)
        ntime      = len(time_tmp)

        if i == 0:
            time = time_tmp
        else:
            time = np.concatenate((time, time_tmp), axis=0)
        # #print(time)

        if loc_lat == None:
            Var_tmp = var_file.variables[var_name][:]
            if hasattr(var_file.variables[var_name], '_FillValue'):
                # hasattr(a,"b"): check whether object a has attribute 'b'
                def_val = var_file.variables[var_name]._FillValue
                Var = np.where(Var_tmp == def_val, np.nan, Var_tmp)
            elif hasattr(var_file.variables[var_name], '_fillvalue'):
                def_val = var_file.variables[var_name]._fillvalue
                Var = np.where(Var_tmp == def_val, np.nan, Var_tmp)
            else:
                Var = Var_tmp
            # #print('np.unique(Var)',np.unique(Var))
        else:
            # selected region
            # read var except lat or lat
            if i == 0:
                mask = mask_by_lat_lon(file_path, loc_lat, loc_lon, lat_name, lon_name)

            mask_multi = [ mask ] * ntime

            # #print("shape of mask_multi",np.shape(mask_multi))

            if 'v3-6a' in file_paths[0]:
                #print("1: 'v3-6a' in file_paths[0]")
                tmp = var_file.variables[var_name][:,::-1,:]
            elif 'GLEAM' in file_paths[0]:
                #print("2: 'GLEAM' in file_paths[0]")
                # change GLEAM's coordinates from (time, lon, lat) to (time, lat, lon)
                tmp = np.moveaxis(var_file.variables[var_name], -1, 1)
                tmp = tmp[:,::-1,:]
            else:
                #print("3")
                tmp = var_file.variables[var_name][:]

            # plt.contourf(tmp[0])
            # plt.show()

            if var_name in ["SoilMoist_inst","SoilTemp_inst", "SoilMoist", "SoilTemp"]:
                nlat    = len(mask[:,0])
                nlon    = len(mask[0,:])
                Var_tmp = np.zeros((ntime,6,nlat,nlon))
                for j in np.arange(6):
                    Var_tmp[:,j,:,:] = np.where(mask_multi,tmp[:,j,:,:],np.nan)
            else:
                Var_tmp = np.where(mask_multi,tmp,np.nan)

            if hasattr(var_file.variables[var_name], '_FillValue'):
                def_val = var_file.variables[var_name]._FillValue
                Var = np.where(Var_tmp == def_val, np.nan, Var_tmp)
            elif hasattr(var_file.variables[var_name], '_fillvalue'):
                def_val = var_file.variables[var_name]._fillvalue
                Var = np.where(Var_tmp == def_val, np.nan, Var_tmp)
            else:
                Var = Var_tmp
        if i == 0:
            var = Var
        else:
            var = np.concatenate((var, Var), axis=0)

    #print("=== In read_var_multi_file ===")
    # #print("time = ", np.shape(time))
    # #print("var = ", np.shape(var))

    return time,var

def read_wrf_time(file_path):

    '''
    output: AEST time
    '''

    # Open the NetCDF file
    encoding = 'utf-8' # Times in WRF output is btype, convert to string

    wrf      = Dataset(file_path)
    ntime    = len(wrf.variables['Times'][:,0])
    time_tmp = []

    for i in np.arange(ntime):
        time_temp = datetime.strptime(str(wrf.variables['Times'][i,:], encoding),'%Y-%m-%d_%H:%M:%S')
        time_tmp.append(UTC_to_AEST(time_temp) - datetime(2000,1,1))

    time = np.array(time_tmp)

    return time

def read_wrf_surf_var(file_path, var_name, loc_lat=None, loc_lon=None, mask_map=None):

    '''
    output: [time,lat,lon]
    '''

    #print("read "+var_name+" from wrf output")

    var_3D = [
                'rh2',  # 2m Relative Humidity
                'T2',   # 2m Temperature
                'td2',  # 2m Dew Point Temperature
                'slp',  # Sea Level Pressure
                'ter',  # Model Terrain Height
                'ctt',  # Cloud Top Temperature
                'mdbz', # Maximum Reflectivity
                'pw',   # Precipitable Water
                'updraft_helicity', # Updraft Helicity
                'helicity',        # Storm Relative Helicity
              ]

    wrf_file = Dataset(file_path)
    p        = getvar(wrf_file, "pressure",timeidx=ALL_TIMES)
    if var_name == 'cape_2d':
        # 'cape_2d', # 2D CAPE (MCAPE/MCIN/LCL/LFC)
        var_tmp  = getvar(wrf_file, var_name, timeidx=ALL_TIMES)[0]
        #print("======= cape_2d =======")
        # #print(var_tmp)
    elif var_name == 'cloudfrac':
        # 'cloudfrac', # Cloud Fraction
        var_tmp  = getvar(wrf_file, var_name, timeidx=ALL_TIMES)[0]
        #print("======= cloudfrac =======")
        # #print(var_tmp)
    else:
        var_tmp  = getvar(wrf_file, var_name, timeidx=ALL_TIMES)

    if loc_lat == None:
        if mask_map is not None:
            ntime      = len(p[:,0,0,0])
            mask_multi = [ mask_map ] * ntime
            var        = np.where(mask_multi,var_tmp,np.nan)
        else:
            var        = var_tmp
    else:
        ### need to fix, not work
        ntime      = len(p[:,0,0,0])
        mask       = mask_by_lat_lon(file_path, loc_lat, loc_lon, 'XLAT', 'XLONG')
        if mask_map is not None:
            mask   = np.where(np.all([mask,mask_map],axis=0), True, False)
        # np.savetxt("check",mask)
        mask_multi = [ mask ] * ntime
        var        = np.where(mask_multi,var_tmp,np.nan)

    return var

def read_wrf_surf_var_multi_files(file_paths, var_name, loc_lat=None, loc_lon=None, mask_map=None):

    '''
    output: [time,lat,lon]
    '''

    #print("read "+var_name+" from wrf output")

    var_3D = [
                'rh2',  # 2m Relative Humidity
                'T2',   # 2m Temperature
                'td2',  # 2m Dew Point Temperature
                'slp',  # Sea Level Pressure
                'ter',  # Model Terrain Height
                'ctt',  # Cloud Top Temperature
                'mdbz', # Maximum Reflectivity
                'pw',   # Precipitable Water
                'updraft_helicity', # Updraft Helicity
                'helicity',        # Storm Relative Helicity
              ]

    for i, file_path in enumerate(file_paths):
        #print(file_path)
        wrf_file = Dataset(file_path)
        # p        = getvar(wrf_file, "pressure",timeidx=ALL_TIMES)
        time_tmp = read_wrf_time(file_path)

        if var_name == 'cape_2d':
            # 'cape_2d', # 2D CAPE (MCAPE/MCIN/LCL/LFC)
            var_tmp  = getvar(wrf_file, var_name, timeidx=ALL_TIMES)[0]
            #print("======= cape_2d =======")
        elif var_name == 'cloudfrac':
            # 'cloudfrac', # Cloud Fraction
            var_tmp  = getvar(wrf_file, var_name, timeidx=ALL_TIMES)[0]
            #print("======= cloudfrac =======")
        else:
            var_tmp  = getvar(wrf_file, var_name, timeidx=ALL_TIMES)

        if loc_lat == None:
            if mask_map is not None:
                ntime      = len(var_tmp[:,0,0,0])
                mask_multi = [ mask_map ] * ntime
                Var        = np.where(mask_multi,var_tmp,np.nan)
            else:
                Var        = var_tmp
        else:
            ### need to fix, not work
            ntime      = len(var_tmp[:,0,0,0])
            mask       = mask_by_lat_lon(file_path, loc_lat, loc_lon, 'XLAT', 'XLONG')
            if mask_map is not None:
                mask   = np.where(np.all([mask,mask_map],axis=0), True, False)
            # np.savetxt("check",mask)
            mask_multi = [ mask ] * ntime
            Var        = np.where(mask_multi,var_tmp,np.nan)

        if i == 0:
            var = Var
            time= time_tmp
        else:
            var = np.concatenate((var, Var), axis=0)
            time = np.concatenate((time, time_tmp), axis=0)

    return time, var

def read_wrf_hgt_var(file_path, var_name, var_unit=None, height=None, loc_lat=None, loc_lon=None, p_hgt="p"):

    #print("read "+var_name+" from wrf output")

    var_4D =  [
                'p',    # Full Model Pressure
                'avo',    # Absolute Vorticity
                'eth',    # Equivalent Potential Temperature
                'dbz',    # Reflectivity
                'geopt',  # Geopotential for the Mass Grid
                'omg',  # Omega
                'pvo',  # Potential Vorticity
                'rh',   # Relative Humidity
                'td',   # Dew Point Temperature
                'tc',   # Temperature in Celsius
                'th',   # Potential Temperature
                'temp', # Temperature (in specified units)
                'tv',   # Virtual Temperature
                'twb',  # Wet Bulb Temperature
                'ua',   # U-component of Wind on Mass Points
                'va',   # V-component of Wind on Mass Points
                'wa',   # W-component of Wind on Mass Points
                'z',    # Model Height for Mass Grid
                'cape_3d',# 3D CAPE and CIN
                'height_agl', # Model Height for Mass Grid (AGL)
                ]

    wrf_file = Dataset(file_path)
    if p_hgt == "p":
        p   = getvar(wrf_file, "pressure",timeidx=ALL_TIMES)
    if p_hgt == "hgt":
        hgt = getvar(wrf_file, "height_agl",timeidx=ALL_TIMES)

    # if var_name in var_4D:
    if var_unit == None:
        if var_name == 'cape_3d':
            Var  = getvar(wrf_file, var_name, timeidx=ALL_TIMES)[0]
            #print("======= Var =======")
            # #print(Var)
        else:
            Var  = getvar(wrf_file, var_name, timeidx=ALL_TIMES)
    else:
        Var      = getvar(wrf_file, var_name, units=var_unit, timeidx=ALL_TIMES)
    # else:
    #     Var  = wrf_file.variables[var_name][:]

    if height == None:
        var_tmp  = Var
    else:
        if p_hgt == "p":
            var_tmp  = interplevel(Var, p, height)
        if p_hgt == "hgt":
            var_tmp  = interplevel(Var, hgt, height)

    if loc_lat == None:
        var  = var_tmp
    else:
        # here only suit 2D and 3D var
        ### need to fix, not work
        ntime  = len(var_tmp[:,0,0])
        # #print("ntime",ntime)
        mask   = mask_by_lat_lon(file_path, loc_lat, loc_lon, 'XLAT', 'XLONG')
        # np.savetxt("check",mask)
        mask_multi = [ mask ] * ntime
        var    = np.where(mask_multi,var_tmp,np.nan)

    return var

def read_wrf_hgt_var_multi_files(file_paths, var_name, var_unit=None, height=None, loc_lat=None, loc_lon=None, p_hgt="p"):

    #print("read "+var_name+" from wrf output")

    var_4D =  [
                'p',    # Full Model Pressure
                'avo',    # Absolute Vorticity
                'eth',    # Equivalent Potential Temperature
                'dbz',    # Reflectivity
                'geopt',  # Geopotential for the Mass Grid
                'omg',  # Omega
                'pvo',  # Potential Vorticity
                'rh',   # Relative Humidity
                'td',   # Dew Point Temperature
                'tc',   # Temperature in Celsius
                'th',   # Potential Temperature
                'temp', # Temperature (in specified units)
                'tv',   # Virtual Temperature
                'twb',  # Wet Bulb Temperature
                'ua',   # U-component of Wind on Mass Points
                'va',   # V-component of Wind on Mass Points
                'wa',   # W-component of Wind on Mass Points
                'z',    # Model Height for Mass Grid
                'cape_3d',# 3D CAPE and CIN
                'height_agl', # Model Height for Mass Grid (AGL)
                ]

    for i, file_path in enumerate(file_paths):

        wrf_file = Dataset(file_path)
        time_tmp = read_wrf_time(file_path)

        if p_hgt == "p":
            p   = getvar(wrf_file, "pressure",timeidx=ALL_TIMES)
        if p_hgt == "hgt":
            hgt = getvar(wrf_file, "height_agl",timeidx=ALL_TIMES)

        # if var_name in var_4D:
        if var_unit == None:
            if var_name == 'cape_3d':
                Var  = getvar(wrf_file, var_name, timeidx=ALL_TIMES)[0]
            else:
                Var  = getvar(wrf_file, var_name, timeidx=ALL_TIMES)
        else:
            Var      = getvar(wrf_file, var_name, units=var_unit, timeidx=ALL_TIMES)

        if height == None:
            var_tmp  = Var
        else:
            if p_hgt == "p":
                var_tmp  = interplevel(Var, p, height)
            if p_hgt == "hgt":
                var_tmp  = interplevel(Var, hgt, height)

        if loc_lat == None:
            var  = var_tmp
        else:
            # here only suit 2D and 3D var
            ntime  = np.shape(var_tmp)[0]
            #print("ntime",ntime)
            mask   = mask_by_lat_lon(file_path, loc_lat, loc_lon, 'XLAT', 'XLONG')
            mask_multi = [ mask ] * ntime
            var    = np.where(mask_multi,var_tmp,np.nan)

        if i == 0:
            var_e  = var
            time   = time_tmp
        else:
            var_e  = np.concatenate((var_e, var), axis=0)
            time   = np.concatenate((time, time_tmp), axis=0)

    return var_e

# ========================= Spitial & temporal Average =========================
def time_clip_to_day(time, Var, time_s, time_e, seconds=None):

    time_cood = time_mask(time, time_s, time_e, seconds)
    time_slt  = time[time_cood]
    if len(np.shape(Var))==3:
        #print("len(np.shape(Var))==3")
        var_slt  = Var[time_cood,:,:]
        days     = []
        for t in time_slt:
            days.append(t.days)
        cnt = 0
        var_tmp  = np.zeros([len(np.unique(days)),len(var_slt[0,:,0]),len(var_slt[0,0,:])])
        for d in np.unique(days):
            var_tmp[cnt,:,:] = np.nanmean(var_slt[days == d,:,:],axis=0)
            cnt              = cnt +1
    elif len(np.shape(Var))==4:
        #print("len(np.shape(Var))==4")
        nlayer   = len(Var[0,:,0,0])
        nlat     = len(Var[0,0,:,0])
        nlon     = len(Var[0,0,0,:])

        var_slt  = Var[time_cood,:,:,:]
        days     = []
        for t in time_slt:
            days.append(t.days)
        cnt = 0
        var_tmp  = np.zeros([len(np.unique(days)),nlayer,nlat,nlon])
        for d in np.unique(days):
            var_tmp[cnt,:,:,:] = np.nanmean(var_slt[days == d,:,:,:],axis=0)
            cnt              = cnt +1
    # #print("var_tmp", var_tmp)
    return var_tmp

def time_clip_to_day_sum(time, Var, time_s, time_e, seconds=None):

    time_cood = time_mask(time, time_s, time_e, seconds)
    time_slt  = time[time_cood]
    if len(np.shape(Var))==3:
        var_slt  = Var[time_cood,:,:]
        days     = []
        for t in time_slt:
            days.append(t.days)
        cnt = 0
        var_tmp  = np.zeros([len(np.unique(days)),len(var_slt[0,:,0]),len(var_slt[0,0,:])])
        for d in np.unique(days):
            var_tmp[cnt,:,:] = np.nansum(var_slt[days == d,:,:],axis=0)
            cnt              = cnt +1
    elif len(np.shape(Var))==4:
        nlayer   = len(Var[0,:,0,0])
        nlat     = len(Var[0,0,:,0])
        nlon     = len(Var[0,0,0,:])

        var_slt  = Var[time_cood,:,:,:]
        days     = []
        for t in time_slt:
            days.append(t.days)
        cnt = 0
        var_tmp  = np.zeros([len(np.unique(days)),nlayer,nlat,nlon])
        for d in np.unique(days):
            var_tmp[cnt,:,:,:] = np.nansum(var_slt[days == d,:,:,:],axis=0)
            cnt              = cnt +1

    return var_tmp

def time_clip_to_day_dates(time, Var, time_s, time_e, seconds=None):

    time_cood = time_mask(time, time_s, time_e, seconds)
    time_slt  = time[time_cood]

    if len(np.shape(Var))==3:
        #print("len(np.shape(Var))==3")
        var_slt  = Var[time_cood,:,:]
        days     = []
        for t in time_slt:
            days.append(t.days)

        cnt      = 0
        var_tmp  = np.zeros([len(np.unique(days)),len(var_slt[0,:,0]),len(var_slt[0,0,:])])
        dates    = []
        for d in np.unique(days):
            dates.append(d)
            var_tmp[cnt,:,:] = np.nanmean(var_slt[days == d,:,:],axis=0)
            cnt              = cnt +1

    elif len(np.shape(Var))==4:
        #print("len(np.shape(Var))==4")
        nlayer   = len(Var[0,:,0,0])
        nlat     = len(Var[0,0,:,0])
        nlon     = len(Var[0,0,0,:])

        var_slt  = Var[time_cood,:,:,:]
        days     = []
        for t in time_slt:
            days.append(t.days)

        cnt      = 0
        var_tmp  = np.zeros([len(np.unique(days)),nlayer,nlat,nlon])
        dates    = []
        for d in np.unique(days):
            dates.append(d)
            var_tmp[cnt,:,:,:] = np.nanmean(var_slt[days == d,:,:,:],axis=0)
            cnt              = cnt +1

    # #print("var_tmp", var_tmp)
    dates = np.array(dates)
    #print("dates",dates)

    return var_tmp,dates

def spatial_var(time, Var, time_s, time_e, seconds=None):

    # time should be AEST

    time_cood = time_mask(time, time_s, time_e, seconds)
    var       = np.nanmean(Var[time_cood],axis=0)

    #print('time[time_cood]',time[time_cood])
    # np.savetxt("test_var.txt",var,delimiter=",")
    return var

def spatial_var_mean(time, Var, time_s, time_e, seconds=None):

    # time should be AEST

    time_cood = time_mask(time, time_s, time_e, seconds)
    var       = np.nanmean(Var[time_cood],axis=0)

    # np.savetxt("test_var.txt",var,delimiter=",")
    return var

def spatial_var_sum(time, Var, time_s, time_e, seconds=None):

    # time should be AEST
    # ATTENTION np.nansum has problems - it produces odd values

    time_cood = time_mask(time, time_s, time_e, seconds)
    #print('time[time_cood]',time[time_cood])
    var       = np.nansum(Var[time_cood],axis=0)

    # np.savetxt("test_var.txt",var,delimiter=",")
    return var

def time_clip_to_day_max(time, Var, time_s, time_e, seconds=None):

    time_cood = time_mask(time, time_s, time_e, seconds)
    time_slt  = time[time_cood]

    # #print("time_slt",time_slt)

    var_slt  = Var[time_cood,:,:]

    days     = []

    for t in time_slt:
        days.append(t.days)

    cnt = 0
    var_tmp  = np.zeros([len(np.unique(days)),len(var_slt[0,:,0]),len(var_slt[0,0,:])])
    var_tmp[:,:,:]  = np.nan

    for d in np.unique(days):
        var_tmp[cnt,:,:] = np.nanmax(var_slt[days == d,:,:],axis=0)
        cnt              = cnt +1

    return var_tmp

def spatial_var_max(time, Var, time_s, time_e, seconds=None):

    # time should be AEST

    var_tmp = time_clip_to_day_max(time, Var, time_s, time_e, seconds=seconds)

    var = np.nanmean(var_tmp,axis=0)

    return var

def spatial_var_Xmax(time, Var, time_s, time_e, seconds=None):

    # time should be AEST

    time_cood = time_mask(time, time_s, time_e, seconds)

    var_slt   = Var[time_cood,:,:]

    var       = np.nanmax(var_slt,axis=0)

    return var

def time_clip_to_day_min(time, Var, time_s, time_e, seconds=None):

    time_cood = time_mask(time, time_s, time_e, seconds)
    time_slt  = time[time_cood]

    var_slt  = Var[time_cood,:,:]

    days     = []

    for t in time_slt:
        days.append(t.days)

    cnt = 0
    var_tmp  = np.zeros([len(np.unique(days)),len(var_slt[0,:,0]),len(var_slt[0,0,:])])
    var_tmp[:,:,:]  = np.nan

    for d in np.unique(days):
        var_tmp[cnt,:,:] = np.nanmin(var_slt[days == d,:,:],axis=0)
        cnt              = cnt +1

    return var_tmp

def spatial_var_min(time, Var, time_s, time_e, seconds=None):

    # time should be AEST

    var_tmp = time_clip_to_day_min(time, Var, time_s, time_e, seconds=seconds)

    var = np.nanmean(var_tmp,axis=0)

    return var

def spital_ERAI_tp(time,Var,time_s,time_e):

    Time_s = time_s - datetime(2000,1,1,0,0,0)
    Time_e = time_e - datetime(2000,1,1,0,0,0)

    is_12_24 = []
    for i in np.arange(len(time)):
        is_12_24.append((time[i].seconds == 43200) | (time[i].seconds == 0))
    time_cood = (time>=Time_s) & (time<Time_e) & is_12_24
    # #print(float(np.count_nonzero(time_cood==True)))
    var = np.nanmean(Var[time_cood,:,:],axis=0) * float(np.count_nonzero(time_cood==True))

    # #print(np.nansum(var))
    return var

def time_series_var(time,Var,time_s,time_e):

    time_cood = time_mask(time, time_s, time_e)

    if len(np.shape(Var)) == 3:
        var_tmp  = Var[time_cood,:,:]
        var      = np.nanmean(var_tmp,axis=(1,2))
    if len(np.shape(Var)) == 4:
        var_tmp  = Var[time_cood,:,:,:]
        var      = np.nanmean(var_tmp,axis=(2,3))
    Time     = time[time_cood]

    return Time,var

def time_series_statistic(time,Var,time_s,time_e):

    Time_s = time_s - datetime(2000,1,1,0,0,0)
    Time_e = time_e - datetime(2000,1,1,0,0,0)

    var_tmp  = Var[(time>=Time_s) & (time<=Time_e),:,:]

    ##### how to keep these multi lines rather than average????
    var_mean = np.nanmean(var_tmp,axis=(1,2))
    var_max  = np.nanmax(var_tmp,axis=(1,2))
    var_min  = np.nanmin(var_tmp,axis=(1,2))

    return var_mean, var_max, var_min

def time_series_time(time,time_s,time_e):

    Time_s = time_s - datetime(2000,1,1,0,0,0)
    Time_e = time_e - datetime(2000,1,1,0,0,0)
    Time     = time[(time>=Time_s) & (time<=Time_e)]

    return Time

# =============================== Plots setting ================================
def get_reverse_colormap(var_name):

    '''
    To tell whether it needs a reversed colormap
    '''

    var_reverse_yes = [ "Rainf_f_inst","Rainf_tavg","Evap_tavg","ECanop_tavg","TVeg_tavg","ESoil_tavg",
                        "Qs_tavg","Qsb_tavg", "Snowf_tavg","GPP_tavg","Qle_tavg","SoilMoist_inst",
                        "FWsoil_tavg","SnowCover_inst","Qair_f_inst","Wind_f_inst","SWE_inst",
                        "SnowDepth_inst","SoilWet_inst", "EF",
                        "rh2"]
    var_reverse_no  = [ "Qh_tavg","Qg_tavg","Swnet_tavg","Lwnet_tavg","SWdown_f_inst","LWdown_f_inst",
                        "VegT_tavg","AvgSurfT_tavg","Tair_f_inst","SoilTemp_inst","Albedo_inst",
                        "Psurf_f_inst",
                        "T2"]

    if var_name in var_reverse_yes:
        return_value = True
    elif var_name in var_reverse_no:
        return_value = False
    else:
        return_value = None

    return return_value

def get_wrf_var_range_diff(var_name):

    '''
    Get range
    '''

    var_degc       = ["T2"]
    var_percent    = ["rh2"]
    ranges         = [0.0,0.0]

    if var_name in var_degc:
        ranges[0] = -2.
        ranges[1] = 2.
    elif var_name in var_percent:
        ranges[0] = -20.
        ranges[1] = 20.
    else:
        ranges = None
    return ranges

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):

    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

# =============================== Regrid ================================
def regrid_data(lat_in, lon_in, lat_out, lon_out, input_data, method='linear',threshold=None):

    if len(np.shape(lat_in)) == 1:
        lon_in_2D, lat_in_2D = np.meshgrid(lon_in,lat_in)
        lon_in_1D            = np.reshape(lon_in_2D,-1)
        lat_in_1D            = np.reshape(lat_in_2D,-1)
    elif len(np.shape(lat_in)) == 2:
        lon_in_1D            = np.reshape(lon_in,-1)
        lat_in_1D            = np.reshape(lat_in,-1)
    else:
        print("ERROR: lon_in has ", len(np.shape(lat_in)), "dimensions")

    if len(np.shape(lat_out)) == 1:
        lon_out_2D, lat_out_2D = np.meshgrid(lon_out,lat_out)
    elif len(np.shape(lat_out)) == 2:
        lon_out_2D            = lon_out
        lat_out_2D            = lat_out
    else:
        print("ERROR: lon_out has ", len(np.shape(lat_in)), "dimensions")

    value_tmp = np.reshape(input_data,-1)

    # Check NaN - input array shouldn't have NaN
    if threshold is None:
        mask_values     = ~np.isnan(value_tmp)
    else:
        #print("np.all([~np.isnan(value_tmp),value_tmp>threshold],axis=0) ",np.all([~np.isnan(value_tmp),value_tmp>threshold],axis=0))
        mask_values     = np.all([~np.isnan(value_tmp),value_tmp>threshold],axis=0)

    value     = value_tmp[mask_values]
    # ======= CAUTION =======
    lat_in_1D = lat_in_1D[mask_values]  # here I make nan in values as the standard
    lon_in_1D = lon_in_1D[mask_values]
    #print("shape value = ", np.shape(value))
    #print("shape lat_in_1D = ", np.shape(lat_in_1D))
    #print("shape lon_in_1D = ", np.shape(lon_in_1D))
    # =======================
    #print("value =",value)
    #print("lat_in_1D =",lat_in_1D)
    #print("lon_in_1D =",lon_in_1D)
    Value = griddata((lon_in_1D, lat_in_1D), value, (lon_out_2D, lat_out_2D), method=method)

    return Value

def regrid_to_PlateCarree(var, mask_val, lat_in, lon_in, lat_out_1D, lon_out_1D, method='nearest'):

    '''
    Regrid to lat-lon projection
    mask_val: -1: mask out, 1: kept
    '''
    #print('in regrid_to_PlateCarree')
    # Set up the mask_val

    mask_val    = np.where(np.isnan(mask_val), 0, 1)
    mask_val    = np.where(mask_val < 0, 0, 1)

    # Set the output lat and lon
    lon_out_2D, lat_out_2D = np.meshgrid(lon_out_1D,lat_out_1D)

    # Set up input
    var_1D_tmp  = var.flatten()
    var_1D      = var_1D_tmp[~np.isnan(var_1D_tmp)]

    lat_in_1D   = lat_in.flatten()
    lon_in_1D   = lon_in.flatten()
    lat_in_1D   = lat_in_1D[~np.isnan(var_1D_tmp)]
    lon_in_1D   = lon_in_1D[~np.isnan(var_1D_tmp)]

    var_regrid  = griddata((lat_in_1D, lon_in_1D), var_1D, (lat_out_2D, lon_out_2D), method=method)

    lat_mask_1D = lat_in.flatten()
    lon_mask_1D = lon_in.flatten()
    mask_val_1D = mask_val.flatten()
    mask_regrid = griddata((lat_mask_1D, lon_mask_1D), mask_val_1D, (lat_out_2D, lon_out_2D), method=method)

    var_regrid  = np.where(mask_regrid==1, var_regrid, np.nan)
    #print('np.any(np.isnan(var_regrid))',np.any(np.isnan(var_regrid)))
    return var_regrid

# =============================== Calculate Var ===============================
def qair_to_vpd(qair, tair, press):
    '''
    calculate vpd
    Input:
          qair: kg/kg
          tair: K
          press: hPa
    Output:
          vpd: kPa
    '''

    # set nan values
    qair = np.where(qair==-9999.,np.nan,qair)
    tair = np.where(tair==-9999.,np.nan,tair)
    press= np.where(press<-9999.,np.nan,press)

    DEG_2_KELVIN = 273.15
    PA_TO_KPA    = 0.001
    PA_TO_HPA    = 0.01

    # convert back to Pa
    press        /= PA_TO_HPA
    tair         -= DEG_2_KELVIN

    # saturation vapor pressure
    es = 100.0 * 6.112 * np.exp((17.67 * tair) / (243.5 + tair))

    # vapor pressure
    ea = (qair * press) / (0.622 + (1.0 - 0.622) * qair)

    vpd = (es - ea) * PA_TO_KPA
    vpd = np.where(vpd < 0.05, 0.05, vpd)

    return vpd
