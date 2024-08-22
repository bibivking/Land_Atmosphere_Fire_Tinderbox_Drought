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

def get_time_cood(file_path, file_name_wrf, time_s ,time_e):

    # ==================== process time ====================
    # Read in time steps
    ncfile      = Dataset(file_path+"z/"+file_name_wrf)
    time_tmp    = nc.num2date(ncfile.variables['time'][:],ncfile.variables['time'].units,
                  only_use_cftime_datetimes=False, only_use_python_datetimes=True)

    # Adjust from UTC to local time
    time        = UTC_to_AEST(time_tmp) - datetime(2000,1,1,0,0,0)
    time        = np.array(time)
    ntime       = len(time)
    print("time", time)

    # Change datetime to timedelta
    Time_s      = time_s - datetime(2000,1,1,0,0,0)
    Time_e      = time_e - datetime(2000,1,1,0,0,0)

    # ==================== add three time coordiation ====================
    # for all days
    time_cood_all = (time>=Time_s) & (time<Time_e)
    time_t        = time[time_cood_all]
    doy_all       = [ time_t[i].days for i in np.arange(len(time_t)) ]
    time_t        = None

    # Check whether the current time step is at day time
    seconds       = [6.*60.*60., 18.*60.*60.]
    time_cood_day = []
    for j in np.arange(ntime):
        if_day    = (time[j].seconds >= seconds[0]) & (time[j].seconds < seconds[1])
        time_cood_day.append((time[j]>=Time_s) & (time[j]<Time_e) & if_day)
    time_t        = time[time_cood_day]
    doy_day       = [ time_t[i].days for i in np.arange(len(time_t)) ]
    time_t        = None

    # Check whether the current time step is at night time
    seconds       = [18.*60.*60., 6.*60.*60.]
    time_cood_night = []
    for j in np.arange(ntime):
        if_night  = (time[j].seconds >= seconds[0]) | (time[j].seconds < seconds[1])
        time_cood_night.append((time[j]>=Time_s) & (time[j]<Time_e) & if_night)
    time_t        = time[time_cood_night]
    doy_night     = [ time_t[i].days for i in np.arange(len(time_t)) ]
    time_t        = None

    return time_cood_all, time_cood_day, time_cood_night, doy_all, doy_day, doy_night

def read_wrf_variable(atmo_path,wrf_path,file_name_wrf):

    z_file  = Dataset(atmo_path + "z/" + file_name_wrf, mode='r')
    time    = z_file.variables['time'][:]
    Z       = z_file.variables['z'][:]
    z_file.close()

    wa_file = Dataset(atmo_path + "wa/" + file_name_wrf, mode='r')
    Wa      = wa_file.variables['wa'][:]
    wa_file.close()

    ua_file = Dataset(atmo_path + "ua/" + file_name_wrf, mode='r')
    Ua      = ua_file.variables['ua'][:]
    ua_file.close()

    T_file  = Dataset(atmo_path + "th/" + file_name_wrf, mode='r')
    T       = T_file.variables['th'][:]
    T_file.close()

    rh_file = Dataset(atmo_path + "rh/" + file_name_wrf, mode='r')
    S       = rh_file.variables['rh'][:]
    rh_file.close()

    PBLH_file = Dataset(atmo_path + "PBLH/" + file_name_wrf, mode='r')
    PBL       = PBLH_file.variables['PBLH'][:]
    PBLH_file.close()

    # Get coordiate info
    ncfile   = Dataset(wrf_path)
    template = getvar(ncfile, 'th', timeidx=1)

    # Extract the coordinate values from A
    coord_values = {'Time': time,
                    'bottom_top': template['bottom_top'].values,
                    'south_north': template['south_north'].values,
                    'west_east': template['west_east'].values}

    # Create the DataArray B with the same dimensions and coordinate information as A
    z       = xr.DataArray(Z, dims=('Time', 'bottom_top', 'south_north', 'west_east'),
                            coords=coord_values)
    wa      = xr.DataArray(Wa, dims=('Time', 'bottom_top', 'south_north', 'west_east'),
                            coords=coord_values)
    ua      = xr.DataArray(Ua, dims=('Time', 'bottom_top', 'south_north', 'west_east'),
                            coords=coord_values)
    t       = xr.DataArray(T, dims=('Time', 'bottom_top', 'south_north', 'west_east'),
                            coords=coord_values)
    s       = xr.DataArray(S, dims=('Time', 'bottom_top', 'south_north', 'west_east'),
                            coords=coord_values)

    coord_values = {'Time': time,
                    'south_north': template['south_north'].values,
                    'west_east': template['west_east'].values}

    pbl     = xr.DataArray(PBL, dims=('Time', 'south_north', 'west_east'),
                            coords=coord_values)

    return z, wa, ua, t, s, pbl

def get_time_masked(Z,T,S,Ua,Wa,PBL,time_cood):

    z  = Z[time_cood,:,:,:]
    t  = T[time_cood,:,:,:]
    s  = S[time_cood,:,:,:]
    ua = Ua[time_cood,:,:,:]
    wa = Wa[time_cood,:,:,:]
    pbl= PBL[time_cood,:,:]

    return z, t, s, ua, wa, pbl

def get_vertcross(wrf_path, z, t, s, ua, wa, lat_slt, lon_min, lon_max, doy, seconds=None):

    # Compute the vertical cross-section interpolation.  Also, include the
    # lat/lon points along the cross-section in the metadata by setting latlon
    # to True.

    # Set up time dimension
    ntime       = np.shape(z)[0]

    # Set up the starting and ending points of the line
    start_point = CoordPair(lat=lat_slt, lon=lon_min)
    end_point   = CoordPair(lat=lat_slt, lon=lon_max)

    # ============== calc interpolation ==============
    ncfile      = Dataset(wrf_path)

    for i in np.arange(ntime):

        print("i = ", i)

        # get the transect for each time step
        t_crs  = vertcross(t[i],  z[i], wrfin=ncfile, start_point=start_point,
                            end_point=end_point, latlon=True, meta=True, autolevels=160)
        s_crs  = vertcross(s[i],  z[i], wrfin=ncfile, start_point=start_point,
                            end_point=end_point, latlon=True, meta=True, autolevels=160)
        ua_crs = vertcross(ua[i], z[i], wrfin=ncfile, start_point=start_point,
                            end_point=end_point, latlon=True, meta=True, autolevels=160)
        wa_crs = vertcross(wa[i], z[i], wrfin=ncfile, start_point=start_point,
                            end_point=end_point, latlon=True, meta=True, autolevels=160)
        print("t_crs",t_crs)

        if i == 0:
            # setting the number of pixels this transect includes
            loct = np.linspace(lon_min, lon_max, np.shape(t_crs)[1])
            print("loct", loct)

            # setting the vertical levels to interpolate
            vrt = np.arange(0, 3010., 10.)

        # Interpolate to make the contour plot more smooth
        t_tmp, s_tmp, ua_tmp, wa_tmp = \
                get_interpolation(t_crs, s_crs, ua_crs, wa_crs, loct, vrt)

        # Free the memory
        t_crs, s_crs, ua_crs, wa_crs = None, None, None, None

        if i == 0:
            # Set the dimension for the output variables
            print("np.shape(t_tmp)",np.shape(t_tmp))
            nx     = np.shape(t_tmp)[0]
            ny     = np.shape(t_tmp)[1]
            t_out  = np.zeros((ntime, nx, ny))
            s_out  = np.zeros((ntime, nx, ny))
            ua_out = np.zeros((ntime, nx, ny))
            wa_out = np.zeros((ntime, nx, ny))

        # Pass to out variable
        t_out[i,:,:]  = t_tmp
        s_out[i,:,:]  = s_tmp
        ua_out[i,:,:] = ua_tmp
        wa_out[i,:,:] = wa_tmp

        # Free the memory
        t_tmp, s_tmp, ua_tmp, wa_tmp = None, None, None, None


    # ============== Calculate the average over the selected period ==============
    if seconds == None:
        t_cross  = np.nanmean(t_out, axis=0)

    elif seconds[0] < seconds[1]:
        # daytime - Tmax
        day_num = len(np.unique(doy))
        t       = np.zeros((day_num,nx,ny))

        # find Tmax in the day time periods
        for i in np.arange(day_num):
            is_the_day = [ doy[j] == np.unique(doy)[i] for j in np.arange(len(doy)) ]
            t[i,:,:]   = np.nanmax(t_out[is_the_day,:,:],axis=0)

        # average Tmax for the selected period
        t_cross  = np.nanmean(t, axis=0)

    elif seconds[1] < seconds[0]:
        # night - Tmin
        day_num = len(np.unique(doy))
        t       = np.zeros((day_num,nx,ny))

        # find Tmin in the night time periods
        for i in np.arange(day_num):
            is_the_day = [ doy[j] == np.unique(doy)[i] for j in np.arange(len(doy)) ]
            t[i,:,:]   = np.nanmin(t_out[is_the_day,:,:],axis=0)

        # average Tmin for the selected period
        t_cross  = np.nanmean(t, axis=0)

    # average variables for the selected period
    s_cross  = np.nanmean(s_out, axis=0)
    ua_cross = np.nanmean(ua_out, axis=0)
    wa_cross = np.nanmean(wa_out, axis=0)

    return t_cross, s_cross, ua_cross, wa_cross, loct, vrt

def get_PBL(wrf_path, land_path, z, pbl, lat_slt, lon_min, lon_max):

    # Compute the vertical cross-section interpolation.  Also, include the
    # lat/lon points along the cross-section in the metadata by setting latlon
    # to True.

    # Get time dimension
    ntime       = np.shape(z)[0]

    # Set transect line
    start_point = CoordPair(lat=lat_slt, lon=lon_min)
    end_point   = CoordPair(lat=lat_slt, lon=lon_max)

    # ================= calc interpolation ===============
    '''
    In vertcross, autolevels=100(default), then vertical profile is evenly spaced to 100 levels
    '''

    # Open the file
    ncfile      = Dataset(wrf_path)

    # Extend the 3D PBL data into 4D for plotting reason
    pbl_4D      = np.expand_dims(pbl,axis=1).repeat(29,axis=1)

    # Get the elevation data
    landfile    = Dataset(land_path, mode='r')
    elev        = landfile.variables['HGT_M'][:]

    for i in np.arange(ntime):
        print(np.shape(z[i]))
        print(np.shape(pbl_4D[i]))

        # Get the transect from 4D PBLH dataset
        pbl_crs = vertcross(pbl_4D[i]+elev, z[i], wrfin=ncfile, start_point=start_point,
                            end_point=end_point, latlon=True, meta=True, autolevels=10) #

        if i == 0:
            print("np.shape(pbl_crs)",np.shape(pbl_crs))
            # Set the dimension for pbl_out
            nx      = np.shape(pbl_crs)[0]
            ny      = np.shape(pbl_crs)[1]
            pbl_out = np.zeros((ntime, nx, ny))

        pbl_out[i,:,:] = pbl_crs
        pbl_crs        = None

    # nx refers to the vertical level and ny refers to lons, 9 is height level selected to get the PBLH
    # since PBLH values are the same for different levels, so 9 is a random number I select. It can be any
    # other levels.
    pbl_cross = np.nanmean(pbl_out[:,9,:], axis=0)
    # print(pbl_cross)

    return pbl_cross

def get_LAI_ALBEDO(wrf_path, z, lai, albedo, lat_slt, lon_min, lon_max):

    # Compute the vertical cross-section interpolation.  Also, include the
    # lat/lon points along the cross-section in the metadata by setting latlon
    # to True.

    # Set transect line
    start_point = CoordPair(lat=lat_slt, lon=lon_min)
    end_point   = CoordPair(lat=lat_slt, lon=lon_max)

    # ================= calc interpolation ===============

    # Open the file
    ncfile      = Dataset(wrf_path)

    # Extend to 3D
    lai_3D      = np.expand_dims(lai,axis=0).repeat(29,axis=0)
    alb_3D      = np.expand_dims(albedo,axis=0).repeat(29,axis=0)

    print("np.shape(lai_3D)",lai_3D)

    # Get the transect from 3D dataset
    lai_crs = vertcross(lai_3D, z, wrfin=ncfile, start_point=start_point,
                        end_point=end_point, latlon=True, meta=True, autolevels=10) #
    alb_crs = vertcross(alb_3D, z, wrfin=ncfile, start_point=start_point,
                        end_point=end_point, latlon=True, meta=True, autolevels=10) #

    print("np.shape(lai_crs)",np.shape(lai_crs))
    print("np.shape(alb_crs)",np.shape(alb_crs))

    # nx refers to the vertical level and ny refers to lons, 9 is height level selected to get the LAI/albedo
    # since LAI/albedo values are the same for different levels, so 9 is a random number I select. It can be any
    # other levels.

    lai_cross = lai_crs[9,:]
    alb_cross = alb_crs[9,:]

    print(lai_cross)
    print(alb_cross)

    return lai_cross, alb_cross

def get_interpolation(t_crs, s_crs, ua_crs, wa_crs, loct, vrt):

    print("get_interpolation")
    grid_X, grid_Y = np.meshgrid(loct,vrt)
    vertical_tmp   = to_np(t_crs.coords['vertical'])[:]

    grid_x, grid_y = np.meshgrid(loct,vertical_tmp)
    x              = np.reshape(grid_x,-1)
    y              = np.reshape(grid_y,-1)

    t_out  = griddata((x, y), np.reshape(to_np(t_crs),-1), (grid_X, grid_Y), method="linear")
    s_out  = griddata((x, y), np.reshape(to_np(s_crs),-1), (grid_X, grid_Y), method="linear")
    ua_out = griddata((x, y), np.reshape(to_np(ua_crs),-1), (grid_X, grid_Y), method="linear")
    wa_out = griddata((x, y), np.reshape(to_np(wa_crs),-1), (grid_X, grid_Y), method="linear")

    return t_out, s_out, ua_out, wa_out

def output_profile_wrf_wind(file_out, atmo_path_ctl, atmo_path_sen, land_path_ctl, land_path_sen,
                            wrf_path, land_path, file_name_wrf, file_name_lis, time_s, time_e,
                            lat_slt=36, lon_min=130, lon_max=160):

    # ================ Get time coordiation ================
    time_cood_all, time_cood_day, time_cood_night, doy_all, doy_day, doy_night = \
                                        get_time_cood(atmo_path_ctl, file_name_wrf, time_s, time_e)

    print("time_cood_day",time_cood_day)
    print("time_cood_day",time_cood_night)

    # ================ Get the WRF variables ================
    Z1,Wa1,Ua1,T1,S1,PBL1 = read_wrf_variable(atmo_path_ctl,wrf_path,file_name_wrf)
    print("Z1", Z1)

    # ================ Get time masked ================
    # Day time
    z1_day, t1_day, s1_day, ua1_day, wa1_day, pbl1_day = \
                       get_time_masked(Z1, T1, S1, Ua1, Wa1, PBL1, time_cood_day)

    # Night time
    z1_night, t1_night, s1_night, ua1_night, wa1_night, pbl1_night = \
                       get_time_masked(Z1, T1, S1, Ua1, Wa1, PBL1, time_cood_night)

    print("z1_day",z1_day)

    # ================ Vertcross, interpolate and mean ================
    seconds         = [6.*60.*60.,18.*60.*60.]
    t1_day_crs, s1_day_crs, ua1_day_crs, wa1_day_crs, xy_loc, vertical =\
                      get_vertcross(wrf_path, z1_day, t1_day, s1_day, ua1_day,
                      wa1_day, lat_slt, lon_min, lon_max, doy_day, seconds)
    pbl1_day_crs    = get_PBL(wrf_path, land_path, z1_day, pbl1_day, lat_slt, lon_min, lon_max)

    seconds         = [18.*60.*60.,6.*60.*60.]
    t1_night_crs, s1_night_crs, ua1_night_crs, wa1_night_crs, xy_loc, vertical =\
                      get_vertcross(wrf_path, z1_night, t1_night, s1_night, ua1_night,
                      wa1_night, lat_slt, lon_min, lon_max, doy_night, seconds)
    pbl1_night_crs  = get_PBL(wrf_path, land_path, z1_night, pbl1_night, lat_slt, lon_min, lon_max)


    # ================ read second file ================
    Z2,Wa2,Ua2,T2,S2,PBL2 = read_wrf_variable(atmo_path_sen,wrf_path,file_name_wrf)

    z2_day, t2_day, s2_day, ua2_day, wa2_day, pbl2_day = \
                    get_time_masked(Z2,T2,S2,Ua2,Wa2,PBL2, time_cood_day)

    z2_night, t2_night, s2_night, ua2_night, wa2_night, pbl2_night = \
                    get_time_masked(Z2, T2, S2, Ua2, Wa2, PBL2, time_cood_night)

    seconds         = [6.*60.*60.,18.*60.*60.]
    t2_day_crs, s2_day_crs, ua2_day_crs, wa2_day_crs, xy_loc, vertical =\
        get_vertcross(wrf_path, z2_day, t2_day, s2_day, ua2_day, wa2_day,
                      lat_slt, lon_min, lon_max, doy_day, seconds)

    pbl2_day_crs = get_PBL(wrf_path, land_path, z2_day, pbl2_day, lat_slt, lon_min, lon_max)

    seconds         = [18.*60.*60.,6.*60.*60.]
    t2_night_crs, s2_night_crs, ua2_night_crs, wa2_night_crs, xy_loc, vertical =\
        get_vertcross(wrf_path, z2_night, t2_night, s2_night, ua2_night,
                      wa2_night, lat_slt, lon_min, lon_max, doy_night, seconds)

    pbl2_night_crs = get_PBL(wrf_path, land_path, z2_night, pbl2_night, lat_slt, lon_min, lon_max)

    # ================== calc difference ==================
    # Sen - Ctl
    t_day_crs    = t2_day_crs  - t1_day_crs
    s_day_crs    = s2_day_crs  - s1_day_crs
    ua_day_crs   = ua2_day_crs - ua1_day_crs
    wa_day_crs   = wa2_day_crs - wa1_day_crs

    t_night_crs  = t2_night_crs  - t1_night_crs
    s_night_crs  = s2_night_crs  - s1_night_crs
    ua_night_crs = ua2_night_crs - ua1_night_crs
    wa_night_crs = wa2_night_crs - wa1_night_crs

    print("t_day_crs", t_day_crs)


    # ================ Get LAI and Albedo ================
    # Get coordiate info
    ncfile       = Dataset(wrf_path)
    template     = getvar(ncfile, 'th', timeidx=1)
    ncfile.close()

    # Extract the coordinate values from template
    coord_values = {'south_north': template['south_north'].values,
                    'west_east': template['west_east'].values}

    # Read LAI and Albedo data
    lai_file     = Dataset(land_path_ctl + "LAI_inst/" + file_name_lis, mode='r')
    LAI_ctl      = lai_file.variables['LAI_inst'][:]

    time_tmp     = nc.num2date(lai_file.variables['time'][:],lai_file.variables['time'].units,
                   only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    time         = UTC_to_AEST(time_tmp) - datetime(2000,1,1,0,0,0)
    time         = np.array(time)

    lai_file.close()

    lai_file     = Dataset(land_path_sen + "LAI_inst/" + file_name_lis, mode='r')
    LAI_sen      = lai_file.variables['LAI_inst'][:]
    lai_file.close()

    alb_file     = Dataset(land_path_ctl + "Albedo_inst/" + file_name_lis, mode='r')
    ALB_ctl      = alb_file.variables['Albedo_inst'][:]
    alb_file.close()

    alb_file     = Dataset(land_path_sen + "Albedo_inst/" + file_name_lis, mode='r')
    ALB_sen      = alb_file.variables['Albedo_inst'][:]
    alb_file.close()

    # Calculate difference
    LAI_diff     = LAI_sen - LAI_ctl
    ALB_diff     = ALB_sen - ALB_ctl

    # Change datetime to timedelta
    Time_s       = time_s - datetime(2000,1,1,0,0,0)
    Time_e       = time_e - datetime(2000,1,1,0,0,0)

    # ==================== add three time coordiation ====================
    # for all days
    time_lai_alb_mask = (time>=Time_s) & (time<Time_e)

    # Get time masked
    LAI          = np.nanmean(LAI_diff[time_lai_alb_mask,:,:],axis=0)
    ALB          = np.nanmean(ALB_diff[time_lai_alb_mask,:,:],axis=0)

    # Create the DataArray B with the same dimensions and coordinate information as A
    lai          = xr.DataArray(LAI, dims=('south_north', 'west_east'), coords=coord_values)
    albedo       = xr.DataArray(ALB, dims=('south_north', 'west_east'), coords=coord_values)

    # Get the transect
    lai_crs, alb_crs = get_LAI_ALBEDO(wrf_path, z1_day[0,:,:,:], lai, albedo, lat_slt, lon_min, lon_max) # z1_day can be replaced as others

    # ================== make output file ==================

    # create file and write global attributes
    f = nc.Dataset(file_out, 'w', format='NETCDF4')

    ### Create nc file ###
    f.history           = "Created by: %s" % (os.path.basename(__file__))
    f.creation_date     = "%s" % (datetime.now())
    f.description       = 'WRF-CABLE output for making transect plot, made by MU Mengyuan'
    f.Conventions       = "CF-1.0"

    # set dimensions
    nvrt                = len(vertical)
    nlon                = len(xy_loc)

    f.createDimension('bottom_top', nvrt)
    f.createDimension('east_west',  nlon)

    # Set cooridiates
    level               = f.createVariable('level', 'f4', ('bottom_top'))
    level.standard_name = "Height_MSL"
    level.long_name     = "heights above mean sea level"
    level.units         = "m"
    level[:]            = vertical

    lon                 = f.createVariable('lon', 'f4', ('east_west'))
    lon.standard_name   = "longitude"
    lon.units           = "degree_east"
    lon[:]              = xy_loc

    # Set variables
    th_day_diff                 = f.createVariable('th_day_diff', 'f4', ('bottom_top', 'east_west'))
    th_day_diff.standard_name   = "Potential temperature difference at daytime (sen-ctl)"
    th_day_diff.units           = "deg C"
    th_day_diff[:]              = t_day_crs

    th_night_diff               = f.createVariable('th_night_diff', 'f4', ('bottom_top', 'east_west'))
    th_night_diff.standard_name = "Potential temperature difference at night (sen-ctl)"
    th_night_diff.units         = "deg C"
    th_night_diff[:]            = t_night_crs

    rh_day_diff                 = f.createVariable('rh_day_diff', 'f4', ('bottom_top', 'east_west'))
    rh_day_diff.standard_name   = "Relative humidity difference at daytime (sen-ctl)"
    rh_day_diff.units           = "%"
    rh_day_diff[:]              = s_day_crs

    rh_night_diff               = f.createVariable('rh_night_diff', 'f4', ('bottom_top', 'east_west'))
    rh_night_diff.standard_name = "Relative humidity difference at night (sen-ctl)"
    rh_night_diff.units         = "%"
    rh_night_diff[:]            = s_night_crs

    ua_day_diff                 = f.createVariable('ua_day_diff', 'f4', ('bottom_top', 'east_west'))
    ua_day_diff.standard_name   = "u-wind difference at daytime (sen-ctl)"
    ua_day_diff.units           = "m s-1"
    ua_day_diff[:]              = ua_day_crs

    ua_night_diff               = f.createVariable('ua_night_diff', 'f4', ('bottom_top', 'east_west'))
    ua_night_diff.standard_name = "u-wind difference at night (sen-ctl)"
    ua_night_diff.units         = "m s-1"
    ua_night_diff[:]            = ua_night_crs

    wa_day_diff                 = f.createVariable('wa_day_diff', 'f4', ('bottom_top', 'east_west'))
    wa_day_diff.standard_name   = "w-wind difference at daytime (sen-ctl)"
    wa_day_diff.units           = "m s-1"
    wa_day_diff[:]              = wa_day_crs

    wa_night_diff               = f.createVariable('wa_night_diff', 'f4', ('bottom_top', 'east_west'))
    wa_night_diff.standard_name = "w-wind difference at night (sen-ctl)"
    wa_night_diff.units         = "m s-1"
    wa_night_diff[:]            = wa_night_crs


    lai_diff                    = f.createVariable('lai_diff', 'f4', ('east_west'))
    lai_diff.standard_name      = "LAI difference (sen-ctl)"
    lai_diff.units              = "m3 m-3"
    lai_diff[:]                 = lai_crs

    alb_diff                    = f.createVariable('alb_diff', 'f4', ('east_west'))
    alb_diff.standard_name      = "albedo difference (sen-ctl)"
    alb_diff.units              = "-"
    alb_diff[:]                 = alb_crs

    pbl_day_ctl                 = f.createVariable('pbl_day_ctl', 'f4', ('east_west'))
    pbl_day_ctl.standard_name   = "PBL height in ctl at daytime"
    pbl_day_ctl.units           = "m"
    pbl_day_ctl[:]              = pbl1_day_crs

    pbl_night_ctl               = f.createVariable('pbl_night_ctl', 'f4', ('east_west'))
    pbl_night_ctl.standard_name = "PBL height in ctl at night"
    pbl_night_ctl.units         = "m"
    pbl_night_ctl[:]            = pbl1_night_crs

    pbl_day_sen                 = f.createVariable('pbl_day_sen', 'f4', ('east_west'))
    pbl_day_sen.standard_name   = "PBL height in sen at daytime"
    pbl_day_sen.units           = "m"
    pbl_day_sen[:]              = pbl1_day_crs

    pbl_night_sen               = f.createVariable('pbl_night_sen', 'f4', ('east_west'))
    pbl_night_sen.standard_name = "PBL height in sen at night"
    pbl_night_sen.units         = "m"
    pbl_night_sen[:]            = pbl1_night_crs

    f.close()

def plot_profile_wrf_wind(file_out,message=None):

    # ****************** Read data *****************
    ncfile        = Dataset(file_out, mode='r')

    vertical      = ncfile.variables['level'][:]
    xy_loc        = ncfile.variables['lon'][:]

    th_day_diff   = ncfile.variables['th_day_diff'][:]
    th_night_diff = ncfile.variables['th_night_diff'][:]
    rh_day_diff   = ncfile.variables['rh_day_diff'][:]
    rh_night_diff = ncfile.variables['rh_night_diff'][:]
    ua_day_diff   = ncfile.variables['ua_day_diff'][:]
    ua_night_diff = ncfile.variables['ua_night_diff'][:]
    wa_day_diff   = ncfile.variables['wa_day_diff'][:]
    wa_night_diff = ncfile.variables['wa_night_diff'][:]

    lai_diff      = ncfile.variables['lai_diff'][:]
    alb_diff      = ncfile.variables['alb_diff'][:]
    pbl_day_ctl   = ncfile.variables['pbl_day_ctl'][:]
    pbl_night_ctl = ncfile.variables['pbl_night_ctl'][:]
    pbl_day_sen   = ncfile.variables['pbl_day_sen'][:]
    pbl_night_sen = ncfile.variables['pbl_night_sen'][:]

    # ****************** plotting ******************
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=[8,10],sharex=True, sharey=True, squeeze=True)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    plt.rcParams['text.usetex']     = False
    plt.rcParams['font.family']     = "sans-serif"
    plt.rcParams['font.serif']      = "Helvetica"
    plt.rcParams['axes.linewidth']  = 1.5
    plt.rcParams['axes.labelsize']  = 14
    plt.rcParams['font.size']       = 14
    plt.rcParams['legend.fontsize'] = 12
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

    props = dict(boxstyle="round", facecolor='white', alpha=0.0, ec='white')

    # ===== set plot =====
    # color map
    color_map_1   = get_cmap("seismic")
    blue_map_neg  = truncate_colormap(color_map_1, minval=0., maxval=0.5)
    color_map_2   = get_cmap("seismic").reversed()
    blue_map_pos  = truncate_colormap(color_map_2, minval=0.5, maxval=1.)
    cmap          = color_map_1
    cmap1         = plt.cm.BrBG #YlGnBu_r #get_cmap("Greens").reversed()
    cmap2         = plt.cm.BrBG_r #YlGnBu_r #get_cmap("Greens").reversed()

    # quiver scale
    scale = 1.

    # contour levels
    levels1    = [-1.2,-1.1,-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.,1.1,1.2]
    levels2    = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9,10]
    levels_lai = [-2,-1.8,-1.6,-1.4,-1.2,-1.,-0.8,-0.6,-0.4,-0.2,0.2,0.4,0.6,0.8,1.,1.2,1.4,1.6,1.8,2]
    levels_alb = [-0.08,-0.07,-0.06,-0.05,-0.04,-0.03,-0.02,-0.01,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08]

    # Water table depth height
    lai_hgt          = [130,260]
    lai_diff_2D      = np.zeros((2,len(lai_diff)))
    lai_diff_2D[0,:] = lai_diff
    lai_diff_2D[1,:] = lai_diff

    alb_hgt          = [0,130]
    alb_diff_2D      = np.zeros((2,len(alb_diff)))
    alb_diff_2D[0,:] = alb_diff
    alb_diff_2D[1,:] = alb_diff


    # Day temperature
    ax[0,0].set_facecolor('lightgray')
    contour   = ax[0,0].contourf(xy_loc, vertical, th_day_diff, levels=levels1, cmap=color_map_1, extend='both')
    cntr_lai  = ax[0,0].contourf(xy_loc, lai_hgt,  lai_diff_2D, levels=levels_lai, cmap=cmap1, extend='both')
    cntr_alb  = ax[0,0].contourf(xy_loc, alb_hgt,  alb_diff_2D, levels=levels_alb, cmap=cmap2, extend='both')

    line1     = ax[0,0].plot(xy_loc,pbl_day_ctl,ls="-", color="black")
    line2     = ax[0,0].plot(xy_loc,pbl_day_sen,ls="--", color="black")

    q         = ax[0,0].quiver(xy_loc[::30], vertical[::3], ua_day_diff[::3,::30],
                              wa_day_diff[::3,::30], angles='xy', scale_units='xy',
                              scale=scale, pivot='middle', color="white")
    ax[0,0].text(0.02, 0.95, "(a) Δθ$\mathregular{_{max}}$", transform=ax[0,0].transAxes, verticalalignment='top', bbox=props) # fontsize=14,

    ax[0,0].set_ylabel("Geopotential Height (m)")#, fontsize=12)
    ax[0,0].set_ylim(0,3000)
    # ax[0,0].axhline(y=300, color='white', linestyle='-')
    # ax[0,0].axhline(y=100, color='white', linestyle='-')


    # Night temperature
    ax[0,1].set_facecolor('lightgray')
    contour   = ax[0,1].contourf(xy_loc, vertical, th_night_diff, levels=levels1, cmap=color_map_1, extend='both')
    cntr_lai  = ax[0,1].contourf(xy_loc, lai_hgt,  lai_diff_2D, levels=levels_lai, cmap=cmap1, extend='both')
    cntr_alb  = ax[0,1].contourf(xy_loc, alb_hgt,  alb_diff_2D, levels=levels_alb, cmap=cmap2, extend='both')

    line1     = ax[0,1].plot(xy_loc,pbl_night_ctl,ls="-", color="black")
    line2     = ax[0,1].plot(xy_loc,pbl_night_sen,ls="--", color="black")

    q         = ax[0,1].quiver(xy_loc[::30], vertical[::3], ua_night_diff[::3,::30],
                              wa_night_diff[::3,::30], angles='xy', scale_units='xy',
                              scale=scale, pivot='middle', color="white")
    ax[0,1].text(0.02, 0.95, "(b) Δθ$\mathregular{_{min}}$", transform=ax[0,1].transAxes, verticalalignment='top', bbox=props) # fontsize=14,
    cb_var    = fig.colorbar(contour, ax=ax[0], pad=0.01, orientation="vertical", aspect=20, shrink=0.88)
    cb_var.set_label('Δθ (${^o}$C)', loc='center') # rotation=270,
    ax[0,1].set_ylim(0,3000)

    # ax[0,1].axhline(y=300, color='white', linestyle='-')
    # ax[0,1].axhline(y=100, color='white', linestyle='-')

    # Day relative humidity
    color_map = get_cmap("seismic")
    cmap      = color_map.reversed()

    ax[1,0].set_facecolor('lightgray')
    contour   = ax[1,0].contourf(xy_loc, vertical, rh_day_diff, levels=levels2, cmap=color_map_2, extend='both')
    cntr_lai  = ax[1,0].contourf(xy_loc, lai_hgt,  lai_diff_2D, levels=levels_lai, cmap=cmap1, extend='both')
    cntr_alb  = ax[1,0].contourf(xy_loc, alb_hgt,  alb_diff_2D, levels=levels_alb, cmap=cmap2, extend='both')


    line1     = ax[1,0].plot(xy_loc,pbl_day_ctl,ls="-", color="black")
    line2     = ax[1,0].plot(xy_loc,pbl_day_sen,ls="--", color="black")

    q         = ax[1,0].quiver(xy_loc[::30], vertical[::3], ua_day_diff[::3,::30],
                              wa_day_diff[::3,::30], angles='xy', scale_units='xy',
                              scale=scale, pivot='middle', color="white")
    ax[1,0].text(0.02, 0.95, "(c) ΔRH$\mathregular{_{day}}$", transform=ax[1,0].transAxes, verticalalignment='top', bbox=props) # fontsize=14,
    # ax[1,0].set_xlabel("Longitude")#, fontsize=12)
    ax[1,0].set_ylabel("Geopotential Height (m)")#, fontsize=12)
    ax[1,0].set_ylim(0,3000)
    # ax[1,0].axhline(y=300, color='white', linestyle='-')
    # ax[1,0].axhline(y=100, color='white', linestyle='-')

    # Day relative humidity
    ax[1,1].set_facecolor('lightgray')
    contour   = ax[1,1].contourf(xy_loc, vertical, rh_night_diff, levels=levels2, cmap=color_map_2, extend='both')
    cntr_lai  = ax[1,1].contourf(xy_loc, lai_hgt,  lai_diff_2D, levels=levels_lai, cmap=cmap1, extend='both')
    cntr_alb  = ax[1,1].contourf(xy_loc, alb_hgt,  alb_diff_2D, levels=levels_alb, cmap=cmap2, extend='both')


    line1     = ax[1,1].plot(xy_loc,pbl_night_ctl,ls="-", color="black")
    line2     = ax[1,1].plot(xy_loc,pbl_night_sen,ls="--", color="black")

    q         = ax[1,1].quiver(xy_loc[::30], vertical[::3], ua_night_diff[::3,::30],
                              wa_night_diff[::3,::30], angles='xy', scale_units='xy',
                              scale=scale, pivot='middle', color="white")

    ax[1,1].quiverkey(q,X=0.80, Y=2.1, U=scale, label=str(scale)+' m/s', labelpos='E', color="black")
    ax[1,1].text(0.02, 0.95, "(d) ΔRH$\mathregular{_{night}}$", transform=ax[1,1].transAxes, verticalalignment='top', bbox=props) # fontsize=14,
    # ax[1,1].set_xlabel("Longitude")#, fontsize=12)

    cb_var    = fig.colorbar(contour, ax=ax[1], pad=0.01, orientation="vertical", aspect=20, shrink=0.88)
    cb_var.set_label('ΔRH (%)', loc='center')
    ax[1,1].set_ylim(0,3000)
    # ax[1,1].axhline(y=300, color='white', linestyle='-')
    # ax[1,1].axhline(y=100, color='white', linestyle='-')

    # colorbar position
    position_lai  = fig.add_axes([0.14, 0.06, 0.62, 0.01]) # [left, bottom, width, height]
    cb_lai        = fig.colorbar(cntr_lai, ax=ax, pad=0.07, cax=position_lai, orientation="horizontal", aspect=60, shrink=0.8)
    cb_lai.set_label('LAI (m$\mathregular{^{3}}$ m$\mathregular{^{-3}}$)', loc='center',size=10)# rotation=270,
    cb_lai.ax.tick_params(labelsize=12)

    position_alb  = fig.add_axes([0.14, 0.02, 0.62, 0.01]) # [left, bottom, width, height]
    cb_alb        = fig.colorbar(cntr_alb, ax=ax, pad=0.07, cax=position_alb, orientation="horizontal", aspect=60, shrink=0.8)
    cb_alb.set_label('Albedo', loc='center',size=10)# rotation=270,
    cb_alb.ax.tick_params(labelsize=12)

    fig.savefig("./plots/profile_wrf_"+message, bbox_inches='tight', pad_inches=0.3)

if __name__ == "__main__":

    lat_slt        = -37.5
    lon_min        = 145.0
    lon_max        = 154.0

    file_name_wrf  = "wrfout_201912-202002_hourly.nc"
    file_name_lis  = "LIS.CABLE.201912-202002.nc"
    path           = '/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/'


    wrf_path      = path+ 'drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/WRF_output/wrfout_d01_2019-12-01_01:00:00'
    land_path     = path+ 'drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/geo_em.d01.nc'

    atmo_path_ctl = path + 'drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/WRF_output/'
    atmo_path_sen = path + 'drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2_obs_LAI_ALB/WRF_output/'

    land_path_ctl = path + 'drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/LIS_output/'
    land_path_sen = path + 'drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2_obs_LAI_ALB/LIS_output/'

    # message        = "201920_Dec_lat"+str(-375)
    # time_s         = datetime(2019,12,1,0,0,0,0)
    # time_e         = datetime(2020,1,1,0,0,0,0)
    # file_out       = "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/transect_"+message+"_houly.nc"

    # output_profile_wrf_wind(file_out, atmo_path_ctl, atmo_path_sen,
    #                         land_path_ctl, land_path_sen,
    #                         wrf_path, land_path,
    #                         file_name_wrf, file_name_lis,
    #                         time_s, time_e, lat_slt=lat_slt, lon_min=lon_min, lon_max=lon_max)

    message        = "201920_Jan_lat"+str(-375)
    time_s         = datetime(2020,1,1,0,0,0,0)
    time_e         = datetime(2020,2,1,0,0,0,0)
    file_out       = "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/transect_"+message+"_houly.nc"

    output_profile_wrf_wind(file_out, atmo_path_ctl, atmo_path_sen,
                            land_path_ctl, land_path_sen,
                            wrf_path, land_path,
                            file_name_wrf, file_name_lis,
                            time_s, time_e, lat_slt=lat_slt, lon_min=lon_min, lon_max=lon_max)

    # message        = "201920_Feb_lat"+str(-375)
    # time_s         = datetime(2020,2,1,0,0,0,0)
    # time_e         = datetime(2020,3,1,0,0,0,0)
    # file_out       = "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/transect_"+message+"_houly.nc"

    # output_profile_wrf_wind(file_out, atmo_path_ctl, atmo_path_sen,
    #                         land_path_ctl, land_path_sen,
    #                         wrf_path, land_path,
    #                         file_name_wrf, file_name_lis,
    #                         time_s, time_e, lat_slt=lat_slt, lon_min=lon_min, lon_max=lon_max)

    # plot_profile_wrf_wind(file_out,message=message)
