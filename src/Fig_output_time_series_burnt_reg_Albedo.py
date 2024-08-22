import os
import cartopy
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.patches import Polygon
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from scipy.interpolate import griddata, interp1d
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature, OCEAN
from common_utils import *

def regrid_to_fire_map_resolution(fire_path, var_in, lat_in, lon_in, loc_lat=None, loc_lon=None, burn=0):

    # =========== Read in fire data ============
    print("burn=",burn)
    fire_file  = Dataset(fire_path, mode='r')
    Burn_Date  = fire_file.variables['Burn_Date'][0:8,:,:]  # 2019-07 - 2020-02
    lat_out    = fire_file.variables['lat'][:]
    lon_out    = fire_file.variables['lon'][:]

    var_regrid = regrid_data(lat_in, lon_in, lat_out, lon_out, var_in, method='nearest')

    # burnt region from 2019-07 to 2020-02
    burn_area  = np.where( Burn_Date[0,:,:] + Burn_Date[1,:,:] + Burn_Date[2,:,:] + Burn_Date[3,:,:] +
                           Burn_Date[4,:,:] + Burn_Date[5,:,:] + Burn_Date[6,:,:] + Burn_Date[7,:,:] > 0, 1, Burn_Date[0,:,:])
    if burn == 1:
        # burnt region
        var_regrid = np.where(burn_area==1, var_regrid, np.nan )
    elif burn == 0:
        # all region
        var_regrid = var_regrid
    elif burn == -1:
        # unburnt region
        var_regrid = np.where(burn_area==0, var_regrid, np.nan )

    var_regrid = np.where(var_regrid==-9999.,np.nan,var_regrid)

    if loc_lat !=None:
        lons_2D, lats_2D = np.meshgrid(lon_out, lat_out)
        var_regrid = np.where(np.all(( lats_2D>loc_lat[0],
                                       lats_2D<loc_lat[1],
                                       lons_2D>loc_lon[0],
                                       lons_2D<loc_lon[1]), axis=0),
                                       var_regrid, np.nan)
        lat_out    = lats_2D
        lon_out    = lons_2D

    return var_regrid, lat_out, lon_out

def output_time_series_burn_region(var_name, var_unit, file_out, fire_path, wrf_path, file_name, land_ctl_path, land_sen_path,
                                    time_s=None, time_e=None, loc_lats=None, loc_lons=None,
                                    lat_name=None, lon_name=None, message=None, burn=0):

    # Read all pixels in the dataset
    if var_name =='Tmax':
        time_ctl, Tmax_ctl = read_var_multi_file([land_ctl_path +"Tair_f_inst/"+ file_name], "Tair_f_inst",
                                                loc_lat=None, loc_lon=None, lat_name=lat_name, lon_name=lon_name)
        time_sen, Tmax_sen = read_var_multi_file([land_sen_path +"Tair_f_inst/"+ file_name], "Tair_f_inst",
                                                loc_lat=None, loc_lon=None, lat_name=lat_name, lon_name=lon_name)

        # Calculate daily values
        var_daily_ctl = time_clip_to_day_max(time_ctl, Tmax_ctl, time_s, time_e, seconds=None)
        var_daily_sen = time_clip_to_day_max(time_sen, Tmax_sen, time_s, time_e, seconds=None)

    elif var_name =='LAI':
        time_ctl, LAI_ctl = read_var_multi_file([land_ctl_path +"LAI_inst/"+ file_name], 'LAI_inst',
                                                loc_lat=None, loc_lon=None, lat_name=lat_name, lon_name=lon_name)
        time_sen, LAI_sen = read_var_multi_file([land_sen_path +"LAI_inst/"+ file_name], 'LAI_inst',
                                                loc_lat=None, loc_lon=None, lat_name=lat_name, lon_name=lon_name)

        var_daily_ctl  = time_clip_to_day(time_ctl, LAI_ctl, time_s, time_e, seconds=None)
        var_daily_sen  = time_clip_to_day(time_sen, LAI_sen, time_s, time_e, seconds=None)

    elif var_name =='Albedo':
        time_ctl, ALB_ctl = read_var_multi_file([land_ctl_path +"Albedo_inst/"+ file_name], 'Albedo_inst',
                                                loc_lat=None, loc_lon=None, lat_name=lat_name, lon_name=lon_name)
        time_sen, ALB_sen = read_var_multi_file([land_sen_path +"Albedo_inst/"+ file_name], 'Albedo_inst',
                                                loc_lat=None, loc_lon=None, lat_name=lat_name, lon_name=lon_name)

        var_daily_ctl  = time_clip_to_day(time_ctl, ALB_ctl, time_s, time_e, seconds=None)
        var_daily_sen  = time_clip_to_day(time_sen, ALB_sen, time_s, time_e, seconds=None)

    elif var_name =='SMtop':
        time_ctl, SM_ctl = read_var_multi_file([land_ctl_path +"SoilMoist_inst/"+ file_name], 'SoilMoist_inst',
                                                loc_lat=None, loc_lon=None, lat_name=lat_name, lon_name=lon_name)
        time_sen, SM_sen = read_var_multi_file([land_sen_path +"SoilMoist_inst/"+ file_name], 'SoilMoist_inst',
                                                loc_lat=None, loc_lon=None, lat_name=lat_name, lon_name=lon_name)

        SMtop_ctl    = (SM_ctl[:,0,:,:]*0.022 + SM_ctl[:,1,:,:]*0.058 + SM_ctl[:,2,:,:]*0.154 + SM_ctl[:,3,:,:]*0.266)/0.5
        SMtop_sen    = (SM_sen[:,0,:,:]*0.022 + SM_sen[:,1,:,:]*0.058 + SM_sen[:,2,:,:]*0.154 + SM_sen[:,3,:,:]*0.266)/0.5

        var_daily_ctl = time_clip_to_day(time_ctl, SMtop_ctl, time_s, time_e, seconds=None)
        var_daily_sen = time_clip_to_day(time_sen, SMtop_sen, time_s, time_e, seconds=None)

    # Set lat and lon input
    wrf     = Dataset(wrf_path,  mode='r')
    lon_in  = wrf.variables['XLONG'][0,:,:]
    lat_in  = wrf.variables['XLAT'][0,:,:]

    ntime   = np.shape(var_daily_ctl)[0]
    print("ntime =",ntime)

    for i in np.arange(ntime):
        print("i=",i)

        # regrid to burn map resolution ~ 400m
        if i == 0:
            var_regrid_ctl_tmp, lats, lons  = regrid_to_fire_map_resolution(fire_path, var_daily_ctl[i,:,:], lat_in, lon_in, loc_lat=None, loc_lon=None, burn=burn)
            var_regrid_sen_tmp, lats, lons  = regrid_to_fire_map_resolution(fire_path, var_daily_sen[i,:,:], lat_in, lon_in, loc_lat=None, loc_lon=None, burn=burn)

            # Set up array
            nlat = np.shape(var_regrid_ctl_tmp)[0]
            nlon = np.shape(var_regrid_ctl_tmp)[1]

            var_regrid_ctl = np.zeros((ntime, nlat, nlon))
            var_regrid_sen = np.zeros((ntime, nlat, nlon))

            # Assign the first time step value
            var_regrid_ctl[i,:,:]  = var_regrid_ctl_tmp
            var_regrid_sen[i,:,:]  = var_regrid_sen_tmp

        else:
            var_regrid_ctl[i,:,:], lats, lons = regrid_to_fire_map_resolution(fire_path, var_daily_ctl[i,:,:], lat_in, lon_in, loc_lat=None, loc_lon=None, burn=burn)
            var_regrid_sen[i,:,:], lats, lons = regrid_to_fire_map_resolution(fire_path, var_daily_sen[i,:,:], lat_in, lon_in, loc_lat=None, loc_lon=None, burn=burn)

    # ===== Make masks for three regions =====
    # make fire lats and lons into 2 D
    lons_2D, lats_2D = np.meshgrid(lons, lats)
    mask_val         = np.zeros((3,np.shape(lons_2D)[0],np.shape(lons_2D)[1]),dtype=bool)

    for i in np.arange(3):
        mask_val[i,:,:]  = np.all(( lats_2D>loc_lats[i][0],lats_2D<loc_lats[i][1],
                                    lons_2D>loc_lons[i][0],lons_2D<loc_lons[i][1]), axis=0)

    # Extend the 3D mask (nreg, nlat, nlon) to 4D (nreg, ntime, nlat, nlon)
    mask_val_4D      = np.expand_dims(mask_val,axis=1).repeat(ntime,axis=1)

    # Set up the output variables
    nreg         = 3
    var_mean_ctl = np.zeros((nreg,ntime))
    var_std_ctl  = np.zeros((nreg,ntime))
    var_mean_sen = np.zeros((nreg,ntime))
    var_std_sen  = np.zeros((nreg,ntime))

    if 1:
        fig1, ax1 = plt.subplots(nrows=1, ncols=2, figsize=[5,4],sharex=True, sharey=True, squeeze=True,
                                    subplot_kw={'projection': ccrs.PlateCarree()})

        states    = NaturalEarthFeature(category="cultural", scale="50m",
                                            facecolor="none",
                                            name="admin_1_states_provinces_shp")

        # ======================= Set colormap =======================
        cmap    = plt.cm.BrBG
        cmap.set_bad(color='lightgrey')
        for i in np.arange(2):
            ax1[i].coastlines(resolution="50m",linewidth=1)
            ax1[i].set_extent([135,155,-39,-23])
            ax1[i].add_feature(states, linewidth=.5, edgecolor="black")

        plot1  = ax1[0].contourf( lons, lats, np.nanmean(var_regrid_ctl,axis=0), transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
        plot2  = ax1[1].contourf( lons, lats, np.nanmean(var_regrid_sen,axis=0), transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
        cbar1  = plt.colorbar(plot1, ax=ax1, ticklocation="right", pad=0.08, orientation="horizontal",
                            aspect=40, shrink=0.6)
        cbar1.ax.tick_params(labelsize=8, labelrotation=45)

        plt.savefig('./plots/spatial_map_check_burn_region.png',dpi=300)

    # Mask out three regions
    for i in np.arange(3):
        print("process reg",i)

        var_masked_ctl  = np.where( mask_val_4D[i,:,:,:], var_regrid_ctl, np.nan)
        var_masked_sen  = np.where( mask_val_4D[i,:,:,:], var_regrid_sen, np.nan)

        var_mean_ctl[i,:] = np.nanmean(var_masked_ctl,axis=(1,2))
        var_std_ctl[i,:]  = np.nanstd(var_masked_ctl, axis=(1,2))
        var_mean_sen[i,:] = np.nanmean(var_masked_sen,axis=(1,2))
        var_std_sen[i,:]  = np.nanstd(var_masked_sen, axis=(1,2))

        if 1:
            print('i=',i)
            fig1, ax1 = plt.subplots(nrows=2, ncols=2, figsize=[5,4],sharex=True, sharey=True, squeeze=True,
                                        subplot_kw={'projection': ccrs.PlateCarree()})

            states    = NaturalEarthFeature(category="cultural", scale="50m",
                                                facecolor="none",
                                                name="admin_1_states_provinces_shp")

            # ======================= Set colormap =======================
            cmap    = plt.cm.BrBG
            cmap.set_bad(color='lightgrey')
            for j in np.arange(2):
                ax1[0,j].coastlines(resolution="50m",linewidth=1)
                ax1[0,j].set_extent([135,155,-39,-23])
                ax1[0,j].add_feature(states, linewidth=.5, edgecolor="black")

            print('np.unique(var_masked_ctl)',np.unique(var_masked_ctl))
            print('np.unique(var_masked_sen)',np.unique(var_masked_sen))

            plot1  = ax1[0,0].contourf( lons, lats, np.nanmean(var_masked_ctl,axis=0), transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
            plot2  = ax1[0,1].contourf( lons, lats, np.nanmean(var_masked_sen,axis=0), transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
            plot3  = ax1[1,0].plot(var_mean_ctl[i,:],color='green')
            plot3  = ax1[1,0].plot(var_mean_sen[i,:],color='orange')
            plot4  = ax1[1,1].plot(var_std_ctl[i,:],color='green')
            plot4  = ax1[1,1].plot(var_std_sen[i,:],color='orange')
            cbar1  = plt.colorbar(plot1, ax=ax1, ticklocation="right", pad=0.08, orientation="horizontal",
                                aspect=40, shrink=0.6)
            cbar1.ax.tick_params(labelsize=8, labelrotation=45)

            plt.savefig('./plots/spatial_map_check_burn_region_'+str(i)+'.png',dpi=300)


    # ================== make output file ==================

    # create file and write global attributes
    f = nc.Dataset(file_out, 'w', format='NETCDF4')

    ### Create nc file ###
    f.history           = "Created by: %s" % (os.path.basename(__file__))
    f.creation_date     = "%s" % (datetime.now())
    f.description       = '201909-202002 Δ'+var_name+'in three burnt regions, made by MU Mengyuan'
    f.Conventions       = "CF-1.0"

    # set dimensions
    f.createDimension('region', 3)
    f.createDimension('time',  ntime)

    # Set cooridiates
    region               = f.createVariable('region', 'S7', ('region'))
    region.standard_name = "Burnt regions"
    region.long_name     = "Name of the burnt regions"
    region[:]            = np.array(['North  ', 'Central', 'South  '], dtype='S7')

    time                 = f.createVariable('time', 'f4', ('time'))
    time.standard_name   = "time"
    time.units           = "days since 2019-09-01 00:00:00"
    time[:]              = np.arange((time_e-time_s).days)
    print("np.arange((time_e-time_s).days)",np.arange((time_e-time_s).days))

    Var_mean_ctl               = f.createVariable( var_name+'_mean_ctl', 'f4', ('region','time'))
    Var_mean_ctl.standard_name = var_name+" in ctl"
    Var_mean_ctl.units         = var_unit
    Var_mean_ctl[:]            = var_mean_ctl

    Var_std_ctl               = f.createVariable(var_name+'_std_ctl', 'f4', ('region','time'))
    Var_std_ctl.standard_name = "standard deviation of "+var_name+" in burnt region in ctl"
    Var_std_ctl.units         = var_unit
    Var_std_ctl[:]            = var_std_ctl

    Var_mean_sen               = f.createVariable(var_name+'_mean_sen', 'f4', ('region','time'))
    Var_mean_sen.standard_name = var_name+" in sen"
    Var_mean_sen.units         = var_unit
    Var_mean_sen[:]            = var_mean_sen

    Var_std_sen               = f.createVariable(var_name+'_std_sen', 'f4', ('region','time'))
    Var_std_sen.standard_name = "standard deviation of " +var_name+ " in burnt region in sen"
    Var_std_sen.units         = var_unit
    Var_std_sen[:]            = var_std_sen

    f.close()

    return

if __name__ == "__main__":

    # plot burnt region time series
    case_ctl       = "drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2"
    case_sen       = "drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2_obs_LAI_ALB"

    file_name      = 'LIS.CABLE.201701-202002.nc'
    fire_path      = '/g/data/w97/mm3972/data/MODIS/MODIS_fire/MCD64A1.061_500m_aid0001.nc'
    wrf_path       = "/scratch/w97/mm3972/model/NUWRF/Tinderbox_drght_LAI_ALB/output/drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/WRF_output/wrfout_d01_2017-02-01_06:00:00"

    land_sen_path  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_sen+"/LIS_output/"
    land_ctl_path  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_ctl+"/LIS_output/"

    time_s         = datetime(2019,9,1,0,0,0,0)
    time_e         = datetime(2020,3,1,0,0,0,0)

    #                   North ,        Central,       South
    loc_lats       = [[-32,-28.5],   [-34.5,-32.5], [-38,-34.5]]
    loc_lons       = [[151.5,153.5], [149.5,151.5], [146.5,151]]

    message        = "time_series_201909-202002_burnt"

    var_name       = "Albedo"
    var_unit       = "-"
    file_out       = "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/times_series_201909-202002_"+var_name+"_unburnt.nc"

    output_time_series_burn_region(var_name, var_unit, file_out, fire_path, wrf_path, file_name, land_ctl_path, land_sen_path,
                        time_s=time_s, time_e=time_e, loc_lats=loc_lats, loc_lons=loc_lons,
                        lat_name="lat", lon_name="lon", message=message, burn=-1)
