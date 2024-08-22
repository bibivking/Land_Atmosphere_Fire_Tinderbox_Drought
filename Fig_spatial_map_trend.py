import os
import sys
import cartopy
import numpy as np
from netCDF4 import Dataset
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.cm import get_cmap
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature, OCEAN
from common_utils import *

def read_variable(file_path,var_name,time_s,time_e,time_init,threshold):

    # READ LAI or Albedo
    time,var_tmp = read_var_multi_file(file_path, var_name)
    var_daily    = time_clip_to_day(time, var_tmp, time_s, time_e)
    ts           = time_s - time_init
    te           = time_e - time_init
    time_daily   = np.arange(ts.days, te.days)
    var_daily    = np.where(var_daily < threshold, np.nan, var_daily)

    print('np.shape(time_daily)',np.shape(time_daily))
    print('np.shape(var_daily)',np.shape(var_daily))

    return time_daily, var_daily

def calculate_trend(ctl_file_path,sen_file_path,var_name,time_s,time_e,threshold):

    time_init             = datetime(2000,1,1,0,0,0,0)

    # READ WRF-CABLE output, and calcuate to daily data
    time_daily, ctl_daily = read_variable(ctl_file_path,var_name,time_s,time_e,time_init,threshold) # Var(nday,nlat,nlon)
    time_daily, sen_daily = read_variable(sen_file_path,var_name,time_s,time_e,time_init,threshold) # Var(nday,nlat,nlon)

    # Calculate from daily to seasonal
    time, lats  = read_var(ctl_file_path[0], 'lat', lat_name='lat', lon_name='lon')
    time, lons  = read_var(ctl_file_path[0], 'lon', lat_name='lat', lon_name='lon')

    nlat        = np.shape(lats)[0]
    nlon        = np.shape(lats)[1]

    ctl_season  = np.zeros((3,4,nlat,nlon))
    sen_season  = np.zeros((3,4,nlat,nlon))

    season_tmp  = [ [datetime(2017,3,1,0,0,0,0), datetime(2017,6,1,0,0,0,0)],
                    [datetime(2017,6,1,0,0,0,0), datetime(2017,9,1,0,0,0,0)],
                    [datetime(2017,9,1,0,0,0,0), datetime(2017,12,1,0,0,0,0)],
                    [datetime(2017,12,1,0,0,0,0),datetime(2018,3,1,0,0,0,0)],
                    [datetime(2018,3,1,0,0,0,0), datetime(2018,6,1,0,0,0,0)],
                    [datetime(2018,6,1,0,0,0,0), datetime(2018,9,1,0,0,0,0)],
                    [datetime(2018,9,1,0,0,0,0), datetime(2018,12,1,0,0,0,0)],
                    [datetime(2018,12,1,0,0,0,0),datetime(2019,3,1,0,0,0,0)],
                    [datetime(2019,3,1,0,0,0,0), datetime(2019,6,1,0,0,0,0)],
                    [datetime(2019,6,1,0,0,0,0), datetime(2019,9,1,0,0,0,0)],
                    [datetime(2019,9,1,0,0,0,0), datetime(2019,12,1,0,0,0,0)],
                    [datetime(2019,12,1,0,0,0,0),datetime(2020,3,1,0,0,0,0)],
                    ]

    season_tmp = np.array(season_tmp)
    season_s_e = np.zeros((12,2))

    for i in np.arange(12):
        for j in np.arange(2):
            season_s_e[i,j] = (season_tmp[i,j]-time_init).days
    print("season_s_e",season_s_e)

    for i in np.arange(12):
        year               = int(i / 4)
        season             = int(i % 4)
        print("ATTENTION!!! check year=", year, "season=", season, "season_s",season_s_e[i,0],"season_e",season_s_e[i,1] )

        time_mask          = np.all([time_daily >= season_s_e[i,0], time_daily < season_s_e[i,1]], axis=0)
        print("ATTENTION!!! check time_mask=", time_mask)

        ctl_season[year,season,:,:] = np.nanmean(ctl_daily[time_mask,:,:],axis=0)
        sen_season[year,season,:,:] = np.nanmean(sen_daily[time_mask,:,:],axis=0)

    # Calculate trend
    nseason    = 4 # MAM, JJA, SON, DJF
    nindex     = 4 # trend type, (2nd-1st)/1st, (3rd-2nd)/1st, and (3rd-1st)/1st
    var_trend  = np.zeros((nseason,nindex,nlat,nlon))

    var_trend[:,1,:,:] = (sen_season[1,:,:,:]-sen_season[0,:,:,:])/ctl_season[0,:,:,:] # (2nd-1st)/1st yr
    var_trend[:,2,:,:] = (sen_season[2,:,:,:]-sen_season[1,:,:,:])/ctl_season[0,:,:,:] # (3rd-2nd)/1st yr
    var_trend[:,3,:,:] = (sen_season[2,:,:,:]-sen_season[0,:,:,:])/ctl_season[0,:,:,:] # (3rd-1st)/1st yr

    # classifiy increase or decrease types:
    var_trend[:,0,:,:] = np.where(np.all([var_trend[:,1,:,:] > 0.05, var_trend[:,2,:,:] > 0.05], axis=0),
                                  1, var_trend[:,0,:,:]) # increase
    var_trend[:,0,:,:] = np.where(np.all([var_trend[:,1,:,:] < -0.05, var_trend[:,2,:,:] < -0.05], axis=0),
                                  2, var_trend[:,0,:,:]) # decrease
    var_trend[:,0,:,:] = np.where(np.all([var_trend[:,1,:,:] > 0.10, var_trend[:,2,:,:] < -0.10], axis=0),
                                  3, var_trend[:,0,:,:]) # increase then decrease
    var_trend[:,0,:,:] = np.where(np.all([var_trend[:,1,:,:] < - 0.10, var_trend[:,2,:,:] > 0.10], axis=0),
                                  4, var_trend[:,0,:,:]) # decrease then increase
    var_trend[:,0,:,:] = np.where(np.all([abs(var_trend[:,2,:,:]) < 0.05, var_trend[:,3,:,:] > 0.10], axis=0),
                                  5, var_trend[:,0,:,:]) # 2st has no obvious trend, 1nd year inrease a lot
    var_trend[:,0,:,:] = np.where(np.all([abs(var_trend[:,2,:,:]) < 0.05, var_trend[:,3,:,:] < -0.10], axis=0),
                                  6, var_trend[:,0,:,:]) # 2st has no obvious trend, 1nd year decrease a lot
    var_trend[:,0,:,:] = np.where(np.all([abs(var_trend[:,1,:,:]) < 0.05, var_trend[:,3,:,:] > 0.10], axis=0),
                                  7, var_trend[:,0,:,:]) # 1st has no obvious trend, 2nd year inrease a lot
    var_trend[:,0,:,:] = np.where(np.all([abs(var_trend[:,1,:,:]) < 0.05, var_trend[:,3,:,:] < -0.10], axis=0),
                                  8, var_trend[:,0,:,:]) # 1st has no obvious trend, 2nd year decrease a lot

    return lats, lons, var_trend

def make_nc_file(ctl_file_path,sen_file_path,trend_file,var_name,time_s,time_e,threshold):

    # calculate trend
    lats, lons, var_trend = calculate_trend(ctl_file_path,sen_file_path,var_name,time_s,time_e,threshold)

    # set the dimension
    nseason  = 4
    ntype    = 4
    nLat     = np.shape(lats)[0]
    nLon     = np.shape(lats)[1]
    print("nLat=",nLat,"nLon=",nLon)

    # create file and write global attributes
    f                    = Dataset(trend_file, 'w', format='NETCDF4')
    f.history            = "Created by: %s" % (os.path.basename(__file__))
    f.creation_date      = "%s" % (datetime.now())
    f.description        = 'the '+var_name+' trend for 2017-03~2020-02 in '+sen_file_path[0]+', made by MU Mengyuan'

    # set dimensions
    f.createDimension('season', nseason)
    f.createDimension('type',  ntype)
    f.createDimension('north_south', nLat)
    f.createDimension('east_west', nLon)
    f.Conventions        = "CF-1.0"

    season               = f.createVariable('season', 'S3', ('season'))
    season.standard_name = "four seasons"
    season[:]            = np.array(['MAM', 'JJA', 'SON', 'DJF'], dtype='S4')

    Type                = f.createVariable('type', 'f4', ('type'))
    Type.standard_name  = "0: trend type;                            \
                           1: (sen 2nd yr - sen 1st yr)/ctl 1st yr;  \
                           3: (sen 3rd yr - sen 2nd yr)/ctl 1st yr;  \
                           4: (sen 3rd yr - sen 1st yr)/ctl 1st yr"
    Type[:]             = [0, 1, 2, 3]

    latitude            = f.createVariable('lat', 'f4', ('north_south', 'east_west'))
    latitude.long_name  = "latitude"
    latitude.units      = "degree_north"
    latitude._CoordinateAxisType = "Lat"
    latitude[:]         = lats

    longitude           = f.createVariable('lon', 'f4', ('north_south', 'east_west'))
    longitude.long_name = "longitude"
    longitude.units     = "degree_east"
    longitude._CoordinateAxisType = "Lon"
    longitude[:]        = lons

    var                 = f.createVariable('trend', 'f4', ('season', 'type', 'north_south', 'east_west'))
    var.standard_name   = 'the trend in '+var_name
    var.description     = "0: no trend; 1: increase; 2: decrease;                                     \
                           3: increase then decrease; 4: decrease then increase;                      \
                           5: 2st no trend, 1nd inrease a lot; 6: 2st no trend, 1nd decrease a lot; \
                           7: 1st no trend, 2nd inrease a lot; 8: 1st on trend, 2nd decrease a lot;"
    var[:]              = var_trend

    f.close()

def plot_trend_map(LAI_trend_file, ALB_trend_file, wrf_path, message=None, contour=True):

    # =================== Plotting spatial map ===================

    # Read in trend plots on WRF domain
    f_LAI     = Dataset(LAI_trend_file, 'r')
    LAI_trend = f_LAI.variables['trend'][:,0,:,:] # nseason, nindex, nlat, nlon
    LAI_Mask  = f_LAI.variables['lon'][:,:]
    LAI_trend = np.where(LAI_Mask <0, np.nan, LAI_trend)
    f_LAI.close()

    f_ALB     = Dataset(ALB_trend_file, 'r')
    ALB_trend = f_ALB.variables['trend'][:,0,:,:] # nseason, nindex, nlat, nlon
    ALB_Mask  = f_ALB.variables['lon'][:,:]
    ALB_trend = np.where(ALB_Mask <0, np.nan, ALB_trend)
    f_ALB.close()

    # Read in no Nan WRF lat and lon
    wrf   = Dataset(wrf_path, 'r')
    lat   = wrf.variables['lat'][:,:]
    lon   = wrf.variables['lon'][:,:]

    if contour == False:
        # Set the output lat and lon
        lat_out     = np.arange(-39,-24,0.04)
        lon_out     = np.arange(135,155,0.04)
        lon_out_2D,lat_out_2D = np.meshgrid(lon_out,lat_out)

        # Regrid to lat-lon projection
        LAI_trend_regrid= np.zeros((4,len(lat_out),len(lon_out)))
        LAI_Mask        = np.where(LAI_Mask < 0, 0, 1)

        ALB_trend_regrid= np.zeros((4,len(lat_out),len(lon_out)))
        ALB_Mask        = np.where(ALB_Mask < 0, 0, 1)

        Mask_regrid     = np.zeros((len(lat_out),len(lon_out)))

        for i in np.arange(4):
            LAI_trend_in_1D_tmp = LAI_trend[i,:,:].flatten()
            ALB_trend_in_1D_tmp = ALB_trend[i,:,:].flatten()
            lat_in_1D           = lat.flatten()
            lon_in_1D           = lon.flatten()

            LAI_trend_in_1D     = LAI_trend_in_1D_tmp[~np.isnan(LAI_trend_in_1D_tmp)]
            ALB_trend_in_1D     = ALB_trend_in_1D_tmp[~np.isnan(ALB_trend_in_1D_tmp)]
            lat_in_1D           = lat_in_1D[~np.isnan(LAI_trend_in_1D_tmp)]    # here I make nan in values as the standard
            lon_in_1D           = lon_in_1D[~np.isnan(LAI_trend_in_1D_tmp)]


            LAI_trend_regrid[i,:,:] = griddata((lat_in_1D, lon_in_1D), LAI_trend_in_1D, (lat_out_2D, lon_out_2D), method='nearest')
            ALB_trend_regrid[i,:,:] = griddata((lat_in_1D, lon_in_1D), ALB_trend_in_1D, (lat_out_2D, lon_out_2D), method='nearest')
            if i == 0:
                LAI_Mask_in_1D  = LAI_Mask.flatten()
                lat_mask_1D     = lat.flatten()
                lon_mask_1D     = lon.flatten()
                Mask_regrid     = griddata((lat_mask_1D, lon_mask_1D), LAI_Mask_in_1D, (lat_out_2D, lon_out_2D), method='nearest')
            LAI_trend_regrid[i,:,:] = np.where(Mask_regrid==1,LAI_trend_regrid[i,:,:],np.nan)
            ALB_trend_regrid[i,:,:] = np.where(Mask_regrid==1,ALB_trend_regrid[i,:,:],np.nan)

    # ============ Plotting ============
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=[7,6],sharex=True, sharey=True, squeeze=True,
                            subplot_kw={'projection': ccrs.PlateCarree()})
    plt.subplots_adjust(wspace=-0.05, hspace=0.05) # left=0.15,right=0.95,top=0.85,bottom=0.05,

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

    states= NaturalEarthFeature(category="cultural", scale="50m",
                                        facecolor="none",
                                        name="admin_1_states_provinces_shp")


    # ======================= Set colormap =======================
    cmap    = ListedColormap(["lightgray", # 0: no trend
                              "green",             # 1: increase
                              "red",               # 2: decrease
                              "orange",            # 3: increase then decrease
                              "lightseagreen",     # 4: decrease then increase
                              "mediumspringgreen", # 5: 2st no trend, 1nd inrease a lot
                              "lightcoral",        # 6: 2st no trend, 1nd decrease a lot
                              "lawngreen",             # 7: 1st no trend, 2nd inrease a lot
                              "hotpink",               # 8: 1st on trend, 2nd decrease a lot
                              ]) #plt.cm.tab10 #BrBG
    label_x = [ "MAM", "JJA", "SON", "DJF",]

                     #  0,   1,   2,   3,   4,  11,   12,   21,   22,
    clevs_trend = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
    cnt         = 0

    for i in np.arange(2):
        for j in np.arange(2):

            ax[i,j].coastlines(resolution="50m",linewidth=1)
            ax[i,j].set_extent([135,155,-39,-24])
            ax[i,j].add_feature(states, linewidth=.5, edgecolor="black")

            # Add gridlines
            gl = ax[i,j].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color=almost_black, linestyle='--')
            gl.top_labels   = False
            gl.right_labels = False
            if i == 0:
                gl.bottom_labels= False
            else:
                gl.bottom_labels= True
            if j ==0:
                gl.left_labels  = True
            else:
                gl.left_labels  = False
            gl.xlines       = False
            gl.ylines       = False
            gl.xlocator     = mticker.FixedLocator(np.arange(125,160,1))
            gl.ylocator     = mticker.FixedLocator(np.arange(-40,-20,1))
            gl.xformatter   = LONGITUDE_FORMATTER
            gl.yformatter   = LATITUDE_FORMATTER
            gl.xlabel_style = {'size':10, 'color':almost_black}#,'rotation': 90}
            gl.ylabel_style = {'size':10, 'color':almost_black}

    if contour== False:
        extent=(135, 155, -39, -24)
        print(np.min(lon),np.max(lon), np.min(lat), np.max(lat))
        plot1 = ax[0,0].imshow(LAI_trend_regrid[1,:,:], origin="lower", extent=extent, interpolation="none", vmin=-0.5, vmax=8.5, transform=ccrs.PlateCarree(), cmap=cmap) # resample=False,
        plot1 = ax[0,1].imshow(LAI_trend_regrid[3,:,:], origin="lower", extent=extent, interpolation="none", vmin=-0.5, vmax=8.5, transform=ccrs.PlateCarree(), cmap=cmap)
        plot2 = ax[1,0].imshow(ALB_trend_regrid[1,:,:], origin="lower", extent=extent, interpolation="none", vmin=-0.5, vmax=8.5, transform=ccrs.PlateCarree(), cmap=cmap)
        plot2 = ax[1,1].imshow(ALB_trend_regrid[3,:,:], origin="lower", extent=extent, interpolation="none",  vmin=-0.5, vmax=8.5, transform=ccrs.PlateCarree(), cmap=cmap)
    else:
        plt.rcParams["lines.linewidth"] = 0
        plot1 = ax[0,0].contourf(lon_out, lat_out, LAI_trend_regrid[1,:,:], clevs_trend, transform=ccrs.PlateCarree(),antialiased=False, cmap=cmap, extend='neither')
        plot1 = ax[0,1].contourf(lon_out, lat_out, LAI_trend_regrid[3,:,:], clevs_trend, transform=ccrs.PlateCarree(),antialiased=False, cmap=cmap, extend='neither')
        plot2 = ax[1,0].contourf(lon_out, lat_out, ALB_trend_regrid[1,:,:], clevs_trend, transform=ccrs.PlateCarree(),antialiased=False, cmap=cmap, extend='neither')
        plot2 = ax[1,1].contourf(lon_out, lat_out, ALB_trend_regrid[3,:,:], clevs_trend, transform=ccrs.PlateCarree(),antialiased=False, cmap=cmap, extend='neither')

        # ax[0].add_feature(OCEAN,edgecolor='none', facecolor="lightgray")
        # ax[i,j].add_feature(OCEAN,edgecolor='none', facecolor="lightgray")

    # Add titles

    ax[0,0].set_title("Winter", fontsize=12)
    ax[0,1].set_title("Summer", fontsize=12)

    ax[0,0].set_ylabel("ΔLAI$ (m\mathregular{^{2}}$ m\mathregular{^{-2}}$")
    ax[1,0].set_ylabel("Δ$α$ (-)")

    ax[0,0].text(0.02, 0.15, "(a)", transform=ax[0,0].transAxes, fontsize=12, verticalalignment='top', bbox=props)
    ax[0,1].text(0.02, 0.15, "(b)", transform=ax[0,1].transAxes, fontsize=12, verticalalignment='top', bbox=props)
    ax[1,0].text(0.02, 0.15, "(c)", transform=ax[1,0].transAxes, fontsize=12, verticalalignment='top', bbox=props)
    ax[1,1].text(0.02, 0.15, "(d)", transform=ax[1,1].transAxes, fontsize=12, verticalalignment='top', bbox=props)

    cbar = plt.colorbar(plot1, ax=ax, ticklocation="bottom", pad=0.05, orientation="horizontal", aspect=25, shrink=1.) # cax=cax,
    cbar.ax.tick_params(labelsize=12, labelrotation=45)

    plt.savefig('./plots/spatial_map_trend_'+message+'.png',dpi=300)
    return

if __name__ == "__main__":

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

    lat_name       = "lat"
    lon_name       = "lon"

    if 1:
        var_name       = 'LAI_inst'
        path           = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"
        ctl_file_path  = [ path+"drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/LIS_output/"+var_name+"/LIS.CABLE.201701-202002.nc",]
        sen_file_path  = [ path+"drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2_obs_LAI_ALB/LIS_output/"+var_name+"/LIS.CABLE.201701-202006.nc",]
        LAI_trend_file = "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/LAI_trend.nc"
        ALB_trend_file = "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/Albedo_trend.nc"
        time_s         = datetime(2017,3,1,0,0,0,0)
        time_e         = datetime(2020,3,1,0,0,0,0)
        threshold      = 0.000001
        wrf_path       = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/WRF_output/p/wrfout_201912-202002.nc"

        message        = "LAI_ALB_trend_2017-2020"
        plot_trend_map(LAI_trend_file, ALB_trend_file, wrf_path, message,contour=False)
