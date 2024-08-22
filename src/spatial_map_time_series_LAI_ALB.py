import sys
import cartopy
import numpy as np
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

def read_LAI(filename,LAI_type,year_s,year_e,time_s,time_e,nlat=439,nlon=529):

    # read LAI_one_input LAI
    time_init    = datetime(2000,1,1,0,0,0)

    if LAI_type == "obs":
        # read obs LAI
        time_obs,LAI_obs_tmp = read_var(filename, "LAI")
        LAI_obs              = time_clip_to_day(time_obs, LAI_obs_tmp, time_s, time_e)
        LAI_out              = np.where(LAI_obs ==0, np.nan, LAI_obs)
        print("obs len(LAI_out[:,0,0]) = ", len(LAI_out[:,0,0]))

    if LAI_type == "lis_out":
        # read obs LAI
        time_obs,LAI_obs_tmp = read_var(filename, "LAI_inst")
        LAI_obs              = time_clip_to_day(time_obs, LAI_obs_tmp, time_s, time_e)
        LAI_out              = np.where(LAI_obs ==0, np.nan, LAI_obs)
        print("obs len(LAI_out[:,0,0]) = ", len(LAI_out[:,0,0]))

    elif LAI_type == "lis_multi_out":
        # read obs LAI
        time_obs,LAI_obs_tmp = read_var_multi_file(filename, "LAI_inst")
        LAI_obs              = time_clip_to_day(time_obs, LAI_obs_tmp, time_s, time_e)
        LAI_out              = np.where(LAI_obs ==0, np.nan, LAI_obs)
        print("obs len(LAI_out[:,0,0]) = ", len(LAI_out[:,0,0]))

    elif LAI_type == "lis":
        # read lis_input.nc
        lis_file             = Dataset(filename, mode='r')
        LAI_lis_tmp          = lis_file.variables['LAI'][:]
        LAI_lis_t            = np.zeros((12*(year_e+1-year_s)+2,nlat,nlon))
        LAI_lis_t[0,:,:]     = LAI_lis_tmp[11,:,:] # Dec before the start year
        LAI_lis_t[-1,:,:]    = LAI_lis_tmp[0,:,:]  # Jan after the end year
        for yr in np.arange(year_s,year_e+1):
            diff                          = (yr-year_s)*12
            LAI_lis_t[1+diff:13+diff,:,:] = LAI_lis_tmp

        print("lis len(LAI_lis_t[:,0,0]) = ", len(LAI_lis_t[:,0,0]))

        # middle of each month during 2002 Dec -2017 Jan
        time_lis_in  = [(datetime(year_s-1,12,14,0,0,0,0)-time_init).days]
        for yr in np.arange(year_s,year_e+1):
            for mon in np.arange(1,13,1):
                time_lis_in.append((datetime(yr,mon,14,0,0,0,0)-time_init).days)
        time_lis_in.append((datetime(year_e+1,1,14,0,0,0,0)-time_init).days)
        time_lis_in  = np.array(time_lis_in)

        print("len(time_lis_in) = ", len(time_lis_in))

        deltatime_s  = time_s - time_init
        deltatime_e  = time_e - time_init
        time_out     = np.arange(deltatime_s.days,deltatime_e.days+1,1)

        f            = interp1d(time_lis_in, LAI_lis_t, kind='linear',axis=0)
        LAI_out      = f(time_out)
        LAI_out      = np.where(LAI_out ==-9999., np.nan, LAI_out)

        print("len(LAI_out[:,0,0]) = ", len(LAI_out[:,0,0]))

    elif LAI_type == "modis_clim":
        # read 2003-2016 daily climatology
        deltatime_s  = time_s - time_init
        deltatime_e  = time_e - time_init
        time_out     = np.arange(deltatime_s.days,deltatime_e.days+1,1)
        print("modis_clim len(time_out) = ", len(time_out))

        lai_file     = Dataset(filename, mode='r')
        LAI_input    = lai_file.variables['LAI'][:]
        t            = 0
        year_tot     = 0

        for yr in np.arange(year_s,year_e+1,1):
            if yr%4 == 0:
                year_tot = year_tot + 366
            else:
                year_tot = year_tot + 365
        LAI_out     = np.zeros((year_tot,nlat,nlon))

        for yr in np.arange(year_s,year_e+1,1):
            if yr%4 == 0:
                LAI_out[t:t+366,:,:] = LAI_input[:]
                t   = t + 366
            else:
                LAI_out[t:t+59,:,:]     = LAI_input[0:59,:,:]
                LAI_out[t+59:t+365,:,:] = LAI_input[60:,:,:]
                t   = t + 365

        LAI_out = np.where(LAI_out ==-9999., np.nan, LAI_out)

        print("modis_clim len(LAI_out[:,0,0]) = ", len(LAI_out[:,0,0]))

    # # read WRF LAI
        # LAI_wrf = np.zeros((365*3,nlat,nlon))
        # dom     = [31,28,31,30,31,30,31,31,30,31,30,31]
        # d_s     = 0
        # for yr in np.arange(2017,2020):
        #     print("year="+str(yr))
        #     for mth in np.arange(1,13):
        #         d_e = d_s + dom[mth-1]
        #         print("month="+str(mth))
        #         if mth < 10:
        #             LAI_wrf_file = LIS_PFT_path + "LIS.CABLE."+str(yr)+"0"+str(mth)+"-"+str(yr)+"0"+str(mth)+".d01.nc"
        #         else:
        #             LAI_wrf_file = LIS_PFT_path + "LIS.CABLE."+str(yr)+str(mth)+"-"+str(yr)+str(mth)+".d01.nc"
        #
        #         time_wrf, LAI_wrf_tmp = read_var(LAI_wrf_file, "LAI_inst")
        #
        #     print("np.shape(LAI_wrf)")
        #
        #     LAI_wrf[d_s:d_e,:,:]  = time_clip_to_day(time_wrf[:-10], LAI_wrf_tmp[:-10,:,:], time_s, time_e)
        #     print(np.shape(LAI_wrf))
        #     # # ================= testing plot ===================
        #     # for i in np.arange(dom[mth-1]):
        #     #     fig, axs = plt.subplots(nrows=1, ncols=1, figsize=[4,4],sharex=True, sharey=True, squeeze=True,
        #     #                             subplot_kw={'projection': ccrs.PlateCarree()})
        #     #     plot     = axs.contourf( lon, lat, LAI_wrf[i,:,:], transform=ccrs.PlateCarree(),cmap=plt.cm.seismic,extend='both') # levels=clevs,
        #     #     cbar     = plt.colorbar(plot, ax=axs, ticklocation="right", pad=0.05, orientation="horizontal",aspect=40, shrink=0.8) # cax=cax,
        #     #     plt.savefig('./plots/spatial_map_LAI_WRF_i='+str(i)+'.png',dpi=300)
        #     #     fig = None
        #     #     axs = None
        #     #     plot = None

    return LAI_out

def read_ALBEDO(filename,ALBEDO_type,year_s,year_e,time_s,time_e,loc_lat=None, loc_lon=None, lat_name="lat", lon_name="lon"):

    nlat=439
    nlon=529
    # read ALB_one_input ALB
    time_init    = datetime(2000,1,1,0,0,0)

    if ALBEDO_type == "obs":
        # read obs ALB
        time_obs,ALB_obs_tmp = read_var_multi_file(filename, "ALBEDO")#,loc_lat, loc_lon, lat_name, lon_name)
        ALB_obs              = time_clip_to_day(time_obs, ALB_obs_tmp, time_s, time_e)
        ALB_out              = np.where(ALB_obs <=0, np.nan, ALB_obs)
        print("obs len(ALB_out[:,0,0]) = ", len(ALB_out[:,0,0]))

    elif ALBEDO_type == "lis_out":
        # read obs ALB
        time_obs,ALB_obs_tmp = read_var(filename, "Albedo_inst")#, loc_lat, loc_lon, lat_name, lon_name)
        ALB_obs              = time_clip_to_day(time_obs, ALB_obs_tmp, time_s, time_e)
        ALB_out              = np.where(ALB_obs <=0, np.nan, ALB_obs)
        print("obs len(ALB_out[:,0,0]) = ", len(ALB_out[:,0,0]))

    elif ALBEDO_type == "lis_multi_out":
        # read obs ALB
        time_obs,ALB_obs_tmp = read_var_multi_file(filename, "Albedo_inst")#,loc_lat, loc_lon, lat_name, lon_name)
        ALB_obs              = time_clip_to_day(time_obs, ALB_obs_tmp, time_s, time_e)
        ALB_out              = np.where(ALB_obs <=0, np.nan, ALB_obs)
        print("obs len(ALB_out[:,0,0]) = ", len(ALB_out[:,0,0]))

    elif ALBEDO_type == "modis_clim":
        # read 2003-2016 daily climatology
        deltatime_s  = time_s - time_init
        deltatime_e  = time_e - time_init
        time_out     = np.arange(deltatime_s.days,deltatime_e.days+1,1)
        print("modis_clim len(time_out) = ", len(time_out))

        ALB_file     = Dataset(filename, mode='r')
        ALB_input    = ALB_file.variables['ALBEDO'][:]
        t            = 0
        year_tot     = 0

        for yr in np.arange(year_s,year_e+1,1):
            if yr%4 == 0:
                year_tot = year_tot + 366
            else:
                year_tot = year_tot + 365
        ALB_out     = np.zeros((year_tot,nlat,nlon))

        for yr in np.arange(year_s,year_e+1,1):
            if yr%4 == 0:
                ALB_out[t:t+366,:,:] = ALB_input[:]
                t   = t + 366
            else:
                ALB_out[t:t+59,:,:]     = ALB_input[0:59,:,:]
                ALB_out[t+59:t+365,:,:] = ALB_input[60:,:,:]
                t   = t + 365

        ALB_out = np.where(ALB_out ==-9999., np.nan, ALB_out)

        print("modis_clim len(ALB_out[:,0,0]) = ", len(ALB_out[:,0,0]))

    elif ALBEDO_type == "modis_blue_sky":
        # read 2003-2016 daily climatology
        deltatime_s  = time_s - time_init
        deltatime_e  = time_e - time_init
        time_out     = np.arange(deltatime_s.days,deltatime_e.days+1,1)
        print("modis_clim len(time_out) = ", len(time_out))

        ALB_file     = Dataset(filename, mode='r')
        ALB_input    = ALB_file.variables['Albedo'][:]
        t            = 0
        year_tot     = 0

        for yr in np.arange(year_s,year_e+1,1):
            if yr%4 == 0:
                year_tot = year_tot + 366
            else:
                year_tot = year_tot + 365
        ALB_out     = np.zeros((year_tot,nlat,nlon))

        for yr in np.arange(year_s,year_e+1,1):
            if yr%4 == 0:
                ALB_out[t:t+59,:,:] = ALB_input[:59,:,:]
                ALB_out[t+59,:,:]   = (ALB_input[58,:,:] + ALB_input[59,:,:])/2.
                ALB_out[t+60:,:,:]   = ALB_input[59:,:,:]
                t   = t + 366
            else:
                ALB_out[t:t+365,:,:]= ALB_input[:,:,:]
                t   = t + 365

        ALB_out = np.where(ALB_out ==0, np.nan, ALB_out)

        print("modis_clim len(ALB_out[:,0,0]) = ", len(ALB_out[:,0,0]))

    elif ALBEDO_type == "modis_wrflowinp":
        # read WRF ALB from wrflowinp
        # !!! Please note that this code haven't been tested so it doesn't not work at all
        ALB_out = np.zeros((365*3,nlat,nlon)) # 20170101-20200101

        for yr in np.arange(2017,2020):
            print("year="+str(yr))
            if yr == 2020:
                mth_e = 7
            else:
                mth_e = 13
            for mth in np.arange(1,mth_e):
                # d_e = d_s + dom[mth-1]
                print("month="+str(mth))
                if mth < 10:
                    ALB_wrf_file = ALB_wrf_path + "LIS.CABLE."+str(yr)+"0"+str(mth)+"-"+str(yr)+"0"+str(mth)+".d01.nc"
                else:
                    ALB_wrf_file = ALB_wrf_path + "LIS.CABLE."+str(yr)+str(mth)+"-"+str(yr)+str(mth)+".d01.nc"

                time_wrf, ALB_wrf_tmp = read_var(ALB_wrf_file, "Albedo_inst")

                time_from_2017     = datetime(2017,1,1,0,0,0) - time_init
                time_from_2017_day = time_from_2017.days
                d_s                = time_wrf[0].days - time_from_2017_day
                if yr == 2019 and mth_e == 12:
                    d_e  = time_wrf[-1].days - time_from_2017_day
                else:
                    d_e  = time_wrf[-1].days - time_from_2017_day +1
                print("d_s=",d_s)
                print("d_e=",d_e)

                ALB_out[d_s:d_e,:,:]  = time_clip_to_day(time_wrf[:], ALB_wrf_tmp[:,:,:], time_s, time_e)
                print(np.shape(ALB_out))
                print("ALB_lis",ALB_out)

    return ALB_out

def plot_LAI(LAI_obs_path, LAI_one_path, LAI_two_path, LIS_PFT_file, wrf_path, time_s, time_e, LAI_types=["obs","modis_clim","modis_clim"],PFT=True,message=None):

    year_s     = 2017
    year_e     = 2019

    nlat       = 439
    nlon       = 529

    LAI_obs       = read_LAI(LAI_obs_path,LAI_types[0],year_s,year_e,time_s,time_e,nlat,nlon)
    LAI_obs       = np.where(LAI_obs >0.0001, LAI_obs, np.nan)
    LAI_obs_mean  = np.nanmean(LAI_obs,axis=0)
    LAI_obs_time_series = np.nanmean(LAI_obs,axis=(1,2))

    LAI_one       = read_LAI(LAI_one_path,LAI_types[1],year_s,year_e,time_s,time_e,nlat,nlon)
    LAI_one       = np.where(LAI_one >0.0001, LAI_one, np.nan)
    LAI_one_mean  = np.nanmean(LAI_one,axis=0)
    LAI_one_time_series = np.nanmean(LAI_one,axis=(1,2))

    ntime_obs     = len(LAI_obs[:,0,0])
    ntime_one     = len(LAI_one[:,0,0])

    if LAI_two_path != None:
        LAI_two    = read_LAI(LAI_two_path,LAI_types[2],year_s,year_e,time_s,time_e,nlat,nlon)
        LAI_two    = np.where(LAI_two >0.0001, LAI_two, np.nan)
        LAI_two_mean  = np.nanmean(LAI_two,axis=0)
        LAI_two_time_series = np.nanmean(LAI_two,axis=(1,2))
        ntime_two  = len(LAI_two[:,0,0])

    if PFT:
        # read PFT type

        year_sum     = year_e+1-year_s
        pft_wrf      = Dataset(LIS_PFT_file, mode='r')
        PFT_tmp      = pft_wrf.variables['Landcover_inst'][0,:,:]
        PFT_obs      = [PFT_tmp] * ntime_obs
        PFT_one      = [PFT_tmp] * ntime_one
        #(365*year_sum)
        print("np.shape(PFT)")
        print(np.shape(PFT))
        LAI_obs_pft  = np.zeros((17,ntime_obs,nlat,nlon)) # 365*year_sum
        LAI_one_pft  = np.zeros((17,ntime_one,nlat,nlon))
        if LAI_two_path != None:
            PFT_two      = [PFT_tmp] * ntime_two
            LAI_two_pft  = np.zeros((17,ntime_two,nlat,nlon))

        for i in np.arange(1,18,1):
            LAI_obs_pft[i-1,:,:,:] = np.where(PFT_obs == i, LAI_obs, np.nan)
            LAI_one_pft[i-1,:,:,:] = np.where(PFT_one == i, LAI_one, np.nan)
            if LAI_two_path != None:
                LAI_two_pft[i-1,:,:,:] = np.where(PFT_two == i, LAI_two, np.nan)
        LAI_obs_mean_pft           = np.nanmean(LAI_obs_pft,axis=1)
        LAI_one_mean_pft           = np.nanmean(LAI_one_pft,axis=1)
        if LAI_two_path != None:
            LAI_two_mean_pft           = np.nanmean(LAI_two_pft,axis=1)

        print("np.shape(LAI_obs_mean_pft)")
        print(np.shape(LAI_obs_mean_pft))

        LAI_obs_time_series_pft    = np.nanmean(LAI_obs_pft,axis=(2,3))
        LAI_one_time_series_pft    = np.nanmean(LAI_one_pft,axis=(2,3))
        if LAI_two_path != None:
            LAI_two_time_series_pft= np.nanmean(LAI_two_pft,axis=(2,3))

        print("np.shape(LAI_obs_time_series_pft)")
        print(np.shape(LAI_obs_time_series_pft))

    # read lat and lon outs
    wrf                  = Dataset(wrf_path,  mode='r')
    lon                  = wrf.variables['XLONG'][0,:,:]
    lat                  = wrf.variables['XLAT'][0,:,:]

    # # ================= testing plot ===================
    if 0:
        for i in np.arange(1,18,1):
            fig, axs = plt.subplots(nrows=1, ncols=1, figsize=[10,8],sharex=True, sharey=True, squeeze=True,
                                    subplot_kw={'projection': ccrs.PlateCarree()})
            plot     = axs.contourf( lon, lat, LAI_obs_mean_pft[i-1,:,:], transform=ccrs.PlateCarree(),cmap=plt.cm.seismic,extend='both') # levels=clevs,
            cbar     = plt.colorbar(plot, ax=axs, ticklocation="right", pad=0.05, orientation="horizontal",aspect=40, shrink=0.8) # cax=cax,
            plt.savefig('./plots/spatial_map_OBS_PFT_i='+str(i)+'.png',dpi=300)
            fig = None
            axs = None
            plot = None

        for i in np.arange(1,18,1):
            fig, axs = plt.subplots(nrows=1, ncols=1, figsize=[10,8],sharex=True, sharey=True, squeeze=True,
                                    subplot_kw={'projection': ccrs.PlateCarree()})
            plot     = axs.contourf( lon, lat, LAI_one_mean_pft[i-1,:,:], transform=ccrs.PlateCarree(),cmap=plt.cm.seismic,extend='both') # levels=clevs,
            cbar     = plt.colorbar(plot, ax=axs, ticklocation="right", pad=0.05, orientation="horizontal",aspect=40, shrink=0.8) # cax=cax,
            plt.savefig('./plots/spatial_map_WRF_PFT_i='+str(i)+'.png',dpi=300)
            fig = None
            axs = None
            plot = None

    # =================== Plotting time series ===================
    if 1:
        ls_mark       = ['-','--','-.',':','--',',','o','v','^','<','>','1','2','3','4','s','p']
        cleaner_dates = ["2017","2018", "2019", "2020" ]
        xtickslocs    = [0,     365,      730,   1095  ]

        fig, ax       = plt.subplots(nrows=1, ncols=1, figsize=[10,8],sharex=True, sharey=True, squeeze=True)

        print("np.shape(LAI_obs_time_series)")
        print(np.shape(LAI_obs_time_series))
        if PFT:
            cnt = 0
            for i in [2,5,6,9,14]:
                ax.plot(np.arange(ntime_obs), LAI_obs_time_series_pft[i-1,:], c = 'red', ls=ls_mark[cnt], label='obs'+str(i), alpha=0.5) #
                ax.plot(np.arange(ntime_one), LAI_one_time_series_pft[i-1,:], c = 'green', ls=ls_mark[cnt], alpha=0.5) #, label='2003-2022'+str(i)
                if LAI_two_path != None:
                    ax.plot(np.arange(ntime_two), LAI_two_time_series_pft[i-1,:], c = 'blue', ls=ls_mark[cnt], alpha=0.5) #, label='2014-2016'+str(i)
                cnt = cnt + 1
        else:
            ax.plot(np.arange(ntime_obs), LAI_obs_time_series, c = 'red', ls=ls_mark[0], label='obs', alpha=0.5) #
            ax.plot(np.arange(ntime_one), LAI_one_time_series, c = 'green', ls=ls_mark[0], label='one', alpha=0.5) #, label='2003-2022'+str(i)
            if LAI_two_path != None:
                ax.plot(np.arange(ntime_two), LAI_two_time_series, c = 'blue', ls=ls_mark[0],label='two', alpha=0.5) #, label='2014-2016'+str(i)
        ax.legend()
        fig.tight_layout()

        if message != None:
            plt.savefig('./plots/time_series_'+message+'_LAI.png',dpi=300)
        else:
            plt.savefig('./plots/time_series_LAI.png',dpi=300)

    # =================== Plotting spatial map ===================
    if 1:
        # for j in np.arange(17):
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

        states= NaturalEarthFeature(category="cultural", scale="50m",
                                            facecolor="none",
                                            name="admin_1_states_provinces_shp")

        # ======================= Set colormap =======================
        cmap    = plt.cm.BrBG
        texts   = [ "(a)","(b)","(c)","(d)","(e)",
                    "(f)","(g)","(h)","(i)","(j)",
                    "(k)","(l)","(m)","(n)","(o)",
                    "(p)","(q)","(r)","(s)","(t)"]

        label_x = ["LAI$\mathregular{_{obs}}$",
                    "LAI$\mathregular{_{ctl-obs}}$",
                    "ΔLAI$\mathregular{_{alb-ctl}}$",]

        label_y = ["Annual LAI","Spring LAI","Summer LAI","Autumn LAI","Winter LAI"]
        loc_y   = [0.63,0.55,0.47,0.38]
        cnt     = 0


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


        clevs          = np.arange(0,6,0.2)
        clevs_diff     = [-0.5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5]

        # left: LAI_obs_mean
        plot1    = ax[0].contourf(lon, lat, LAI_obs_mean[:,:], levels=clevs, transform=ccrs.PlateCarree(),cmap=cmap,extend='both') #
        ax[0].text(0.02, 0.15, texts[0], transform=ax[0].transAxes, fontsize=14, verticalalignment='top', bbox=props)
        ax[0].add_feature(OCEAN,edgecolor='none', facecolor="lightgray")

        # middle: LAI_one_mean - LAI_obs_mean
        plot2    = ax[1].contourf(lon, lat, LAI_one_mean[:,:] - LAI_obs_mean[:,:], levels=clevs_diff,transform=ccrs.PlateCarree(),cmap=cmap,extend='both') # levels=clevs,
        ax[1].text(0.02, 0.15, texts[1], transform=ax[1].transAxes, fontsize=14, verticalalignment='top', bbox=props)
        ax[1].add_feature(OCEAN,edgecolor='none', facecolor="lightgray")

        # right:  LAI_two_mean - LAI_one_mean
        plot3   = ax[2].contourf(lon, lat, LAI_two_mean[:,:]-LAI_one_mean[:,:],levels=clevs_diff, transform=ccrs.PlateCarree(),cmap=cmap,extend='both') #  levels=clevs_diff,
        ax[2].text(0.02, 0.15, texts[2], transform=ax[2].transAxes, fontsize=14, verticalalignment='top', bbox=props)
        ax[2].add_feature(OCEAN,edgecolor='none', facecolor="lightgray")

        cbar = plt.colorbar(plot1, ax=ax[0], ticklocation="right", pad=0.08, orientation="horizontal",
                            aspect=40, shrink=1) # cax=cax,
        cbar.ax.tick_params(labelsize=8, labelrotation=45)

        cbar = plt.colorbar(plot2, ax=ax[1], ticklocation="right", pad=0.08, orientation="horizontal",
                            aspect=40, shrink=1) # cax=cax,
        cbar.ax.tick_params(labelsize=8, labelrotation=45)

        cbar = plt.colorbar(plot3, ax=ax[2], ticklocation="right", pad=0.08, orientation="horizontal",
                            aspect=40, shrink=1)
        cbar.ax.tick_params(labelsize=8, labelrotation=45)

        # set top x label
        ax[0].set_title(label_x[0])#,labelpad=-0.1)#, fontsize=12)
        ax[1].set_title(label_x[1])#,labelpad=-0.1)#, fontsize=12)
        ax[2].set_title(label_x[2])#,labelpad=-0.1)#, fontsize=12)
        if message != None:
            plt.savefig('./plots/spatial_map_LAI_'+message+'_compare.png',dpi=300)
        else:
            plt.savefig('./plots/spatial_map_LAI_compare_2017_2019.png',dpi=300)
        fig = None
        ax  = None
        plot1 = None
        plot2 = None
        plot3 = None

def plot_ALBEDO(ALB_obs_paths, ALB_one_path, ALB_two_path, LIS_PFT_file, wrf_path, time_s, time_e, loc_lat=None, loc_lon=None, lat_name="lat", lon_name="lon", ALB_types=["obs","modis_clim","modis_clim"],PFT=True,message=None):

    year_s        = 2017
    year_e        = 2019

    nlat          = 439
    nlon          = 529

    # ========== read ALBEDO ==========
    ALB_obs       = read_ALBEDO(ALB_obs_paths,ALB_types[0],year_s,year_e,time_s,time_e,loc_lat, loc_lon, lat_name, lon_name)
    ALB_obs       = np.where(ALB_obs > 0.0001, ALB_obs, np.nan)
    ALB_obs_mean  = np.nanmean(ALB_obs,axis=0)
    ALB_obs_time_series = np.nanmean(ALB_obs,axis=(1,2))

    ALB_one       = read_ALBEDO(ALB_one_path,ALB_types[1],year_s,year_e,time_s,time_e,loc_lat, loc_lon, lat_name, lon_name)
    ALB_one       = np.where(ALB_one > 0.0001, ALB_one, np.nan)
    ALB_one_mean  = np.nanmean(ALB_one,axis=0)
    ALB_one_time_series = np.nanmean(ALB_one,axis=(1,2))

    ntime_obs     = len(ALB_obs[:,0,0])
    ntime_one     = len(ALB_one[:,0,0])

    if ALB_two_path != None:
        ALB_two       = read_ALBEDO(ALB_two_path,ALB_types[2],year_s,year_e,time_s,time_e,loc_lat, loc_lon, lat_name, lon_name)
        ALB_two       = np.where(ALB_two > 0.0001, ALB_two, np.nan)
        ALB_two_mean  = np.nanmean(ALB_two,axis=0)
        ALB_two_time_series = np.nanmean(ALB_two,axis=(1,2))
        ntime_two     = len(ALB_two[:,0,0])

    # ========== read PFT ==========
    if PFT:

        # read PFT type
        year_sum     = year_e+1-year_s
        pft_wrf      = Dataset(LIS_PFT_file, mode='r')
        PFT_tmp      = pft_wrf.variables['Landcover_inst'][0,:,:]
        PFT_obs      = [PFT_tmp] * ntime_obs #(365*year_sum)
        PFT_one      = [PFT_tmp] * ntime_one #(365*year_sum)

        ALB_obs_pft  = np.zeros((17,ntime_obs,nlat,nlon))
        ALB_one_pft  = np.zeros((17,ntime_one,nlat,nlon))
        if ALB_two_path != None:
            PFT_two      = [PFT_tmp] * ntime_two #(365*year_sum)
            ALB_two_pft  = np.zeros((17,ntime_two,nlat,nlon))
            print("ntime_obs = ", ntime_obs, ", ntime_one=",ntime_one, ", ntime_two=",ntime_two)

        for i in np.arange(1,18,1):
            ALB_obs_pft[i-1,:,:,:] = np.where(PFT_obs == i, ALB_obs, np.nan)
            ALB_one_pft[i-1,:,:,:] = np.where(PFT_one == i, ALB_one, np.nan)
            if ALB_two_path != None:
                ALB_two_pft[i-1,:,:,:] = np.where(PFT_two == i, ALB_two, np.nan)

        ALB_obs_mean_pft           = np.nanmean(ALB_obs_pft,axis=1)
        ALB_one_mean_pft           = np.nanmean(ALB_one_pft,axis=1)
        if ALB_two_path != None:
            ALB_two_mean_pft       = np.nanmean(ALB_two_pft,axis=1)

        print("np.shape(ALB_obs_mean_pft)")
        print(np.shape(ALB_obs_mean_pft))

        ALB_obs_time_series_pft    = np.nanmean(ALB_obs_pft,axis=(2,3))
        ALB_one_time_series_pft    = np.nanmean(ALB_one_pft,axis=(2,3))
        if ALB_two_path != None:
            ALB_two_time_series_pft= np.nanmean(ALB_two_pft,axis=(2,3))

        print("np.shape(ALB_obs_time_series_pft)")
        print(np.shape(ALB_obs_time_series_pft))

    # read lat and lon outs
    wrf                  = Dataset(wrf_path,  mode='r')
    lon                  = wrf.variables['XLONG'][0,:,:]
    lat                  = wrf.variables['XLAT'][0,:,:]

    # # ================= testing plot ===================
    if 0:
        # I haven't tested these codes
        for i in np.arange(1,18,1):
            fig, axs = plt.subplots(nrows=1, ncols=1, figsize=[10,8],sharex=True, sharey=True, squeeze=True,
                                    subplot_kw={'projection': ccrs.PlateCarree()})
            plot     = axs.contourf( lon, lat, ALB_obs_mean_pft[i-1,:,:], transform=ccrs.PlateCarree(),cmap=plt.cm.seismic,extend='both') # levels=clevs,
            cbar     = plt.colorbar(plot, ax=axs, ticklocation="right", pad=0.05, orientation="horizontal",aspect=40, shrink=0.8) # cax=cax,
            plt.savefig('./plots/spatial_map_OBS_PFT_i='+str(i)+'.png',dpi=300)
            fig = None
            axs = None
            plot = None

        for i in np.arange(1,18,1):
            fig, axs = plt.subplots(nrows=1, ncols=1, figsize=[10,8],sharex=True, sharey=True, squeeze=True,
                                    subplot_kw={'projection': ccrs.PlateCarree()})
            plot     = axs.contourf( lon, lat, ALB_one_mean_pft[i-1,:,:], transform=ccrs.PlateCarree(),cmap=plt.cm.seismic,extend='both') # levels=clevs,
            cbar     = plt.colorbar(plot, ax=axs, ticklocation="right", pad=0.05, orientation="horizontal",aspect=40, shrink=0.8) # cax=cax,
            plt.savefig('./plots/spatial_map_WRF_PFT_i='+str(i)+'.png',dpi=300)
            fig = None
            axs = None
            plot = None

    # =================== Plotting time series ===================
    if 1:
        ls_mark       = ['-','--','-.',':','--',',','o','v','^','<','>','1','2','3','4','s','p']
        cleaner_dates = ["2017","2018", "2019", "2020" ]
        xtickslocs    = [0,     365,      730,   1095  ]

        fig, ax       = plt.subplots(nrows=1, ncols=1, figsize=[10,8],sharex=True, sharey=True, squeeze=True)

        print("np.shape(ALB_obs_time_series)")
        print(np.shape(ALB_obs_time_series))
        if PFT:
            cnt = 0
            for i in [2,5,6,9,14]:
                ax.plot(np.arange(ntime_obs), ALB_obs_time_series_pft[i-1,:], c = 'red', ls=ls_mark[cnt], label='obs'+str(i), alpha=0.5) #
                ax.plot(np.arange(ntime_one), ALB_one_time_series_pft[i-1,:], c = 'green', ls=ls_mark[cnt], alpha=0.5) # , label='2003-2022'+str(i)
                if ALB_two_path != None:
                    ax.plot(np.arange(ntime_two), ALB_two_time_series_pft[i-1,:], c = 'blue', ls=ls_mark[cnt], alpha=0.5) #, label='2014-2016'+str(i)
                cnt = cnt + 1
        else:
            ax.plot(np.arange(ntime_obs), ALB_obs_time_series, c = 'red', ls=ls_mark[0], label='obs', alpha=0.5) #
            ax.plot(np.arange(ntime_one), ALB_one_time_series, c = 'green', ls=ls_mark[0],label='one', alpha=0.5) # , label='2003-2022'+str(i)
            if ALB_two_path != None:
                ax.plot(np.arange(ntime_two), ALB_two_time_series, c = 'blue', ls=ls_mark[0], label='two', alpha=0.5) #, label='2014-2016'+str(i)

        ax.legend()
        fig.tight_layout()

        if message != None:
            plt.savefig('./plots/time_series_'+message+'_ALB.png',dpi=300)
        else:
            plt.savefig('./plots/time_series_ALB.png',dpi=300)


    # =================== Plotting spatial map ===================
    if 1:
        # for j in np.arange(17):
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

        states= NaturalEarthFeature(category="cultural", scale="50m",
                                            facecolor="none",
                                            name="admin_1_states_provinces_shp")

        # ======================= Set colormap =======================
        texts   = [ "(a)","(b)","(c)","(d)","(e)",
                    "(f)","(g)","(h)","(i)","(j)",
                    "(k)","(l)","(m)","(n)","(o)",
                    "(p)","(q)","(r)","(s)","(t)"]

        label_x = ["ALB$\mathregular{_{obs}}$",
                    "ALB$\mathregular{_{ctl-obs}}$",
                    "ΔALB$\mathregular{_{abl-ctl}}$",]

        label_y = ["Annual ALB","Spring ALB","Summer ALB","Autumn ALB","Winter ALB"]
        loc_y   = [0.63,0.55,0.47,0.38]
        cnt     = 0


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
            # gl.xlocator     = mticker.FixedLocator([130,135,140,145,150,155,160])
            # gl.ylocator     = mticker.FixedLocator([-40,-35,-30,-25,-20])
            gl.xformatter   = LONGITUDE_FORMATTER
            gl.yformatter   = LATITUDE_FORMATTER
            gl.xlabel_style = {'size':12, 'color':almost_black}#,'rotation': 90}
            gl.ylabel_style = {'size':12, 'color':almost_black}

            gl.xlabels_bottom = True
            gl.ylabels_left   = True


        clevs          = np.arange(0,0.3,0.005)
        clevs_diff     = [-0.035,-0.03,-0.025,-0.02,-0.015,-0.01,-0.005,0.005,0.01,0.015,0.02,0.025,0.03,0.035]

        cmap           = plt.cm.BrBG

        # left: ALB_obs_mean
        plot1    = ax[0].contourf(lon, lat, ALB_obs_mean[:,:], levels=clevs, transform=ccrs.PlateCarree(),cmap=cmap,extend='both') #
        ax[0].text(0.02, 0.15, texts[0], transform=ax[0].transAxes, fontsize=14, verticalalignment='top', bbox=props)
        ax[0].add_feature(OCEAN,edgecolor='none', facecolor="lightgray")

        # middle: ALB_clim - ALB_obs_mean
        plot2    = ax[1].contourf(lon, lat, ALB_one_mean[:,:]-ALB_obs_mean[:,:], levels=clevs_diff,transform=ccrs.PlateCarree(),cmap=cmap,extend='both') # levels=clevs,
        ax[1].text(0.02, 0.15, texts[1], transform=ax[1].transAxes, fontsize=14, verticalalignment='top', bbox=props)
        ax[1].add_feature(OCEAN,edgecolor='none', facecolor="lightgray")

        # right: ALB_1416 - ALB_clim
        plot3   = ax[2].contourf(lon, lat, ALB_two_mean[:,:]-ALB_one_mean[:,:],levels=clevs_diff, transform=ccrs.PlateCarree(),cmap=cmap,extend='both') #  levels=clevs_diff,
        ax[2].text(0.02, 0.15, texts[2], transform=ax[2].transAxes, fontsize=14, verticalalignment='top', bbox=props)
        ax[2].add_feature(OCEAN,edgecolor='none', facecolor="lightgray")

        cbar = plt.colorbar(plot1, ax=ax[0], ticklocation="right", pad=0.08, orientation="horizontal",
                            aspect=40, shrink=1) # cax=cax,
        cbar.ax.tick_params(labelsize=8, labelrotation=45)

        cbar = plt.colorbar(plot2, ax=ax[1], ticklocation="right", pad=0.08, orientation="horizontal",
                            aspect=40, shrink=1) # cax=cax,
        cbar.ax.tick_params(labelsize=8, labelrotation=45)

        cbar = plt.colorbar(plot3, ax=ax[2], ticklocation="right", pad=0.08, orientation="horizontal",
                            aspect=40, shrink=1)
        cbar.ax.tick_params(labelsize=8, labelrotation=45)

        # set top x label
        ax[0].set_title(label_x[0])#,labelpad=-0.1)#, fontsize=12)
        ax[1].set_title(label_x[1])#,labelpad=-0.1)#, fontsize=12)
        ax[2].set_title(label_x[2])#,labelpad=-0.1)#, fontsize=12)
        if message != None:
            plt.savefig('./plots/spatial_map_ALB_'+message+'_compare.png',dpi=300)
        else:
            plt.savefig('./plots/spatial_map_ALB_2017_2019_compare.png',dpi=300)
        fig = None
        ax  = None
        plot1 = None
        plot2 = None
        plot3 = None

def plot_periods(obs_path, one_path, two_path=None, LIS_PFT_file=None, wrf_path=None, var_name=None, PFT=True,message=None):

    year_s        = 2017
    year_e        = 2019

    nlat          = 439
    nlon          = 529

    time_obs,var_obs = read_var(obs_path, var_name)
    time_one,var_one = read_var(one_path, var_name)
    obs_time_series = np.nanmean(var_obs,axis=(1,2))
    one_time_series = np.nanmean(var_one,axis=(1,2))
    if two_path != None:
        time_two,var_two = read_var(two_path, var_name)
        two_time_series = np.nanmean(var_two,axis=(1,2))


    t_obs = []
    t_one = []
    for t in time_obs:
        t_obs.append(t.days)
    for t in time_one:
        t_one.append(t.days)
    t_obs = np.array(t_obs)
    t_one = np.array(t_one)

    if two_path != None:
        t_two = []
        for t in time_two:
            t_two.append(t.days)
        t_two = np.array(t_two)


    if PFT:
        # read PFT type
        pft_wrf      = Dataset(LIS_PFT_file, mode='r')
        PFT_tmp      = pft_wrf.variables['Landcover_inst'][0,:,:]
        PFT_obs      = [PFT_tmp] * len(time_obs)
        PFT_one      = [PFT_tmp] * len(time_one)
        obs_pft      = np.zeros((17,len(time_obs),nlat,nlon))
        one_pft      = np.zeros((17,len(time_one),nlat,nlon))

        for i in np.arange(1,18,1):
            obs_pft[i-1,:,:,:] = np.where(PFT_obs == i, var_obs, np.nan)
            one_pft[i-1,:,:,:] = np.where(PFT_one == i, var_one, np.nan)

        obs_mean_pft           = np.nanmean(obs_pft,axis=1)
        one_mean_pft           = np.nanmean(one_pft,axis=1)

        obs_time_series_pft    = np.nanmean(obs_pft,axis=(2,3))
        one_time_series_pft    = np.nanmean(one_pft,axis=(2,3))

        if two_path !=None:
            PFT_two  = [PFT_tmp] * len(time_two)
            two_pft      = np.zeros((17,len(time_two),nlat,nlon))
            for i in np.arange(1,18,1):
                two_pft[i-1,:,:,:] = np.where(PFT_two == i, var_two, np.nan)
            two_mean_pft           = np.nanmean(two_pft,axis=1)
            two_time_series_pft    = np.nanmean(two_pft,axis=(2,3))


    # read lat and lon outs
    wrf                  = Dataset(wrf_path,  mode='r')
    lon                  = wrf.variables['XLONG'][0,:,:]
    lat                  = wrf.variables['XLAT'][0,:,:]

    # =================== Plotting time series ===================
    if 1:
        ls_mark       = ['-','--','-.',':','--',',','o','v','^','<','>','1','2','3','4','s','p']
        cleaner_dates = ["2017","2018", "2019", "2020" ]
        xtickslocs    = [0,     365,      730,   1095  ]

        fig, ax       = plt.subplots(nrows=1, ncols=1, figsize=[10,8],sharex=True, sharey=True, squeeze=True)

        print("np.shape(obs_time_series)")
        print(np.shape(obs_time_series))
        if PFT:
            cnt = 0
            for i in [2,5,6,9,14]:
                ax.plot(t_obs, obs_time_series_pft[i-1,:], c = 'red', ls=ls_mark[cnt], label='2003-2016'+str(i), alpha=0.5) #
                ax.plot(t_one, one_time_series_pft[i-1,:], c = 'green', ls=ls_mark[cnt], label='2016-2020'+str(i), alpha=0.5) #
                if two_path !=None:
                    ax.plot(t_two, two_time_series_pft[i-1,:], c = 'blue', ls=ls_mark[cnt], label='2003-2022'+str(i), alpha=0.5) #
                cnt = cnt + 1
        else:
            ax.plot(t_obs, obs_time_series, c = 'red', ls=ls_mark[0], label='2003-2016', alpha=0.5) #
            ax.plot(t_one, one_time_series, c = 'green', ls=ls_mark[0], label='2016-2020', alpha=0.5) #
            if two_path !=None:
                ax.plot(t_two, two_time_series, c = 'blue', ls=ls_mark[0], label='2003-2022', alpha=0.5) #

        ax.legend()
        fig.tight_layout()

        if message != None:
            plt.savefig('./plots/time_series_'+message+'.png',dpi=300)
        else:
            plt.savefig('./plots/time_series_periods_compare.png',dpi=300)

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


    # small box
    # loc_lat    = [-33,-29]
    # loc_lon    = [147,149]

    # # east coast
    # loc_lat    = [-33,-27]
    # loc_lon    = [152,154]

    PFT        = False
    if 1:
        '''
        Compare LIS LAI/albedo output with MODIS observed LAI/albedo
        '''
        ###### to make spatial summer plots ######
        # LAI
        if 1:
            '''
            Compare LIS output in ALB_LAI sim and in default sim and observed time-varying MODIS LAI
            '''

            LAI_types     = ["obs","lis_multi_out","lis_multi_out"]

            path          = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"

            LAI_obs_path  = "/g/data/w97/mm3972/data/MODIS/MODIS_LAI/regrid_daily_fill/MCD15A3H_c61_bigWRFroi_LAI_AVHRR_fill_for_WRF_daily_20170101_20201231.nc"
                            #"/g/data/w97/mm3972/data/MODIS/MODIS_LAI/refuse/whittakerSmoothed_MCD15A3H_c61_LAI_for_WRF_daily_20170101_20200630.nc"
            LAI_one_path  = [ path+"drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/LIS_output/LAI_inst/LIS.CABLE.201701-201912.nc",]
            LAI_two_path  = [ path+"drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2_obs_LAI_ALB/LIS_output/LAI_inst/LIS.CABLE.201701-201912.nc",]

            LIS_PFT_file  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/LIS_output/LIS.CABLE.201701-201701.d01.nc"
            wrf_path      = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/WRF_output/wrfout_d01_2017-02-01_06:00:00"

            time_s        = datetime(2017,12,1,0,0,0,0)
            time_e        = datetime(2018,3,1,0,0,0,0)
            message       = "LAI_201718_summer"
            plot_LAI( LAI_obs_path, LAI_one_path, LAI_two_path, LIS_PFT_file, wrf_path, time_s, time_e, LAI_types,PFT=PFT,message=message)


            time_s        = datetime(2018,12,1,0,0,0,0)
            time_e        = datetime(2019,3,1,0,0,0,0)
            message       = "LAI_201819_summer"
            plot_LAI( LAI_obs_path, LAI_one_path, LAI_two_path, LIS_PFT_file, wrf_path, time_s, time_e, LAI_types,PFT=PFT,message=message)


            time_s        = datetime(2019,12,1,0,0,0,0)
            time_e        = datetime(2020,3,1,0,0,0,0)
            message       = "LAI_201920_summer"
            plot_LAI( LAI_obs_path, LAI_one_path, LAI_two_path, LIS_PFT_file, wrf_path, time_s, time_e, LAI_types,PFT=PFT,message=message)

        # ALBEDO
        if 1:
            '''
            Compare LIS ALBEDO with time-varying MODIS ALBEDO
            '''

            ALB_types     = ["modis_blue_sky","lis_multi_out","lis_multi_out"]

            ALB_obs_paths = "/g/data/w97/mm3972/data/MODIS/MODIS_Albedo/refuse/albedo_climatology_0.05CMG/MODIS_Blue_Sky_Albedo_Climatology.CMG005_for_WRF.nc"
            # ALB_obs_paths = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB_before_Mar2023/drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2_obs_LAI_ALB/LIS_output/LIS.CABLE.201701-202006_ALB_LAI.nc"
            # ALB_one_path  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB_before_Mar2023/drght_2017_2019_bl_pbl2_mp4_sf_sfclay2/LIS_output/LIS.CABLE.201701-202006_ALB_LAI.nc"
            # ALB_two_path  = "/g/data/w97/mm3972/data/MODIS/MODIS_Albedo/refuse/albedo_climatology_0.05CMG/MODIS_Blue_Sky_Albedo_Climatology.CMG005_for_WRF.nc"

            path          = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"
            ALB_one_path  = [ path+"drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/LIS_output/Albedo_inst/LIS.CABLE.201701-201912.nc",]
            ALB_two_path  = [ path+"drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2_obs_LAI_ALB/LIS_output/Albedo_inst/LIS.CABLE.201701-201912.nc",]

            LIS_PFT_file  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/LIS_output/LIS.CABLE.201701-201701.d01.nc"
            wrf_path      = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/WRF_output/wrfout_d01_2017-02-01_06:00:00"

            message       = "Albedo_201718_summer"
            time_s        = datetime(2017,12,1,0,0,0,0)
            time_e        = datetime(2018,3,1,0,0,0,0)
            plot_ALBEDO(ALB_obs_paths, ALB_one_path, ALB_two_path, LIS_PFT_file, wrf_path, time_s, time_e, loc_lat=loc_lat, loc_lon=loc_lon,
                                  lat_name=lat_name, lon_name=lon_name, ALB_types=ALB_types,PFT=PFT,message=message)

            message       = "Albedo_201819_summer"
            time_s        = datetime(2018,12,1,0,0,0,0)
            time_e        = datetime(2019,3,1,0,0,0,0)
            plot_ALBEDO(ALB_obs_paths, ALB_one_path, ALB_two_path, LIS_PFT_file, wrf_path, time_s, time_e, loc_lat=loc_lat, loc_lon=loc_lon,
                                  lat_name=lat_name, lon_name=lon_name, ALB_types=ALB_types,PFT=PFT,message=message)

            message       = "Albedo_201920_summer"
            time_s        = datetime(2019,12,1,0,0,0,0)
            time_e        = datetime(2020,3,1,0,0,0,0)
            plot_ALBEDO(ALB_obs_paths, ALB_one_path, ALB_two_path, LIS_PFT_file, wrf_path, time_s, time_e, loc_lat=loc_lat, loc_lon=loc_lon,
                                  lat_name=lat_name, lon_name=lon_name, ALB_types=ALB_types,PFT=PFT,message=message)


        ###### to make time series plots ######
        # LAI
        if 1:
            '''
            Compare LIS output in ALB_LAI sim and in default sim and observed time-varying MODIS LAI
            '''

            LAI_types     = ["obs","lis_multi_out","lis_multi_out"]

            path          = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"

            LAI_obs_path  = "/g/data/w97/mm3972/data/MODIS/MODIS_LAI/regrid_daily_fill/MCD15A3H_c61_bigWRFroi_LAI_AVHRR_fill_for_WRF_daily_20170101_20201231.nc"
                            #"/g/data/w97/mm3972/data/MODIS/MODIS_LAI/refuse/whittakerSmoothed_MCD15A3H_c61_LAI_for_WRF_daily_20170101_20200630.nc"
            LAI_one_path  = [ path+"drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/LIS_output/LAI_inst/LIS.CABLE.201701-201912.nc",]
            LAI_two_path  = [ path+"drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2_obs_LAI_ALB/LIS_output/LAI_inst/LIS.CABLE.201701-201912.nc",]

            LIS_PFT_file  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/LIS_output/LIS.CABLE.201701-201701.d01.nc"
            wrf_path      = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/WRF_output/wrfout_d01_2017-02-01_06:00:00"

            time_s        = datetime(2017,1,1,0,0,0,0)
            time_e        = datetime(2020,3,1,0,0,0,0)
            message       = "LAI_201701_202002"
            plot_LAI( LAI_obs_path, LAI_one_path, LAI_two_path, LIS_PFT_file, wrf_path, time_s, time_e, LAI_types,PFT=PFT,message=message)


        # ALBEDO
        if 1:
            '''
            Compare LIS ALBEDO with time-varying MODIS ALBEDO
            '''

            ALB_types     = ["modis_blue_sky","lis_multi_out","lis_multi_out"]

            ALB_obs_paths = "/g/data/w97/mm3972/data/MODIS/MODIS_Albedo/refuse/albedo_climatology_0.05CMG/MODIS_Blue_Sky_Albedo_Climatology.CMG005_for_WRF.nc"
            # ALB_obs_paths = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB_before_Mar2023/drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2_obs_LAI_ALB/LIS_output/LIS.CABLE.201701-202006_ALB_LAI.nc"
            # ALB_one_path  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB_before_Mar2023/drght_2017_2019_bl_pbl2_mp4_sf_sfclay2/LIS_output/LIS.CABLE.201701-202006_ALB_LAI.nc"
            # ALB_two_path  = "/g/data/w97/mm3972/data/MODIS/MODIS_Albedo/refuse/albedo_climatology_0.05CMG/MODIS_Blue_Sky_Albedo_Climatology.CMG005_for_WRF.nc"

            path          = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"
            ALB_one_path  = [ path+"drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/LIS_output/Albedo_inst/LIS.CABLE.201701-201912.nc",]
            ALB_two_path  = [ path+"drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2_obs_LAI_ALB/LIS_output/Albedo_inst/LIS.CABLE.201701-201912.nc",]

            LIS_PFT_file  = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/LIS_output/LIS.CABLE.201701-201701.d01.nc"
            wrf_path      = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/WRF_output/wrfout_d01_2017-02-01_06:00:00"

            message       = "Albedo_201701_202002"
            time_s        = datetime(2017,1,1,0,0,0,0)
            time_e        = datetime(2020,3,1,0,0,0,0)
            plot_ALBEDO(ALB_obs_paths, ALB_one_path, ALB_two_path, LIS_PFT_file, wrf_path, time_s, time_e, loc_lat=loc_lat, loc_lon=loc_lon,
                                  lat_name=lat_name, lon_name=lon_name, ALB_types=ALB_types,PFT=PFT,message=message)
