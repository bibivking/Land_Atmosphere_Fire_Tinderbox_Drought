import sys
import cartopy
import numpy as np
import pandas as pd
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

def read_LIS_diff(var_name,file_name,land_ctl_path, land_sen_path, lat_names, lon_names,
                  loc_lat=None, loc_lon=None, time_s=None,time_e=None):

    print("plotting "+var_name)

    if var_name in ["Tmax","Tmin","TDR"]:
        land_ctl_files= [land_ctl_path+'Tair_f_inst/'+file_name]
        land_sen_files= [land_sen_path+'Tair_f_inst/'+file_name]
        time, Ctl_tmp = read_var_multi_file(land_ctl_files, 'Tair_f_inst', loc_lat, loc_lon, lat_names, lon_names)
        time, Sen_tmp = read_var_multi_file(land_sen_files, 'Tair_f_inst', loc_lat, loc_lon, lat_names, lon_names)
        Ctl_tmp       = Ctl_tmp -273.15
        Sen_tmp       = Sen_tmp -273.15
    elif var_name in ["VegTmax","VegTmin","VegTDR"]:
        land_ctl_files= [land_ctl_path+'VegT_tavg/'+file_name]
        land_sen_files= [land_sen_path+'VegT_tavg/'+file_name]
        time, Ctl_tmp = read_var_multi_file(land_ctl_files, 'VegT_tavg', loc_lat, loc_lon, lat_names, lon_names)
        time, Sen_tmp = read_var_multi_file(land_sen_files, 'VegT_tavg', loc_lat, loc_lon, lat_names, lon_names)
        Ctl_tmp       = Ctl_tmp -273.15
        Sen_tmp       = Sen_tmp -273.15
    elif var_name in ["SurfTmax","SurfTmin","SurfTDR"]:
        land_ctl_files= [land_ctl_path+'AvgSurfT_tavg/'+file_name]
        land_sen_files= [land_sen_path+'AvgSurfT_tavg/'+file_name]
        time, Ctl_tmp = read_var_multi_file(land_ctl_files, 'AvgSurfT_tavg', loc_lat, loc_lon, lat_names, lon_names)
        time, Sen_tmp = read_var_multi_file(land_sen_files, 'AvgSurfT_tavg', loc_lat, loc_lon, lat_names, lon_names)
        Ctl_tmp       = Ctl_tmp -273.15
        Sen_tmp       = Sen_tmp -273.15
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
    elif var_name in ['VPD','VPDmax','VPDmin']:
        tair_ctl_files= [land_ctl_path+'Tair_f_inst/'+file_name]
        tair_sen_files= [land_sen_path+'Tair_f_inst/'+file_name]
        qair_ctl_files= [land_ctl_path+'Qair_f_inst/'+file_name]
        qair_sen_files= [land_sen_path+'Qair_f_inst/'+file_name]
        pres_ctl_files= ['/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/'
                            +'WRF_output/slp/wrfout_201701-202002.nc']
        pres_sen_files= ['/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2_obs_LAI_ALB/'
                            +'WRF_output/slp/wrfout_201701-202002.nc']

        time, Tair_ctl    = read_var_multi_file(tair_ctl_files, "Tair_f_inst", loc_lat, loc_lon, lat_names, lon_names)
        time, Tair_sen    = read_var_multi_file(tair_sen_files, "Tair_f_inst", loc_lat, loc_lon, lat_names, lon_names)
        time, Qair_ctl    = read_var_multi_file(qair_ctl_files, "Qair_f_inst", loc_lat, loc_lon, lat_names, lon_names)
        time, Qair_sen    = read_var_multi_file(qair_sen_files, "Qair_f_inst", loc_lat, loc_lon, lat_names, lon_names)
        time_wrf, Pres_ctl_tmp= read_var_multi_file(pres_ctl_files, "slp", loc_lat, loc_lon, lat_names, lon_names)
        time_wrf, Pres_sen_tmp= read_var_multi_file(pres_sen_files, "slp", loc_lat, loc_lon, lat_names, lon_names)

        time_in = []
        time_out= []
        for t in time_wrf:
            time_in.append(t.total_seconds())
        for t in time:
            time_out.append(t.total_seconds())

        f_ctl             = interp1d(np.array(time_in), Pres_ctl_tmp[:], kind='linear',fill_value='extrapolate', axis=0)
        f_sen             = interp1d(np.array(time_in), Pres_sen_tmp[:],kind='linear', fill_value='extrapolate', axis=0)
        Pres_ctl          = f_ctl(np.array(time_out))
        Pres_sen          = f_sen(np.array(time_out))
        Ctl_tmp           = qair_to_vpd(Qair_ctl, Tair_ctl, Pres_ctl)
        Sen_tmp           = qair_to_vpd(Qair_sen, Tair_sen, Pres_sen)
    else:
        land_ctl_files= [land_ctl_path+var_name+'/'+file_name]
        land_sen_files= [land_sen_path+var_name+'/'+file_name]
        time, Ctl_tmp = read_var_multi_file(land_ctl_files, var_name, loc_lat, loc_lon, lat_names, lon_names)
        time, Sen_tmp = read_var_multi_file(land_sen_files, var_name, loc_lat, loc_lon, lat_names, lon_names)

    if 'max' in var_name:
        # average of daily max
        ctl_in       = spatial_var_max(time,Ctl_tmp,time_s,time_e)
        sen_in       = spatial_var_max(time,Sen_tmp,time_s,time_e)
    elif 'min' in var_name:
        # average of daily min
        ctl_in       = spatial_var_min(time,Ctl_tmp,time_s,time_e)
        sen_in       = spatial_var_min(time,Sen_tmp,time_s,time_e)
    elif 'TDR' in var_name:
        # average of daily min
        ctl_in_max   = spatial_var_max(time,Ctl_tmp,time_s,time_e)
        sen_in_max   = spatial_var_max(time,Sen_tmp,time_s,time_e)
        ctl_in_min   = spatial_var_min(time,Ctl_tmp,time_s,time_e)
        sen_in_min   = spatial_var_min(time,Sen_tmp,time_s,time_e)
        ctl_in       = ctl_in_max - ctl_in_min
        sen_in       = sen_in_max - sen_in_min
    else:
        ctl_in       = spatial_var(time,Ctl_tmp,time_s,time_e)
        sen_in       = spatial_var(time,Sen_tmp,time_s,time_e)

    if var_name in ['WaterTableD_tavg','WatTable']:
        ctl_in     = ctl_in/1000.
        sen_in     = sen_in/1000.
    if var_name in ['ESoil_tavg','Evap_tavg',"ECanop_tavg",'TVeg_tavg',"Rainf_tavg","Snowf_tavg","Qs_tavg","Qsb_tavg"]:
        t_s        = time_s - datetime(2000,1,1,0,0,0,0)
        t_e        = time_e - datetime(2000,1,1,0,0,0,0)
        ctl_in     = ctl_in*3600*24*(t_e.days - t_s.days)
        sen_in     = sen_in*3600*24*(t_e.days - t_s.days)
    if var_name in ['Qair_f_inst']:
        ctl_in     = ctl_in*1000
        sen_in     = sen_in*1000
    if var_name in ['GPP_tavg','NPP_tavg']:
        t_s        = time_s - datetime(2000,1,1,0,0,0,0)
        t_e        = time_e - datetime(2000,1,1,0,0,0,0)
        s2d        = 3600*24.          # s-1 to d-1
        GPP_scale  = -0.000001*12*s2d   # umol s-1 to g d-1
        ctl_in     = ctl_in*GPP_scale*(t_e.days - t_s.days)
        sen_in     = sen_in*GPP_scale*(t_e.days - t_s.days)

    var_diff     = sen_in - ctl_in

    # Define the RGB values as a 2D array
    rgb_17colors= np.array([
                        [0.338024, 0.193310, 0.020377],
                        [0.458593, 0.264360, 0.031142],
                        [0.576471, 0.343483, 0.058055],
                        [0.686275, 0.446828, 0.133410],
                        [0.778547, 0.565859, 0.250288],
                        [0.847443, 0.705805, 0.422530],
                        [0.932872, 0.857209, 0.667820],
                        [0.964091, 0.917801, 0.795463],
                        [0.955517, 0.959016, 0.9570165],
                        [0.808689, 0.924414, 0.907882],
                        [0.627528, 0.855210, 0.820531],
                        [0.426990, 0.749942, 0.706882],
                        [0.265513, 0.633679, 0.599231],
                        [0.135871, 0.524337, 0.492964],
                        [0.023914, 0.418839, 0.387466],
                        [0.002153, 0.325721, 0.287274],
                        [0.000000, 0.235294, 0.188235]
                    ])

    rgb_21colors = np.array([
                [0.338024, 0.193310, 0.020377],
                [0.441369, 0.254210, 0.029604],
                [0.544714, 0.315110, 0.038831],
                [0.631373, 0.395156, 0.095732],
                [0.733333, 0.491119, 0.165706],
                [0.793310, 0.595848, 0.287197],
                [0.857286, 0.725798, 0.447136],
                [0.904575, 0.810458, 0.581699],
                [0.947020, 0.880584, 0.710880],
                [0.963629, 0.923799, 0.818531],
                [0.955517, 0.959016, 0.9570165],
                [0.822837, 0.927797, 0.912803],
                [0.714879, 0.890888, 0.864821],
                [0.583852, 0.837370, 0.798385],
                [0.461592, 0.774856, 0.729950],
                [0.311649, 0.666897, 0.629988],
                [0.183852, 0.569550, 0.538178],
                [0.087889, 0.479123, 0.447751],
                [0.003691, 0.390311, 0.358016],
                [0.001845, 0.312803, 0.273126],
                [0.000000, 0.235294, 0.188235]
            ])


    # Create a colormap from the RGB values
    cmap17 = plt.cm.colors.ListedColormap(rgb_17colors)
    cmap21 = plt.cm.colors.ListedColormap(rgb_21colors)

    # Select clevs
    cmap  = plt.cm.BrBG
    if var_name in ['WaterTableD_tavg','WatTable']:
        clevs = [-4,-3,-2,-1.5,-1,-0.5,0.5,1,1.5,2,3,4]
    elif var_name in ['GWwb_tavg','GWMoist']:
        clevs = [-0.05,-0.04,-0.03,-0.02,-0.01,-0.005,0.005,0.01,0.02,0.03,0.04,0.05]
    elif  var_name in ["Qair_f_inst"]:
        clevs = [-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]
    elif var_name in ['SoilMoist_inst','SoilMoist',"SM_top50cm"]:
        clevs = [-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0.05,0.1,0.15,0.2,0.25,0.3]
    elif var_name in ['ESoil_tavg','Evap_tavg',"ECanop_tavg",'TVeg_tavg',"Rainf_tavg","Snowf_tavg","Qs_tavg","Qsb_tavg"]:
        clevs = [-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,-5,5,10,20.,30,40,50,60,70,80,90,100]
    elif var_name in ["GPP_tavg","NPP_tavg",]:
        clevs = [-200,-190,-180,-170,-160,-150,-140,-130,-120,-110,-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,
                    -5,5,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
        clevs_percentage =  [-70,-60,-50,-40,-30,-20,-10,10,20,30,40,50,60,70]
        cmap  = plt.cm.BrBG
    elif var_name in ["CanopInt_inst","SnowCover_inst"]:
        clevs = [-2.,-1.8,-1.6,-1.4,-1.2,-1.,-0.8,-0.6,-0.4,-0.2,0.2,0.4,0.6,0.8,1.,1.2,1.4,1.6,1.8,2.]
    elif var_name in ["Qle_tavg","Qh_tavg","Qg_tavg","Rnet",]:
        clevs = [-140,-120,-100,-80,-60,-40,-20,-10,-5,5,10,20,40,60,80,100,120,140]
    elif var_name in ["Swnet_tavg","Lwnet_tavg","SWdown_f_inst","LWdown_f_inst","Rnet"]:
        clevs = [-22,-18,-14,-10,-6,-2,2,6,10,14,18,22]
    elif var_name in ["Wind_f_inst",]:
        clevs = [-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4]
    elif var_name in ["Psurf_f_inst"]:
        clevs = [-22,-18,-14,-10,-6,-2,2,6,10,14,18,22]
    elif var_name in ["Tair_f_inst","Tmax","Tmin","VegT_tavg","VegTmax","VegTmin",
                        "AvgSurfT_tavg","SurfTmax","SurfTmin","SoilTemp_inst",'TDR','VegTDR','SurfTDR']:
        clevs = [-1.2,-1.1,-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.,1.1,1.2]
        cmap  = plt.cm.seismic
    elif var_name in ["Wind_f_inst",]:
        clevs = [-2.,-1.5,-1,-0.5,-0.1,0.1,0.5,1.,1.5,2.]
    elif var_name in ["FWsoil_tavg","SmLiqFrac_inst","SmFrozFrac_inst"]:
        clevs = [-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0.05,0.1,0.15,0.2,0.25,0.3,0.35]
    elif var_name in ["LAI_inst"]:
        clevs = [-2,-1.8,-1.6,-1.4,-1.2,-1.,-0.8,-0.6,-0.4,-0.2,0.2,0.4,0.6,0.8,1.,1.2,1.4,1.6,1.8,2]
        clevs_percentage =  [-70,-60,-50,-40,-30,-20,-10,-5,5,10,20,30,40,50,60,70]
        cmap  = cmap21
    elif var_name in ["VPD","VPDmax","VPDmin",]:
        clevs = [-0.1,-0.09,-0.08,-0.07,-0.06,-0.05,-0.04,-0.03,-0.02,-0.01,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
        cmap  = plt.cm.BrBG
    elif var_name in ["Albedo_inst"]:
        clevs = [-0.08,-0.07,-0.06,-0.05,-0.04,-0.03,-0.02,-0.01,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08]
        clevs_percentage =   [-70,-60,-50,-40,-30,-20,-10,-5,5,10,20,30,40,50,60,70]
        cmap  = cmap17
    else:
        clevs = [-0.5,-0.4,-0.3,-0.2,-0.1,-0.05,0.05,0.1,0.2,0.3,0.4,0.5]

    return var_diff, clevs, cmap

def regrid_to_fire_map_resolution(fire_path, var_in, lat_in, lon_in, loc_lat=None, loc_lon=None, burn=0):

    # =========== Read in fire data ============
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

def plot_Tmax_LAI_Albedo_diff(fire_path, file_name, land_ctl_path, land_sen_path, time_ss=None, time_es=None,
                              lat_names="lat", lon_names="lon", loc_lat=None, loc_lon=None, reg_lats=None,
                              reg_lons=None, wrf_path=None, message=None, burn=0):

    '''
    plot LIS variables in burnt / unburnt / all regions
    '''

    # Read in WRF lat and lon
    wrf            = Dataset(wrf_path,  mode='r')
    lon_in         = wrf.variables['XLONG'][0,:,:]
    lat_in         = wrf.variables['XLAT'][0,:,:]

    # Read in var
    Tmax_diff_Dec, clevs_Tmax, cmap_Tmax = read_LIS_diff("Tmax", file_name, land_ctl_path, land_sen_path,
                                                          lat_names, lon_names, loc_lat, loc_lon, time_ss[0], time_es[0])
    LAI_diff_Dec,  clevs_LAI,  cmap_LAI  = read_LIS_diff("LAI_inst", file_name, land_ctl_path, land_sen_path,
                                                          lat_names, lon_names, loc_lat, loc_lon, time_ss[0], time_es[0])
    ALB_diff_Dec,  clevs_ALB,  cmap_ALB  = read_LIS_diff("Albedo_inst", file_name, land_ctl_path, land_sen_path,
                                                          lat_names, lon_names, loc_lat, loc_lon, time_ss[0], time_es[0])
    FW_diff_Dec,   clevs_FW,   cmap_FW   = read_LIS_diff("FWsoil_tavg", file_name, land_ctl_path, land_sen_path,
                                                          lat_names, lon_names, loc_lat, loc_lon, time_ss[0], time_es[0])

    Tmax_diff_Jan, clevs_Tmax, cmap_Tmax = read_LIS_diff("Tmax", file_name, land_ctl_path, land_sen_path,
                                                          lat_names, lon_names, loc_lat, loc_lon, time_ss[1], time_es[1])
    LAI_diff_Jan,  clevs_LAI,  cmap_LAI  = read_LIS_diff("LAI_inst", file_name, land_ctl_path, land_sen_path,
                                                          lat_names, lon_names, loc_lat, loc_lon, time_ss[1], time_es[1])
    ALB_diff_Jan,  clevs_ALB,  cmap_ALB  = read_LIS_diff("Albedo_inst", file_name, land_ctl_path, land_sen_path,
                                                          lat_names, lon_names, loc_lat, loc_lon, time_ss[1], time_es[1])
    FW_diff_Jan,   clevs_FW,   cmap_FW   = read_LIS_diff("FWsoil_tavg", file_name, land_ctl_path, land_sen_path,
                                                          lat_names, lon_names, loc_lat, loc_lon, time_ss[1], time_es[1])

    Tmax_diff_Feb, clevs_Tmax, cmap_Tmax = read_LIS_diff("Tmax", file_name, land_ctl_path, land_sen_path,
                                                          lat_names, lon_names, loc_lat, loc_lon, time_ss[2], time_es[2])
    LAI_diff_Feb,  clevs_LAI,  cmap_LAI  = read_LIS_diff("LAI_inst", file_name, land_ctl_path, land_sen_path,
                                                          lat_names, lon_names, loc_lat, loc_lon, time_ss[2], time_es[2])
    ALB_diff_Feb,  clevs_ALB,  cmap_ALB  = read_LIS_diff("Albedo_inst", file_name, land_ctl_path, land_sen_path,
                                                          lat_names, lon_names, loc_lat, loc_lon, time_ss[2], time_es[2])
    FW_diff_Feb,   clevs_FW,   cmap_FW   = read_LIS_diff("FWsoil_tavg", file_name, land_ctl_path, land_sen_path,
                                                          lat_names, lon_names, loc_lat, loc_lon, time_ss[2], time_es[2])

    # regrid to burn map resolution ~ 400m
    Tmax_regrid_Dec, lats, lons = regrid_to_fire_map_resolution(fire_path, Tmax_diff_Dec, lat_in, lon_in, burn=burn)
    LAI_regrid_Dec,  lats, lons = regrid_to_fire_map_resolution(fire_path, LAI_diff_Dec, lat_in, lon_in, burn=burn)
    ALB_regrid_Dec,  lats, lons = regrid_to_fire_map_resolution(fire_path, ALB_diff_Dec, lat_in, lon_in, burn=burn)
    FW_regrid_Dec,   lats, lons = regrid_to_fire_map_resolution(fire_path, FW_diff_Dec, lat_in, lon_in, burn=burn)

    Tmax_regrid_Jan, lats, lons = regrid_to_fire_map_resolution(fire_path, Tmax_diff_Jan, lat_in, lon_in, burn=burn)
    LAI_regrid_Jan,  lats, lons = regrid_to_fire_map_resolution(fire_path, LAI_diff_Jan, lat_in, lon_in, burn=burn)
    ALB_regrid_Jan,  lats, lons = regrid_to_fire_map_resolution(fire_path, ALB_diff_Jan, lat_in, lon_in, burn=burn)
    FW_regrid_Jan,   lats, lons = regrid_to_fire_map_resolution(fire_path, FW_diff_Jan, lat_in, lon_in, burn=burn)

    Tmax_regrid_Feb, lats, lons = regrid_to_fire_map_resolution(fire_path, Tmax_diff_Feb, lat_in, lon_in, burn=burn)
    LAI_regrid_Feb,  lats, lons = regrid_to_fire_map_resolution(fire_path, LAI_diff_Feb, lat_in, lon_in, burn=burn)
    ALB_regrid_Feb,  lats, lons = regrid_to_fire_map_resolution(fire_path, ALB_diff_Feb, lat_in, lon_in, burn=burn)
    FW_regrid_Feb,   lats, lons = regrid_to_fire_map_resolution(fire_path, FW_diff_Feb, lat_in, lon_in, burn=burn)

    # ================== Start Plotting =================
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=[7,9],sharex=False,
                sharey=False, squeeze=True, subplot_kw={'projection': ccrs.PlateCarree()})

    plt.subplots_adjust(wspace=-0.6, hspace=-0.15)

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

    # =============== CHANGE HERE ===============
    x_ticks = [146, 148, 150, 152, 154]      # Example x-axis tick positions
    y_ticks = [-38, -36, -34, -32, -30,-28]  # Example y-axis tick positions

    for i in np.arange(3):
        for j in np.arange(3):
            axs[i,j].set_facecolor('lightgray')
            axs[i,j].coastlines(resolution="50m",linewidth=1)
            axs[i,j].set_extent([146,154,-39,-27])
            axs[i,j].add_feature(states, linewidth=.5, edgecolor="black")

            # Set the ticks on the x-axis and y-axis
            axs[i,j].tick_params(axis='x', direction='out')
            axs[i,j].tick_params(axis='y', direction='out')
            x_ticks = np.arange(148, 155, 4)
            y_ticks = np.arange(-40, -26, 4)
            axs[i,j].set_xticks(x_ticks)
            axs[i,j].set_yticks(y_ticks)

            if i==2:
                axs[i, j].set_xticklabels(['148$\mathregular{^{o}}$E','152$\mathregular{^{o}}$E'])
            else:
                axs[i, j].set_xticklabels([])

            if j==0:
                axs[i, j].set_yticklabels(['40$\mathregular{^{o}}$S','36$\mathregular{^{o}}$S',
                                           '32$\mathregular{^{o}}$S','28$\mathregular{^{o}}$S'])
            else:
                axs[i, j].set_yticklabels([])


    Tmax_regrid_Dec, lats, lons = regrid_to_fire_map_resolution(fire_path, Tmax_diff_Dec, lat_in, lon_in, burn=burn)
    LAI_regrid_Dec,  lats, lons = regrid_to_fire_map_resolution(fire_path, LAI_diff_Dec, lat_in, lon_in, burn=burn)
    ALB_regrid_Dec,  lats, lons = regrid_to_fire_map_resolution(fire_path, ALB_diff_Dec, lat_in, lon_in, burn=burn)

    # Tmax
    plot1 = axs[0,0].contourf(lons, lats, Tmax_regrid_Dec, clevs_Tmax, transform=ccrs.PlateCarree(), cmap=cmap_Tmax, extend='both')
    plot1 = axs[0,1].contourf(lons, lats, Tmax_regrid_Jan, clevs_Tmax, transform=ccrs.PlateCarree(), cmap=cmap_Tmax, extend='both')
    plot1 = axs[0,2].contourf(lons, lats, Tmax_regrid_Feb, clevs_Tmax, transform=ccrs.PlateCarree(), cmap=cmap_Tmax, extend='both')

    cbar = plt.colorbar(plot1, ax=axs[0,2], ticklocation="right", pad=0.05, orientation="vertical",
            aspect=30, shrink=0.9) # cax=cax,
    cbar.set_label('$\mathregular{^{o}}$C', loc='center',size=12)# rotation=270,
    cbar.ax.tick_params(labelsize=12)#,labelrotation=45)

    axs[0,0].add_feature(OCEAN,edgecolor='none', facecolor="white")
    axs[0,1].add_feature(OCEAN,edgecolor='none', facecolor="white")
    axs[0,2].add_feature(OCEAN,edgecolor='none', facecolor="white")

    # LAI
    plot2 = axs[1,0].contourf(lons, lats, LAI_regrid_Dec, clevs_LAI, transform=ccrs.PlateCarree(), cmap=cmap_LAI, extend='both')
    plot2 = axs[1,1].contourf(lons, lats, LAI_regrid_Jan, clevs_LAI, transform=ccrs.PlateCarree(), cmap=cmap_LAI, extend='both')
    plot2 = axs[1,2].contourf(lons, lats, LAI_regrid_Feb, clevs_LAI, transform=ccrs.PlateCarree(), cmap=cmap_LAI, extend='both')
    cbar  = plt.colorbar(plot2, ax=axs[1,2], ticklocation="right", pad=0.05, orientation="vertical",
            aspect=30, shrink=0.9) # cax=cax,
    cbar.set_label('m$\mathregular{^{2}}$ m$\mathregular{^{-2}}$', loc='center',size=12)# rotation=270,
    cbar.ax.tick_params(labelsize=12)#,labelrotation=45)

    axs[1,0].add_feature(OCEAN,edgecolor='none', facecolor="white")
    axs[1,1].add_feature(OCEAN,edgecolor='none', facecolor="white")
    axs[1,2].add_feature(OCEAN,edgecolor='none', facecolor="white")

    # Albedo
    plot3 = axs[2,0].contourf(lons, lats, ALB_regrid_Dec, clevs_ALB, transform=ccrs.PlateCarree(), cmap=cmap_ALB, extend='both')
    plot3 = axs[2,1].contourf(lons, lats, ALB_regrid_Jan, clevs_ALB, transform=ccrs.PlateCarree(), cmap=cmap_ALB, extend='both')
    plot3 = axs[2,2].contourf(lons, lats, ALB_regrid_Feb, clevs_ALB, transform=ccrs.PlateCarree(), cmap=cmap_ALB, extend='both')
    cbar  = plt.colorbar(plot3, ax=axs[2,2], ticklocation="right", pad=0.05, orientation="vertical",
            aspect=30, shrink=0.9) # cax=cax,
    cbar.set_label('(-)', loc='center',size=12)# rotation=270,
    cbar.ax.tick_params(labelsize=12)#,labelrotation=45)

    axs[2,0].add_feature(OCEAN,edgecolor='none', facecolor="white")
    axs[2,1].add_feature(OCEAN,edgecolor='none', facecolor="white")
    axs[2,2].add_feature(OCEAN,edgecolor='none', facecolor="white")

    # Add boxes, lines
    for i in np.arange(3):
        axs[0,0].add_patch(Polygon([[reg_lons[i][0], reg_lats[i][0]], [reg_lons[i][1], reg_lats[i][0]],
                                    [reg_lons[i][1], reg_lats[i][1]], [reg_lons[i][0], reg_lats[i][1]]],
                                    closed=True,color=almost_black, fill=False,linewidth=0.8))

    axs[0,1].plot([ 149, 154], [-30, -30],    c=almost_black, lw=0.8, alpha = 1, linestyle="--", transform=ccrs.PlateCarree())
    axs[0,1].plot([ 148, 153], [-33, -33],    c=almost_black, lw=0.8, alpha = 1, linestyle="--", transform=ccrs.PlateCarree())
    axs[0,1].plot([ 146, 151], [-37.5, -37.5],c=almost_black, lw=0.8, alpha = 1, linestyle="--", transform=ccrs.PlateCarree())

    # Adding titles
    axs[0,0].set_title("Dec 2019", fontsize=12)
    axs[0,1].set_title("Jan 2020", fontsize=12)
    axs[0,2].set_title("Feb 2020", fontsize=12)

    axs[0,0].text(-0.32, 0.53, 'ΔT$\mathregular{_{max}}$', va='bottom', ha='center',
              rotation='vertical', rotation_mode='anchor',
              transform=axs[0,0].transAxes, fontsize=12)
    axs[1,0].text(-0.32, 0.50, "ΔLAI", va='bottom', ha='center',
              rotation='vertical', rotation_mode='anchor',
              transform=axs[1,0].transAxes, fontsize=12)
    axs[2,0].text(-0.32, 0.48, "Δ$α$", va='bottom', ha='center',
              rotation='vertical', rotation_mode='anchor',
              transform=axs[2,0].transAxes, fontsize=12)

    # Apply tight layout
    plt.tight_layout()
    plt.savefig('./plots/spatial_map_' +message + '.png',dpi=300)


if __name__ == "__main__":

    if 1:
        '''
        3 X 3 plot
        '''

        loc_lat       = [-40,-25]
        loc_lon       = [135,155]

        case_ctl      = "drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2"
        case_sen      = "drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2_obs_LAI_ALB"

        wrf_path      = "/scratch/w97/mm3972/model/NUWRF/Tinderbox_drght_LAI_ALB/output/drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/WRF_output/wrfout_d01_2017-02-01_06:00:00"
        land_sen_path = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_sen+"/LIS_output/"
        land_ctl_path = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/"+case_ctl+"/LIS_output/"

        fire_path     = '/g/data/w97/mm3972/data/MODIS/MODIS_fire/MCD64A1.061_500m_aid0001.nc'

        reg_lats      = [  [-32,-28.5],
                           [-34.5,-32.5],
                           [-38,-34.5]    ]

        reg_lons      = [  [151.5,153.5],
                           [149.5,151.5],
                           [146.5,151]    ]

        burn         = 1
        message      = "HW_Tmax_LAI_Albedo_burnt"
        file_name    = 'LIS.CABLE.201912-202002.nc'

        time_ss      = [datetime(2019,12,1,0,0,0,0),
                        datetime(2020,1,1,0,0,0,0),
                        datetime(2020,2,1,0,0,0,0)]

        time_es      = [datetime(2020,1,1,0,0,0,0),
                        datetime(2020,2,1,0,0,0,0),
                        datetime(2020,3,1,0,0,0,0)]

        plot_Tmax_LAI_Albedo_diff(fire_path, file_name, land_ctl_path, land_sen_path, time_ss=time_ss, time_es=time_es, lat_names="lat",
                      lon_names="lon", loc_lat=loc_lat, loc_lon=loc_lon,  reg_lats=reg_lats, reg_lons=reg_lons, wrf_path=wrf_path, message=message, burn=burn)
