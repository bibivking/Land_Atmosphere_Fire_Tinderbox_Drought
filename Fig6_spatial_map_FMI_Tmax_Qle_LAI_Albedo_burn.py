import sys
import cartopy
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.patches import Polygon
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
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
    seismic_rgb_24colors= np.array([
                        [0.000000, 0.000000, 0.332941],
                        [0.000000, 0.000000, 0.541569],
                        [0.000000, 0.000000, 0.662353],
                        [0.000000, 0.000000, 0.860000],
                        [0.000000, 0.000000, 0.958824],
                        [0.113725, 0.113725, 1.000000],
                        [0.270588, 0.270588, 1.000000],
                        [0.396078, 0.396078, 1.000000],
                        [0.552941, 0.552941, 1.000000],
                        [0.662745, 0.662745, 1.000000],
                        [0.835294, 0.835294, 1.000000],
                        [0.992157, 0.992157, 1.000000],
                        [0.992157, 0.992157, 1.000000],
                        [1.000000, 0.568627, 0.568627],
                        [1.000000, 0.411765, 0.411765],
                        [1.000000, 0.286275, 0.286275],
                        [1.000000, 0.145098, 0.145098],
                        [1.000000, 0.003922, 0.003922],
                        [0.923529, 0.000000, 0.000000],
                        [0.860784, 0.000000, 0.000000],
                        [0.782353, 0.000000, 0.000000],
                        [0.703922, 0.000000, 0.000000],
                        [0.641176, 0.000000, 0.000000],
                        [0.547059, 0.000000, 0.000000],
                    ])

    # Define the RGB values as a 2D array
    BrBG_rgb_17colors= np.array([
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

    # Define the RGB values as a 2D array
    BrBG_rgb_11colors= np.array([
                        [0.338024, 0.193310, 0.020377],
                        [0.576471, 0.343483, 0.058055],
                        [0.686275, 0.446828, 0.133410],
                        [0.847443, 0.705805, 0.422530],
                        [0.964091, 0.917801, 0.795463],
                        [0.955517, 0.959016, 0.9570165],
                        [0.808689, 0.924414, 0.907882],
                        [0.426990, 0.749942, 0.706882],
                        [0.135871, 0.524337, 0.492964],
                        [0.023914, 0.418839, 0.387466],
                        [0.000000, 0.235294, 0.188235]
                    ])

    BrBG_rgb_16colors= np.array([
                        [0.338024, 0.193310, 0.020377],
                        [0.458593, 0.264360, 0.031142],
                        [0.576471, 0.343483, 0.058055],
                        [0.686275, 0.446828, 0.133410],
                        [0.778547, 0.565859, 0.250288],
                        [0.847443, 0.705805, 0.422530],
                        [0.932872, 0.857209, 0.667820],
                        [0.955517, 0.959016, 0.9570165],
                        [0.955517, 0.959016, 0.9570165],
                        [0.627528, 0.855210, 0.820531],
                        [0.426990, 0.749942, 0.706882],
                        [0.265513, 0.633679, 0.599231],
                        [0.135871, 0.524337, 0.492964],
                        [0.023914, 0.418839, 0.387466],
                        [0.002153, 0.325721, 0.287274],
                        [0.000000, 0.235294, 0.188235]
                    ])

    BrBG_rgb_21colors = np.array([
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


    BrBG_rgb_20colors = np.array([
                [0.338024, 0.193310, 0.020377],
                [0.441369, 0.254210, 0.029604],
                [0.544714, 0.315110, 0.038831],
                [0.631373, 0.395156, 0.095732],
                [0.733333, 0.491119, 0.165706],
                [0.793310, 0.595848, 0.287197],
                [0.857286, 0.725798, 0.447136],
                [0.904575, 0.810458, 0.581699],
                [0.947020, 0.880584, 0.710880],
                [0.955517, 0.959016, 0.9570165],
                [0.955517, 0.959016, 0.9570165],
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
    cmap17 = plt.cm.colors.ListedColormap(BrBG_rgb_17colors)
    cmap21 = plt.cm.colors.ListedColormap(BrBG_rgb_21colors)

    cmap11 = plt.cm.colors.ListedColormap(BrBG_rgb_11colors)
    cmap16 = plt.cm.colors.ListedColormap(BrBG_rgb_16colors)
    cmap20 = plt.cm.colors.ListedColormap(BrBG_rgb_20colors)
    cmap24 = plt.cm.colors.ListedColormap(seismic_rgb_24colors)

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
        clevs = [-22,-18,-14,-10,-6,-2,2,6,10,14,18,22]
        clevs_ticks = ['-22','-18','-14','-10','-6','-2','2','6','10',"14",'18','22']
        cmap  = cmap11 #cmap22
        # clevs = [-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8]
    elif var_name in ["Swnet_tavg","Lwnet_tavg","SWdown_f_inst","LWdown_f_inst"]:
        clevs = [-22,-18,-14,-10,-6,-2,2,6,10,14,18,22]
    elif var_name in ["Wind_f_inst",]:
        clevs = [-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4]
    elif var_name in ["Psurf_f_inst"]:
        clevs = [-22,-18,-14,-10,-6,-2,2,6,10,14,18,22]
    elif var_name in ["Tair_f_inst","Tmax","Tmin","VegT_tavg","VegTmax","VegTmin",
                        "AvgSurfT_tavg","SurfTmax","SurfTmin","SoilTemp_inst",'TDR','VegTDR','SurfTDR']:
        clevs = [-1.2,-1.1,-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.,1.1,1.2]
        clevs_ticks = ['','-1.1','','-0.9','','-0.7','','-0.5','','-0.3','','-0.1','',
                      '0.1','','0.3','','0.5','','0.7','','0.9','','1.1','']
        cmap  = cmap24 #plt.cm.seismic
    elif var_name in ["Wind_f_inst",]:
        clevs = [-2.,-1.5,-1,-0.5,-0.1,0.1,0.5,1.,1.5,2.]
    elif var_name in ["FWsoil_tavg","SmLiqFrac_inst","SmFrozFrac_inst"]:
        clevs = [-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0.05,0.1,0.15,0.2,0.25,0.3,0.35]
    elif var_name in ["LAI_inst"]:
        clevs       = [-2,-1.8,-1.6,-1.4,-1.2,-1.,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.,1.2,1.4,1.6,1.8,2]
        clevs_ticks = ['','-1.8','','-1.4','','-1.0','','-0.6','','-0.2','','0.2','','0.6','','1.0','','1.4','','1.8','']
        clevs_percentage =  [-70,-60,-50,-40,-30,-20,-10,-5,5,10,20,30,40,50,60,70]
        cmap  = cmap20 #cmap22
    elif var_name in ["VPD","VPDmax","VPDmin",]:
        clevs = [-0.1,-0.09,-0.08,-0.07,-0.06,-0.05,-0.04,-0.03,-0.02,-0.01,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
        cmap  = plt.cm.BrBG
    elif var_name in ["Albedo_inst"]:
        clevs = [-0.08,-0.07,-0.06,-0.05,-0.04,-0.03,-0.02,-0.01,0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08]
        clevs_ticks = ['','-0.07','','-0.05','','-0.03','','-0.01','','0.01','','0.03','','0.05','','0.07','']
        clevs_percentage =   [-70,-60,-50,-40,-30,-20,-10,-5,5,10,20,30,40,50,60,70]
        cmap  = cmap16 #cmap18
    else:
        clevs = [-0.5,-0.4,-0.3,-0.2,-0.1,-0.05,0.05,0.1,0.2,0.3,0.4,0.5]

    return var_diff, clevs, clevs_ticks, cmap

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

def plot_Tmax_LAI_Albedo_diff(fire_path, file_name, FMI_path, land_ctl_path, land_sen_path, time_ss=None, time_es=None,
                              lat_names="lat", lon_names="lon", loc_lat=None, loc_lon=None, reg_lats=None,
                              reg_lons=None, wrf_path=None, message=None, burn_shadow=False, burn=0):

    '''
    plot LIS variables in burnt / unburnt / all regions
    '''

    # Read in WRF lat and lon
    wrf            = Dataset(wrf_path,  mode='r')
    lon_in         = wrf.variables['XLONG'][0,:,:]
    lat_in         = wrf.variables['XLAT'][0,:,:]

    # Read in var
    Tmax_diff_month1, clevs_Tmax, clevs_Tmax_tick, cmap_Tmax = read_LIS_diff("Tmax", file_name, land_ctl_path, land_sen_path,
                                                          lat_names, lon_names, loc_lat, loc_lon, time_ss[0], time_es[0])
    LAI_diff_month1,  clevs_LAI,  clevs_LAI_tick, cmap_LAI  = read_LIS_diff("LAI_inst", file_name, land_ctl_path, land_sen_path,
                                                          lat_names, lon_names, loc_lat, loc_lon, time_ss[0], time_es[0])
    ALB_diff_month1,  clevs_ALB,  clevs_ALB_tick, cmap_ALB  = read_LIS_diff("Albedo_inst", file_name, land_ctl_path, land_sen_path,
                                                          lat_names, lon_names, loc_lat, loc_lon, time_ss[0], time_es[0])
    Qle_diff_month1,  clevs_Qle,  clevs_Qle_tick, cmap_Qle  = read_LIS_diff("Qle_tavg", file_name, land_ctl_path, land_sen_path,
                                                          lat_names, lon_names, loc_lat, loc_lon, time_ss[0], time_es[0])

    Tmax_diff_month2, clevs_Tmax, clevs_Tmax_tick,cmap_Tmax = read_LIS_diff("Tmax", file_name, land_ctl_path, land_sen_path,
                                                          lat_names, lon_names, loc_lat, loc_lon, time_ss[1], time_es[1])
    LAI_diff_month2,  clevs_LAI,  clevs_LAI_tick,cmap_LAI  = read_LIS_diff("LAI_inst", file_name, land_ctl_path, land_sen_path,
                                                          lat_names, lon_names, loc_lat, loc_lon, time_ss[1], time_es[1])
    ALB_diff_month2,  clevs_ALB,  clevs_ALB_tick,cmap_ALB  = read_LIS_diff("Albedo_inst", file_name, land_ctl_path, land_sen_path,
                                                          lat_names, lon_names, loc_lat, loc_lon, time_ss[1], time_es[1])
    Qle_diff_month2,  clevs_Qle,  clevs_Qle_tick,cmap_Qle  = read_LIS_diff("Qle_tavg", file_name, land_ctl_path, land_sen_path,
                                                          lat_names, lon_names, loc_lat, loc_lon, time_ss[1], time_es[1])

    Tmax_diff_month3, clevs_Tmax, clevs_Tmax_tick,cmap_Tmax = read_LIS_diff("Tmax", file_name, land_ctl_path, land_sen_path,
                                                          lat_names, lon_names, loc_lat, loc_lon, time_ss[2], time_es[2])
    LAI_diff_month3,  clevs_LAI,  clevs_LAI_tick,cmap_LAI  = read_LIS_diff("LAI_inst", file_name, land_ctl_path, land_sen_path,
                                                          lat_names, lon_names, loc_lat, loc_lon, time_ss[2], time_es[2])
    ALB_diff_month3,  clevs_ALB,  clevs_ALB_tick,cmap_ALB  = read_LIS_diff("Albedo_inst", file_name, land_ctl_path, land_sen_path,
                                                          lat_names, lon_names, loc_lat, loc_lon, time_ss[2], time_es[2])
    Qle_diff_month3,  clevs_Qle,  clevs_Qle_tick,cmap_Qle  = read_LIS_diff("Qle_tavg", file_name, land_ctl_path, land_sen_path,
                                                          lat_names, lon_names, loc_lat, loc_lon, time_ss[2], time_es[2])

    # Read in FMI
    try:
        # plot hourly data based daily FMI max value from Dec 2019 to Feb 2020
        time, FMI_ctl = read_var_multi_file([FMI_path], "max_FMI_ctl", loc_lat, loc_lon, lat_names, lon_names)
        time, FMI_sen = read_var_multi_file([FMI_path], "max_FMI_sen", loc_lat, loc_lon, lat_names, lon_names)
        FMI_diff      = FMI_sen-FMI_ctl
        FMI_diff_month1  = np.nanmean(FMI_diff[:30,:,:],axis=0)
        FMI_diff_month2  = np.nanmean(FMI_diff[30:61,:,:],axis=0)
        FMI_diff_month3  = np.nanmean(FMI_diff[61:,:,:],axis=0)
    except:
        # plot 6-hourly data based daytime FMI value from Sep 2019 to Nov 2019
        time, FMI_ctl = read_var_multi_file([FMI_path], "FMI_ctl", loc_lat, loc_lon, lat_names, lon_names)
        time, FMI_sen = read_var_multi_file([FMI_path], "FMI_sen", loc_lat, loc_lon, lat_names, lon_names)
        FMI_diff      = FMI_sen-FMI_ctl
        FMI_diff_month1  = np.nanmean(FMI_diff[:29,:,:],axis=0) # the file start from 2nd Sept, these index match to the file
        FMI_diff_month2  = np.nanmean(FMI_diff[29:60,:,:],axis=0)
        FMI_diff_month3  = np.nanmean(FMI_diff[60:90,:,:],axis=0)

    BrBG_rgb_20colors = np.array([
                [0.338024, 0.193310, 0.020377],
                [0.441369, 0.254210, 0.029604],
                [0.544714, 0.315110, 0.038831],
                [0.631373, 0.395156, 0.095732],
                [0.733333, 0.491119, 0.165706],
                [0.793310, 0.595848, 0.287197],
                [0.857286, 0.725798, 0.447136],
                [0.904575, 0.810458, 0.581699],
                [0.947020, 0.880584, 0.710880],
                [0.955517, 0.959016, 0.9570165],
                [0.955517, 0.959016, 0.9570165],
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
    cmap20 = plt.cm.colors.ListedColormap(BrBG_rgb_20colors)

    clevs_FMI     = [-2.,-1.8,-1.6,-1.4,-1.2,-1.0,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0]
    clevs_FMI_tick= ['','-1.8','','-1.4','','-1.0','','-0.6','','-0.2','','0.2','','0.6','','1.0','','1.4','','1.8','']
    # clevs_FMI     = [-4.,-3.5,-3,-2.5,-2.,-1.5,-1,-0.5,0.5,1.,1.5,2.,2.5,3,3.5,4]
    cmap_FMI      = cmap20#plt.cm.BrBG #cmap_ALB # cmap20 #


    # regrid to burn map resolution ~ 400m
    Tmax_regrid_month1, lats, lons = regrid_to_fire_map_resolution(fire_path, Tmax_diff_month1,lat_in, lon_in, burn=burn)
    LAI_regrid_month1,  lats, lons = regrid_to_fire_map_resolution(fire_path, LAI_diff_month1, lat_in, lon_in, burn=burn)
    ALB_regrid_month1,  lats, lons = regrid_to_fire_map_resolution(fire_path, ALB_diff_month1, lat_in, lon_in, burn=burn)
    FMI_regrid_month1,  lats, lons = regrid_to_fire_map_resolution(fire_path, FMI_diff_month1, lat_in, lon_in, burn=burn)
    Qle_regrid_month1,  lats, lons = regrid_to_fire_map_resolution(fire_path, Qle_diff_month1, lat_in, lon_in, burn=burn)

    Tmax_regrid_month2, lats, lons = regrid_to_fire_map_resolution(fire_path, Tmax_diff_month2,lat_in, lon_in, burn=burn)
    LAI_regrid_month2,  lats, lons = regrid_to_fire_map_resolution(fire_path, LAI_diff_month2, lat_in, lon_in, burn=burn)
    ALB_regrid_month2,  lats, lons = regrid_to_fire_map_resolution(fire_path, ALB_diff_month2, lat_in, lon_in, burn=burn)
    FMI_regrid_month2,  lats, lons = regrid_to_fire_map_resolution(fire_path, FMI_diff_month2, lat_in, lon_in, burn=burn)
    Qle_regrid_month2,  lats, lons = regrid_to_fire_map_resolution(fire_path, Qle_diff_month2, lat_in, lon_in, burn=burn)

    Tmax_regrid_month3, lats, lons = regrid_to_fire_map_resolution(fire_path, Tmax_diff_month3,lat_in, lon_in, burn=burn)
    LAI_regrid_month3,  lats, lons = regrid_to_fire_map_resolution(fire_path, LAI_diff_month3, lat_in, lon_in, burn=burn)
    ALB_regrid_month3,  lats, lons = regrid_to_fire_map_resolution(fire_path, ALB_diff_month3, lat_in, lon_in, burn=burn)
    FMI_regrid_month3,  lats, lons = regrid_to_fire_map_resolution(fire_path, FMI_diff_month3, lat_in, lon_in, burn=burn)
    Qle_regrid_month3,  lats, lons = regrid_to_fire_map_resolution(fire_path, Qle_diff_month3, lat_in, lon_in, burn=burn)

    if burn_shadow:
        # print('burn_shadow=',burn_shadow)
        # =========== Read in data ===========
        fire_path      = '/g/data/w97/mm3972/data/MODIS/MODIS_fire/MCD64A1.061_500m_aid0001.nc'
        fire_file      = Dataset(fire_path, mode='r')
        Burn_Date_tmp  = fire_file.variables['Burn_Date'][2:8,::-1,:]  # 2019-09 - 2020-02
        lat_fire       = fire_file.variables['lat'][:]
        lon_fire       = fire_file.variables['lon'][:]

        # print('lat_fire',lat_fire)
        # print('lats',lats)

        Burn_Date      = Burn_Date_tmp.astype(float)
        Burn_Date      = np.where(Burn_Date<=0, 99999, Burn_Date)

        Burn_Date[4:,:,:] = Burn_Date[4:,:,:]+365 # Add 365 to Jan-Feb 2020

        Burn_Date_min  = np.nanmin(Burn_Date, axis=0)
        unique_values  = np.unique(Burn_Date)
        # print("Unique values:", unique_values)

        Burn_Date_index = np.where(Burn_Date_min<=273,   1009, Burn_Date_min)
        Burn_Date_index = np.where(Burn_Date_index<=304, 1010, Burn_Date_index)
        Burn_Date_index = np.where(Burn_Date_index<=334, 1011, Burn_Date_index)
        Burn_Date_index = np.where(Burn_Date_index<=365, 1012, Burn_Date_index)
        Burn_Date_index = np.where(Burn_Date_index<=396, 1013, Burn_Date_index)
        Burn_Date_index = np.where(Burn_Date_index<=425, 1014, Burn_Date_index)
        Burn_Date_index = np.where(Burn_Date_index>=99999, np.nan, Burn_Date_index)

        # # Sep - Nov
        expected_index = [1009,1010,1011]

        # Dec - Feb
        # expected_index = [1012,1013,1014]

    # ================== Start Plotting =================
    fig, axs = plt.subplots(nrows=3, ncols=5, figsize=[11,11],sharex=True,
                sharey=False, squeeze=True, subplot_kw={'projection': ccrs.PlateCarree()})
                # gridspec_kw={'width_ratios': [1, 1, 1, 1]})  # Adjust the width ratios as needed

    plt.subplots_adjust(wspace=0.05, hspace=0.05)

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
        for j in np.arange(5):
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
                axs[i, j].set_xticklabels(['148$\mathregular{^{o}}$E','152$\mathregular{^{o}}$E'],color=almost_black,alpha=1.)
            else:
                axs[i, j].set_xticklabels(['148$\mathregular{^{o}}$E','152$\mathregular{^{o}}$E'],color='white',alpha=0)

            if j==0:
                axs[i, j].set_yticklabels(['40$\mathregular{^{o}}$S','36$\mathregular{^{o}}$S',
                                        '32$\mathregular{^{o}}$S','28$\mathregular{^{o}}$S'])
            else:
                axs[i, j].set_yticklabels([])
    if 0:
        # LAI
        col = 0
        plot1 = axs[0,col].contourf(lons, lats, LAI_regrid_month1, clevs_LAI, transform=ccrs.PlateCarree(), cmap=cmap_LAI, extend='both')
        plot1 = axs[1,col].contourf(lons, lats, LAI_regrid_month2, clevs_LAI, transform=ccrs.PlateCarree(), cmap=cmap_LAI, extend='both')
        plot1 = axs[2,col].contourf(lons, lats, LAI_regrid_month3, clevs_LAI, transform=ccrs.PlateCarree(), cmap=cmap_LAI, extend='both')
        # Albedo
        col = 1
        plot2 = axs[0,col].contourf(lons, lats, ALB_regrid_month1, clevs_ALB, transform=ccrs.PlateCarree(), cmap=cmap_ALB, extend='both')
        plot2 = axs[1,col].contourf(lons, lats, ALB_regrid_month2, clevs_ALB, transform=ccrs.PlateCarree(), cmap=cmap_ALB, extend='both')
        plot2 = axs[2,col].contourf(lons, lats, ALB_regrid_month3, clevs_ALB, transform=ccrs.PlateCarree(), cmap=cmap_ALB, extend='both')

        # Qle
        col = 2
        plot3 = axs[0,col].contourf(lons, lats, Qle_regrid_month1, clevs_Qle, transform=ccrs.PlateCarree(), cmap=cmap_Qle, extend='both')
        plot3 = axs[1,col].contourf(lons, lats, Qle_regrid_month2, clevs_Qle, transform=ccrs.PlateCarree(), cmap=cmap_Qle, extend='both')
        plot3 = axs[2,col].contourf(lons, lats, Qle_regrid_month3, clevs_Qle, transform=ccrs.PlateCarree(), cmap=cmap_Qle, extend='both')

        # Tmax
        col = 3
        plot4 = axs[0,col].contourf(lons, lats, Tmax_regrid_month1, clevs_Tmax, transform=ccrs.PlateCarree(), cmap=cmap_Tmax, extend='both')
        plot4 = axs[1,col].contourf(lons, lats, Tmax_regrid_month2, clevs_Tmax, transform=ccrs.PlateCarree(), cmap=cmap_Tmax, extend='both')
        plot4 = axs[2,col].contourf(lons, lats, Tmax_regrid_month3, clevs_Tmax, transform=ccrs.PlateCarree(), cmap=cmap_Tmax, extend='both')

        # FMI
        col = 4
        plot5 = axs[0,col].contourf(lons, lats, FMI_regrid_month1, clevs_FMI, transform=ccrs.PlateCarree(), cmap=cmap_FMI, extend='both')
        plot5 = axs[1,col].contourf(lons, lats, FMI_regrid_month2, clevs_FMI, transform=ccrs.PlateCarree(), cmap=cmap_FMI, extend='both')
        plot5 = axs[2,col].contourf(lons, lats, FMI_regrid_month3, clevs_FMI, transform=ccrs.PlateCarree(), cmap=cmap_FMI, extend='both')

    else:
        extent   = (min(lons), max(lons), min(lats), max(lats))
        # LAI
        col = 0
        plot1 = axs[0,col].imshow(LAI_regrid_month1[::-1,:], origin="lower", extent=extent, interpolation="none", vmin=np.min(clevs_LAI), vmax=np.max(clevs_LAI), transform=ccrs.PlateCarree(), cmap=cmap_LAI) # resample=False,
        plot1 = axs[1,col].imshow(LAI_regrid_month2[::-1,:], origin="lower", extent=extent, interpolation="none", vmin=np.min(clevs_LAI), vmax=np.max(clevs_LAI), transform=ccrs.PlateCarree(), cmap=cmap_LAI)
        plot1 = axs[2,col].imshow(LAI_regrid_month3[::-1,:], origin="lower", extent=extent, interpolation="none", vmin=np.min(clevs_LAI), vmax=np.max(clevs_LAI), transform=ccrs.PlateCarree(), cmap=cmap_LAI)

        # Albedo
        col = 1
        plot2 = axs[0,col].imshow(ALB_regrid_month1[::-1,:], origin="lower", extent=extent, interpolation="none", vmin=np.min(clevs_ALB), vmax=np.max(clevs_ALB), transform=ccrs.PlateCarree(), cmap=cmap_ALB)
        plot2 = axs[1,col].imshow(ALB_regrid_month2[::-1,:], origin="lower", extent=extent, interpolation="none", vmin=np.min(clevs_ALB), vmax=np.max(clevs_ALB), transform=ccrs.PlateCarree(), cmap=cmap_ALB)
        plot2 = axs[2,col].imshow(ALB_regrid_month3[::-1,:], origin="lower", extent=extent, interpolation="none", vmin=np.min(clevs_ALB), vmax=np.max(clevs_ALB), transform=ccrs.PlateCarree(), cmap=cmap_ALB)

        # Qle
        col = 2
        plot3 = axs[0,col].imshow(Qle_regrid_month1[::-1,:], origin="lower", extent=extent, interpolation="none", vmin=np.min(clevs_Qle), vmax=np.max(clevs_Qle), transform=ccrs.PlateCarree(), cmap=cmap_Qle)
        plot3 = axs[1,col].imshow(Qle_regrid_month2[::-1,:], origin="lower", extent=extent, interpolation="none", vmin=np.min(clevs_Qle), vmax=np.max(clevs_Qle), transform=ccrs.PlateCarree(), cmap=cmap_Qle)
        plot3 = axs[2,col].imshow(Qle_regrid_month3[::-1,:], origin="lower", extent=extent, interpolation="none", vmin=np.min(clevs_Qle), vmax=np.max(clevs_Qle), transform=ccrs.PlateCarree(), cmap=cmap_Qle)

        # Tmax
        col = 3
        plot4 = axs[0,col].imshow(Tmax_regrid_month1[::-1,:], origin="lower", extent=extent, interpolation="none", vmin=np.min(clevs_Tmax), vmax=np.max(clevs_Tmax), transform=ccrs.PlateCarree(), cmap=cmap_Tmax)
        plot4 = axs[1,col].imshow(Tmax_regrid_month2[::-1,:], origin="lower", extent=extent, interpolation="none", vmin=np.min(clevs_Tmax), vmax=np.max(clevs_Tmax), transform=ccrs.PlateCarree(), cmap=cmap_Tmax)
        plot4 = axs[2,col].imshow(Tmax_regrid_month3[::-1,:], origin="lower", extent=extent, interpolation="none", vmin=np.min(clevs_Tmax), vmax=np.max(clevs_Tmax), transform=ccrs.PlateCarree(), cmap=cmap_Tmax)

        # FMI
        col = 4
        plot5 = axs[0,col].imshow(FMI_regrid_month1[::-1,:], origin="lower", extent=extent, interpolation="none", vmin=np.min(clevs_FMI), vmax=np.max(clevs_FMI), transform=ccrs.PlateCarree(), cmap=cmap_FMI)
        plot5 = axs[1,col].imshow(FMI_regrid_month2[::-1,:], origin="lower", extent=extent, interpolation="none", vmin=np.min(clevs_FMI), vmax=np.max(clevs_FMI), transform=ccrs.PlateCarree(), cmap=cmap_FMI)
        plot5 = axs[2,col].imshow(FMI_regrid_month3[::-1,:], origin="lower", extent=extent, interpolation="none", vmin=np.min(clevs_FMI), vmax=np.max(clevs_FMI), transform=ccrs.PlateCarree(), cmap=cmap_FMI)
        # [::-1,:]
    # Add boxes, lines
    for i in np.arange(3):
        axs[0,0].add_patch(Polygon([[reg_lons[i][0], reg_lats[i][0]], [reg_lons[i][1], reg_lats[i][0]],
                                    [reg_lons[i][1], reg_lats[i][1]], [reg_lons[i][0], reg_lats[i][1]]],
                                    closed=True,color=almost_black, fill=False,linewidth=0.8))

    position_LAI  = fig.add_axes([0.13, 0.07, 0.13, 0.011] ) # [left, bottom, width, height]
    cb_lai        = fig.colorbar(plot1, ax=axs[:,0], pad=0.08, cax=position_LAI, orientation="horizontal", aspect=60, shrink=0.8)
    cb_lai.set_label('m$\mathregular{^{2}}$ m$\mathregular{^{-2}}$', loc='center',size=12)# rotation=270,
    cb_lai.ax.tick_params(labelsize=9,rotation=90)
    cb_lai.set_ticks(clevs_LAI)
    cb_lai.set_ticklabels(clevs_LAI_tick) # cax=cax,

    position_ALB  = fig.add_axes([0.28, 0.07, 0.13, 0.011]) # [left, bottom, width, height]
    cb_lai        = fig.colorbar(plot2, ax=axs[:,1], pad=0.08, cax=position_ALB, orientation="horizontal", aspect=60, shrink=0.8)
    cb_lai.set_label('', loc='center',size=12)# rotation=270,
    cb_lai.ax.tick_params(labelsize=9,rotation=90)
    cb_lai.set_ticks(clevs_ALB)
    cb_lai.set_ticklabels(clevs_ALB_tick) # cax=cax,

    position_Qle  = fig.add_axes([0.44, 0.07, 0.13, 0.011]) # [left, bottom, width, height]
    cb_lai        = fig.colorbar(plot3, ax=axs[:,2], pad=0.08, cax=position_Qle, orientation="horizontal", aspect=60, shrink=0.8)
    cb_lai.set_label('W m$\mathregular{^{-2}}$', loc='center',size=12)# rotation=270,
    cb_lai.ax.tick_params(labelsize=9,rotation=90)
    cb_lai.set_ticks(clevs_Qle)
    cb_lai.set_ticklabels(clevs_Qle_tick) # cax=cax,

    position_Tmax  = fig.add_axes([0.605, 0.07, 0.13, 0.011]) # [left, bottom, width, height]
    cb_lai        = fig.colorbar(plot4, ax=axs[:,3], pad=0.08, cax=position_Tmax, orientation="horizontal", aspect=60, shrink=0.8)
    cb_lai.set_label('$\mathregular{^{o}}$C', loc='center',size=12)# rotation=270,
    cb_lai.ax.tick_params(labelsize=9,rotation=90)
    cb_lai.set_ticks(clevs_Tmax)
    cb_lai.set_ticklabels(clevs_Tmax_tick) # cax=cax,

    position_FMI  = fig.add_axes([0.76, 0.07, 0.13, 0.011]) # [left, bottom, width, height]
    cb_lai        = fig.colorbar(plot5, ax=axs[:,4], pad=0.08, cax=position_FMI, orientation="horizontal", aspect=60, shrink=0.8)
    cb_lai.set_label('-', loc='center',size=12)# rotation=270,
    cb_lai.ax.tick_params(labelsize=9,rotation=90)
    cb_lai.set_ticks(clevs_FMI)
    cb_lai.set_ticklabels(clevs_FMI_tick) # cax=cax,

    if burn_shadow:
        print('Point2 burn_shadow')
        Burn_Date_index_1 = np.where( Burn_Date_index <= expected_index[0], 1, np.nan)
        Burn_Date_index_2 = np.where( Burn_Date_index <= expected_index[1], 1, np.nan)
        Burn_Date_index_3 = np.where( Burn_Date_index <= expected_index[2], 1, np.nan)
        # print('np.nansum(Burn_Date_index_1)',np.nansum(Burn_Date_index_1))

        extent   = (min(lon_fire), max(lon_fire), min(lat_fire), max(lat_fire))

        colors = ['darkgreen','black','darkgreen']#,'black'] # 'coral',
        custom_cmap = ListedColormap(colors)

        # LAI
        col = 0
        plot1 = axs[0,col].imshow(Burn_Date_index_1, origin="lower", extent=extent, interpolation="none", vmin=0.5, vmax=1.5, transform=ccrs.PlateCarree(), cmap=custom_cmap) # resample=False,
        plot1 = axs[1,col].imshow(Burn_Date_index_2, origin="lower", extent=extent, interpolation="none", vmin=0.5, vmax=1.5, transform=ccrs.PlateCarree(), cmap=custom_cmap) # resample=False,
        plot1 = axs[2,col].imshow(Burn_Date_index_3, origin="lower", extent=extent, interpolation="none", vmin=0.5, vmax=1.5, transform=ccrs.PlateCarree(), cmap=custom_cmap) # resample=False,

        # Albedo
        col = 1
        plot2 = axs[0,col].imshow(Burn_Date_index_1, origin="lower", extent=extent, interpolation="none", vmin=0.5, vmax=1.5, transform=ccrs.PlateCarree(), cmap=custom_cmap) # resample=False,
        plot2 = axs[1,col].imshow(Burn_Date_index_2, origin="lower", extent=extent, interpolation="none", vmin=0.5, vmax=1.5, transform=ccrs.PlateCarree(), cmap=custom_cmap) # resample=False,
        plot2 = axs[2,col].imshow(Burn_Date_index_3, origin="lower", extent=extent, interpolation="none", vmin=0.5, vmax=1.5, transform=ccrs.PlateCarree(), cmap=custom_cmap) # resample=False,

        # Qle
        col = 2
        plot3 = axs[0,col].imshow(Burn_Date_index_1, origin="lower", extent=extent, interpolation="none", vmin=0.5, vmax=1.5, transform=ccrs.PlateCarree(), cmap=custom_cmap) # resample=False,
        plot3 = axs[1,col].imshow(Burn_Date_index_2, origin="lower", extent=extent, interpolation="none", vmin=0.5, vmax=1.5, transform=ccrs.PlateCarree(), cmap=custom_cmap) # resample=False,
        plot3 = axs[2,col].imshow(Burn_Date_index_3, origin="lower", extent=extent, interpolation="none", vmin=0.5, vmax=1.5, transform=ccrs.PlateCarree(), cmap=custom_cmap) # resample=False,

        # Tmax
        col = 3
        plot4 = axs[0,col].imshow(Burn_Date_index_1, origin="lower", extent=extent, interpolation="none", vmin=0.5, vmax=1.5, transform=ccrs.PlateCarree(), cmap=custom_cmap) # resample=False,
        plot4 = axs[1,col].imshow(Burn_Date_index_2, origin="lower", extent=extent, interpolation="none", vmin=0.5, vmax=1.5, transform=ccrs.PlateCarree(), cmap=custom_cmap) # resample=False,
        plot4 = axs[2,col].imshow(Burn_Date_index_3, origin="lower", extent=extent, interpolation="none", vmin=0.5, vmax=1.5, transform=ccrs.PlateCarree(), cmap=custom_cmap) # resample=False,

        # FMI
        col = 4
        plot5 = axs[0,col].imshow(Burn_Date_index_1, origin="lower", extent=extent, interpolation="none", vmin=0.5, vmax=1.5, transform=ccrs.PlateCarree(), cmap=custom_cmap) # resample=False,
        plot5 = axs[1,col].imshow(Burn_Date_index_2, origin="lower", extent=extent, interpolation="none", vmin=0.5, vmax=1.5, transform=ccrs.PlateCarree(), cmap=custom_cmap) # resample=False,
        plot5 = axs[2,col].imshow(Burn_Date_index_3, origin="lower", extent=extent, interpolation="none", vmin=0.5, vmax=1.5, transform=ccrs.PlateCarree(), cmap=custom_cmap) # resample=False,


    # Add order number
    col=0
    axs[0,col].text(0.75, 0.12, "(a)", transform=axs[0,col].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    axs[1,col].text(0.75, 0.12, "(f)", transform=axs[1,col].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    axs[2,col].text(0.75, 0.12, "(j)", transform=axs[2,col].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    axs[0,col].add_feature(OCEAN,edgecolor='none', facecolor="white")
    axs[1,col].add_feature(OCEAN,edgecolor='none', facecolor="white")
    axs[2,col].add_feature(OCEAN,edgecolor='none', facecolor="white")

    col=1
    axs[0,col].text(0.75, 0.12, "(b)", transform=axs[0,col].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    axs[1,col].text(0.75, 0.12, "(g)", transform=axs[1,col].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    axs[2,col].text(0.75, 0.12, "(k)", transform=axs[2,col].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    axs[0,col].add_feature(OCEAN,edgecolor='none', facecolor="white")
    axs[1,col].add_feature(OCEAN,edgecolor='none', facecolor="white")
    axs[2,col].add_feature(OCEAN,edgecolor='none', facecolor="white")

    col=2
    axs[0,col].text(0.75, 0.12, "(c)", transform=axs[0,col].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    axs[1,col].text(0.75, 0.12, "(h)", transform=axs[1,col].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    axs[2,col].text(0.75, 0.12, "(l)", transform=axs[2,col].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    axs[0,col].add_feature(OCEAN,edgecolor='none', facecolor="white")
    axs[1,col].add_feature(OCEAN,edgecolor='none', facecolor="white")
    axs[2,col].add_feature(OCEAN,edgecolor='none', facecolor="white")

    col=3
    axs[0,col].text(0.75, 0.12, "(d)", transform=axs[0,col].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    axs[1,col].text(0.75, 0.12, "(i)", transform=axs[1,col].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    axs[2,col].text(0.75, 0.12, "(m)", transform=axs[2,col].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    axs[0,col].add_feature(OCEAN,edgecolor='none', facecolor="white")
    axs[1,col].add_feature(OCEAN,edgecolor='none', facecolor="white")
    axs[2,col].add_feature(OCEAN,edgecolor='none', facecolor="white")

    col=4
    axs[0,col].text(0.75, 0.12, "(e)", transform=axs[0,col].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    axs[1,col].text(0.75, 0.12, "(h)", transform=axs[1,col].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    axs[2,col].text(0.75, 0.12, "(n)", transform=axs[2,col].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    axs[0,col].add_feature(OCEAN,edgecolor='none', facecolor="white")
    axs[1,col].add_feature(OCEAN,edgecolor='none', facecolor="white")
    axs[2,col].add_feature(OCEAN,edgecolor='none', facecolor="white")


    # Adding titles
    axs[0,0].set_title("ΔLAI", fontsize=12)
    axs[0,1].set_title("Δ$α$ ", fontsize=12)
    axs[0,2].set_title("ΔLH", fontsize=12)
    # axs[0,2].set_title("ΔRnet", fontsize=12)
    axs[0,3].set_title("ΔT$\mathregular{_{max}}$", fontsize=12)
    axs[0,4].set_title("ΔFMI", fontsize=12)

    axs[0,0].text(-0.32, 0.53, 'Sep 2019', va='bottom', ha='center',
              rotation='vertical', rotation_mode='anchor',
              transform=axs[0,0].transAxes, fontsize=12)
    axs[1,0].text(-0.32, 0.50, "Oct 2019", va='bottom', ha='center',
              rotation='vertical', rotation_mode='anchor',
              transform=axs[1,0].transAxes, fontsize=12)
    axs[2,0].text(-0.32, 0.48, "Nov 2019", va='bottom', ha='center',
              rotation='vertical', rotation_mode='anchor',
              transform=axs[2,0].transAxes, fontsize=12)

    # Apply tight layout
    # plt.tight_layout()
    plt.savefig('./plots/spatial_map_' +message + '_Sep_Nov_2019_version3.png',dpi=300)

    #
    # axs[0,0].text(-0.32, 0.53, 'Dec 2019', va='bottom', ha='center',
    #           rotation='vertical', rotation_mode='anchor',
    #           transform=axs[0,0].transAxes, fontsize=12)
    # axs[1,0].text(-0.32, 0.50, "Jan 2020", va='bottom', ha='center',
    #           rotation='vertical', rotation_mode='anchor',
    #           transform=axs[1,0].transAxes, fontsize=12)
    # axs[2,0].text(-0.32, 0.48, "Feb 2020", va='bottom', ha='center',
    #           rotation='vertical', rotation_mode='anchor',
    #           transform=axs[2,0].transAxes, fontsize=12)
    #
    #
    # # Apply tight layout
    # # plt.tight_layout()
    # plt.savefig('./plots/spatial_map_' +message + '.png',dpi=300)

def plot_FMI_diff(fire_path, file_name, FMI_path, lat_names="lat", lon_names="lon", loc_lat=None, loc_lon=None, reg_lats=None,
                  reg_lons=None, wrf_path=None, message=None, burn_shadow=False, burn=0):

    '''
    plot LIS variables in burnt / unburnt / all regions
    '''

    # Read in WRF lat and lon
    wrf            = Dataset(wrf_path,  mode='r')
    lon_in         = wrf.variables['XLONG'][0,:,:]
    lat_in         = wrf.variables['XLAT'][0,:,:]

    # Read in FMI
    # plot 6-hourly data based daytime FMI value from Sep 2019 to Nov 2019
    time, FMI_ctl    = read_var_multi_file([FMI_path], "FMI_ctl", loc_lat, loc_lon, lat_names, lon_names)
    time, FMI_sen    = read_var_multi_file([FMI_path], "FMI_sen", loc_lat, loc_lon, lat_names, lon_names)
    FMI_diff         = FMI_sen-FMI_ctl
    FMI_diff_month1  = np.nanmean(FMI_diff[:29,:,:],axis=0) # the file start from 2nd Sept, these index match to the file
    FMI_diff_month2  = np.nanmean(FMI_diff[29:60,:,:],axis=0)
    FMI_diff_month3  = np.nanmean(FMI_diff[60:90,:,:],axis=0)
    FMI_diff_month4  = np.nanmean(FMI_diff[90:121,:,:],axis=0)

    BrBG_rgb_20colors = np.array([
                [0.338024, 0.193310, 0.020377],
                [0.441369, 0.254210, 0.029604],
                [0.544714, 0.315110, 0.038831],
                [0.631373, 0.395156, 0.095732],
                [0.733333, 0.491119, 0.165706],
                [0.793310, 0.595848, 0.287197],
                [0.857286, 0.725798, 0.447136],
                [0.904575, 0.810458, 0.581699],
                [0.947020, 0.880584, 0.710880],
                [0.955517, 0.959016, 0.9570165],
                [0.955517, 0.959016, 0.9570165],
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

    cmap20 = plt.cm.colors.ListedColormap(BrBG_rgb_20colors)

    clevs_FMI     = [-1.,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    clevs_FMI_tick= ['-1.0','','-0.8','','-0.6','','-0.4','','-0.2','','0.0','','0.2','','0.4','','0.6','','0.8','','1.0']
    cmap_FMI      = cmap20#plt.cm.BrBG #cmap_ALB # cmap20 #


    # regrid to burn map resolution ~ 400m
    FMI_regrid_month1,  lats, lons = regrid_to_fire_map_resolution(fire_path, FMI_diff_month1, lat_in, lon_in, burn=burn)
    FMI_regrid_month2,  lats, lons = regrid_to_fire_map_resolution(fire_path, FMI_diff_month2, lat_in, lon_in, burn=burn)
    FMI_regrid_month3,  lats, lons = regrid_to_fire_map_resolution(fire_path, FMI_diff_month3, lat_in, lon_in, burn=burn)
    FMI_regrid_month4,  lats, lons = regrid_to_fire_map_resolution(fire_path, FMI_diff_month4, lat_in, lon_in, burn=burn)

    if burn_shadow:
        # print('burn_shadow=',burn_shadow)
        # =========== Read in data ===========
        fire_path      = '/g/data/w97/mm3972/data/MODIS/MODIS_fire/MCD64A1.061_500m_aid0001.nc'
        fire_file      = Dataset(fire_path, mode='r')
        Burn_Date_tmp  = fire_file.variables['Burn_Date'][2:8,::-1,:]  # 2019-09 - 2020-02
        lat_fire       = fire_file.variables['lat'][:]
        lon_fire       = fire_file.variables['lon'][:]

        Burn_Date      = Burn_Date_tmp.astype(float)
        Burn_Date      = np.where(Burn_Date<=0, 99999, Burn_Date)

        Burn_Date[4:,:,:] = Burn_Date[4:,:,:]+365 # Add 365 to Jan-Feb 2020

        Burn_Date_min  = np.nanmin(Burn_Date, axis=0)
        unique_values  = np.unique(Burn_Date)
        # print("Unique values:", unique_values)

        Burn_Date_index = np.where(Burn_Date_min<=273,   1009, Burn_Date_min)
        Burn_Date_index = np.where(Burn_Date_index<=304, 1010, Burn_Date_index)
        Burn_Date_index = np.where(Burn_Date_index<=334, 1011, Burn_Date_index)
        Burn_Date_index = np.where(Burn_Date_index<=365, 1012, Burn_Date_index)
        Burn_Date_index = np.where(Burn_Date_index<=396, 1013, Burn_Date_index)
        Burn_Date_index = np.where(Burn_Date_index<=425, 1014, Burn_Date_index)
        Burn_Date_index = np.where(Burn_Date_index>=99999, np.nan, Burn_Date_index)

        # # Sep - Dec
        expected_index = [1009,1010,1011,1012]

    # ================== Start Plotting =================
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=[9,3],sharex=True,
                sharey=False, squeeze=True, subplot_kw={'projection': ccrs.PlateCarree()})
                # gridspec_kw={'width_ratios': [1, 1, 1, 1]})  # Adjust the width ratios as needed

    plt.subplots_adjust(wspace=-0.13, hspace=0)

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
    y_ticks = [-38, -36, -34, -32, -30, -28]  # Example y-axis tick positions

    for i in np.arange(4):
        axs[i].set_facecolor('lightgray')
        axs[i].coastlines(resolution="50m",linewidth=1)
        axs[i].set_extent([146,154,-39,-27])
        axs[i].add_feature(states, linewidth=.5, edgecolor="black")

        # Set the ticks on the x-axis and y-axis
        axs[i].tick_params(axis='x', direction='out')
        axs[i].tick_params(axis='y', direction='out')
        x_ticks = np.arange(148, 155, 4)
        y_ticks = np.arange(-40, -26, 4)
        axs[i].set_xticks(x_ticks)
        axs[i].set_yticks(y_ticks)

        axs[i].set_xticklabels(['148$\mathregular{^{o}}$E','152$\mathregular{^{o}}$E'],color=almost_black,alpha=1.)

        if i==0:
            axs[i].set_yticklabels(['40$\mathregular{^{o}}$S','36$\mathregular{^{o}}$S',
                                    '32$\mathregular{^{o}}$S','28$\mathregular{^{o}}$S'])
        else:
            axs[i].set_yticklabels([])

    extent = (min(lons), max(lons), min(lats), max(lats))
  
    # FMI
    plot1 = axs[0].imshow(FMI_regrid_month1[::-1,:], origin="lower", extent=extent, interpolation="none", vmin=np.min(clevs_FMI), vmax=np.max(clevs_FMI), transform=ccrs.PlateCarree(), cmap=cmap_FMI)
    plot2 = axs[1].imshow(FMI_regrid_month2[::-1,:], origin="lower", extent=extent, interpolation="none", vmin=np.min(clevs_FMI), vmax=np.max(clevs_FMI), transform=ccrs.PlateCarree(), cmap=cmap_FMI)
    plot3 = axs[2].imshow(FMI_regrid_month3[::-1,:], origin="lower", extent=extent, interpolation="none", vmin=np.min(clevs_FMI), vmax=np.max(clevs_FMI), transform=ccrs.PlateCarree(), cmap=cmap_FMI)
    plot4 = axs[3].imshow(FMI_regrid_month4[::-1,:], origin="lower", extent=extent, interpolation="none", vmin=np.min(clevs_FMI), vmax=np.max(clevs_FMI), transform=ccrs.PlateCarree(), cmap=cmap_FMI)

    # Add boxes, lines
    for i in np.arange(4):
        for j in np.arange(3):
            axs[i].add_patch(Polygon([[reg_lons[j][0], reg_lats[j][0]], [reg_lons[j][1], reg_lats[j][0]],
                                        [reg_lons[j][1], reg_lats[j][1]], [reg_lons[j][0], reg_lats[j][1]]],
                                        closed=True,color=almost_black, fill=False,linewidth=0.8))

    # position_FMI  = fig.add_axes([0.76, 0.07, 0.13, 0.011]) # [left, bottom, width, height]
    # cb_lai        = fig.colorbar(plot4, ax=axs[3], pad=0.08, cax=position_FMI, orientation="vertical", aspect=60, shrink=0.8)
    # cb_lai.set_label('-', loc='center',size=12)# rotation=270,
    # cb_lai.ax.tick_params(labelsize=9)#,rotation=90)
    # cb_lai.set_ticks(clevs_FMI)
    # cb_lai.set_ticklabels(clevs_FMI_tick) # cax=cax,

    cbar  = plt.colorbar(plot4, ax=axs[3], ticklocation="right", pad=0.04, orientation="vertical",
            aspect=30, shrink=1.0) # cax=cax,
    cbar.set_label('-', loc='center',size=12)# rotation=270,
    cbar.ax.tick_params(labelsize=9)#,rotation=90)
    cbar.set_ticks(clevs_FMI)
    cbar.set_ticklabels(clevs_FMI_tick) # cax=cax,


    if burn_shadow:
        print('Point2 burn_shadow')
        Burn_Date_index_1 = np.where( Burn_Date_index <= expected_index[0], 1, np.nan)
        Burn_Date_index_2 = np.where( Burn_Date_index <= expected_index[1], 1, np.nan)
        Burn_Date_index_3 = np.where( Burn_Date_index <= expected_index[2], 1, np.nan)
        Burn_Date_index_4 = np.where( Burn_Date_index <= expected_index[3], 1, np.nan)
        # print('np.nansum(Burn_Date_index_1)',np.nansum(Burn_Date_index_1))

        extent   = (min(lon_fire), max(lon_fire), min(lat_fire), max(lat_fire))

        colors = ['darkgreen','black','darkgreen']#,'black'] # 'coral',
        custom_cmap = ListedColormap(colors)

        # FMI
        plot1 = axs[0].imshow(Burn_Date_index_1, origin="lower", extent=extent, interpolation="none", vmin=0.5, vmax=1.5, transform=ccrs.PlateCarree(), cmap=custom_cmap) # resample=False,
        plot2 = axs[1].imshow(Burn_Date_index_2, origin="lower", extent=extent, interpolation="none", vmin=0.5, vmax=1.5, transform=ccrs.PlateCarree(), cmap=custom_cmap) # resample=False,
        plot3 = axs[2].imshow(Burn_Date_index_3, origin="lower", extent=extent, interpolation="none", vmin=0.5, vmax=1.5, transform=ccrs.PlateCarree(), cmap=custom_cmap) # resample=False,
        plot4 = axs[3].imshow(Burn_Date_index_4, origin="lower", extent=extent, interpolation="none", vmin=0.5, vmax=1.5, transform=ccrs.PlateCarree(), cmap=custom_cmap) # resample=False,


    # Add order number
    axs[0].text(0.75, 0.12, "(a)", transform=axs[0].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    axs[1].text(0.75, 0.12, "(b)", transform=axs[1].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    axs[2].text(0.75, 0.12, "(c)", transform=axs[2].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    axs[3].text(0.75, 0.12, "(d)", transform=axs[3].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    axs[0].add_feature(OCEAN,edgecolor='none', facecolor="white")
    axs[1].add_feature(OCEAN,edgecolor='none', facecolor="white")
    axs[2].add_feature(OCEAN,edgecolor='none', facecolor="white")
    axs[3].add_feature(OCEAN,edgecolor='none', facecolor="white")

    # Adding titles
    axs[0].set_title("Sep 2019", fontsize=12)
    axs[1].set_title("Oct 2019", fontsize=12)
    axs[2].set_title("Nov 2019", fontsize=12)
    axs[3].set_title("Dec 2019", fontsize=12)

    axs[0].text(-0.3, 0.53, 'ΔFMI', va='bottom', ha='center',
              rotation='vertical', rotation_mode='anchor',
              transform=axs[0].transAxes, fontsize=12)

    # Apply tight layout
    # plt.tight_layout()
    plt.savefig('./plots/Fig6_spatial_map_' +message + '_Sep_Dec_2019.png',dpi=300)

if __name__ == "__main__":

    if 1:
        '''
        3 X 3 plot
        '''

        loc_lat       = [-40,-25]
        loc_lon       = [135,155]

        case_ctl      = "drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2"
        case_sen      = "drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2_obs_LAI_ALB"

        wrf_path      = "/g/data/w97/mm3972/model/wrf/NUWRF/LISWRF_configs/Tinderbox_drght_LAI_ALB/drght_2017_2019_bl_pbl2_mp4_ra5_sf_sfclay2/WRF_output/wrfout_d01_2019-12-01_01:00:00"
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
        burn_shadow  = True
        message      = "HW_FMI_Tmax_LAI_Albedo_Qle_burnt"


        # # Sep to Nov
        file_name    = 'LIS.CABLE.201909-201911.nc'

        time_ss      = [datetime(2019,9,1,0,0,0,0),
                        datetime(2019,10,1,0,0,0,0),
                        datetime(2019,11,1,0,0,0,0)]

        time_es      = [datetime(2019,10,1,0,0,0,0),
                        datetime(2019,11,1,0,0,0,0),
                        datetime(2019,12,1,0,0,0,0)]
        FMI_path    = "/g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files/daytime_FMI_201909_202002.nc"

        # plot_Tmax_LAI_Albedo_diff(fire_path, file_name, FMI_path, land_ctl_path, land_sen_path, time_ss=time_ss, time_es=time_es, lat_names="lat",
        #               lon_names="lon", loc_lat=loc_lat, loc_lon=loc_lon,  reg_lats=reg_lats, reg_lons=reg_lons, wrf_path=wrf_path, message=message,
        #               burn_shadow=burn_shadow, burn=burn)

        plot_FMI_diff(fire_path, file_name, FMI_path, lat_names="lat", lon_names="lon", 
                      loc_lat=loc_lat, loc_lon=loc_lon, reg_lats=reg_lats,
                    reg_lons=reg_lons, wrf_path=wrf_path, message=message, burn_shadow=burn_shadow, burn=burn)