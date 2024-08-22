#!/bin/bash

#PBS -m ae
#PBS -P w97
#PBS -q normalsr
#PBS -l walltime=2:00:00
#PBS -l mem=500GB
#PBS -l ncpus=1
#PBS -j oe
#PBS -l wd
#PBS -l storage=gdata/rt52+gdata/zz93+gdata/hh5+gdata/w97+scratch/w97+gdata/w97

module use /g/data/hh5/public/modules
module load conda/analysis3-22.04

cd /g/data/w97/mm3972/scripts/Drought/drght_2017-2019/nc_files

cdo daymean /g/data/rt52/era5/single-levels/reanalysis/2t/2018/2t_era5_oper_sfc_20180101-20180131.nc 2t_era5_oper_sfc_201801_day_mean.nc
cdo daymean /g/data/rt52/era5/single-levels/reanalysis/2t/2018/2t_era5_oper_sfc_20180201-20180228.nc 2t_era5_oper_sfc_201802_day_mean.nc
cdo daymean /g/data/rt52/era5/single-levels/reanalysis/2t/2018/2t_era5_oper_sfc_20180301-20180331.nc 2t_era5_oper_sfc_201803_day_mean.nc
cdo daymean /g/data/rt52/era5/single-levels/reanalysis/2t/2018/2t_era5_oper_sfc_20180401-20180430.nc 2t_era5_oper_sfc_201804_day_mean.nc
cdo daymean /g/data/rt52/era5/single-levels/reanalysis/2t/2018/2t_era5_oper_sfc_20180501-20180531.nc 2t_era5_oper_sfc_201805_day_mean.nc
cdo daymean /g/data/rt52/era5/single-levels/reanalysis/2t/2018/2t_era5_oper_sfc_20180601-20180630.nc 2t_era5_oper_sfc_201806_day_mean.nc
cdo daymean /g/data/rt52/era5/single-levels/reanalysis/2t/2018/2t_era5_oper_sfc_20180701-20180731.nc 2t_era5_oper_sfc_201807_day_mean.nc
cdo daymean /g/data/rt52/era5/single-levels/reanalysis/2t/2018/2t_era5_oper_sfc_20180801-20180831.nc 2t_era5_oper_sfc_201808_day_mean.nc
cdo daymean /g/data/rt52/era5/single-levels/reanalysis/2t/2018/2t_era5_oper_sfc_20180901-20180930.nc 2t_era5_oper_sfc_201809_day_mean.nc
cdo daymean /g/data/rt52/era5/single-levels/reanalysis/2t/2018/2t_era5_oper_sfc_20181001-20181031.nc 2t_era5_oper_sfc_201810_day_mean.nc
cdo daymean /g/data/rt52/era5/single-levels/reanalysis/2t/2018/2t_era5_oper_sfc_20181101-20181130.nc 2t_era5_oper_sfc_201811_day_mean.nc
cdo daymean /g/data/rt52/era5/single-levels/reanalysis/2t/2018/2t_era5_oper_sfc_20181201-20181231.nc 2t_era5_oper_sfc_201812_day_mean.nc
#
# cdo daymax /g/data/rt52/era5/single-levels/reanalysis/2t/2018/2t_era5_oper_sfc_20180101-20180131.nc 2t_era5_oper_sfc_201801_day_max.nc
# cdo daymax /g/data/rt52/era5/single-levels/reanalysis/2t/2018/2t_era5_oper_sfc_20180201-20180228.nc 2t_era5_oper_sfc_201802_day_max.nc
# cdo daymax /g/data/rt52/era5/single-levels/reanalysis/2t/2018/2t_era5_oper_sfc_20180301-20180331.nc 2t_era5_oper_sfc_201803_day_max.nc
# cdo daymax /g/data/rt52/era5/single-levels/reanalysis/2t/2018/2t_era5_oper_sfc_20180401-20180430.nc 2t_era5_oper_sfc_201804_day_max.nc
# cdo daymax /g/data/rt52/era5/single-levels/reanalysis/2t/2018/2t_era5_oper_sfc_20180501-20180531.nc 2t_era5_oper_sfc_201805_day_max.nc
# cdo daymax /g/data/rt52/era5/single-levels/reanalysis/2t/2018/2t_era5_oper_sfc_20180601-20180630.nc 2t_era5_oper_sfc_201806_day_max.nc
# cdo daymax /g/data/rt52/era5/single-levels/reanalysis/2t/2018/2t_era5_oper_sfc_20180701-20180731.nc 2t_era5_oper_sfc_201807_day_max.nc
# cdo daymax /g/data/rt52/era5/single-levels/reanalysis/2t/2018/2t_era5_oper_sfc_20180801-20180831.nc 2t_era5_oper_sfc_201808_day_max.nc
# cdo daymax /g/data/rt52/era5/single-levels/reanalysis/2t/2018/2t_era5_oper_sfc_20180901-20180930.nc 2t_era5_oper_sfc_201809_day_max.nc
# cdo daymax /g/data/rt52/era5/single-levels/reanalysis/2t/2018/2t_era5_oper_sfc_20181001-20181031.nc 2t_era5_oper_sfc_201810_day_max.nc
# cdo daymax /g/data/rt52/era5/single-levels/reanalysis/2t/2018/2t_era5_oper_sfc_20181101-20181130.nc 2t_era5_oper_sfc_201811_day_max.nc
# cdo daymax /g/data/rt52/era5/single-levels/reanalysis/2t/2018/2t_era5_oper_sfc_20181201-20181231.nc 2t_era5_oper_sfc_201812_day_max.nc

# cdo mergetime 2t_era5_oper_sfc_2018??_day_max.nc 2t_era5_oper_sfc_2018_day_max.nc
