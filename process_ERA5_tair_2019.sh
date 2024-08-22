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

cdo daymean /g/data/rt52/era5/single-levels/reanalysis/2t/2019/2t_era5_oper_sfc_20190101-20190131.nc 2t_era5_oper_sfc_201901_day_mean.nc
cdo daymean /g/data/rt52/era5/single-levels/reanalysis/2t/2019/2t_era5_oper_sfc_20190201-20190228.nc 2t_era5_oper_sfc_201902_day_mean.nc
cdo daymean /g/data/rt52/era5/single-levels/reanalysis/2t/2019/2t_era5_oper_sfc_20190301-20190331.nc 2t_era5_oper_sfc_201903_day_mean.nc
cdo daymean /g/data/rt52/era5/single-levels/reanalysis/2t/2019/2t_era5_oper_sfc_20190401-20190430.nc 2t_era5_oper_sfc_201904_day_mean.nc
cdo daymean /g/data/rt52/era5/single-levels/reanalysis/2t/2019/2t_era5_oper_sfc_20190501-20190531.nc 2t_era5_oper_sfc_201905_day_mean.nc
cdo daymean /g/data/rt52/era5/single-levels/reanalysis/2t/2019/2t_era5_oper_sfc_20190601-20190630.nc 2t_era5_oper_sfc_201906_day_mean.nc
cdo daymean /g/data/rt52/era5/single-levels/reanalysis/2t/2019/2t_era5_oper_sfc_20190701-20190731.nc 2t_era5_oper_sfc_201907_day_mean.nc
cdo daymean /g/data/rt52/era5/single-levels/reanalysis/2t/2019/2t_era5_oper_sfc_20190801-20190831.nc 2t_era5_oper_sfc_201908_day_mean.nc
cdo daymean /g/data/rt52/era5/single-levels/reanalysis/2t/2019/2t_era5_oper_sfc_20190901-20190930.nc 2t_era5_oper_sfc_201909_day_mean.nc
cdo daymean /g/data/rt52/era5/single-levels/reanalysis/2t/2019/2t_era5_oper_sfc_20191001-20191031.nc 2t_era5_oper_sfc_201910_day_mean.nc
cdo daymean /g/data/rt52/era5/single-levels/reanalysis/2t/2019/2t_era5_oper_sfc_20191101-20191130.nc 2t_era5_oper_sfc_201911_day_mean.nc
cdo daymean /g/data/rt52/era5/single-levels/reanalysis/2t/2019/2t_era5_oper_sfc_20191201-20191231.nc 2t_era5_oper_sfc_201912_day_mean.nc
#
# cdo daymax /g/data/rt52/era5/single-levels/reanalysis/2t/2019/2t_era5_oper_sfc_20190101-20190131.nc 2t_era5_oper_sfc_201901_day_max.nc
# cdo daymax /g/data/rt52/era5/single-levels/reanalysis/2t/2019/2t_era5_oper_sfc_20190201-20190228.nc 2t_era5_oper_sfc_201902_day_max.nc
# cdo daymax /g/data/rt52/era5/single-levels/reanalysis/2t/2019/2t_era5_oper_sfc_20190301-20190331.nc 2t_era5_oper_sfc_201903_day_max.nc
# cdo daymax /g/data/rt52/era5/single-levels/reanalysis/2t/2019/2t_era5_oper_sfc_20190401-20190430.nc 2t_era5_oper_sfc_201904_day_max.nc
# cdo daymax /g/data/rt52/era5/single-levels/reanalysis/2t/2019/2t_era5_oper_sfc_20190501-20190531.nc 2t_era5_oper_sfc_201905_day_max.nc
# cdo daymax /g/data/rt52/era5/single-levels/reanalysis/2t/2019/2t_era5_oper_sfc_20190601-20190630.nc 2t_era5_oper_sfc_201906_day_max.nc
# cdo daymax /g/data/rt52/era5/single-levels/reanalysis/2t/2019/2t_era5_oper_sfc_20190701-20190731.nc 2t_era5_oper_sfc_201907_day_max.nc
# cdo daymax /g/data/rt52/era5/single-levels/reanalysis/2t/2019/2t_era5_oper_sfc_20190801-20190831.nc 2t_era5_oper_sfc_201908_day_max.nc
# cdo daymax /g/data/rt52/era5/single-levels/reanalysis/2t/2019/2t_era5_oper_sfc_20190901-20190930.nc 2t_era5_oper_sfc_201909_day_max.nc
# cdo daymax /g/data/rt52/era5/single-levels/reanalysis/2t/2019/2t_era5_oper_sfc_20191001-20191031.nc 2t_era5_oper_sfc_201910_day_max.nc
# cdo daymax /g/data/rt52/era5/single-levels/reanalysis/2t/2019/2t_era5_oper_sfc_20191101-20191130.nc 2t_era5_oper_sfc_201911_day_max.nc
# cdo daymax /g/data/rt52/era5/single-levels/reanalysis/2t/2019/2t_era5_oper_sfc_20191201-20191231.nc 2t_era5_oper_sfc_201912_day_max.nc

# cdo mergetime 2t_era5_oper_sfc_2019??_day_max.nc 2t_era5_oper_sfc_2019_day_max.nc
