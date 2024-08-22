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

cdo daymean /g/data/rt52/era5/single-levels/reanalysis/2t/2017/2t_era5_oper_sfc_20170101-20170131.nc 2t_era5_oper_sfc_201701_day_mean.nc
cdo daymean /g/data/rt52/era5/single-levels/reanalysis/2t/2017/2t_era5_oper_sfc_20170201-20170228.nc 2t_era5_oper_sfc_201702_day_mean.nc
cdo daymean /g/data/rt52/era5/single-levels/reanalysis/2t/2017/2t_era5_oper_sfc_20170301-20170331.nc 2t_era5_oper_sfc_201703_day_mean.nc
cdo daymean /g/data/rt52/era5/single-levels/reanalysis/2t/2017/2t_era5_oper_sfc_20170401-20170430.nc 2t_era5_oper_sfc_201704_day_mean.nc
cdo daymean /g/data/rt52/era5/single-levels/reanalysis/2t/2017/2t_era5_oper_sfc_20170501-20170531.nc 2t_era5_oper_sfc_201705_day_mean.nc
cdo daymean /g/data/rt52/era5/single-levels/reanalysis/2t/2017/2t_era5_oper_sfc_20170601-20170630.nc 2t_era5_oper_sfc_201706_day_mean.nc
cdo daymean /g/data/rt52/era5/single-levels/reanalysis/2t/2017/2t_era5_oper_sfc_20170701-20170731.nc 2t_era5_oper_sfc_201707_day_mean.nc
cdo daymean /g/data/rt52/era5/single-levels/reanalysis/2t/2017/2t_era5_oper_sfc_20170801-20170831.nc 2t_era5_oper_sfc_201708_day_mean.nc
cdo daymean /g/data/rt52/era5/single-levels/reanalysis/2t/2017/2t_era5_oper_sfc_20170901-20170930.nc 2t_era5_oper_sfc_201709_day_mean.nc
cdo daymean /g/data/rt52/era5/single-levels/reanalysis/2t/2017/2t_era5_oper_sfc_20171001-20171031.nc 2t_era5_oper_sfc_201710_day_mean.nc
cdo daymean /g/data/rt52/era5/single-levels/reanalysis/2t/2017/2t_era5_oper_sfc_20171101-20171130.nc 2t_era5_oper_sfc_201711_day_mean.nc
cdo daymean /g/data/rt52/era5/single-levels/reanalysis/2t/2017/2t_era5_oper_sfc_20171201-20171231.nc 2t_era5_oper_sfc_201712_day_mean.nc

# cdo daymax /g/data/rt52/era5/single-levels/reanalysis/2t/2017/2t_era5_oper_sfc_20170101-20170131.nc 2t_era5_oper_sfc_201701_day_max.nc
# cdo daymax /g/data/rt52/era5/single-levels/reanalysis/2t/2017/2t_era5_oper_sfc_20170201-20170228.nc 2t_era5_oper_sfc_201702_day_max.nc
# cdo daymax /g/data/rt52/era5/single-levels/reanalysis/2t/2017/2t_era5_oper_sfc_20170301-20170331.nc 2t_era5_oper_sfc_201703_day_max.nc
# cdo daymax /g/data/rt52/era5/single-levels/reanalysis/2t/2017/2t_era5_oper_sfc_20170401-20170430.nc 2t_era5_oper_sfc_201704_day_max.nc
# cdo daymax /g/data/rt52/era5/single-levels/reanalysis/2t/2017/2t_era5_oper_sfc_20170501-20170531.nc 2t_era5_oper_sfc_201705_day_max.nc
# cdo daymax /g/data/rt52/era5/single-levels/reanalysis/2t/2017/2t_era5_oper_sfc_20170601-20170630.nc 2t_era5_oper_sfc_201706_day_max.nc
# cdo daymax /g/data/rt52/era5/single-levels/reanalysis/2t/2017/2t_era5_oper_sfc_20170701-20170731.nc 2t_era5_oper_sfc_201707_day_max.nc
# cdo daymax /g/data/rt52/era5/single-levels/reanalysis/2t/2017/2t_era5_oper_sfc_20170801-20170831.nc 2t_era5_oper_sfc_201708_day_max.nc
# cdo daymax /g/data/rt52/era5/single-levels/reanalysis/2t/2017/2t_era5_oper_sfc_20170901-20170930.nc 2t_era5_oper_sfc_201709_day_max.nc
# cdo daymax /g/data/rt52/era5/single-levels/reanalysis/2t/2017/2t_era5_oper_sfc_20171001-20171031.nc 2t_era5_oper_sfc_201710_day_max.nc
# cdo daymax /g/data/rt52/era5/single-levels/reanalysis/2t/2017/2t_era5_oper_sfc_20171101-20171130.nc 2t_era5_oper_sfc_201711_day_max.nc
# cdo daymax /g/data/rt52/era5/single-levels/reanalysis/2t/2017/2t_era5_oper_sfc_20171201-20171231.nc 2t_era5_oper_sfc_201712_day_max.nc

# cdo mergetime 2t_era5_oper_sfc_2017??_day_max.nc 2t_era5_oper_sfc_2017_day_max.nc
