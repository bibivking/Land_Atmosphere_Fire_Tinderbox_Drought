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


cdo daymean /g/data/rt52/era5/single-levels/reanalysis/2t/2020/2t_era5_oper_sfc_20200101-20200131.nc 2t_era5_oper_sfc_202001_day_mean.nc
cdo daymean /g/data/rt52/era5/single-levels/reanalysis/2t/2020/2t_era5_oper_sfc_20200201-20200229.nc 2t_era5_oper_sfc_202002_day_mean.nc
cdo daymean /g/data/rt52/era5/single-levels/reanalysis/2t/2020/2t_era5_oper_sfc_20200301-20200331.nc 2t_era5_oper_sfc_202003_day_mean.nc

# cdo daymax /g/data/rt52/era5/single-levels/reanalysis/2t/2020/2t_era5_oper_sfc_20200101-20200131.nc 2t_era5_oper_sfc_202001_day_max.nc
# cdo daymax /g/data/rt52/era5/single-levels/reanalysis/2t/2020/2t_era5_oper_sfc_20200201-20200229.nc 2t_era5_oper_sfc_202002_day_max.nc
# cdo daymax /g/data/rt52/era5/single-levels/reanalysis/2t/2020/2t_era5_oper_sfc_20200301-20200331.nc 2t_era5_oper_sfc_202003_day_max.nc

# cdo mergetime 2t_era5_oper_sfc_2020??_day_max.nc 2t_era5_oper_sfc_202001-03_day_max.nc
