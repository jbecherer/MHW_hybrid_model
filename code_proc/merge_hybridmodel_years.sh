#!/bin/bash

module load cdo

cdo mergetime ../data/hybrid_model/histssp585/sst_hybrid_????.nc ../data/hybrid_model/sst_hybrid_histssp585_1850-2100.nc
cdo mergetime ../data/hybrid_model/ssp126/sst_hybrid_????.nc ../data/hybrid_model/sst_hybrid_ssp126_2015-2100.nc
cdo mergetime ../data/hybrid_model/ssp245/sst_hybrid_????.nc ../data/hybrid_model/sst_hybrid_ssp245_2015-2100.nc
cdo mergetime ../data/hybrid_model/ssp370/sst_hybrid_????.nc ../data/hybrid_model/sst_hybrid_ssp370_2015-2100.nc


