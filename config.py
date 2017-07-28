"""
This file contains user defined paths and settings.
Normally, this should not be committed unless your changes are major.
"""
import os
import pwd

username = pwd.getpwuid(os.getuid()).pw_name
author= username+"@pik-potsdam.de"

working_dir = os.path.join("/p/tmp/",username,"pism_out")
ensemble_name = "pismpik_036_initmip08km"

## Input file, as the basis for to calculate thickness, bed topography differences
## TODO: write test that ensure that PISM output and the here referred
## input file are on same grid
grid_id = "initmip8km"
## for creation of input data, see github.com/pism/pism-ais project.
input_file = "bedmap2_albmap_racmo_hadcm3_I2S_schmidtko_tillphi_pism_"+grid_id+"_bhflxcorr.nc"

## To update: velocity file on same grid as output data.
# velobsfile    = workpath+"Velocity/rignot_mouginot11/regrid_cdo/antarctica_ice_velocity_"+str(resolution)+"km.nc"


#### No edits needed below that line. ####
project_root = os.path.dirname(os.path.abspath(__file__))

