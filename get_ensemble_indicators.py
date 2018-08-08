import joblib
import numpy as np
# from matplotlib import colors
# import netCDF4 as nc
# import dimarray as da
import sys, os
# import glob
import pandas as pd
# import itertools
import glob
import collections
# import skfmm
# import scipy.ndimage
# import seaborn.apionly as sns
import importlib
import time

# our custom imports
import settings as s; importlib.reload(s)
# import pism_ens_analysis.timeseries as ts; importlib.reload(ts)
import pism_ens_analysis.pism_ens_analysis as ea; importlib.reload(ea)

ensemble_id = "dev_058_initmip4km_resoensemble5"

# move to settings.py later.
grid_id = "initmip4km"
input_root_dir = "/gpfs/work/pr94ga/di36lav/pism_input"
resolution = 4 # in km

data_path = "/gss/scratch/pr94ga/di36lav/pism_out"
experiments = glob.glob(os.path.join(data_path, ensemble_id+"_*"))

## get the reference data for a specific grid.

bedm_mask, bedm_thk, bedm_thk_grounded, basins, rignot_velsurf_mag = \
    ea.get_reference_data(input_root_dir, grid_id)

rigmag_grounded_above100 = ea.get_data_on_maskval_above_threshold(rignot_velsurf_mag, bedm_mask, 0, 100)

distance_to_observed_gl = ea.get_distance_to_observed_gl(bedm_mask, resolution)

# main functionality: get the quality indicators for each run.

def get_ens_indicator(experiment, year):

    ensemble_indicator = pd.Series(index = [
                                   'Amundsen Stream Velocity', 'Ross Stream Velocity',
       'Amundsen Thk Anomaly', 'Ross Thk Anomaly', 'Total Grounded Area',
       'Total Floating Area', 'Amundsen Grounded Area',
       'Amundsen Floating Area', 'Ross Grounded Area', 'Total Grounding Line',
       'Amundsen Grounding Line', 'Ross Grounding Line'])

    ehash = experiment.split("_")[-1]
    print (ehash, year)

    try:
        pism_data = ea.get_spatial_variables(
                os.path.join(experiment,"extra_"+str(year)+".000.nc"),["velsurf_mag","thk","mask"])
    except IOError as error:
        print (error)
        return ehash, year, ensemble_indicator

    pismvel_above100 = ea.get_data_on_maskval_above_threshold(
        pism_data["velsurf_mag"], bedm_mask, 0, 100)

    rms_vel = np.sqrt((rigmag_grounded_above100-pismvel_above100)**2.)
    velrms_per_basin = ea.get_sum_per_basin(rms_vel, basins, basin_range=[12,14])

    rms_thk_gr = np.sqrt((pism_data["thk"] - bedm_thk_grounded)**2)
    thkrms_per_basin = ea.get_sum_per_basin(rms_thk_gr, basins, basin_range=[12,14],
                                            weigh_by_size=True)

    area_errors = ea.get_area_errors(pism_data["mask"], bedm_mask)

    garea_err_per_basin = ea.get_sum_per_basin(
        area_errors["grounded_area_error"], basins, basin_range=[12,14])
    farea_err_per_basin = ea.get_sum_per_basin(
        area_errors["floating_area_error"], basins, basin_range=[12,14])

    ensemble_indicator.loc["Amundsen Stream Velocity"] = \
        velrms_per_basin.loc[14]

    ensemble_indicator.loc["Ross Stream Velocity"] = \
        velrms_per_basin.loc[12]

    ensemble_indicator.loc["Amundsen Thk Anomaly"] = \
        thkrms_per_basin.loc[14]

    ensemble_indicator.loc["Ross Thk Anomaly"] = \
        thkrms_per_basin.loc[12]

    ensemble_indicator.loc["Total Grounded Area"] = \
        area_errors["grounded_area_error"].sum()

    ensemble_indicator.loc["Total Floating Area"] = \
        area_errors["floating_area_error"].sum()

    ensemble_indicator.loc["Amundsen Grounded Area"] = \
        garea_err_per_basin.loc[14]

    ensemble_indicator.loc["Amundsen Floating Area"] = \
        farea_err_per_basin.loc[14]

    ensemble_indicator.loc["Ross Grounded Area"] = \
        garea_err_per_basin.loc[12]

    gl_deviation = ea.get_grounding_line_deviaton(
        pism_data["mask"], distance_to_observed_gl, basins, basin_range=[12,14])

    ensemble_indicator.loc["Total Grounding Line"] = \
        gl_deviation.loc["total"]
    ensemble_indicator.loc["Amundsen Grounding Line"] = \
        gl_deviation.loc[14]
    ensemble_indicator.loc["Ross Grounding Line"] = \
        gl_deviation.loc[12]

    return ehash, year, ensemble_indicator


if __name__ == "__main__":

    years = np.arange(2100,2900,50)

    start = time.perf_counter()
    ensemble_indicators_par = joblib.Parallel(n_jobs=28)(
        joblib.delayed(get_ens_indicator)(e, y) for e in experiments[0:] for y in years)
    print("elapsed time",time.perf_counter()-start)

    ensemble_indicators = pd.DataFrame(
        {(el[0], el[1]): el[2] for el in ensemble_indicators_par}).T

    ensemble_indicators.index.names = ["ehash", "year"]
    ensemble_indicators.columns.names = ["indicator"]
    ensemble_indicators.to_csv(os.path.join("data/",ensemble_id+"_allyears.csv"))



