import os
import collections
import numpy as np
# import pandas as pd
import netCDF4 as nc
import dimarray as da
import glob


def get_netcdf_as_dataset(ts_file_name):

    """ Read all timeseries variable from ts_file_name
    into one dimarray Dataset. Handle some errors if it
    not existent or empty. """

    try:
        ncf = nc.Dataset(ts_file_name,"r")
    except IOError as e:
        print spath, "has no", ts_file_name, "file, skip"
        raise IOError(ts_file_name + "does not exist.")

    try:
        nct = ncf.variables["time"]
    except KeyError:
        print name, "contains no data, skip."
        raise KeyError(ts_file_name+" contains no data.")

    ts_variables = ncf.variables.keys()
    ## ignore some variables
    ts_variables = [var for var in ts_variables if var not in
                        ["pism_config","run_stats","time_bounds","time"]]

    dataset = {}

    for var in ts_variables:
        dataset[var] = ncf.variables[var][:]

    dataset = da.Dataset(dataset)
    ## the following for a nice time axis.
    datetm = nc.num2date(nct[:],units = nct.units,calendar = nct.calendar)
    # takes long for long timeseries
    years = [d.year for d in datetm]
    dataset.set_axis(years)
    dataset.rename_axes({"x0":"time"})

    return dataset


def get_timeseries_data2(ensemble_members, ts_file_pattern="timeseries"):

    """ loop over all ensemble members, get their ts file data,
        merge them and collect them into an ordered dict.
    """

    ensemble_data = {}

    for em in ensemble_members:

        name = em.split("/")[-1]
        # these are the files like timeseries_308000.nc from restarts
        available_ts_restart_files = sorted(glob.glob(
            os.path.join(em,ts_file_pattern+"_*nc")))

        # append the timeseries.nc files
        available_ts_files = available_ts_restart_files + [
            os.path.join(em,ts_file_pattern+".nc")]

        for ts_file in available_ts_files:

            if ts_file == available_ts_files[0]:
                dataset = get_netcdf_as_dataset(ts_file)
            else:
                ds = get_netcdf_as_dataset(ts_file)
                dataset = da.concatenate_ds((dataset,ds),axis="time")

        ensemble_data[name] = dataset

    ensemble_data = collections.OrderedDict(sorted(ensemble_data.items()))

    return ensemble_data


def get_last_common_time(ts_data):

    """ find the last timestep that is present in all runs,
    also find the longest run and the respective timestep """

    lasttm = []
    for tsd in ts_data:
        lasttm.append(ts_data[tsd].time[-1])
    lasttm = np.array(lasttm)

    last_common_time = lasttm.min()
    longest_run_time = lasttm.max()

    return last_common_time, longest_run_time



def imshow_variable(fname,varname,**kwargs):

    ncfname = os.path.join(ensemble_base_path,fname)
    try:
        ncf = nc.Dataset(ncfname,"r")
    except IOError as error:
        print ncfname, "not found."
        raise error
    plt.imshow(np.squeeze(ncf.variables[varname][0:150,0:150]),origin="lower",
               interpolation="nearest",**kwargs)


def contour_variable(fname,varname,**kwargs):

    ncfname = os.path.join(ensemble_base_path,fname)
    try:
        ncf = nc.Dataset(ncfname,"r")
    except IOError as error:
        print ncfname, "not found."
        raise error
    plt.contour(np.squeeze(ncf.variables[varname][0:150,0:150]),origin="lower",
               interpolation="nearest",**kwargs)