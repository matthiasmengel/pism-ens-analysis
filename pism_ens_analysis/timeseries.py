import os
import collections
import numpy as np
# import pandas as pd
import netCDF4 as nc
import dimarray as da


def get_timeseries_data(datapath, ts_variables = ["slvol","area_glacierized_shelf"],
                        ts_file_name="timeseries.nc"):

    """ get time series variables ts_variables from PISM timeseries
    file in datapath.
    TODO: merge with get_timeseries_data2
    """

    name = datapath.split("/")[-1]

    ts_data = {}

    try:
        ncf = nc.Dataset(os.path.join(datapath,ts_file_name),"r")
    except IOError as error:
        print name, "has no timeseries file, skip"
        # raise error
    try:
        nct = ncf.variables["time"]
    except KeyError as error:
        print name, "contains no data, skip."
        # raise error

    for var in ts_variables:
        ts_data[var] = ncf.variables[var][:]
    ts_data = da.Dataset(ts_data)

    datetm = nc.num2date(nct[:],units = nct.units,calendar = nct.calendar)
    # takes long for long timeseries
    years = [d.year for d in datetm]
    ts_data.set_axis(years)
    ts_data.rename_axes({"x0":"time"})

    return ts_data


def get_timeseries_data2(ensemble_members, ts_file_name="timeseries.nc"):

    """ loop over all ensemble members, check if they have a ts_file_name,
    read its data and save it to an ordered dictionary.
    """

    ## assume that first entry has all variables
    ncf = nc.Dataset(os.path.join(ensemble_members[0],ts_file_name),"r")
    ts_variables = ncf.variables.keys()
    ## ignore some variables
    ts_variables = [var for var in ts_variables if var not in
                        ["pism_config","run_stats","time_bounds","time"]]
    ncf.close()

    ensemble_data = {}
    for em in ensemble_members:
        name = em.split("/")[-1]
        try:
            ncf = nc.Dataset(os.path.join(em,ts_file_name),"r")
        except IOError:
            print name, "has no", ts_file_name, "file, skip"
            continue
        try:
            nct = ncf.variables["time"]
        except KeyError:
            print name, "contains no data, skip."
            continue

        ensemble_data[name] = {}

        for var in ts_variables:
            ensemble_data[name][var] = ncf.variables[var][:]
        ensemble_data[name] = da.Dataset(ensemble_data[name])

        datetm = nc.num2date(nct[:],units = nct.units,calendar = nct.calendar)
        # takes long for long timeseries
        years = [d.year for d in datetm]
        ensemble_data[name].set_axis(years)
        ensemble_data[name].rename_axes({"x0":"time"})

    # self.data = ensemble_data
    # keep it ordered
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