
import os
import numpy as np
import netCDF4 as nc
import collections
import pandas as pd
import glob


def get_spatial_variable(fname,varname):

    try:
        ncf = nc.Dataset(fname,"r")
    except IOError as error:
        print fname, "not found."
        raise error
    return np.squeeze(ncf.variables[varname])


def get_rms_error(score, varname, ncr, refncr, spatial=True):

    """ get the root mean square error between variable and
    refvariable. Also get the sum of the rms for
    grounded+floating, and grounded and floating alone.
    mask=2 is grounded,
    mask=3 is floating,
    mask=4 is ocean.
    TODO: weight the RMS:
          the closer to the grounding line, the more important.
    """

    variable = np.squeeze(ncr.variables[varname][:])
    mask = np.squeeze(ncr.variables["mask"][:])
    refvariable = np.squeeze(refncr.variables[varname][:])

    rms = np.sqrt((variable-refvariable)**2.)

    ## do not count where now is ocean (could have been icy in ref)
    ## FIXME: is this a reasonable decision?
    rms[mask==4] = 0.

    if spatial:
        return rms

    rms_floating = np.array(rms, copy=True)
    rms_floating[mask==2] = 0.

    rms_grounded = np.array(rms, copy=True)
    rms_grounded[mask==3] = 0.

    rms_error_sums = {"rms_"+varname:rms.sum(),
                        "rms_"+varname+"_grounded":rms_grounded.sum(),
                        "rms_"+varname+"_floating":rms_floating.sum()}

    # ensure same order for all dicts
    score.update((sorted(rms_error_sums.items())))
    return score


def get_area_errors(score, ncr, refncr, spatial=True):

    """ find difference in areas of floating and observed,
        between mask and refmask. sum the erros if spatial=False.
        mask=2 is grounded,
        mask=3 is floating,
        mask=4 is ocean. """

    ad = collections.OrderedDict()

    mask = np.squeeze(ncr.variables["mask"][:])
    refmask = np.squeeze(refncr.variables["mask"][:])

    ad["floating_in_obs_now_not"] = np.array((refmask == 3) &
        (mask !=3),dtype=np.float)
    ad["floating_now_not_in_obs"] = np.array((refmask != 3) &
        (mask ==3),dtype=np.float)

    ad["grounded_in_obs_now_not"] = np.array((refmask == 2) &
         (mask !=2),dtype=np.float)
    ad["grounded_now_not_in_obs"] = np.array((refmask != 2) &
         (mask ==2),dtype=np.float)

    if spatial:
        return ad
    else:
        for name, measure in ad.iteritems():
            ad[name] = measure.sum()
        score.update(ad)
        return score


def get_wais_ungrounded_area(score, ncr, refncr,
                             wais_latbounds = [-180,-30]):

    """ This is a good measure for detecting collapsed WAIS states,
        but it is not necessarily a good measure to define a good
        (stable) WAIS. This is because it does not take into
        account when new areas ground that should not.
    """

    lon = np.squeeze(refncr.variables["lon"][:])
    wais_msk = (lon < wais_latbounds[0]) | (lon > wais_latbounds[1])

    mask = np.squeeze(ncr.variables["mask"][:])
    refmask = np.squeeze(refncr.variables["mask"][:])

    floating_or_ocean_now_grounded_in_obs = np.ma.masked_array((refmask == 2) &
            ((mask==3)|(mask==4)),dtype=np.float,mask=wais_msk)

    score.update({"wais_ungrounded":
        floating_or_ocean_now_grounded_in_obs.sum()})

    return score


def mean_melt_rate_deviation(score, ncr, basins, rignot_bmr_data, basins_for_score,
                            spatial=False, absolute_values=False):

    rho_ice = 910. # in kg/m^3

    pism_melt_rates = pd.DataFrame(index=rignot_bmr_data.index,
                                   columns=["mean basal melt rate per basin"])

    effshelfbmassflux = np.squeeze(
        ncr.variables['effective_shelf_base_mass_flux'][:])

    mask = np.squeeze(ncr.variables['mask'][:])

    # all basins, hardcoded for now
    for basin_id in np.arange(1,20,1):
        # select only floating ice in basin
        data_basin = np.ma.masked_array(effshelfbmassflux,
            mask = np.logical_or(basins!=basin_id, mask!=3) )
        pism_melt_rates.iloc[basin_id-1] = data_basin.mean()/rho_ice

    if absolute_values:
        return pism_melt_rates

    scorem = (pism_melt_rates - rignot_bmr_data)["mean basal melt rate per basin"]
    # root mean square
    scorem = np.power(scorem.loc[basins_for_score]**2.,0.5)

    if spatial:
        return scorem

    else:
        score.update({"basal_melt_per_basin":scorem.sum()})
        return score

def collect_scores_to_arrays(measures):

    """ this is a kind of resorting: use a dictionary of
        the measures at top level and sort all the single
        runs into a numpy array. We thus can easily use
        numpy methods. """

    run_names = measures.keys()
    measure_names = measures[measures.keys()[0]].keys()

    measure_arrays = collections.OrderedDict()
    for mn in measure_names:
        measure_arrays[mn] = np.zeros(len(run_names))
        for i,run in enumerate(run_names):
            measure_arrays[mn][i] = measures[run][mn]

    return measure_arrays


def normalize_scores(measure_arrays):

    """ calculate the ensemble mean per measure, and
        normalize all runs with that ensemble mean, i.e. divide it. """

    measure_arrays_mean = [arr.mean() for m,arr in measure_arrays.iteritems()]

    measure_arrays_normal = collections.OrderedDict()

    for i,nm in enumerate(measure_arrays.keys()):
        measure_arrays_normal[nm] = np.zeros(len(measure_arrays[nm]))
        if measure_arrays_mean[i] != 0.0:
            measure_arrays_normal[nm] = measure_arrays[nm]/measure_arrays_mean[i]

    return measure_arrays_normal



def collect_scores(ensemble_members, varnames_for_rms,
                   refncr, basins, rignot_bmr_data, basins_for_score,
                   fixed_analysis_year=None):

    """ run all score measures and collect them in the scores ordered
    dictionary.
    """

    scores = collections.OrderedDict()

    for em in ensemble_members:
        run = em.split("/")[-1]
        print run,

        if fixed_analysis_year != None:
            ncr = nc.Dataset(os.path.join(em,"snapshots_"+str(fixed_analysis_year)+".000.nc"),"r")
        else:
            analysis_year = get_last_snap_year(em, pattern="snapshots_")
            ncr = nc.Dataset(os.path.join(em,"snapshots_"+str(analysis_year)+".000.nc"),"r")

        scores[run] = collections.OrderedDict()
        for varname in varnames_for_rms:
            scores[run] = get_rms_error(scores[run], varname, ncr, refncr, spatial=False)
        scores[run] = get_area_errors(scores[run], ncr, refncr, spatial=False)
        scores[run] = get_wais_ungrounded_area(scores[run], ncr, refncr)
        scores[run] = mean_melt_rate_deviation(scores[run], ncr, basins, rignot_bmr_data,
                        basins_for_score)
        ncr.close()

    return scores


def get_last_snap_year(ensemble_member, pattern="snapshots_"):

    all_files = sorted(glob.glob(os.path.join(
        ensemble_member,pattern+"[0-9]*")))

    last_avail_year = all_files[-1].split(pattern)[-1].split(".000.nc")[0]
    return int(last_avail_year)