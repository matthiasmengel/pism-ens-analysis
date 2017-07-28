
import numpy as np
import netCDF4 as nc

def get_spatial_variable(fname,varname):

    try:
        ncf = nc.Dataset(fname,"r")
    except IOError as error:
        print fname, "not found."
        raise error
    return np.squeeze(ncf.variables[varname])


def get_spatial_rms_error(variable,refvariable,mask,
                          restrict_to="both"):

    """ get the root mean square error between variable and
    refvariable. Optional: do rms calculation only on floating
    or grounded area, as determined from mask.
    mask=2 is grounded,
    mask=3 is floating,
    mask=4 is ocean.
    TODO: weight the RMS:
          the closer to the grounding line, the more important.
    """

    rms = np.sqrt((variable-refvariable)**2.)

    ## do not count where now is ocean (could have been icy in ref)
    rms[mask==4] = 0.

    if restrict_to == "floating":
        rms[mask==2] = 0.

    elif restrict_to == "grounded":
        rms[mask==3] = 0.

    elif restrict_to == "none":
        pass

    else:
        raise NotImplemented

    return rms


def get_summed_rms_error(fname, varname, reference_field,
                         restrict_to="none"):

    """ use the fields from fname to calculate rms against
        reference field. sum up the rms and return.
        restrict_to can be passed to the get_spatial_rms_error
        function.
    """

    variable = get_spatial_variable(fname, varname)
    mask = get_spatial_variable(fname, "mask")

    rms_of_var = get_spatial_rms_error(variable,reference_field, mask,
                 restrict_to=restrict_to)

    return rms_of_var.sum()