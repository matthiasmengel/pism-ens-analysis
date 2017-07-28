
import numpy as np
import netCDF4 as nc

def get_spatial_variable(fname,varname):

    try:
        ncf = nc.Dataset(fname,"r")
    except IOError as error:
        print fname, "not found."
        raise error
    return np.squeeze(ncf.variables[varname])


def get_rms_error(variable,refvariable,mask, spatial=True):

    """ get the root mean square error between variable and
    refvariable. Also get the sum of the rms for
    grounded+floating, and grounded and floating alone.
    mask=2 is grounded,
    mask=3 is floating,
    mask=4 is ocean.
    TODO: weight the RMS:
          the closer to the grounding line, the more important.
    """

    rms = np.sqrt((variable-refvariable)**2.)

    ## do not count where now is ocean (could have been icy in ref)
    ## FIXME: is this a reasonable decision?
    rms[mask==4] = 0.

    if spatial:
        return rms

    rms_floating = rms.copy()
    rms_floating[mask==2] = 0.

    rms_grounded = rms.copy()
    rms_grounded[mask==3] = 0.

    return {"rms_sum":rms.sum(),
            "rms_sum_grounded":rms_grounded.sum(),
            "rms_sum_floating":rms_floating.sum()}


def get_summed_rms_error(fname, varname, reference_field,
                         restrict_to="none"):

    """ use the fields from fname to calculate rms against
        reference field.
    """

    variable = get_spatial_variable(fname, varname)
    mask = get_spatial_variable(fname, "mask")

    rms_of_var = get_rms_error(variable,reference_field, mask,
                 spatial = False)

    return rms_of_var