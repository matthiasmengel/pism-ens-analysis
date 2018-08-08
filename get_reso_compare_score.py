import joblib
import numpy as np
import sys, os
import pandas as pd
import glob
import collections
import importlib
import time
import itertools
import logging

# our custom imports
import settings as s; importlib.reload(s)
import pism_ens_analysis.pism_ens_analysis as ea; importlib.reload(ea)

ensemble_id_a = "dev_058_initmip4km_resoensemble5"
ensemble_id_b = "dev_058_initmip8km_resoensemble4"
year = 2300

input_root_dir = "/gpfs/work/pr94ga/di36lav/pism_input"
data_path = "/gss/scratch/pr94ga/di36lav/pism_out"

experiments = glob.glob(os.path.join(data_path, ensemble_id_a+"_*"))


def get_comparison(ehash_a, ehash_b, year):

    logger = logging.getLogger('get_reso_compare')
    logger.info("do_comparison")
    logger.warning("compare %s to %s",ehash_a,ehash_b)

    file_a = os.path.join(data_path,
        ensemble_id_a+"_"+ehash_a,"extra_"+str(year)+".000.nc")
    file_b = os.path.join(data_path,
        ensemble_id_b+"_"+ehash_b,"extra_"+str(year)+".000.nc")

    data_a = ea.get_spatial_variables(file_a, ["thk"])
    data_b = ea.get_spatial_variables(file_b, ["thk"])

    rms_thk = np.sqrt((data_a["thk"][::2,::2] - data_b["thk"])**2).sum()

    return ehash_a, ehash_b, rms_thk


if __name__ == "__main__":

    print("start here")

    available_in_a = glob.glob(os.path.join(data_path,
                           ensemble_id_a+"_*","extra_"+str(year)+".000.nc"))

    # assume that b has more than a
    available_hashes = [f.split("/")[-2].split("_")[-1] for f in available_in_a]

    # this ensures to not double calculate
    combinations = list(itertools.combinations(available_hashes,2))

    start = time.perf_counter()

    logger = logging.getLogger('get_reso_compare')
    fh = logging.FileHandler('test.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # logger.basicConfig(filename='get_reso_compare_score.log',level=logging.DEBUG)
    logger.info("start here")
    output = joblib.Parallel(n_jobs=28)(
        joblib.delayed(get_comparison)(c[0], c[1], year) for c in combinations[0:])

    # output = [get_comparison(c[0], c[1], year) for c in combinations[0:3]]
    logger.info("end here")


    print("elapsed time",time.perf_counter()-start)

    compare_score = pd.DataFrame(index=available_hashes,columns=available_hashes)

    for c in output:
        compare_score.loc[c[0],c[1]] = c[2]
        # the reverse is also true
        compare_score.loc[c[1],c[0]] = c[2]

    compare_score.to_csv(os.path.join("data",
                                     ensemble_id_a+"_"+ensemble_id_b+".csv"))


