import os
import pandas as pd
import numpy as np

""" Save data preprocessed by Ronja Reese for Antarctic ice shelves into csv.
    Variable: basal melt rate.
"""

reese_scorefolder = "/home/reese/projects/pico07/Plots/prepare_RignotData2012/"

basin_names =[
'Filchner-Ronne', 'Riiser-Larsen, Brunt','Fimbul','Baudouin, Lazarev','Prince Harald',
 'Amery','West, Shackleton','Totten, Moscow University',
'Cook, Ninnis, Mertz','Rennick','Drygalski, Nansen','Ross','Getz',
'Pine-Island, Thwaites, Dotson, Crosson','Abbot, Cosgrove','Wilkins Stange',
'Wordie','Larsen B, C','Larsen D, E, F']

def get_basal_melt_score(folder, npyfile):

    data = np.load(os.path.join(folder, npyfile))
    data = pd.DataFrame(data,index=np.arange(len(data)))

    # delete zero entry, which says nothing
    data = data.drop(0)
    return data

mean_bmr = get_basal_melt_score(reese_scorefolder, "Rignot_mean_bmr_per_basin.npy")
bmr_range = get_basal_melt_score(reese_scorefolder, "Rignot_mean_bmr_range_per_basin.npy")

bmr_data = pd.concat([mean_bmr,bmr_range],axis=1)
bmr_data.columns = ["mean basal melt rate per basin",
                      "range of basal melt rates per basin"]
bmr_data.index = basin_names
bmr_data.to_csv("../data/rignot_basal_melt.csv")