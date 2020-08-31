"""
This file contains user defined paths and settings.
Normally, this should not be committed unless your changes are major.
"""
import os
import pwd

username = pwd.getpwuid(os.getuid()).pw_name

is_pikcluster = False
if username=="reese":
    is_pikcluster = True
    from pikcluster_settings import *
else:
    from supermuc_settings import *

#ensemble_name = "dev_058_initmip4km_resoensemble4"
#grid_id = "initmip8km"


#### No edits needed below that line. ####
project_root = os.path.dirname(os.path.abspath(__file__))

