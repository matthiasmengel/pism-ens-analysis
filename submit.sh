#!/bin/bash

#@ job_name = smuc_$(cluster)_$(stepid)
#@ class = micro
#@ group = pn69ru
#@ notify_user = mengel@pik-potsdam.de
#@ job_type = MPICH
#@ output = ./loadl.out
#@ error  = ./loadl.err
#@ wall_clock_limit = 1:59:00
#@ notification=always
#@ network.MPI = sn_all,not_shared,us
#@ node = 1
#@ tasks_per_node = 28
#@ island_count = 1
#@ energy_policy_tag = albrecht_pism_2015
#@ minimize_time_to_solution = yes
#@ queue


number_of_cores=`echo $LOADL_PROCESSOR_LIST | wc -w`
echo $number_of_cores

echo `which python`

# python test_submission.py
python get_reso_compare_score.py
