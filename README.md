## PISM-ENS-ANALYSIS

This collection of functions and jupyter notebooks aims at making
the analysis of PISM ensembles more convenient. Main functions are
in `pism_ens_analysis/pism_ens_analysis.py`. 

### Usage

Have a look to the example folder, try out the jupyter notebooks.

Start with [get_ensemble_indicators.ipynb](examples/get_ensemble_indicators.ipynb).

For more analyses, look at [indicator_dependencies.ipynb](indicator_dependencies.ipynb).

### Starting jupyter notebooks from cluster

You would like to fire up a notebook on the machine that hosts the PISM files.
On your local machine, this would just be:

```jupyter notebook```

For running jupyter on the cluster, you need an ssh tunnel
to connect your browser with jupyter running on the cluster:

from your local shell:

```ssh -L localhost:choose_a_port1:localhost:choose_a_port2 username@cluster2015.pik-potsdam.de```

And then, on the cluster:

```jupyter notebook --no-browser --port=choose_a_port2```


Open your browser and go to

```localhost:choose_a_port1```

Choose_a_port2 should be chosen randomly for different users,
so that they do not interfere.
There are probably restrictions to the ports that can be
used, staying between 7000 and 9000 is probably safe. For a long
list, see https://en.wikipedia.org/wiki/List_of_TCP_and_UDP_port_numbers

### License

This code is licensed under GPLv3, see the LICENSE.txt. See the commit history for authors.
