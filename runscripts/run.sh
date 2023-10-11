#!/bin/bash

# the following should just exit without an analysis
configpath='configs/config_dev.yml '
python ~/deepskies/cmbcosmo/cmbcosmo/run_inference.py \
          					--config-path=${configpath}

# the following should run mcmc
configpath='configs/config_dev.yml '
python ~/deepskies/cmbcosmo/cmbcosmo/run_inference.py \
          					--config-path=${configpath} \
          					--mcmc
