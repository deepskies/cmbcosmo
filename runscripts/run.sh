#!/bin/bash

# the following should just exit without an analysis
configpath='configs/config_dev.yml '
python ~/deepskies/cmbcosmo/cmbcosmo/run_inference.py \
          					--config-path=${configpath}

# the following should run mcmc and sbi
# can choose to run one of the other
python ~/deepskies/cmbcosmo/cmbcosmo/run_inference.py \
          					--config-path=${configpath} \
							--mcmc --sbi