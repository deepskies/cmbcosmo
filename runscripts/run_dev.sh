#!/bin/bash
# ------------------------------------------------------------
# this script should be run either in an interactive node
# or a local machine (after conda activate cmbcosmo)
# ------------------------------------------------------------
# the following should just exit without an analysis
# although data folder will be set up with cls-plots (no cov).
configpath='configs/config_dev.yml '
python ~/deepskies/cmbcosmo/cmbcosmo/run_inference.py \
          					--config-path=${configpath}

# the following should run mcmc and sbi
# can choose to run one or the other
python ~/deepskies/cmbcosmo/cmbcosmo/run_inference.py \
          					--config-path=${configpath} \
							--mcmc --sbi