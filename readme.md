The goal of this repository is to host the code for CMB cosmology inference - in particular, comparing SBI to other methods. More details to follow.

---
Details re setting up the conda environment for the code:
- Run `conda env create -f conda-env.yml`. This will install the various packages needed (those for the code here as well as for for `DeepCMBSim` and `sbi`).
- Install `DeepCMBSim` via:
    - `git clone git@github.com:deepskies/DeepCMBsim.git`
    - `cd DeepCMBsim`
    - `pip install --user -e .` (or `pip install .` if no user development is expected)
- Set up the code here via `pip install -e .` (or `pip install .` if no user development is expected)