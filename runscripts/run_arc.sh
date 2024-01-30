#!/bin/bash
# ------------------------------------------------------------
# this script is for things running on a cluster (arc)
# update the two for-loops to decide which configs to run
# ------------------------------------------------------------
# ------------------------------------------------------------
# repopath
repopath='/home/hawan/deepskies/cmbcosmo/'
this_code_path=${repopath}'/runscripts/'
outdir=${repopath}'/outdir/'
# ------------------------------------------------------------
# create directories to hold the scripts and sbatch outputs
mkdir -p ${outdir}'/sbatch_out'
mkdir -p ${outdir}'/scripts'
cd ${outdir}'/scripts'

# ----------------------------------------
for tech in 0 1
do
    # ----------------------------------------
    if [ $tech == 0 ];
    then
        fname_techtag='mcmc'
        jobname_techtag='mc'
        option='mcmc'
        cpus=1
        hrs=6
    elif [ $tech == 1 ];
    then
        fname_techtag='sbi'
        jobname_techtag='s'
        option='sbi'
        cpus=1
        hrs=6
    else
        echo 'somethings wrong: tech = ' ${tech}
    fi
    # ----------------------------------------
    # now loop over the cases
    for case in 0 1 2
    do
        if [ $case == 0 ];
        then
            configtag='r-only'
        elif [ $case == 1 ];
        then
            configtag='Alens-only'
        elif [ $case == 2 ];
        then
            configtag='r+Alens'
        else
            echo 'somethings wrong: case = ' ${case}
        fi
        jobname=${jobname_techtag}${case}
        fname=${fname_techtag}'_c'${case}_${configtag}
        cat > ${fname}.sl << EOF
#!/bin/bash
#SBATCH --account=hawan0
#SBATCH --cpus-per-task=${cpus}
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH -t ${hrs}:00:00             # set time limit
#SBATCH --output=${outdir}/sbatch_out/${fname}_%j.out
#SBATCH --job-name=${jobname}
#SBATCH --mem=20000

source /home/hawan/.bashrc
conda activate cmbcosmo

export OMP_NUM_THREADS=1
export USE_SIMPLE_THREADED_LEVEL3=1

srun python ${repopath}/cmbcosmo/run_inference.py \
                --config-path=${this_code_path}/configs/config_arc_${configtag}.yml \
                --${option}

EOF
        sbatch ${fname}.sl
		echo job submitted for ${fname} with configtag ${configtag}
	done
    # ----------------------------------------
done
# ----------------------------------------
