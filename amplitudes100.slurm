#!/bin/sh
#SBATCH --partition=CPUQ           # Use a CPU
#SBATCH --account=ie-idi
#SBATCH --time=8:00:00            # Max wall time
#SBATCH --mem=256000
#SBATCH --nodes=1
#SBATCH -c24
#SBATCH --job-name="AmmplitudesCosmoSim"    
#SBATCH --output=amplitudes.out
#SBATCH --mail-user=hasc@ntnu.no
#SBATCH --mail-type=ALL

 
WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Job Name:          $SLURM_JOB_NAME"
echo "Working directory: $SLURM_SUBMIT_DIR"
echo "Job ID:            $SLURM_JOB_ID"
echo "Nodes used:        $SLURM_JOB_NODELIST"
echo "Number of nodes:   $SLURM_JOB_NUM_NODES"
echo "Cores (per node):  $SLURM_CPUS_ON_NODE"
echo "Total cores:       $SLURM_NTASKS"
echo "Array Task ID:     $SLURM_ARRAY_TASK_ID"

module load SciPy-bundle/2021.05-foss-2021a
module list
source $HOME/torch-3.9.6/bin/activate


time python3 Python/amplitudes.py 100 24

