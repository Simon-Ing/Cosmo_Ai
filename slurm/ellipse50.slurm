#!/bin/sh
#SBATCH --partition=CPUQ           # Use a CPU
#SBATCH --account=ie-idi
#SBATCH --time=96:00:00            # Max wall time
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH -c24
#SBATCH --job-name="ellips50"    
#SBATCH --output=ellips50.out
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


source ./pythonenv.config

amplitudes.py --lens SIE --output sie50.txt 50 24

