#!/bin/sh
#SBATCH --partition=CPUQ           # Use a CPU
#SBATCH --account=share-ie-idi
#SBATCH --time=240:00:00            # Max wall time
#SBATCH --mem=196G
#SBATCH --nodes=1
#SBATCH -c7
#SBATCH --job-name="sim5"
#SBATCH --output=sim5.out
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

cat /proc/cpuinfo

source ./pythonenv.config

time python3 ../../CosmoSimPy/amplitudessim.py --lens SIE --output sim.txt 5
