#!/bin/sh
#SBATCH --partition=CPUQ           # Use a CPU
#SBATCH --account=share-ie-idi
#SBATCH --time=48:00:00            # Max wall time
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH -c7
#SBATCH --array=0-2
#SBATCH --job-name="short-time"
#SBATCH --output=short-time.out
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

case "$SLURM_ARRAY_TASK_ID" in
   0)
      time python3 ../../CosmoSimPy/amplitudes.py --lens SIE --output short-factor.txt 7 
      ;;
   1)
      time python3 ../../CosmoSimPy/amplitudes.py --lens SIE --output short-id.txt --simplify id 7 
      ;;
   2)
      time python3 ../../CosmoSimPy/amplitudes.py --lens SIE --output short-simp.txt --simplify simplify 7 
      ;;
esac
