#!/bin/sh
#SBATCH --partition=CPUQ           # Use a CPU
#SBATCH --account=share-ie-idi
#SBATCH --time=48:00:00            # Max wall time
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH -c16
#SBATCH --array=0-2
#SBATCH --job-name="sis-time"
#SBATCH --output=sis-time.out
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
      time python3 ../../CosmoSimPy/amplitudes.py --lens SIS --output factor.txt 50 16
      ;;
   1)
      time python3 ../../CosmoSimPy/amplitudes.py --lens SIS --output id.txt --simplify id 50 16
      ;;
   2)
      time python3 ../../CosmoSimPy/amplitudes.py --lens SIS --output simp.txt --simplify simplify 50 16
      ;;
esac
