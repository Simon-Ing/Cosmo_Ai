#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --constraint="A100"
#SBATCH --job-name="cosmo_anet"
#SBATCH --output=results.out
#SBATCH --mail-user=einarlau@stud.ntnu.no
#SBATCH --mail-type=ALL

 
WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

module purge
module load torchvision/0.8.2-fosscuda-2020b-PyTorch-1.7.1
module swap NCCL/2.8.3-CUDA-11.1.1 NCCL/2.8.3-GCCcore-10.2.0-CUDA-11.1.1
module swap PyTorch/1.7.1-fosscuda-2020b PyTorch/1.8.1-fosscuda-2020b
module list
python3 Cosmo_Ai/Python/CosmoCNN.py
