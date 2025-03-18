#!/bin/bash
#SBATCH --time=00:02:00
#SBATCH --nodes=2
#SBATCH --ntasks=64
#SBATCH -o slurmjob-%j.out-%N 
#SBATCH -e slurmjob-%j.err-%N 
#SBATCH --account=usucs5890
#SBATCH --partition=notchpeak-freecycle

echo $SLURM_TASKS_PER_NODE
echo $SLURM_JOB_CPUS_PER_NODE

cd /scratch/general/lustre/usucs5030/mpi

mpirun -n 64 ./hello