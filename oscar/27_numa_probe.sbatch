#!/bin/bash
# NUMA layout of an L40S node as a 4-GPU/32-CPU job sees it: SkyRL's Worker._set_numa_affinity
# binds every rank (CPU + memory) to NUMA node (local_rank // (8 // numa_nodes)) regardless of
# the job's cpuset or the GPU's own affinity. What does that resolve to here?
#SBATCH --job-name=numa-probe
#SBATCH --partition=gpu-he
#SBATCH --qos=gpu-he+
#SBATCH --constraint=l40s
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64g
#SBATCH --time=00:10:00
#SBATCH --exclude=gpu2709,gpu3006,gpu3005
#SBATCH --output=/oscar/scratch/mborjigi/skyrl/logs/%x-%j.out
echo "node=$(hostname)"
echo "--- numactl -H"; numactl -H 2>&1 | head -30
echo "--- job cpuset: $(cat /proc/self/status | grep Cpus_allowed_list)"
echo "--- CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "--- nvidia-smi topo -m"; nvidia-smi topo -m 2>&1 | head -20
echo "--- per-NUMA-node cpus inside the job cpuset"
for n in /sys/devices/system/node/node*; do echo "$(basename $n): $(cat $n/cpulist)"; done
