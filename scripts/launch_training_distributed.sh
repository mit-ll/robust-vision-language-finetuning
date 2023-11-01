#!/bin/bash

#SBATCH -p gaia
#SBATCH -C xeon-g6
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:volta:2
#SBATCH --distribution=nopack
#SBATCH --exclusive

# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
#
# This material is based upon work supported by the Under Secretary of Defense for Research
# and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions
# or recommendations expressed in this material are those of the author(s) and do not necessarily
# reflect the views of the Under Secretary of Defense for Research and Engineering.
#
# © 2023 Massachusetts Institute of Technology.

set -o nounset

# Debug flags - disable to improve inference speed
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1

# NCCL network interface
export NCCL_SOCKET_IFNAME=^docker0,lo

function set_master_addr_and_port () {
    JOB_NODELIST=SLURM_JOB_NODELIST
    HOSTNAMES=$(scontrol show hostnames ${!JOB_NODELIST})
    MASTER_ADDR=$(echo ${HOSTNAMES} | tr " " "\n" | head -n 1)
    REST_ADDRS=$(echo ${HOSTNAMES} | tr " " "\n" | tail -n +2)
    REST_ADDRS=$(echo ${REST_ADDRS} | tr " " ",")
    MASTER_PORT=$(srun --nodes=1 --ntasks=1 --exclude=${REST_ADDRS} python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

    export MASTER_ADDR=${MASTER_ADDR}
    export MASTER_PORT=${MASTER_PORT}
    echo MASTER_ADDR=$MASTER_ADDR
    echo MASTER_PORT=$MASTER_PORT
}

set_master_addr_and_port

# launch
echo "Other options: $@"

srun python -u train.py $@

echo "Job finished $(date)"
exit 0