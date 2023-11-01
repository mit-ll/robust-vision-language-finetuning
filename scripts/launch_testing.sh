#!/bin/bash

#SBATCH -p gaia
#SBATCH -C xeon-g6
#SBATCH --gres=gpu:volta:2
#SBATCH --exclusive

# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
#
# This material is based upon work supported by the Under Secretary of Defense for Research
# and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions
# or recommendations expressed in this material are those of the author(s) and do not necessarily
# reflect the views of the Under Secretary of Defense for Research and Engineering.
#
# © 2023 Massachusetts Institute of Technology.

# launch
echo "Other options: $@"
python -u test.py $@