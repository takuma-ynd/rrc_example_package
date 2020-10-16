#! /bin/bash
export PYTHONNOUSERSITE=True
source activate rrc_simulation
if [ -z "$RRC_SIM_ROOT" ]
then
      echo "Please set \$RRC_SIM_ROOT (e.g., export RRC_SIM_ROOT=/path/to/your/rrc_simulation)"
      exit 1
fi
cd $RRC_SIM_ROOT
source init.sh
cd $RRC_SIM_ROOT/code
python train_ppo.py $@
