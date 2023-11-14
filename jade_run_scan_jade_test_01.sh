#! /bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=24:00:00

# set name of job
#SBATCH --job-name=job1

# set number of GPUs
#SBATCH --gres=gpu:1

# set the partition to use
#SBATCH --partition=small

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=ts468@sussex.ac.uk

# run the application
# cd ~/PhD/Intel-Neuromorphic-Research-Project/
# python HD_eventprop_real_time.py jade_test_params_02/params_${SLURM_ARRAY_TASK_ID}.json

PARAM_DIR='jade_test_params_03/'
LINES=`ls -1 | wc -l < $PARAM_DIR`


for i in $(seq ${SLURM_ARRAY_TASK_ID} 30 $LINES); 
do
    ARGS=`head -$i $PARAM_DIR | tail -1`
    python HD_eventprop_real_time.py $PARAM_DIR/$ARGS 

done
