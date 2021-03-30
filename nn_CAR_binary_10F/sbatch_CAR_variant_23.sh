#!/bin/bash
#SBATCH --gres=gpu:1  			# request GPU "generic resource"
#SBATCH --cpus-per-task=4 		# maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham
#SBATCH --mem=32000M			# memory per node
#SBATCH --account=def-miranska		#
#SBATCH --time=03:00:00			# time 01:00:00, (DD-HH:MM) 0-03:00
#SBATCH --array=0-99

# INTERACTIVE JOB
# salloc --account=def-miranska --gres=gpu:1 --cpus-per-task=6 --mem=32000M --time=1:0:0
cp *.json $SLURM_TMPDIR
cp *.py $SLURM_TMPDIR
cp *.sh $SLURM_TMPDIR

###############
## Prepare Data
#cp * $SLURM_TMPDIR
#cd $SLURM_TMPDIR
####tar -xf ~/projects/def-miranska/iwonajs/msr/input.tar.gz
## DO I NEED TO COPY THE CODE?
#tar xf ~/projects/def-miranska/iwonajs/nn_CAR_binary_10F/input.tar.gz -C $SLURM_TMPDIR/input
tar xf input.tar.gz -C $SLURM_TMPDIR


######################
## Prepare VIRTUAL ENV
# https://docs.computecanada.ca/wiki/Python#Creating_and_using_a_virtual_environment
module load python/3.7.4
module load cuda cudnn
#module load python/3.8.2 cuda cudnn
module load scipy-stack
# python --version
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
# https://docs.computecanada.ca/wiki/TensorFlow
# Do not install the tensorflow package (without the _cpu or _gpu suffixes) as it has compatibility issues with other libraries.
pip install --no-index tensorflow_gpu
#pip install --no-index -r $SOURCEDIR/requierements.txt
#### ????????????? pip install --no-index -r $SLURM_TMPDIR/requiremetns.txt
nvidia-smi


###########
## Training
echo "*********************************************************************"
cd $SLURM_TMPDIR
ls -l
echo "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
# python ./run_terminal.py -n 0 -u 10 -m 1 --one 1 --flat 1
### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
### CHANGE x2
### -m X
### results_variant_X
### optional
### #SBATCH --time=03:00:00
### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
python ./run_terminal.py -n $SLURM_ARRAY_TASK_ID -u 10 -m 23 --chkpt True --flat 1
tar -cf ~/projects/def-miranska/iwonajs/nn_CAR_binary_10F/CAR_variant_23_$SLURM_ARRAY_TASK_ID.tar results
