#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=80:00:00
#SBATCH --mem=20000
set -euxo pipefail

my_command="/cm/shared/mit/matlab/matlab-2015a/bin/matlab -nosplash -nodesktop -r 'IT_multi("
my_command+="'\''" 
my_command+=$1
my_command+="'\'', "
my_command+=$2
my_command+=")'"
echo $my_command
eval $my_command
