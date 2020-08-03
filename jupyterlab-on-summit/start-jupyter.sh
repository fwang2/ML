#!/bin/bash

echo "Activating virtual environment..." 
module load ibm-wml-ce
conda activate wm1

unset XDG_RUNTIME_DIR
export HOME=$(pwd)

echo "The compute node home is $HOME" 

jupyter notebook --ip=0.0.0.0 --no-browser --NotebookApp.token='' --port $1 
