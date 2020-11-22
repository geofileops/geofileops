#!/bin/bash

# Init some variables
condadir="$HOME/Miniconda3"
envname="geofileopsdev"

# Init conda and activate environment
. "$condadir/etc/profile.d/conda.sh"
conda activate $envname

# Now run tests
scriptdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
python -m pytest "$scriptdir/tests/"

conda deactivate
conda deactivate

# Pause
read -s -n 1 -p "Press any key to continue . . ."
