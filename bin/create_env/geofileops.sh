#!/bin/bash

# If no parameters passed, show help...
if [ -z "$var" ]
then
  echo
  echo Hello! If you want to override some default options this is possible as such:
  echo 'install_geofileops.sh --envname geofileopsdev --envname_backup geofileopsdev_bck_2020-01-01 --condadir "/ProgramData/Miniconda3" --condaenvsdir "/home/.conda/envs" --fordev Y'
  echo 
  echo The parameters can be used as such:
  echo     - envname: the name the new environment will be given 
  echo     - envname_backup: if the environment already exist, it will be 
  echo       backupped to this environment
  echo     - condadir: the directory where conda is installed
  echo     - condaenvsdir: the directory where conda environments are created
  echo     - fordev: for development: if Y is passed, only the dependencies 
  echo       for geofileops will be installed, not the geofileops package itself
fi

# Extract the parameters passed
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -e|--envname) envname="$2"; shift ;;
        -cd|--condadir) condadir="$2"; shift ;;
        -ced|--condaenvsdir) condaenvsdir="$2"; shift ;;
        -eb|--envname_backup) envname_backup="$2"; shift ;;
        -od|--fordev) fordev="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Format current date
today=$(date +%F)

# If not provided, init parameters with default values
if [ -z "$envname" ]; then envname="geofileops" ; fi
if [ -z "$envname_backup" ]; then envname_backup="${envname}_bck_${today}" ; fi
if [ -z "$condadir" ]; then condadir="/c/ProgramData/Miniconda3" ; fi
if [ -z "$condaenvsdir" ]; then condaenvsdir="$HOME/.conda/envs" ; fi
if [ -z "$fordev" ]; then fordev="N" ; fi

# If no parameters are given, ask if it is ok to use defaults
echo
echo "The script will be ran with the following parameters:"
echo "   - envname=$envname"
echo "   - envname_backup=$envname_backup"
echo "   - condadir=$condadir"
echo "   - condaenvsdir=$condaenvsdir"
echo "   - fordev=$fordev"
echo

read -p "Do you want to move on with these choices? (y/n)" -n 1 -r
echo    
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    [[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1 # handle exits from shell or function but don't exit interactive shell
fi

# Init conda
. "$condadir/etc/profile.d/conda.sh"

#-------------------------------------
# RUN!
#-------------------------------------
echo
echo Backup existing environment
echo -------------------------------------

if [[ ! -z "$envname_backup" ]]
then
  if [[ -d "$condaenvsdir/$envname/" ]]
  then
    echo "Do you want to take a backup from $envname?"
    if [[ -d "$condaenvsdir/$envname_backup/" ]]
    then
      echo "REMARK: $envname_backup exists already, so will be overwritten by a new backup!"
    fi
    
    read -p "y=take backup, n=don't take backup but proceed, c=stop script (y/n/c)" -n 1 -r
    echo    
    if [[ $REPLY =~ ^[Cc]$ ]]
    then
        [[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1 # handle exits from shell or function but don't exit interactive shell
    elif [[ $REPLY =~ ^[Yy]$ ]]
    then  
      conda create --name "$envname_backup" --clone "$envname"
    fi
  else 
    echo "No existing environment $envname found to backup"
  fi
fi

echo
echo Create/overwrite environment
echo -------------------------------------
if [[ -d "$condaenvsdir/$envname/" ]]
then
  echo "First remove conda environment $envname"
  conda env remove -y --name $envname

  echo "Also really delete the env directory, to evade locked file errors"
  rm -rf $condaenvsdir/$envname
fi

echo "Create and install conda environment $envname"
conda create -y --name $envname
conda activate $envname
conda config --env --add channels conda-forge
conda config --env --set channel_priority strict

# Now install what needs to be installed
if [[ ! $fordev =~ ^[Yy]$ ]]
then
  # Conda install
  conda install geofileops
  # For the following packages, no conda package is available.
  pip install simplification
else
  # Dev install...
  echo
  echo Development install: conda install dependencies
  echo
  # List of dependencies + reasons for specific versions.
  # python: 3.8, possibly 3.8 features are used, not sure
  # geopandas: > 0.10 because in sjoin a parameter renamed
  conda install -y python=3.8 cloudpickle "geopandas>=0.10,<0.11" "libspatialite>=5.0" psutil pygeos pyproj topojson

  echo
  echo Prepare for development: conda install dev tools
  echo
  conda install -y pylint pytest pytest-cov rope pydata-sphinx-theme sphinx-automodapi

  echo
  echo Prepare for development: pip install needed dependencies
  echo
  pip install simplification
fi

# Deactivate new env
conda deactivate

# Ask to clean the cache dir + deactivate base env
#conda clean --all
conda deactivate

# Pause
read -s -n 1 -p "Press any key to continue . . ."
