#!/bin/bash

scriptdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
"$scriptdir/geofileops.sh" --envname geofileopsdev --fordev Y

# Pause
#read -s -n 1 -p "Press any key to continue . . ."
