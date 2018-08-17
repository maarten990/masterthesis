#!/bin/bash

set -e
trap abort INT

function abort() {
    echo -e '\nAborting...'
    exit 0
}

# if [[ -n $LABELING_K ]]; then
#     echo "Setting k to $LABELING_K"
#     sed -i'' -E "s/(\s*)k:(\s*)[0-9]+/\1k:\2$LABELING_K/" params.yml
# fi

OUTFOLDER="../clustered_vgmm_pruned/"
XMLFOLDER="../training_data"
PDFFOLDER="../pdfs"
mkdir -p "$OUTFOLDER"
java -jar build/libs/clusterer-0.1-all.jar params.yml $XMLFOLDER $PDFFOLDER $OUTFOLDER
