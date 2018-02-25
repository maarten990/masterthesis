#!/bin/bash

set -e
trap abort INT

function abort() {
    echo -e '\nAborting...'
    exit 0
}

if [[ -n $LABELING_K ]]; then
    echo "Setting k to $LABELING_K"
    sed -i'' -E "s/(\s*)k:(\s*)[0-9]+/\1k:\2$LABELING_K/" params.yml
fi

mkdir -p ../clusterlabeled
filenames=`ls ../training_data`
for file in $filenames; do
    number=`basename $file .xml`
    xml="../training_data/$number.xml"
    pdf="../pdfs/$number.pdf"
    out="../clusterlabeled/$number.xml"

    echo $number
    if [[ ! -f $out ]]; then
        java -jar build/libs/clusterer-0.1-all.jar params.yml $xml $pdf $out > /dev/null
    fi
done
