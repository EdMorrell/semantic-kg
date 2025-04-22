#!/bin/bash

DPATH=${1:-datasets/globi/}

if [ ! -d $DPATH ]; then
    mkdir -p $DPATH
fi

if [[ ! -f $DPATH/interactions.csv.gz && ! -f $DPATH/interactions.csv ]]; then
    echo "Downloading Globi interactions to $DPATH"
    curl -L https://zenodo.org/records/14640564/files/interactions.csv.gz -o $DPATH/interactions.csv.gz
fi

if [ ! -f $DPATH/interactions.csv ]; then
    echo "Unzipping to CSV"
    gzip -v -d $DPATH/interactions.csv.gz
fi