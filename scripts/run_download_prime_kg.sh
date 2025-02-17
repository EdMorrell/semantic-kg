#!/bin/bash

DPATH=${1:-datasets/prime_kg/}

if [ ! -d $DPATH ]; then
    mkdir -p $DPATH
fi


if [ ! -f $DPATH/nodes.csv ]; then
    echo "Downloading PrimeKG nodes to $DPATH"
    curl -L https://dataverse.harvard.edu/api/access/datafile/6180617 -o $DPATH/nodes.csv
fi


if [ ! -f $DPATH/edges.csv ]; then
    echo "Downloading PrimeKG edges to $DPATH"
    curl -L https://dataverse.harvard.edu/api/access/datafile/6180616 -o $DPATH/edges.csv
fi