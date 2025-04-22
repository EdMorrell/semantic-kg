#!/bin/bash

DPATH=${1:-datasets/codex/}
LANG="en"
GH_ROOT_DIR="https://raw.githubusercontent.com/tsafavi/codex/refs/heads/master/data"

if [ ! -d $DPATH ]; then
    mkdir -p $DPATH
fi

# Function to download a file from url if it does not exist
download_file() {
    url=$1
    file=$2
    if [ ! -f $file ]; then
        echo "Downloading $url to $file"
        mkdir -p "$(dirname "$file")"
        curl -L $url -o $file
    fi
}

# Get entity and relation json files
download_file $GH_ROOT_DIR/entities/$LANG/entities.json $DPATH/entities/$LANG/entities.json
download_file $GH_ROOT_DIR/relations/$LANG/relations.json $DPATH/relations/$LANG/relations.json

# Get type information
download_file $GH_ROOT_DIR/types/entity2types.json $DPATH/types/entity2types.json
download_file $GH_ROOT_DIR/types/$LANG/types.json $DPATH/types/$LANG/types.json

for SIZE in l m s; do
    for SPLIT in train valid test; do
        download_file $GH_ROOT_DIR/triples/codex-$SIZE/$SPLIT.txt $DPATH/triples/codex-$SIZE/$SPLIT.txt
    done
done
