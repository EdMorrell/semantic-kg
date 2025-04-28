#!/bin/bash

DPATH=${1:-datasets/findkg/}
GH_ROOT_DIR="https://raw.githubusercontent.com/xiaohui-victor-li/FinDKG/refs/heads/main/FinDKG_dataset/FinDKG/"

if [ ! -d $DPATH ]; then
    mkdir -p $DPATH
fi

# Function to download a file from url if it does not exist
download_file() {
    url=$1
    file=$2
    if [ ! -f $file ]; then
        echo "Downloading $url to $file"
        curl -L $url -o $file
    fi
}

download_file $GH_ROOT_DIR/train.txt $DPATH/train.txt
download_file $GH_ROOT_DIR/entity2id.txt $DPATH/entity2id.txt
download_file $GH_ROOT_DIR/relation2id.txt $DPATH/relation2id.txt