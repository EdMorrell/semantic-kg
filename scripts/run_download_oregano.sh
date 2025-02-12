#!/bin/bash

DPATH=${1:-datasets/oregano/}


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


# Triples
download_file https://ndownloader.figshare.com/files/42612700 $DPATH/OREGANO_V2.1.tsv

# Metadata
download_file https://ndownloader.figshare.com/files/42700234 $DPATH/oreganov2.1_metadata_complete.ttl

# Activity
download_file https://ndownloader.figshare.com/files/42612697 $DPATH/ACTIVITY.tsv

# Compound
download_file https://ndownloader.figshare.com/files/42612676 $DPATH/COMPOUND.tsv

# Diseases
download_file https://ndownloader.figshare.com/files/42612685 $DPATH/DISEASES.tsv

# Effects
download_file https://ndownloader.figshare.com/files/42612694 $DPATH/EFFECT.tsv

# Genes
download_file https://ndownloader.figshare.com/files/42612673 $DPATH/GENES.tsv

# Indication
download_file https://ndownloader.figshare.com/files/42612691 $DPATH/INDICATION.tsv

# Pathways
download_file https://ndownloader.figshare.com/files/42612688 $DPATH/PATHWAYS.tsv

# Phenotype
download_file https://ndownloader.figshare.com/files/42612679 $DPATH/PHENOTYPES.tsv

# Side-effects
download_file https://ndownloader.figshare.com/files/42612670 $DPATH/SIDE_EFFECT.tsv

# Targets
download_file https://ndownloader.figshare.com/files/42612682 $DPATH/TARGET.tsv
