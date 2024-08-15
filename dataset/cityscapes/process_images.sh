#!/bin/bash
if [ ! -d "cityscapesScripts" ]; then
    git clone https://github.com/mcordts/cityscapesScripts.git
    echo "Repo already cloned"
fi

#Set environmental variable necessary to know where is the dataset necessary for processing the images
export CITYSCAPES_DATASET=$(pwd)/data

#run the python script to process imagess
python $(pwd)/cityscapesScripts/cityscapesscripts/preparation/createTrainIdLabelImgs.py

#echo $CITYSCAPES_DATASET
