#! /bin/bash
mkdir $1
cd $1
apt-marker lifecam19 22203 > marker_22203_$1.txt
cd ..