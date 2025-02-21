#!/bin/bash

mkdir data
mkdir data/hutubs
mkdir data/riec
cd data/hutubs
wget https://depositonce.tu-berlin.de/bitstream/11303/9429/3/HRIRs.zip
unzip HRIRs.zip
cd ../riec
wget https://www.riec.tohoku.ac.jp/pub/hrtf/data/RIEC_hrtf_all.zip
unzip RIEC_hrtf_all.zip
cd ../..