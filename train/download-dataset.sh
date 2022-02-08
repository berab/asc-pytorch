#!/bin/bash

mkdir data_2020
for i in {1..21}
do
    wget https://zenodo.org/record/3670185/files/TAU-urban-acoustic-scenes-2020-3class-development.audio.$i.zip?download=1 -O ./data_2020/audio_$i.zip
    
    unzip ./data_2020/audio_$i.zip -d ./data_2020/
    rm -f ./data_2020/audio_$i.zip
done

for i in {1..13}
do
    wget https://zenodo.org/record/3685835/files/TAU-urban-acoustic-scenes-2020-3class-evaluation.audio.$i.zip?download=1 -O ./data_2020/eval_$i.zip
    
    unzip ./data_2020/eval_$i.zip -d ./data_2020/
    rm -f ./data_2020/eval_$i.zip
done

wget https://zenodo.org/record/3670185/files/TAU-urban-acoustic-scenes-2020-3class-development.doc.zip?download=1 -O ./data_2020/doc.zip
wget https://zenodo.org/record/3670185/files/TAU-urban-acoustic-scenes-2020-3class-development.meta.zip?download=1 -O ./data_2020/meta.zip

unzip ./data_2020/meta.zip -d ./data_2020/
unzip ./data_2020/doc.zip -d ./data_2020/
rm -f ./data_2020/meta.zip ./data_2020/doc.zip

mv ./data_2020/TAU-urban-acoustic-scenes-2020-3class-development/* ./data_2020/ && rm -d ./data_2020/TAU-urban-acoustic-scenes-2020-3class-development
mv ./data_2020/TAU-urban-acoustic-scenes-2020-3class-evaluation/audio/* ./data_2020/audio/ && rm -rf ./data_2020/TAU-urban-acoustic-scenes-2020-3class-evaluation
