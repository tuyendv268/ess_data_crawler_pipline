#!/bin/bash
NEMO_ROOT="nemo" ;
mkdir datas
data_path="datas/inference.txt"
ls /home/tuyendv/Desktop/codes/ess_data_crawler_pipline/outputs/speaker_diarization/*/* > ${data_path}
python ../$NEMO_ROOT/scripts/speaker_tasks/filelist_to_manifest.py \
    --filelist ${data_path} \
    --id -2 \
    --out inference.json
