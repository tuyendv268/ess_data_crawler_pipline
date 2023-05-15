#!/bin/bash
NEMO_ROOT="nemo" ;
mkdir datas
data_path="datas/datas.txt"

train_path="datas/train.txt"
test_path="datas/test.txt"
valid_path="datas/valid.txt"

train=10
test=1
valid=1

find ../datas/youtube_datas -iname "*.wav" |shuf > ${data_path} ;
total_lines=$(wc -l < "$data_path")

n_train=$((total_lines * train / (train + test + valid)))
n_test=$((total_lines * test / (train + test + valid)))
n_valid=$((total_lines * valid / (train + test + valid)))

head -n "$n_train" "$data_path" > "$train_path"
tail -n "+$((n_train + 1))" "$data_path" | head -n "$n_test" > "$test_path"
tail -n "+$((n_train + n_test + 1))" "$data_path" | head -n "$n_valid" > "$valid_path"

python ../$NEMO_ROOT/scripts/speaker_tasks/filelist_to_manifest.py \
    --filelist ${test_path} \
    --id -2 \
    --out datas/test.json

python ../$NEMO_ROOT/scripts/speaker_tasks/filelist_to_manifest.py \
    --filelist ${train_path} \
    --id -2 \
    --out datas/train.json

python ../$NEMO_ROOT/scripts/speaker_tasks/filelist_to_manifest.py \
    --filelist ${valid_path} \
    --id -2 \
    --out datas/valid.json
