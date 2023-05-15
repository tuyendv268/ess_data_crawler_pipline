from glob import glob
from tqdm import tqdm
import logging
import os

"""
  preparing dataset for training nemo speaker diarization
"""

def create_rttmlist_and_wavlist(input_path, output_path):
    # clean output dir
    os.system(f'rm -r {output_path}/*')
    logging.info(f'cleaned {output_path}')
    # create rttm list and wav list
    rttms, wavs = [], [] 
    for wav in glob(input_path+"/*.wav"):
        file = wav.replace(".wav", "")
        rttms.append(file+".rttm")
        wavs.append(file+".wav")
      
    with open(f"{output_path}/rttm_list.txt", "w", encoding="utf-8") as tmp:
        tmp.write("\n".join(rttms))
        logging.info(f'saved: {output_path}/rttm_list.txt')
        
    with open(f"{output_path}/wav_list.txt", "w", encoding="utf-8") as tmp:
        tmp.write("\n".join(wavs))
        logging.info(f'saved: {output_path}/wav_list.txt')
        
def convert_to_nemo_data_format(path):
    logging.info(f'create msdd_data.json')
    os.system(f'python ../nemo/scripts/speaker_tasks/pathfiles_to_diarize_manifest.py \
        --paths2audio_files="{path}/wav_list.txt" \
        --paths2rttm_files="{path}/rttm_list.txt" \
        --manifest_filepath="{path}/msdd_data.json"')
    
    logging.info(f'create msdd train dataset')
    os.system(f'python ../nemo/scripts/speaker_tasks/create_msdd_train_dataset.py \
        --input_manifest_path="{path}/msdd_data.json" \
        --output_manifest_path="{path}/msdd_data.50step.json" \
        --pairwise_rttm_output_folder="{path}" \
        --window 0.5 \
        --shift 0.25 \
        --step_count 50')

def main(input_path, output_path):
    create_rttmlist_and_wavlist(
        input_path=input_path,
        output_path=output_path
    )
    convert_to_nemo_data_format(
        path=output_path
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', default="../outputs/simulated_datas" ,type=str, help='path')
    parser.add_argument('--output_path', default="../outputs/diar_datas/valid",type=str, help='path')

    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    
    main(input_path=args.input_path, output_path=args.output_path)

    