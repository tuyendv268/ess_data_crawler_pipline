import pickle as pkl
import pandas as pd
import soundfile as sf
import os
import torch
import librosa
from nemo.collections.asr.parts.utils.nmesc_clustering import COSclustering

def load_pkl_file(path):
    with open(path, "rb") as handle:
        pkl_content = pkl.load(handle)
    return pkl_content

def concat_scale_dict(scale_dict_1, scale_dict_2):
    for key in scale_dict_1.keys():
        scale_dict_1[key]["embeddings"] = torch.cat((scale_dict_1[key]["embeddings"], scale_dict_2[key]["embeddings"]), dim=0)
        scale_dict_1[key]["time_stamps"] += scale_dict_2[key]["time_stamps"]
        if key == 4:
            scale_dict_1[key]["mapping"] += scale_dict_2[key]["mapping"]
    return scale_dict_1

def rttms_to_files(rttms, path):
    for rttm in rttms:
        rttm_name = rttm["wav_segment"][0] + ".rttm"
        rttm_path = os.path.join(path, rttm_name)
        
        rttm.to_csv(rttm_path, sep=" ", na_rep="NaN",header=False, index=False)
        
        print(f'saved: ', rttm_path)


def get_global_embs_and_time_stamps(embs_and_timestamps_path, number_embedding_per_speaker, duration):
    embs_and_timestamps = load_pkl_file(embs_and_timestamps_path)
    
    global_embs_and_timestamps = {}
    multiscale_weights, scale_dict = None, None
    # number_embedding_per_speaker = 512
    # duration = 600.0
    
    segment_offset = 0
    global_indexs = []
    for segment in range(len(embs_and_timestamps.keys())):
        key_1 = f"sub_segment_{segment}"
        value_1 = embs_and_timestamps[key_1]
        length = len(embs_and_timestamps[key_1]["scale_dict"][4]["time_stamps"])
        offset = segment * duration
        
        if multiscale_weights == None:
            multiscale_weights = value_1["multiscale_weights"]
        
        # add offset to each scale time stamp
        for id_ in range(5):
            time_stamps = value_1["scale_dict"][id_]["time_stamps"]
                    
            if id_ != 4:
                time_stamps = [[float(line.split()[0]), float(line.split()[1])] for line in time_stamps]
                time_stamps_df = pd.DataFrame(time_stamps, columns=["start", "end"])
                time_stamps_df[["start", "end"]] = time_stamps_df[["start", "end"]].apply(func=lambda x: x+ offset, axis=1)
            else:
                time_stamps = [[float(line.split()[0]), float(line.split()[1]), line.split()[2]] for line in time_stamps]
                time_stamps_df = pd.DataFrame(time_stamps, columns=["start", "end", "spk_label"])
                sampled_indexs = []
                for _, group in time_stamps_df.groupby("spk_label"):
                    if number_embedding_per_speaker <= group.shape[0]:
                        sampled_indexs += group.sample(number_embedding_per_speaker).index.to_list()
                    else:
                        sampled_indexs += group.index.tolist()
                sampled_indexs.sort()
                time_stamps_df = time_stamps_df.loc[sampled_indexs]
                time_stamps_df[["start", "end"]] = time_stamps_df[["start", "end"]].apply(func=lambda x: x+ offset, axis=1)    
            
            new_time_stamps = []
            # print(key_1, id_ ,time_stamps_df)
            for index in time_stamps_df.index:
                new_time_stamps.append(" ".join([str(ele) for ele in time_stamps_df.loc[index].values.tolist()]))
            
            # print(new_time_stamps)
            embs_and_timestamps[key_1]["scale_dict"][id_]["time_stamps"] = new_time_stamps
        global_indexs += [index + segment_offset for index in sampled_indexs.copy()]
        print("segment_offset: ", length)
        segment_offset += length
        
        value_1["scale_dict"][4]["mapping"] = [key_1]* len(sampled_indexs)
        value_1["scale_dict"][4]["embeddings"] = value_1["scale_dict"][4]["embeddings"][sampled_indexs]
        if scale_dict == None:
            scale_dict = value_1["scale_dict"]
        else:
            scale_dict = concat_scale_dict(scale_dict, value_1["scale_dict"])
            
    global_embs_and_timestamps["multiscale_weights"] = multiscale_weights
    global_embs_and_timestamps["scale_dict"] = scale_dict
    
    return global_embs_and_timestamps, global_indexs

def map_local_speaker_vs_global_speaker(mapping_dict, rttm_dir):
    rttm_files = os.listdir(rttm_dir)
    rttms = []
    for rttm_file in rttm_files:
        rttm_abs_path = os.path.join(rttm_dir, rttm_file)
        names = ["type", "wav_segment", "o1", "offset", "duration","o2", "o3",  "spk_label", "o4", "o5"]
        rttm_df = pd.read_csv(rttm_abs_path, sep="\s+", names=names)
        
        key = rttm_file.replace(".rttm", "")
        for index in rttm_df.index:
            rttm_df["spk_label"][index] = mapping_dict[key][rttm_df["spk_label"][index]]
        
        rttms.append(rttm_df)
        
    return rttms

def global_clustering(embs_and_timestamps_path, number_embedding_per_spk, duration, subsegment_labels_path, predicted_rttm_path, output_rttm_dir):
    """perform sampling embedding and global clustering

    Args:
        embs_and_timestamps_path (_type_): _description_
        number_embedding_per_spk (_type_): _description_
        duration (_type_): _description_
        subsegment_labels_path (_type_): _description_
        predicted_rttm_path (_type_): _description_
        output_rttm_dir (_type_): _description_
    """
    global_embs_and_timestamps, global_indexs = get_global_embs_and_time_stamps(
        embs_and_timestamps_path=embs_and_timestamps_path,
        duration=duration,
        number_embedding_per_speaker=number_embedding_per_spk
        )
    # with open("temp.pkl", "wb") as handle:
    #     pkl.dump(global_embs_and_timestamps, handle)
    # global_embs_and_timestamps, indexs = sample_base_scale_embedding(global_embs_and_timestamps)
    print("###################################")
    print("        Global Clustering")
    print("####################################")
    cluster_labels = COSclustering(
        uniq_embs_and_timestamps=global_embs_and_timestamps,
        oracle_num_speakers=None,
        max_num_speaker=12,
        enhanced_count_thres=80,
        max_rp_threshold=0.25,
        maj_vote_spk_count=False,
        sparse_search_volume=30,
        cuda=True,
    )
    
    clustered_labels = [f"speaker_{index}" for index in cluster_labels]
    
    names = ["wav_id", "start", "end", "label"]
    # label_path = "outputs/speaker_outputs/subsegments_scale4_cluster.label"
    labels = pd.read_csv(subsegment_labels_path, sep="\s+", names=names)
    # print(labels)
    labels = labels.loc[global_indexs].reset_index()
    # print(labels)
    labels["global_label"] = [clustered_labels[index] for index in labels.index]
    mapping_dict = {}
    
    for name, group in labels.groupby("wav_id"):
        # print(group)
        temp = {}
        for sub_name, sub_group in group.groupby("label"):
            global_spk_label = sub_group.groupby("global_label").count()
            global_spk_label = global_spk_label.idxmax()[0]
            temp[sub_name] = global_spk_label
        mapping_dict[name] = temp
    print("mapping_dict: ", mapping_dict)
    mapped_rttms = map_local_speaker_vs_global_speaker(mapping_dict, predicted_rttm_path)
    
    rttms_to_files(rttms=mapped_rttms, path=output_rttm_dir)

def convert_rttms_to_segments(input_path, output_path):
    rttm_path = f'{input_path}/rttms'
    wav_path = f'{input_path}/wavs'
    os.system(f"rm -r {output_path}/*")
    for rttm_file in os.listdir(rttm_path):
        abs_rttm_path = os.path.join(rttm_path, rttm_file)
        abs_wav_path = os.path.join(wav_path, rttm_file.replace(".rttm", ".wav"))
        
        header = ["type", "wav_name", "o1", "offset", "duration", "o2", "o3", "speaker","o4", "o5"]
        rttm = pd.read_csv(abs_rttm_path, sep="\s+", names=header)
        wav, sr = librosa.load(abs_wav_path, sr = 16000)

        spk2utt = {}
        for index in rttm.index:
            if rttm["speaker"][index] not in spk2utt:
                spk2utt[rttm["speaker"][index]] = []
            else:
                start = int(rttm["offset"][index] * sr)
                end = int((rttm["offset"][index] + rttm["duration"][index]) * sr)
                spk2utt[rttm["speaker"][index]].append(wav[start:end])

        for spk, wavs in spk2utt.items():
            spk_wavs = []
            tmp_path = os.path.join(output_path, spk)
            if not os.path.exists(tmp_path):
                os.mkdir(tmp_path)
            for wav in wavs:
                spk_wavs += list(wav)
            abs_path = os.path.join(tmp_path, f"{rttm_file}_{spk}.wav")
            sf.write(abs_path, spk_wavs, sr)
            print("saved: ", abs_path)
                
def get_silence_from_rttms(input_path, output_path):
    rttm_dir = f'{input_path}/rttms'
    wav_dir = f'{input_path}/wavs'
    rttm_names = ["type", "wav_segment", "o1", "offset", "duration","o2", "o3",  "cluster_label", "o4", "o5"]
    sample_rate = 16000

    for file in os.listdir(rttm_dir):
        rttm_file = os.path.join(rttm_dir, file)
        wav_file = os.path.join(wav_dir, file.replace(".rttm", ".wav"))
        
        rttm_df = pd.read_csv(rttm_file, sep="\s+", names=rttm_names)
        wav, _ = librosa.load(wav_file, sr=sample_rate)
        
        start, offset = 0, 0
        wavs = []
        for index in rttm_df.index:
            offset = rttm_df["offset"][index]
            duration = rttm_df["duration"][index]
            wavs += list(wav[int(start*sample_rate):int(offset*sample_rate)])
            start=offset + duration
        sf.write(f'{output_path}/{file}_silence.wav', wavs, samplerate=sample_rate)
        print(f'saved: {output_path}/{file}_silence.wav')
        
if __name__ == "__main__":
    embs_and_timestamps_path = "embs_and_timestamps.pkl"
    predicted_rttm_path = "outputs/pred_rttms"
    output_rttm_dir = "temps"
    subsegment_labels_path = "outputs/speaker_outputs/subsegments_scale4_cluster.label"
    duration = 600.0 # max segment length in second
    number_embedding_per_spk = 512 # number embedding per speaker per segment wav
    
    global_clustering(
        embs_and_timestamps_path=embs_and_timestamps_path,
        predicted_rttm_path=predicted_rttm_path,
        output_rttm_dir=output_rttm_dir,
        subsegment_labels_path=subsegment_labels_path,
        duration=duration,
        number_embedding_per_spk=number_embedding_per_spk
    )