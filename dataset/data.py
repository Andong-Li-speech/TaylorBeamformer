import json
import os
import numpy as np
from random import shuffle
from torch.utils.data import Dataset, DataLoader
import random
import soundfile as sf
import librosa as lib
from utils.utils import BatchInfo, pad_to_longest, logger_print

class InstanceDataset(Dataset):
    def __init__(self,
                 mix_file_path,
                 bf_file_path,
                 target_file_path,
                 mix_json_path,
                 bf_json_path,
                 target_json_path,
                 batch_size,
                 is_check,
                 is_shuffle,
                 is_variance_norm,
                 is_chunk,
                 chunk_length,
                 sr,
                 ):
        super(InstanceDataset, self).__init__()
        self.mix_file_path = mix_file_path
        self.bf_file_path = bf_file_path
        self.target_file_path = target_file_path
        self.mix_json_path = mix_json_path
        self.bf_json_path = bf_json_path
        self.target_json_path = target_json_path
        self.batch_size = batch_size
        self.is_check = is_check
        self.is_shuffle = is_shuffle
        self.is_variance_norm = is_variance_norm
        self.is_chunk = is_chunk
        self.chunk_length = chunk_length
        self.sr = sr

        with open(mix_json_path, "r") as f:
            mix_json_list = json.load(f)
        with open(bf_json_path, "r") as f:
            bf_json_list = json.load(f)
        with open(target_json_path, "r") as f:
            target_json_list = json.load(f)

        # sort
        mix_json_list.sort()
        target_json_list.sort()
        bf_json_list.sort()

        if is_check:
            self.check_align(mix_json_list, target_json_list)
            self.check_align(mix_json_list, bf_json_list)

        if is_shuffle:
            random.seed(1234)  # fixed for reproducibility
            zipped_list = list(zip(mix_json_list, bf_json_list, target_json_list))
            shuffle(zipped_list)
            # the first type
            # mix_json_list, target_json_list = zip(*zipped_list)  # mix_json_list and target_json_list are tuple type
            # the second type
            json_list = list(map(list, zip(*zipped_list)))  # json_list (list:2)
            [mix_json_list, bf_json_list, target_json_list] = json_list

        mix_minibatch, bf_minibatch, target_minibatch = [], [], []
        start = 0
        while True:
            end = min(len(mix_json_list), start+batch_size)
            mix_minibatch.append(mix_json_list[start:end])
            bf_minibatch.append(bf_json_list[start:end])
            target_minibatch.append(target_json_list[start:end])
            start = end
            if end == len(mix_json_list):
                break
        self.mix_minibatch, self.bf_minibatch, self.target_minibatch = mix_minibatch, bf_minibatch, target_minibatch
        self.length = len(mix_minibatch)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        mix_minibatch_list = self.mix_minibatch[index]
        bf_minibatch_list = self.bf_minibatch[index]
        target_minibatch_list = self.target_minibatch[index]
        mix_wav_list, bf_wav_list, target_wav_list, wav_len_list = [], [], [], []
        for id in range(len(mix_minibatch_list)):
            mix_filename = mix_minibatch_list[id]
            bf_filename = bf_minibatch_list[id]
            target_filename = target_minibatch_list[id]
            extracted_filename_from_mix = "_".join(mix_filename.split("_")[:-1])
            extracted_filename_from_target = "_".join(target_filename.split("_")[:-1])
            assert extracted_filename_from_mix == extracted_filename_from_target
            # read speech
            mix_wav, mix_sr = sf.read(os.path.join(self.mix_file_path, "{}.wav".format(mix_filename)))  # (L,M)
            bf_wav, bf_sr = sf.read(os.path.join(self.bf_file_path, "{}.wav".format(bf_filename)))
            target_wav, tar_sr = sf.read(os.path.join(self.target_file_path, "{}.wav".format(target_filename)))
            if mix_sr != self.sr or bf_sr != self.sr or tar_sr != self.sr:
                mix_wav, bf_wav, target_wav = lib.resample(mix_wav, mix_sr, self.sr), \
                                              lib.resample(bf_wav, bf_sr, self.sr),\
                                              lib.resample(target_wav, tar_sr, self.sr)
            if self.is_variance_norm:
                ref_mic = np.mean(mix_wav, axis=-1)   # mean mic is selected as the ref for normalization
                c = np.sqrt(len(ref_mic) / np.sum(ref_mic ** 2.0))
                mix_wav, bf_wav, target_wav = mix_wav * c, bf_wav * c, target_wav * c
            if self.is_chunk and (len(ref_mic) > int(self.sr*self.chunk_length)):
                wav_start = random.randint(0, len(ref_mic)-int(self.sr*self.chunk_length))
                mix_wav = mix_wav[wav_start:wav_start+int(self.sr*self.chunk_length), :]
                bf_wav = bf_wav[wav_start:wav_start+int(self.sr*self.chunk_length)]
                target_wav = target_wav[wav_start:wav_start+int(self.sr*self.chunk_length), :]
            mix_wav_list.append(mix_wav)
            bf_wav_list.append(bf_wav)
            target_wav_list.append(target_wav)
            wav_len_list.append(mix_wav.shape[0])
        return mix_wav_list, bf_wav_list, target_wav_list, wav_len_list

    @staticmethod
    def check_align(mix_list, target_list):
        logger_print("checking.................")
        is_ok = 1
        mix_error_list, target_error_list = [], []
        for i in range(len(mix_list)):
            extracted_filename_from_mix = "_".join(mix_list[i].split("_")[:-1])
            extracted_filename_from_target = "_".join(target_list[i].split("_")[:-1])
            if extracted_filename_from_mix != extracted_filename_from_target:
                is_ok = 0
                mix_error_list.append(extracted_filename_from_mix)
                target_error_list.append(extracted_filename_from_target)
        if is_ok == 0:
            for i in range(min(len(mix_error_list), len(target_error_list))):
                print("mix_file_name:{}, target_file_name:{}".format(mix_error_list[i],
                                                                     target_error_list[i]))
            raise Exception("Datasets between mix and target are not aligned!")
        else:
            logger_print("checking finished..............")

class InstanceDataloader(object):
    def __init__(self,
                 data_set,
                 num_workers,
                 pin_memory,
                 drop_last,
                 shuffle,
                 ):
        self.data_set = data_set
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.shuffle = shuffle

        self.data_loader = DataLoader(dataset=data_set,
                                      num_workers=num_workers,
                                      pin_memory=pin_memory,
                                      drop_last=drop_last,
                                      shuffle=shuffle,
                                      collate_fn=self.collate_fn,
                                      batch_size=1
                                      )
    @staticmethod
    def collate_fn(batch):
        feats, bfs, labels, frame_mask_list = pad_to_longest(batch)
        return BatchInfo(feats, bfs, labels, frame_mask_list)

    def get_data_loader(self):
        return self.data_loader
