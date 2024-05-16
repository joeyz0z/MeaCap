import os
import json
import argparse
from PIL import Image
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from vl_models.CLIP_ViP.src.datasets.dataloader import init_transform_dict_simple
from vl_models.CLIP_ViP.src.datasets.sample_frames import SampleFrames
from decord import VideoReader
from decord import cpu, gpu


class Videodata_vip(Dataset):
    def __init__(self, dir_path, args, match_model):
        self.dir_path = dir_path
        self.video_name_list = os.listdir(dir_path)
        self.args = args
        self.match_model = match_model

        with open(self.args.vip_cfg, 'r') as j:
            json_cfg = json.load(j)
        self.cfg = argparse.Namespace(**json_cfg)

    def __getitem__(self, idx):
        video_name = self.video_name_list[idx]
        video_item_path = os.path.join(self.dir_path, video_name)
        video = self.load_video(video_item_path)
        video = video.to(self.args.device)
        video_embeds = self.match_model.forward_vision(video)
        return video_embeds, video_name

    def __len__(self):
        return len(self.video_name_list)

    def load_video(self, vis_path):
        mode = 'test'
        transform = init_transform_dict_simple(video_res=self.cfg.video_res,
                                               input_res=self.cfg.input_res)[mode]

        # transform = init_transform_dict_simple()[mode]

        vr = VideoReader(vis_path, ctx=cpu(0))
        total_frame_num = len(vr)

        frame_idx = self.get_sample_idx(total_frame_num)
        img_array = vr.get_batch(frame_idx)  # (n_clips*num_frm, H, W, 3)

        # img_array = torch.from_numpy(img_array)
        # img_array_np = img_array.asnumpy()
        img_array_np = img_array.numpy()
        # img_array = img_array.permute(0, 3, 1, 2).float() / 255.
        img_array_np = img_array_np.transpose(0, 3, 1, 2) / 255.
        img_array = torch.from_numpy(img_array_np)
        # img_array = np.transpose(img_array, (0, 3, 1, 2))
        # img_array = array(img_array_np)
        img_array = transform(img_array)
        img_array = img_array.unsqueeze(0)
        # img = img_array.cuda()

        return img_array

    def get_sample_idx(self, total_frame_num):
        """
        sample rate > 0: use SampleFrames, loop default
        sample rate = 0: uniform sampling, temporal jittering
        """
        sample_rate = self.cfg.sample_rate
        # sample_rate = 0
        n_clips = self.cfg.test_n_clips
        # n_clips = 1
        num_frm = self.cfg.test_num_frms
        # num_frm = 12
        mode = 'test'
        frame_sampler = SampleFrames(clip_len=num_frm,
                                     frame_interval=sample_rate,
                                     num_clips=n_clips,
                                     temporal_jitter=True)
        if sample_rate > 0:
            results = {"total_frames": total_frame_num,
                       "start_index": 0}
            results = frame_sampler(results)
            return results["frame_inds"]
        elif sample_rate == 0:
            if hasattr(self.cfg, "sample_jitter") and self.cfg.sample_jitter and mode == "train":
                interval = int(total_frame_num / (n_clips * num_frm - 1))
                start = np.random.randint(0, interval + 1)
                end = np.random.randint(total_frame_num - 1 - interval, total_frame_num)
                return np.linspace(start, end, n_clips * num_frm).astype(int)
            else:
                return np.linspace(0, total_frame_num - 1, n_clips * num_frm).astype(int)


def collate_video_vip(batch_data):
    video_batch_list = list()
    name_batch_list = list()
    for unit in batch_data:
        video_batch_list.append(unit[0].squeeze(0))
        name_batch_list.append(unit[1])
    video_batch = torch.stack(video_batch_list)
    return video_batch, name_batch_list
