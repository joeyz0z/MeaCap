import os
from PIL import Image
import torch
import cv2
from torch.utils.data import Dataset, DataLoader


class Videodata_avg(Dataset):
    def __init__(self, dir_path, clip):
        self.dir_path = dir_path
        self.video_name_list = os.listdir(dir_path)
        self.clip = clip

    def __getitem__(self, idx):
        video_name = self.video_name_list[idx]
        video_item_path = os.path.join(self.dir_path, video_name)
        video_frames = get_clip_video_frames(video_item_path, self.clip)
        with torch.no_grad():
            frames_fts = self.clip.compute_frame_representation_from_tensor(video_frames).detach()
            frames_fts = torch.nn.functional.normalize(frames_fts, dim=-1).detach()

            similiarities = frames_fts @ frames_fts.T
            image_fts, selected_frames_indices = filter_video(frames_fts, similiarities)

        # img = Image.open(img_item_path).convert("RGB")
        return image_fts, video_name

    def __len__(self):
        return len(self.video_name_list)


def collate_video_avg(batch_data):
    video_path_batch_list = list()
    name_batch_list = list()
    for unit in batch_data:
        video_path_batch_list.append(unit[0])
        name_batch_list.append(unit[1])
    batch_frames_emb_list = list()
    for batch in range(len(video_path_batch_list)):
        image_fts = video_path_batch_list[batch]
        frame_avg = torch.mean(image_fts, dim=0)
        batch_frames_emb_list.append(frame_avg)
    batch_frames_emb = torch.stack(batch_frames_emb_list)
    return batch_frames_emb, name_batch_list


def get_clip_video_frames(video_path, clip):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    sample_time = fps // 3
    imgs = []

    i = 0
    while (cap.isOpened()):
        ret, cv2_im = cap.read()

        if ret and i % sample_time == 0:
            converted = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(converted)
            imgs.append(pil_im)
        elif not ret:
            break

        i += 1

    cap.release()

    images = clip.processor(images=imgs, return_tensors='pt')['pixel_values']
    # images = torch.cat([clip.processor(images=x, return_tensors='pt').unsqueeze(0) for x in imgs])

    return images


def filter_video(image_fts, similiarities):
    threshold = 0.9
    groups = []
    curr_group = []
    for i in range(similiarities.size(0)):
        if len(curr_group) == 0:
            curr_group.append(i)

        if i + 1 == similiarities.size(0):
            if len(curr_group) >= 1:
                groups.append(curr_group)
            break

        if similiarities[curr_group[0]][i + 1] > threshold:
            curr_group.append(i + 1)
        else:
            if len(curr_group) >= 1:
                groups.append(curr_group)
            curr_group = []

    result_features = []
    selected_indices = []
    if len(groups) >= 1:
        for i, group in enumerate(groups):
            result_features.append(image_fts[group[0]])
            selected_indices.append(group[0])

    return torch.stack(result_features), selected_indices
