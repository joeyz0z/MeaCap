import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


class Imgdata(Dataset):
    def __init__(self, dir_path, match_model):
        self.dir_path = dir_path
        self.img_name_list = os.listdir(dir_path)
        self.match_model = match_model

    def __getitem__(self, idx):
        img_name = self.img_name_list[idx]
        img_item_path = os.path.join(self.dir_path, img_name)
        img = Image.open(img_item_path).convert("RGB")
        img_embeds = self.match_model.compute_image_representation_from_image_instance(img)
        return img_embeds, img_name

    def __len__(self):
        return len(self.img_name_list)


def collate_img(batch_data):
    img_embeds_batch_list = list()
    name_batch_list = list()
    for unit in batch_data:
        img_embeds_batch_list.append(unit[0])
        name_batch_list.append(unit[1])
    img_batch_embeds = torch.stack(img_embeds_batch_list).squeeze(1)

    return img_batch_embeds, name_batch_list